# Libraries
from benchmark_utils import make_target_details, make_model, mcmc_sample
from benchmark_utils import run_re_sampler, run_smc_sampler
from sde_sampler.additions.hacking import TrainableWrapper
from sde_sampler.distr.checkerboard import Checkerboard
from sde_sampler.distr.funnel import Funnel
from sde_sampler.distr.rings import Rings
from sde_sampler.eval.metrics import get_metrics
import argparse
import math
import numpy as np
import pickle
import pprint
import random
import time
import torch

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--target_type', type=str)
parser.add_argument('--solver_type', type=str)
parser.add_argument('--smc_n_steps', type=int, default=512)
parser.add_argument('--smc_n_particles', type=int, default=2048)
parser.add_argument('--smc_n_mcmc_steps', type=int, default=64)
parser.add_argument('--smc_n_warmup_mcmc_steps', type=int, default=4096)
parser.add_argument('--re_n_steps', type=int, default=512)
parser.add_argument('--re_batch_size', type=int, default=2048)
parser.add_argument('--re_n_mcmc_steps', type=int, default=64)
parser.add_argument('--re_n_warmup_mcmc_steps', type=int, default=8192)
parser.add_argument('--re_swap_frequency', type=int, default=8)
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--terminal_t_pis', type=float, default=5.0)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check the solver type
if not (args.solver_type in ['pis_orig', 'dds_orig', 'dis_orig', 'cmcd', 'smc', 're']):
    print('Solver type {} not supported.'.format(args.solver_type))
    exit(0)
# Check if the target is right
if not (args.target_type in ['funnel', 'rings', 'checkerboard']):
    print('Target {} not supported.'.format(args.target_type))
    exit(0)

# Save the arguments in a dictionnary
config = vars(args)

# Print the config
pprint.pprint(config)

# Make a Pytorch device
device = torch.device('cuda')

# Set the seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Make a filename
filename = ''
filename += 'target_type_' + args.target_type
filename += '_solver_type_' + args.solver_type
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Build the target distribution
if args.target_type == 'funnel':
    target = Funnel(dim=10)
elif args.target_type == 'rings':
    target = Rings(dim=2)
elif args.target_type == 'checkerboard':
    target = Checkerboard(dim=2, width=4)
else:
    print('Target {} not supported.'.format(args.target_type))
    exit(0)
target.to(device)

# Make the initial point
if args.target_type == 'checkerboard':
    x_init = target.loc.clone()
elif args.target_type == 'rings':
    x_init = target.sample_init_points(32)
else:
    uniform = torch.distributions.Uniform(low=target.domain[:, 0], high=target.domain[:, 1])
    x_init = uniform.sample((128,))

# Sample with MCMC
mcmc_start_time = time.time()
if args.target_type == 'checkerboard':
    # Run the MCMC algorithm
    dataset = mcmc_sample(device, target, x_init, mcmc_type='rwmh', dataset_length=args.dataset_size,
                          n_warmup_steps=2048, skip_chain_per_mode=False, n_chains_per_mode=16).cpu()
else:
    # Make the score
    def target_log_prob_and_grad(x):
        return target.unnorm_log_prob(x).flatten(), target.score(x)
    # Run the MCMC algorithm
    dataset = mcmc_sample(device, target, x_init, dataset_length=args.dataset_size, n_warmup_steps=2048,
                          skip_chain_per_mode=True, target_log_prob_and_grad=target_log_prob_and_grad).cpu()
mcmc_end_time = time.time()

# Compute the mean and variance
ref_fitting_start_time = time.time()
mean = torch.mean(dataset, dim=0).to(device)
var = torch.cov(dataset.T).to(device)
var_diag = torch.var(dataset, dim=0).to(device)
ref_fitting_end_time = time.time()

# Define the solver details
if args.solver_type == 'cmcd':
    solver_details = {'mean': mean, 'var': var}
else:
    sigma_opt = math.sqrt((torch.sum(torch.square(mean)) + var_diag.sum()).cpu().item() / target.dim)
    if args.solver_type == 'pis_orig':
        sigma_opt /= math.sqrt(args.terminal_t_pis)
    solver_details = {'sigma': sigma_opt}

# Run VI-based models
if args.solver_type not in ['smc', 're']:
    # Make the model
    model = make_model(
        solver_type=args.solver_type,
        ref_type='gaussian' if args.solver_type == 'cmcd' else 'default',
        loss_type='lv',
        integrator_type='em',
        model_type='base_zero_init' if args.target_type == 'checkerboard' else 'target_informed_zero_init',
        force_base_zero_init=args.target_type == 'checkerboard',
        time_type='uniform',
        solver_details=solver_details,
        target_details=make_target_details(target_name=args.target_type),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        })
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=True)
    results = model_wrapped.run()
    # Get the samples and the metrics
    samples, metrics = results.samples.cpu(), results.metrics
elif args.solver_type == 'smc':
    # Run the SMC sampler
    sample_start_time = time.time()
    samples = run_smc_sampler(
        mean=mean,
        var=var,
        n_steps=args.smc_n_steps,
        step_size=1e-4,
        n_particles=args.smc_n_particles,
        n_mcmc_steps=args.smc_n_mcmc_steps,
        n_warmup_mcmc_steps=args.smc_n_warmup_mcmc_steps,
        target_log_prob=target.unnorm_log_prob,
        target_score=target.score,
        reweight_threshold=1.0,
        target_acceptance=0.75,
        verbose=False)
    sample_end_time = time.time()
    samples = samples.view((-1, target.dim))
    metrics = get_metrics(target, samples, marginal_dims=[0, 1])
    metrics['eval/sample_time'] = sample_end_time - sample_start_time
else:
    # Run the RE sampler
    sample_start_time = time.time()
    samples = run_re_sampler(
        mean=mean,
        var=var,
        n_steps=args.re_n_steps,
        step_size=1e-4,
        batch_size=args.re_batch_size,
        swap_frequency=args.re_swap_frequency,
        n_mcmc_steps=args.re_n_mcmc_steps,
        n_warmup_mcmc_steps=args.re_n_warmup_mcmc_steps,
        target_log_prob=target.unnorm_log_prob,
        target_score=target.score,
        target_acceptance=0.75,
        verbose=False)
    sample_end_time = time.time()
    samples = samples.view((-1, target.dim))
    metrics = get_metrics(target, samples, marginal_dims=[0, 1])
    metrics['eval/sample_time'] = sample_end_time - sample_start_time

# Save the file
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({
        'config': config,
        'metrics': metrics,
        'times': {
            'mcmc': mcmc_end_time - mcmc_start_time,
            'ref': ref_fitting_end_time - ref_fitting_start_time
        },
        'samples': samples.cpu(),
        'params': {'mean': mean.cpu(), 'var': var.cpu()}
    }, f)
