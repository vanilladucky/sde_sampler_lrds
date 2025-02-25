# Libraries
from benchmark_utils import make_target_details, make_model, mcmc_sample
from benchmark_utils import run_re_sampler, run_smc_sampler
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.distr.logistic_regression import LogisticRegression
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
parser.add_argument('--data_type', type=str)
parser.add_argument('--solver_type', type=str)
parser.add_argument('--smc_n_steps', type=int, default=128)
parser.add_argument('--smc_n_particles', type=int, default=1024)
parser.add_argument('--smc_n_mcmc_steps', type=int, default=32)
parser.add_argument('--smc_n_warmup_mcmc_steps', type=int, default=1024)
parser.add_argument('--re_n_steps', type=int, default=128)
parser.add_argument('--re_batch_size', type=int, default=1024)
parser.add_argument('--re_n_mcmc_steps', type=int, default=32)
parser.add_argument('--re_n_warmup_mcmc_steps', type=int, default=4096)
parser.add_argument('--re_swap_frequency', type=int, default=8)
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--terminal_t_pis', type=float, default=5.0)
parser.add_argument('--n_sampling_seeds', type=int, default=16)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check the solver type
if not (args.solver_type in ['pis_orig', 'dds_orig', 'dis_orig', 'cmcd', 'smc', 're']):
    print('Solver type {} not supported.'.format(args.solver_type))
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
filename += 'data_type_' + args.data_type
filename += '_solver_type_' + args.solver_type
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Build the target distribution
if args.data_type == 'cancer':
    target = LogisticRegression(dim=32, data_type='cancer', weight_scale=3.75, intercept_mean=31., intercept_scale=2.)
elif args.data_type == 'credit':
    target = LogisticRegression(dim=25, data_type='credit', weight_scale=1.25, intercept_mean=3.25, intercept_scale=0.5)
elif args.data_type == 'sonar':
    target = LogisticRegression(dim=61, data_type='sonar', weight_scale=4.5, intercept_mean=-2.5, intercept_scale=0.5)
elif args.data_type == 'ionosphere':
    target = LogisticRegression(dim=34, data_type='ionosphere', weight_scale=5.25,
                                intercept_mean=4.25, intercept_scale=0.25)
else:
    print('Data type {} not supported.'.format(args.data_type))
    exit(0)
target.to(device)

# Make the initial point by sampling the prior
x_init = torch.concat([
    target.weights_prior.sample((128,)),
    target.intercept_prior.sample((128,)).unsqueeze(-1),
], dim=-1)

# Make the score


def target_log_prob_and_grad(x):
    return target.unnorm_log_prob(x).flatten(), target.score(x)


# Run the MCMC algorithm
mcmc_start_time = time.time()
dataset = mcmc_sample(device, target, x_init, dataset_length=args.dataset_size, n_warmup_steps=4096,
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
        model_type='target_informed_zero_init',
        time_type='uniform',
        solver_details=solver_details,
        target_details=make_target_details(target_name=args.data_type),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        })
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=True)
    results = model_wrapped.run()
    # Build all the metrics
    all_metrics = [results.metrics]
    for _ in range(args.n_sampling_seeds-1):
        metrics = model_wrapped.trainable.evaluate()
        all_metrics.append(metrics.metrics)
elif args.solver_type == 'smc':
    # Run the SMC sampler
    all_metrics = []
    n_sampling_runs = int((args.eval_batch_size * args.n_sampling_seeds) /
                          (args.smc_n_particles * args.smc_n_mcmc_steps))
    start_sampling = time.time()
    for _ in range(n_sampling_runs):
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
            verbose=True)
        samples = samples.view((-1, target.dim))
        metrics = get_metrics(target, samples, marginal_dims=[0, 1])
        all_metrics.append(metrics)
    end_sampling = time.time()
else:
    # Run the RE sampler
    all_metrics = []
    n_sampling_runs = int((args.eval_batch_size * args.n_sampling_seeds) /
                          (args.re_batch_size * args.re_n_mcmc_steps))
    start_sampling = time.time()
    for _ in range(n_sampling_runs):
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
            verbose=True)
        samples = samples.view((-1, target.dim))
        metrics = get_metrics(target, samples, marginal_dims=[0, 1])
        all_metrics.append(metrics)
    end_sampling = time.time()

# Change the metrics to a dict
all_metrics = list_of_dict_2_dict_of_list(all_metrics)
if args.solver_type in ['re', 'smc']:
    all_metrics['eval/sample_time'] = (end_sampling - start_sampling) / args.n_sampling_seeds

# Save the file
with open(args.results_path + '/' + filename, 'wb') as f:
    pickle.dump({
        'config': config,
        'results': {
            'metrics': all_metrics,
            'times': {
                'mcmc': mcmc_end_time - mcmc_start_time,
                'ref': ref_fitting_end_time - ref_fitting_start_time
            },
            'params': {'data_type': args.data_type},
            'gauss_params': {'mean': mean.cpu(), 'var': var.cpu()}
        },
    }, f)
