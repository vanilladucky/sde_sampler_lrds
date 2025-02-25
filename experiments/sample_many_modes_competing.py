# Libraries
from benchmark_utils import make_target_details, make_model, mcmc_sample
from benchmark_utils import run_re_sampler, run_smc_sampler
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.additions.ks import compute_sliced_ks
from sde_sampler.additions.mmd import mmd_median
from sde_sampler.distr.gauss import ManyModes
from sde_sampler.eval.metrics import get_metrics
from sde_sampler.eval.sinkhorn import Sinkhorn
from tqdm import tqdm
import argparse
import itertools
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
parser.add_argument('--solver_type', type=str)
parser.add_argument('--dim_range', type=str, default='8')
parser.add_argument('--n_modes_range', type=str, default='4,8,16,32,64')
parser.add_argument('--mixture_weight_factor_range', type=str, default='3.0')
parser.add_argument('--var_range', type=str, default='0.5')
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

# Save the arguments in a dictionary
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
filename += 'solver_type_' + args.solver_type
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Change the ranges
dim_range = list(map(int, args.dim_range.split(',')))
n_modes_range = list(map(int, args.n_modes_range.split(',')))
mixture_weight_factor_range = list(map(float, args.mixture_weight_factor_range.split(',')))
var_range = list(map(float, args.var_range.split(',')))

# Browse all the dimension
dump_results = []
for dim, n_modes, mixture_weight_factor, variance in tqdm(
        list(itertools.product(dim_range, n_modes_range, mixture_weight_factor_range, var_range))):

    # Build the target
    target = ManyModes(dim=dim, n_modes=n_modes, mixture_weight_factor=mixture_weight_factor, var=variance)
    target.to(device)

    # Sample the target with MCMC
    mcmc_start_time = time.time()
    dataset = mcmc_sample(device, target, target.loc.to(device), dataset_length=args.dataset_size)
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
        force_vp20 = args.solver_type == 'dis_orig'
        model = make_model(
            solver_type=args.solver_type,
            ref_type='gaussian' if args.solver_type == 'cmcd' else 'default',
            loss_type='lv',
            integrator_type='em',
            model_type='target_informed_zero_init',
            time_type='uniform',
            solver_details=solver_details,
            target_details=make_target_details(
                target_name='many_modes', dim=dim, n_modes=n_modes,
                mixture_weight_factor=mixture_weight_factor, var=variance),
            training_details={
                'train_steps': args.train_steps,
                'train_batch_size': args.train_batch_size,
                'eval_batch_size': args.eval_batch_size
            },
            force_vp20=force_vp20)
        # Train the model
        model_wrapped = TrainableWrapper(model, verbose=False)
        results = model_wrapped.run()
        # Build all the metrics
        all_metrics = [results.metrics]
        results_weight_hists = [model.target.compute_mode_count(results.samples).cpu()]
        for _ in range(args.n_sampling_seeds - 1):
            metrics = model_wrapped.trainable.evaluate()
            results_weight_hists.append(model.target.compute_mode_count(metrics.samples).cpu())
            all_metrics.append(metrics.metrics)
    elif args.solver_type == 'smc':
        # Run the SMC sampler
        compute_ot = Sinkhorn()
        all_metrics = []
        results_weight_hists = []
        n_sampling_runs = int((args.eval_batch_size * args.n_sampling_seeds) /
                              (args.smc_n_particles * args.smc_n_mcmc_steps))
        sampling_time = 0.0
        for _ in range(n_sampling_runs):
            start_sampling = time.time()
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
            sampling_time += time.time() - start_sampling
            samples = samples.view((-1, target.dim))
            for samples_b in torch.chunk(samples, int(samples.shape[0] / args.eval_batch_size)):
                samples_target = target.sample((samples_b.shape[0],))
                metrics = get_metrics(target, samples_b, marginal_dims=[0, 1])
                metrics['error/sinkhorn'] = compute_ot(samples_target, samples_b).cpu().item()
                metrics['error/mmd'] = mmd_median(samples_target, samples_b).cpu().item()
                metrics['error/ks'] = compute_sliced_ks(samples_target, samples_b).cpu().item()
                all_metrics.append(metrics)
                results_weight_hists.append(target.compute_mode_count(samples_b).cpu())
    else:
        # Run the RE sampler
        compute_ot = Sinkhorn()
        all_metrics = []
        results_weight_hists = []
        n_sampling_runs = int((args.eval_batch_size * args.n_sampling_seeds) /
                              (args.re_batch_size * args.re_n_mcmc_steps))
        sampling_time = 0.0
        for _ in range(n_sampling_runs):
            start_sampling = time.time()
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
            sampling_time += time.time() - start_sampling
            samples = samples.view((-1, target.dim))
            for samples_b in torch.chunk(samples, int(samples.shape[0] / args.eval_batch_size)):
                samples_target = target.sample((samples_b.shape[0],))
                metrics = get_metrics(target, samples_b, marginal_dims=[0, 1])
                metrics['error/sinkhorn'] = compute_ot(samples_target, samples_b).cpu().item()
                metrics['error/mmd'] = mmd_median(samples_target, samples_b).cpu().item()
                metrics['error/ks'] = compute_sliced_ks(samples_target, samples_b).cpu().item()
                all_metrics.append(metrics)
                results_weight_hists.append(target.compute_mode_count(samples_b).cpu())

    # Change the metrics to a dict
    all_metrics = list_of_dict_2_dict_of_list(all_metrics)
    if args.solver_type in ['re', 'smc']:
        all_metrics['eval/sample_time'] = sampling_time / args.n_sampling_seeds

    # Dump the results
    dump_results.append({
        'metrics': all_metrics,
        'weight_hists': results_weight_hists,
        'times': {
            'mcmc': mcmc_end_time - mcmc_start_time,
            'ref': ref_fitting_end_time - ref_fitting_start_time
        },
        'params': {'dim': dim,
                   'n_modes': n_modes,
                   'mixture_weight_factor': mixture_weight_factor,
                   'var': var},
        'gauss_params': {'mean': mean.cpu(), 'var': var.cpu()}
    })

    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': dump_results,
        }, f)
