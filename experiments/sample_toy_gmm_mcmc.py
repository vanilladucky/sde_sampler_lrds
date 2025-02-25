# Libraries
from benchmark_utils import make_target_details, make_model, fit_gmm, mcmc_sample
from sde_sampler.additions.hacking import TrainableWrapper
from sde_sampler.distr.checkerboard import Checkerboard
from sde_sampler.distr.funnel import Funnel
from sde_sampler.distr.rings import Rings
from tqdm import tqdm
import argparse
import itertools
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
parser.add_argument('--gmm_type', type=str)
parser.add_argument('--n_components', type=int)
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--use_mcmc_dataset', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check if it is a ref solver
if 'ref' not in args.solver_type:
    print('solver_type has to be a ref one.')
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
filename += '_gmm_type_' + args.gmm_type
filename += '_n_components_' + str(args.n_components)
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Get the ranges
# loss_type_range = ['kl', 'lv']
# integrator_type_range = ['em', 'ei']
# if args.target_type == 'checkerboard':
#     model_type_range = ['base_zero_init']
# else:
#     model_type_range = [
#         'target_informed_zero_init',
#         'target_informed_langevin_init',
#         'base_zero_init'
#     ]
# time_type_range = ['uniform', 'snr']
loss_type_range = ['lv']
integrator_type_range = ['ei']
if args.target_type == 'checkerboard':
    model_type_range = ['base_zero_init']
else:
    model_type_range = [
        # 'target_informed_zero_init',
        'base_zero_init'
    ]
time_type_range = ['uniform']

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

# Sample the target distribution
mcmc_start_time = time.time()
if args.use_mcmc_dataset:
    # Make the initial point
    if args.target_type == 'checkerboard':
        x_init = target.loc.clone()
    elif args.target_type == 'rings':
        x_init = target.sample_init_points(32)
    else:
        uniform = torch.distributions.Uniform(low=target.domain[:, 0], high=target.domain[:, 1])
        x_init = uniform.sample((128,))
    # Sample with MCMC
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
else:
    dataset = target.sample((args.dataset_size,))
mcmc_end_time = time.time()

# Fit a GMM on this dataset
ref_fitting_start_time = time.time()
if args.n_components == 1:
    mean_ref = torch.mean(dataset, dim=0)
    if args.gmm_type == 'full':
        var_ref = torch.cov(dataset.T)
    else:
        var_ref = torch.var(dataset, dim=0)
else:
    weights_ref, means_ref, variances_ref = fit_gmm(
        n_components=args.n_components, dataset=dataset, em_type=args.gmm_type)
ref_fitting_end_time = time.time()

# Make the solver details
if args.n_components == 1:
    solver_details = {
        'mean_ref': mean_ref.to(device),
        'var_ref': torch.linalg.eigh(var_ref.to(device)) if args.gmm_type == 'full' else var_ref.to(device)
    }
else:
    solver_details = {
        'weights_ref': weights_ref.to(device),
        'means_ref': means_ref.to(device),
        'variances_ref': torch.linalg.eigh(variances_ref.to(device)) if args.gmm_type == 'full' else variances_ref.to(device)
    }

# Run the big loop
dump_results = []
params = list(itertools.product(loss_type_range, integrator_type_range, model_type_range, time_type_range))
for loss_type, integrator_type, model_type, time_type in tqdm(params):
    # Check exceptions
    if (model_type == 'target_informed_langevin_init') and (integrator_type == 'ei'):
        continue
    # Make the model
    model = make_model(
        solver_type=args.solver_type,
        ref_type='gaussian' if args.n_components == 1 else 'gmm',
        loss_type=loss_type,
        integrator_type=integrator_type,
        model_type=model_type,
        time_type=time_type,
        solver_details=solver_details,
        target_details=make_target_details(target_name=args.target_type),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        })
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=False)
    results = model_wrapped.run()
    # Save to the list
    dump_results.append({
        'params': {
            'loss_type': loss_type,
            'integrator_type': integrator_type,
            'model_type': model_type,
            'time_type': time_type
        },
        'times': {
            'mcmc': mcmc_end_time - mcmc_start_time,
            'ref': ref_fitting_end_time - ref_fitting_start_time
        },
        'metrics': results.metrics,
        'samples': results.samples.cpu()
    })
    # Save the file
    gmm_params = solver_details
    if args.n_components == 1:
        gmm_params['mean_ref'] = gmm_params['mean_ref'].cpu()
        gmm_params['var_ref'] = var_ref.cpu()
    else:
        gmm_params['weights_ref'] = gmm_params['weights_ref'].cpu()
        gmm_params['means_ref'] = gmm_params['means_ref'].cpu()
        gmm_params['variances_ref'] = variances_ref.cpu()
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': dump_results,
            'gmm_params': gmm_params
        }, f)
