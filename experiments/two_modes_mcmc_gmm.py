# Libraries
from benchmark_utils import make_target_details, make_model, mcmc_sample, fit_gmm
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.distr.gauss import TwoModes, TwoModesFull, GMM, GMMFull
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
parser.add_argument('--solver_type', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--integrator_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--time_type', type=str)
parser.add_argument('--dim_range', type=str, default='8,16,32,64,128')
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--use_full_two_modes', action=argparse.BooleanOptionalAction)
parser.add_argument('--use_mcmc_sampling', action=argparse.BooleanOptionalAction)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--n_sampling_seeds', type=int, default=16)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cpu')
args = parser.parse_args()

# Check if it is a ref solver
if 'ref' not in args.solver_type:
    print('solver_type has to be a ref one.')
    exit(0)

# Save the arguments in a dictionnary
config = vars(args)

# Print the config
pprint.pprint(config)

# Make a Pytorch device
device = torch.device(args.device)

# Set the seed
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Make a filename
filename = ''
filename += 'solver_type_' + args.solver_type
filename += '_loss_type_' + args.loss_type
filename += '_integrator_type_' + args.integrator_type
filename += '_model_type_' + args.model_type
filename += '_time_type_' + args.time_type
filename += 'seed_' + str(args.seed)
filename += '.pkl'

# Get the ranges
dim_range = list(map(int, args.dim_range.split(',')))
if args.use_full_two_modes:
    conditioning_range = ['medium', 'hard']
    em_type_range = ['full']  # ['diag', 'full']
    dim_range = sorted(list(filter(lambda dim: dim <= 32, dim_range)))
else:
    conditioning_range = ['not', 'medium', 'hard']
    em_type_range = ['diag']

# Run the big loop
dump_results = []
for dim, cond_type, em_type in tqdm(list(itertools.product(dim_range, conditioning_range, em_type_range))):
    # Build the target
    if args.use_full_two_modes:
        target = TwoModesFull(dim=dim, ill_conditioned=cond_type)
    else:
        target = TwoModes(dim=dim, ill_conditioned=cond_type)
    target.to(device)
    # Build a dataset
    mcmc_start_time = time.time()
    if args.use_mcmc_sampling:
        # Sample the target with MCMC
        dataset = mcmc_sample(device, target, target.loc.to(device), dataset_length=args.dataset_size)
    else:
        # Build an equilibrated distribution
        if args.use_full_two_modes:
            equi_target = GMMFull(
                dim=target.dim,
                loc=target.loc,
                cov=target.cov,
                mixture_weights=torch.ones_like(target.mixture_weights) / target.mixture_weights.shape[0]
            )
        else:
            equi_target = GMM(
                dim=target.dim,
                loc=target.loc,
                scale=target.scale,
                mixture_weights=torch.ones_like(target.mixture_weights) / target.mixture_weights.shape[0]
            )
        # Sample the equilibrated target
        dataset = equi_target.sample((args.dataset_size,)).cpu()
    mcmc_end_time = time.time()
    # Fit the GMM
    ref_fitting_start_time = time.time()
    weights_ref, means_ref, variances_ref = fit_gmm(
        n_components=2,
        dataset=dataset,
        means_init=target.loc.cpu(),
        em_type=em_type
    )
    ref_fitting_end_time = time.time()
    # Make the solver details
    solver_details = {
        'weights_ref': weights_ref.to(device),
        'means_ref': means_ref.to(device),
        'variances_ref': torch.linalg.eigh(variances_ref.to(device)) if em_type == 'full' else variances_ref.to(device)
    }
    # Make the model
    model = make_model(
        solver_type=args.solver_type,
        ref_type='gmm',
        loss_type=args.loss_type,
        integrator_type=args.integrator_type,
        model_type=args.model_type,
        time_type=args.time_type,
        solver_details=solver_details,
        target_details=make_target_details(target_name='two_modes_full' if args.use_full_two_modes else 'two_modes',
                                           dim=dim, ill_conditioned=cond_type),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        })
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=False)
    final_results, intermediate_training_metrics = model_wrapped.run(keep_training_metrics=True)
    final_results_metrics = [final_results.metrics]
    for _ in range(args.n_sampling_seeds-1):
        final_results_new = model_wrapped.trainable.evaluate()
        final_results_new = model_wrapped.compute_results_eubo(final_results_new)
        final_results_metrics.append(final_results_new.metrics)

    # Save to the list
    dump_results.append({
        'params': {'dim': dim, 'cond_type': cond_type, 'em_type': em_type},
        'times': {
            'mcmc': mcmc_end_time - mcmc_start_time,
            'ref': ref_fitting_end_time - ref_fitting_start_time
        },
        'eval_metrics': list_of_dict_2_dict_of_list(final_results_metrics),
        'intermediate_training_metrics': intermediate_training_metrics,
    })
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({'config': config, 'results': dump_results}, f)
