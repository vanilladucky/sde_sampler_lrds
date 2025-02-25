# Libraries
from benchmark_utils import make_target_details, make_model
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.distr.gauss import TwoModes
from tqdm import tqdm
import argparse
import itertools
import numpy as np
import pickle
import pprint
import random
import torch

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--solver_type', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--integrator_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--time_type', type=str)
parser.add_argument('--dim_range', type=str, default='8')
parser.add_argument('--mean_sensitivity_range', type=str, default='0.,0.25,0.5,0.75,1.0')
parser.add_argument('--std_sensitivity_range', type=str, default='0.5,0.75,1.,1.5,2.')
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--n_sampling_seeds', type=int, default=16)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check if it is a ref solver
if 'ref' not in args.solver_type:
    print('solver_type has to be a ref one.')
    exit(0)

# Save the arguments in a dictionary
config = vars(args)

# Print the config
pprint.pprint(config)

# Make a Pytorch device
device = torch.device('cpu')

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
mean_sensitivity_range = list(map(float, args.mean_sensitivity_range.split(',')))
mean_std_sensitivity_range = [(m, 1.0) for m in mean_sensitivity_range]
std_sensitivity_range = list(map(float, args.std_sensitivity_range.split(',')))
mean_std_sensitivity_range += [(0.0, s) for s in std_sensitivity_range]
mean_std_sensitivity_range = list(set(mean_std_sensitivity_range))
conditioning_range = ['not']
em_type_range = ['diag']

# Run the big loop
dump_results = []
for dim, mean_std_sensitivity, cond_type, em_type in tqdm(list(itertools.product(dim_range,
                                                                                 mean_std_sensitivity_range,
                                                                                 conditioning_range, em_type_range))):
    # Build the target
    target = TwoModes(dim=dim, ill_conditioned=cond_type)
    target.to(device)
    # Perturb the GMM parameters
    mean_param, std_param = mean_std_sensitivity
    perturbed_loc = target.loc + mean_param * torch.randn_like(target.loc)
    perturbed_scale = std_param * target.scale
    # Make the solver details
    solver_details = {
        'weights_ref': target.mixture_weights.to(device),
        'means_ref': perturbed_loc.to(device),
        'variances_ref': torch.square(perturbed_scale).to(device)
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
        target_details=make_target_details(target_name='two_modes',
                                           dim=dim, ill_conditioned=cond_type),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        })
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=False)
    final_results = model_wrapped.run()
    final_results_metrics = [final_results.metrics]
    for _ in range(args.n_sampling_seeds - 1):
        final_results_new = model_wrapped.trainable.evaluate()
        final_results_new = model_wrapped.compute_results_eubo(final_results_new)
        final_results_metrics.append(final_results_new.metrics)

    # Save to the list
    dump_results.append({
        'params': {'dim': dim, 'cond_type': cond_type, 'em_type': em_type,
                   'mean_sensitivity': mean_param, 'std_sensitivity': std_param},
        'eval_metrics': list_of_dict_2_dict_of_list(final_results_metrics),
    })
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({'config': config, 'results': dump_results}, f)
