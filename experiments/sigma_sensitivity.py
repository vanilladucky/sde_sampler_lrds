# Libraries
from sde_sampler.distr.gauss import TwoModes
import pprint
from benchmark_utils import make_target_details, make_model
from sde_sampler.additions.hacking import TrainableWrapper
from tqdm import tqdm
import argparse
import math
import pickle
import torch

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--solver_type', type=str)
parser.add_argument('--loss_type', type=str)
parser.add_argument('--ref_type', type=str)
parser.add_argument('--integrator_type', type=str)
parser.add_argument('--model_type', type=str)
parser.add_argument('--time_type', type=str)
parser.add_argument('--dim_range', type=str, default='8,16,32,64')
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--num_sigmas', type=int, default=8)
parser.add_argument('--terminal_t_pis', type=float, default=5.0)
parser.add_argument('--only_optimal_sigma', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Save the arguments in a dictionnary
config = vars(args)

# Print the config
pprint.pprint(config)

# Make a Pytorch device
device = torch.device('cuda')

# Set the seed
torch.manual_seed(args.seed)

# Make a filename
filename = ''
filename += 'solver_type_' + args.solver_type
filename += '_loss_type_' + args.loss_type
filename += '_ref_type_' + args.ref_type
filename += '_integrator_type_' + args.integrator_type
filename += '_model_type_' + args.model_type
filename += '_time_type_' + args.time_type
filename += 'seed_' + str(args.seed)
filename += '.pkl'

# Get the scalar variances
dim_range = list(map(int, args.dim_range.split(',')))
target_sigmas = {
    dim: TwoModes(dim).distr.stddev.max() for dim in dim_range
}
target_means = {
    dim: TwoModes(dim).distr.mean for dim in dim_range
}

# Make the ranges
if args.only_optimal_sigma:
    sigma_ranges = {
        dim: [target_sigmas[dim]] for dim in dim_range
    }
    if (args.solver_type == 'pis_orig') or ((args.solver_type == 'pbm-ref') and (args.ref_type == 'default')):
        sigma_ranges = {dim: [sigma_ranges[dim][0] / math.sqrt(args.terminal_t_pis)] for dim in dim_range}
else:
    sigma_ranges = {
        dim: target_sigmas[dim] * torch.logspace(-1., 1., args.num_sigmas) for dim in dim_range
    }
    for dim in dim_range:
        idx = torch.searchsorted(sigma_ranges[dim], target_sigmas[dim])
        sigma_ranges[dim] = torch.concat([
            sigma_ranges[dim][:idx],
            torch.FloatTensor([target_sigmas[dim]]),
            sigma_ranges[dim][idx:]
        ])
    if (args.solver_type == 'pis_orig') or ((args.solver_type == 'pbm-ref') and (args.ref_type == 'default')):
        sigma_ranges = {dim: sigma_ranges[dim] / math.sqrt(args.terminal_t_pis) for dim in dim_range}

# Run the big loop
dump_results = []
for dim in dim_range:
    print('dim = ', dim)
    for sigma in tqdm(sigma_ranges[dim]):
        # Make the solver details
        if 'ref' not in args.solver_type:
            solver_details = {'sigma': float(sigma)}
        else:
            if args.ref_type == 'gaussian':
                solver_details = {
                    'mean_ref': target_means[dim].to(device),
                    'var_ref': sigma**2 * torch.ones((dim,), device=device)
                }
            elif args.ref_type == 'default':
                solver_details = {'sigma': float(sigma)}
            else:
                print('Reference type {} not found.'.format(args.ref_type))
                exit(0)
        # Make the model
        model = make_model(
            solver_type=args.solver_type,
            ref_type=args.ref_type,
            loss_type=args.loss_type,
            integrator_type=args.integrator_type,
            model_type=args.model_type,
            time_type=args.time_type,
            solver_details=solver_details,
            target_details=make_target_details(target_name='two_modes', dim=dim),
            training_details={
                'train_steps': args.train_steps,
                'train_batch_size': args.train_batch_size,
                'eval_batch_size': args.eval_batch_size
            })
        # Train the model
        model_wrapped = TrainableWrapper(model, verbose=False)
        eval_metrics, training_metrics = model_wrapped.run(keep_training_metrics=True)
        # Save to the list
        dump_results.append({
            'params': {'dim': dim, 'sigma': sigma},
            'training_metrics': training_metrics,
            'eval_metrics': eval_metrics.metrics
        })
        # Save the file
        with open(args.results_path + '/' + filename, 'wb') as f:
            pickle.dump({'config': config, 'results': dump_results}, f)
