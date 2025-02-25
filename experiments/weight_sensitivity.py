# Libraries
import pprint
from benchmark_utils import make_target_details, make_model
from sde_sampler.additions.hacking import TrainableWrapper
from sde_sampler.distr.gauss import TwoModes
from tqdm import tqdm
import argparse
import pickle
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
parser.add_argument('--num_weights', type=int, default=8)
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
filename += '_integrator_type_' + args.integrator_type
filename += '_model_type_' + args.model_type
filename += '_time_type_' + args.time_type
filename += 'seed_' + str(args.seed)
filename += '.pkl'

# Make the ranges
dim_range = list(map(int, args.dim_range.split(',')))
weights_range = torch.linspace(0.1, 0.9, args.num_weights)

# Run the big loop
dump_results = []
for dim in dim_range:
    print('dim = ', dim)
    target = TwoModes(dim)
    true_weight = float(target.mixture_weights[0])
    true_weight /= target.mixture_weights.sum()
    idx = torch.searchsorted(weights_range, true_weight)
    weights_range_local = torch.concat([
        weights_range[:idx],
        torch.FloatTensor([true_weight]),
        weights_range[idx:]
    ])
    for weight in tqdm(weights_range_local):
        # Make the solver details
        solver_details = {
            'weights_ref': torch.FloatTensor([weight, 1.-weight]).to(device),
            'means_ref': target.loc.to(device),
            'variances_ref': torch.square(target.scale).to(device)
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
            'params': {'dim': dim, 'weight': weight},
            'training_metrics': training_metrics,
            'eval_metrics': eval_metrics.metrics
        })
        # Save the file
        with open(args.results_path + '/' + filename, 'wb') as f:
            pickle.dump({'config': config, 'results': dump_results}, f)
