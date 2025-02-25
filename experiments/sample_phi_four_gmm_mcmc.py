# Libraries
from benchmark_utils import make_target_details, make_model, fit_gmm, mcmc_sample
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.distr.phi_four import PhiFour
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
parser.add_argument('--b', type=float)
parser.add_argument('--dim', type=int, default=100)
parser.add_argument('--solver_type', type=str)
parser.add_argument('--gmm_type', type=str)
parser.add_argument('--n_components_per_mode', type=int)
parser.add_argument('--n_chains_per_component', type=int, default=4)
parser.add_argument('--n_steps', type=int, default=200)
parser.add_argument('--train_steps', type=int, default=8192)
parser.add_argument('--train_batch_size', type=int, default=2048)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=60000)
parser.add_argument('--use_laplace_approx', action=argparse.BooleanOptionalAction)
parser.add_argument('--use_ema', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_sampling_seeds', type=int, default=16)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check if it is a ref solver
if 'ref' not in args.solver_type:
    print('solver_type has to be a ref one.')
    exit(0)

# Check for laplace approx
if args.use_laplace_approx and ((args.gmm_type != 'full') or (args.n_components_per_mode != 1)):
    print('--use_laplace_approx should only be used with --gmm_type full --n_components_per_mode 1')
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
filename += 'phi_four_b_{:.2e}'.format(args.b)
filename += '_solver_type_' + args.solver_type
filename += '_gmm_type_' + args.gmm_type
filename += '_n_components_per_mode_' + str(args.n_components_per_mode)
if args.use_laplace_approx:
    filename += '_use_laplace_approx_'
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Get the ranges
# loss_type_range = ['kl', 'lv']
# integrator_type_range = ['em', 'ei']
# model_type_range = [
#     'target_informed_zero_init',
#     'target_informed_langevin_init',
#     'base_zero_init'
# ]
# time_type_range = ['uniform', 'snr']
loss_type_range = ['lv']
integrator_type_range = ['ei']
model_type_range = [
    'target_informed_zero_init',
    'base_zero_init'
]
time_type_range = ['uniform']

# Build the target distribution
target = PhiFour(dim=args.dim, a=0.1, b=args.b, dim_phys=1, beta=20.)
target.compute_stats_integration()
target.to(device)

if args.use_laplace_approx:
    mcmc_start_time = time.time()
    mcmc_end_time = mcmc_start_time
    ref_fitting_start_time = time.time()
    means_ref = target.x_min.clone()
    variances_ref = torch.stack([
        torch.linalg.inv(target.Hessian(means_ref[i])) for i in range(means_ref.shape[0])
    ], dim=0).to(device) / target.beta
    weights_ref = torch.FloatTensor([0.5, 0.5])
    ref_fitting_end_time = time.time()
else:
    # Sample the target distribution
    def target_log_prob_and_grad(x):
        return target.unnorm_log_prob(x).flatten(), target.score(x)

    x_init = target.x_min.unsqueeze(0).expand(
        (args.n_chains_per_component * args.n_components_per_mode, -1, -1)).reshape((-1, args.dim))
    mcmc_start_time = time.time()
    dataset = mcmc_sample(device, target, x_init.clone(), dataset_length=args.dataset_size,
                          n_warmup_steps=2048, skip_chain_per_mode=True, target_log_prob_and_grad=target_log_prob_and_grad)
    mcmc_end_time = time.time()

    # Fit a GMM on this dataset
    ref_fitting_start_time = time.time()
    weights_ref, means_ref, variances_ref = fit_gmm(n_components=2 * args.n_components_per_mode, dataset=dataset,
                                                    means_init=x_init[::args.n_chains_per_component].clone(), em_type=args.gmm_type)
    ref_fitting_end_time = time.time()

# Make the solver details
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
        ref_type='gmm',
        loss_type=loss_type,
        integrator_type=integrator_type,
        model_type=model_type,
        time_type=time_type,
        solver_details=solver_details,
        target_details=make_target_details(target_name='phi_four', dim=args.dim, b=args.b),
        use_ema=args.use_ema is not None,
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        },
        n_steps=args.n_steps)
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=False)
    results, training_metrics = model_wrapped.run(keep_training_metrics=True)
    all_results_metrics = [results.metrics]
    for _ in range(args.n_sampling_seeds-1):
        results_new = model_wrapped.trainable.evaluate()
        results_new = model_wrapped.compute_results_eubo(results_new)
        all_results_metrics.append(results_new.metrics)

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
        'metrics': list_of_dict_2_dict_of_list(all_results_metrics),
        'training_metrics': training_metrics
    })
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': dump_results,
            'true_weight': target.expectations['true_weight'],
            'true_weight_cor': target.expectations['true_weight_cor']
        }, f)
