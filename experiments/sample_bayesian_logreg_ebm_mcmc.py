# Libraries
from benchmark_utils import make_target_details, make_model, fit_gmm, mcmc_sample
from sde_sampler.additions.ebm_mle import MaximumLikelihoodEBM
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.distr.logistic_regression import LogisticRegression
from sde_sampler.distr.gauss import Gauss
from sde_sampler.eq.sdes import VP
from sde_sampler.models.mlp import FourierMLP
from sde_sampler.models.reparam import GaussTiltedPotential
from sde_sampler.models.utils import init_bias_uniform_zeros, kaiming_uniform_zeros_
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
parser.add_argument('--data_type', type=str)
parser.add_argument('--n_components', type=int, default=32)
parser.add_argument('--swap_frequency', type=int, default=8)
parser.add_argument('--n_mcmc_steps', type=int, default=32)
parser.add_argument('--n_accumulation_steps', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--ebm_batch_size', type=int, default=32)
parser.add_argument('--ebm_n_epochs', type=int, default=300)
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--n_sampling_seeds', type=int, default=16)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check if the target is right
if not (args.data_type in ['cancer', 'credit', 'ionosphere', 'sonar']):
    print('Data type {} not supported.'.format(args.data_type))
    exit(0)

# Check if the number of components is correct
if not (args.n_components > 1):
    print('n_components has to be greater than 1.')
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
filename += '_n_components_' + str(args.n_components)
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Get the ranges
loss_type_range = ['lv']
integrator_type_range = ['ei']
model_type_range = [
    # 'target_informed_zero_init',
    'base_zero_init'
]
time_type_range = ['uniform']

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
dataset = mcmc_sample(device, target, x_init, dataset_length=args.dataset_size, n_warmup_steps=int(args.dataset_size / 2),
                      skip_chain_per_mode=True, target_log_prob_and_grad=target_log_prob_and_grad).cpu()
mcmc_end_time = time.time()

# Fit a GMM on this dataset
ref_fitting_start_time_gmm = time.time()
weights_ref, means_ref, variances_ref = fit_gmm(n_components=args.n_components, dataset=dataset, em_type='diag')
ref_fitting_end_time_gmm = time.time()

# Make an SDE
ref_fitting_start_time_ebm = time.time()
sde = VP(
    diff_coeff_sq_min=0.1,
    diff_coeff_sq_max=10.,
    scale_diff_coeff=1.,
    terminal_t=1.0
)
prior = Gauss(
    dim=target.dim,
    loc=torch.zeros((target.dim,), device=device),
    scale=sde.scale_diff_coeff * torch.ones((target.dim,), device=device),
    domain_tol=None
)

# Make an EBM
net = GaussTiltedPotential(
    base_model=FourierMLP(
        dim=target.dim,
        num_layers=6,
        channels=128,
        activation=torch.nn.GELU(),
        last_bias_init=init_bias_uniform_zeros,
        last_weight_init=kaiming_uniform_zeros_
    ).to(device),
    sde=sde,
    tilt_type='dot',
    mean=torch.mean(dataset, dim=0).to(device),
    variance=torch.linalg.eigh(torch.cov(dataset.T).to(device))
).to(device)


# Make the MLE EBM
ebm = MaximumLikelihoodEBM(
    sde=sde,
    prior=prior,
    net=net,
    target_acceptance=0.75,
    use_snr_adapted_disc=False,
    perc_keep_mcmc=0.5,
    start_eps=0.0,
    end_eps=0.0,
    n_steps=100,
    sampler_type='replica_exchange',
    swap_frequency=args.swap_frequency
).to(device)

# Train the EBM
losses, losses_grad, _ = ebm.train(
    lr=args.lr,
    data=dataset.to(device),
    batch_size=args.ebm_batch_size,
    n_epochs=args.ebm_n_epochs,
    initial_n_warmup_mcmc_steps=512,
    n_mcmc_steps=args.n_mcmc_steps,
    n_accumulation_steps=args.n_accumulation_steps,
    verbose=True
)
ref_fitting_end_time_ebm = time.time()

# Save the parameters
net = net.cpu()
ebm_params = net.state_dict().copy()
net = net.to(device)

# Set the solver_details
solver_details_ebm = {'net': net.to(device)}
solver_details_gmm = {
    'weights_ref': weights_ref.to(device),
    'means_ref': means_ref.to(device),
    'variances_ref': variances_ref.to(device)
    # 'variances_ref': torch.linalg.eigh(variances_ref.to(device))
}

# Run the big loop
dump_results = []
params = list(itertools.product(['gmm', 'nn'], loss_type_range,
              integrator_type_range, model_type_range, time_type_range))
for ref_type, loss_type, integrator_type, model_type, time_type in tqdm(params):
    # Check exceptions
    if (model_type == 'target_informed_langevin_init') and (integrator_type == 'ei'):
        continue
    # Make the model
    model = make_model(
        solver_type='vp-ref',
        ref_type=ref_type,
        loss_type=loss_type,
        integrator_type=integrator_type,
        model_type=model_type,
        time_type=time_type,
        solver_details=solver_details_ebm if ref_type == 'nn' else solver_details_gmm,
        target_details=make_target_details(target_name=args.data_type),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        }, force_vp20=True)
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
        'params': {
            'ref_type': ref_type,
            'loss_type': loss_type,
            'integrator_type': integrator_type,
            'model_type': model_type,
            'time_type': time_type
        },
        'eval_metrics': list_of_dict_2_dict_of_list(final_results_metrics),
        'intermediate_training_metrics': intermediate_training_metrics,
        'times': {
            'mcmc': mcmc_end_time - mcmc_start_time,
            'ref_gmm': ref_fitting_end_time_gmm - ref_fitting_start_time_gmm,
            'ref_ebm': ref_fitting_end_time_ebm - ref_fitting_start_time_ebm
        }
    })
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': dump_results,
            'ebm_params': ebm_params,
            'gmm_params': {k: v.cpu() for k, v in solver_details_gmm.items()}
        }, f)
