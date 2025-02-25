# Libraries
from benchmark_utils import make_target_details, make_model, fit_gmm, mcmc_sample
from sde_sampler.additions.ebm_mle import MaximumLikelihoodEBM
from sde_sampler.additions.hacking import TrainableWrapper
from sde_sampler.distr.checkerboard import Checkerboard
from sde_sampler.distr.funnel import Funnel
from sde_sampler.distr.gauss import Gauss, GMM
from sde_sampler.distr.rings import Rings
from sde_sampler.eq.sdes import VP
from sde_sampler.models.mlp import FourierMLP
from sde_sampler.models.reparam import GMMTitledPotential
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
parser.add_argument('--target_type', type=str)
parser.add_argument('--n_components', type=int)
parser.add_argument('--t_limit', type=float, default=0.2)
parser.add_argument('--swap_frequency', type=int, default=8)
parser.add_argument('--n_mcmc_steps', type=int, default=32)
parser.add_argument('--n_accumulation_steps', type=int, default=5)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--ebm_batch_size', type=int, default=32)
parser.add_argument('--ebm_n_epochs', type=int, default=200)
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--use_mcmc_dataset', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_particles', type=int, default=8192)
parser.add_argument('--n_ess_seeds', type=int, default=128)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check if the target is right
if not (args.target_type in ['funnel', 'rings', 'checkerboard']):
    print('Target {} not supported.'.format(args.target_type))
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
filename += 'target_type_' + args.target_type
filename += '_n_components_' + str(args.n_components)
filename += '_seed_' + str(args.seed)
filename += '.pkl'

# Get the ranges
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
weights_ref, means_ref, variances_ref = fit_gmm(n_components=args.n_components, dataset=dataset, em_type='diag')
ref_fitting_end_time_gmm = time.time()

# Make an SDE
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
net = GMMTitledPotential(
    base_model=FourierMLP(
        dim=target.dim,
        num_layers=6,
        channels=128,
        activation=torch.nn.GELU(),
        last_bias_init=init_bias_uniform_zeros,
        last_weight_init=kaiming_uniform_zeros_
    ),
    t_limit=args.t_limit,
    sde=sde,
    weights=weights_ref,
    means=means_ref,
    variances=variances_ref
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
}

# Compute ESS for the GMM
gmm = GMM(dim=target.dim, loc=means_ref.to(device), scale=variances_ref.to(device).sqrt(),
          mixture_weights=weights_ref.to(device))
forward_ess_gmm = []
for _ in range(args.n_ess_seeds):
    samples_target = target.sample((args.n_particles,))
    log_weights = gmm.unnorm_log_prob(samples_target) - target.unnorm_log_prob(samples_target)
    ess = torch.exp(2. * torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2. * log_weights, dim=0)).cpu().item()
    forward_ess_gmm.append(ess / args.n_particles)

# Compute ESS for the EBM
forward_ess_ebm = []
for _ in range(args.n_ess_seeds):
    samples_target = target.sample((args.n_particles,))
    with torch.no_grad():
        log_weights = net.unnorm_log_prob(torch.zeros(
            (args.n_particles, 1), device=device), samples_target).unsqueeze(-1)
        log_weights -= target.unnorm_log_prob(samples_target)
    ess = torch.exp(2. * torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2. * log_weights, dim=0)).cpu().item()
    forward_ess_ebm.append(ess / args.n_particles)

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
            'ref_type': ref_type,
            'loss_type': loss_type,
            'integrator_type': integrator_type,
            'model_type': model_type,
            'time_type': time_type
        },
        'metrics': results.metrics,
        'times': {
            'mcmc': mcmc_end_time - mcmc_start_time,
            'ref_gmm': ref_fitting_end_time_gmm - ref_fitting_start_time,
            'ref_ebm': ref_fitting_end_time_ebm - ref_fitting_start_time
        },
        'samples': results.samples.cpu()
    })
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': dump_results,
            'forward_ess_gmm': forward_ess_gmm,
            'forward_ess_ebm': forward_ess_ebm,
            'ebm_params': ebm_params,
            'gmm_params': {k: v.cpu() for k, v in solver_details_gmm.items()}
        }, f)
