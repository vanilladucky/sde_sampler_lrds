# Libraries
from sde_sampler.additions.ebm_mle import MaximumLikelihoodEBM
from benchmark_utils import make_target_details, make_model, fit_gmm, mcmc_sample
from sde_sampler.additions.hacking import TrainableWrapper, list_of_dict_2_dict_of_list
from sde_sampler.distr.phi_four import PhiFour
from sde_sampler.distr.gauss import Gauss, GMMFull
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
import torch

# Parse the arguments
parser = argparse.ArgumentParser()
parser.add_argument('--results_path', type=str)
parser.add_argument('--b', type=float)
parser.add_argument('--dim', type=int, default=32)
parser.add_argument('--n_components_per_mode', type=int)
parser.add_argument('--n_steps', type=int, default=200)
parser.add_argument('--t_limit', type=float, default=0.2)
parser.add_argument('--swap_frequency', type=int, default=8)
parser.add_argument('--n_mcmc_steps', type=int, default=32)
parser.add_argument('--n_accumulation_steps', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--decay', type=float, default=1e-2)
parser.add_argument('--ebm_batch_size', type=int, default=32)
parser.add_argument('--ebm_n_epochs', type=int, default=150)
parser.add_argument('--train_steps', type=int, default=4096)
parser.add_argument('--train_batch_size', type=int, default=1024)
parser.add_argument('--eval_batch_size', type=int, default=8192)
parser.add_argument('--dataset_size', type=int, default=40000)
parser.add_argument('--n_particles', type=int, default=8192)
parser.add_argument('--n_ess_seeds', type=int, default=128)
parser.add_argument('--use_laplace_approx', action=argparse.BooleanOptionalAction)
parser.add_argument('--n_sampling_seeds', type=int, default=16)
parser.add_argument('--train_ebm_only', action=argparse.BooleanOptionalAction)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# Check that use_laplace_approx is used correctly
if args.use_laplace_approx and not (args.n_components_per_mode == 1):
    print('Cannot use --use_laplace_approx without --n_components_per_mode 1.')
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
filename += '_n_components_per_mode_' + str(args.n_components_per_mode)
if args.use_laplace_approx:
    filename += '_use_laplace_approx_'
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
target = PhiFour(dim=args.dim, a=0.1, b=args.b, dim_phys=1, beta=20.)
target.compute_stats_integration()
target.to(device)

# Sample the target distribution


def target_log_prob_and_grad(x):
    return target.unnorm_log_prob(x).flatten(), target.score(x)


# Sample with MCMC
dataset = mcmc_sample(device, target, target.x_min.clone(), n_chains_per_mode=8 * args.n_components_per_mode,
                      dataset_length=args.dataset_size, n_warmup_steps=2048, target_log_prob_and_grad=target_log_prob_and_grad)

# Make the GMM approximation
if args.use_laplace_approx:
    means_ref = target.x_min.clone()
    variances_ref = torch.stack([
        torch.linalg.inv(target.Hessian(means_ref[i])) for i in range(means_ref.shape[0])
    ], dim=0).to(device) / target.beta
    weights_ref = torch.FloatTensor([0.5, 0.5])
else:
    # Fit a GMM on this dataset
    weights_ref, means_ref, variances_ref = fit_gmm(n_components=2 * args.n_components_per_mode, dataset=dataset,
                                                    means_init=target.x_min.clone().unsqueeze(0).repeat((args.n_components_per_mode, 1, 1)).view((-1, target.dim)),
                                                    em_type='full')


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
    weights=weights_ref.to(device),
    means=means_ref.to(device),
    variances=variances_ref.to(device)
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
    n_steps=args.n_steps,
    sampler_type='replica_exchange',
    swap_frequency=args.swap_frequency
).to(device)

# Train the EBM
losses_ebm, losses_ebm_grad, _ = ebm.train(
    device=device,
    lr=args.lr,
    decay=args.decay,
    data=dataset.to(device),
    batch_size=args.ebm_batch_size,
    n_epochs=args.ebm_n_epochs,
    initial_n_warmup_mcmc_steps=4096,
    n_warmup_mcmc_steps=0,
    n_mcmc_steps=args.n_mcmc_steps,
    n_accumulation_steps=args.n_accumulation_steps,
    verbose=True
)

# Save the parameters
net = net.cpu()
ebm_params = net.state_dict().copy()
net = net.to(device)

# Stop here if needed
if args.train_ebm_only:
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'true_weight': target.expectations['true_weight'],
            'true_weight_cor': target.expectations['true_weight_cor'],
            'losses_ebm': {'loss': losses_ebm, 'loss_grad': losses_ebm_grad},
            'ebm_params': ebm_params,
            'gmm_params': {
                'weights_ref': weights_ref.cpu(),
                'means_ref': means_ref.cpu(),
                'variances_ref': variances_ref.cpu()
            }
        }, f)
    # Exit
    exit(0)

# Set the solver_details
solver_details_ebm = {'net': net.to(device)}
solver_details_gmm = {
    'weights_ref': weights_ref.to(device),
    'means_ref': means_ref.to(device),
    'variances_ref': variances_ref.to(device)
}

# Compute ESS for the GMM
gmm = GMMFull(dim=target.dim, loc=means_ref.to(device), cov=variances_ref.to(device),
              mixture_weights=weights_ref.to(device))

forward_ess_gmm = []
for _ in range(args.n_ess_seeds):
    samples_target = dataset[torch.randperm(dataset.shape[0])[:args.n_particles]].to(device)
    log_weights = gmm.unnorm_log_prob(samples_target) - target.unnorm_log_prob(samples_target)
    ess = torch.exp(2. * torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2. * log_weights, dim=0)).cpu().item()
    forward_ess_gmm.append(ess / args.n_particles)

# Compute ESS for the EBM
forward_ess_ebm = []
for _ in range(args.n_ess_seeds):
    samples_target = dataset[torch.randperm(dataset.shape[0])[:args.n_particles]].to(device)
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
        target_details=make_target_details(target_name='phi_four', dim=args.dim, b=args.b),
        training_details={
            'train_steps': args.train_steps,
            'train_batch_size': args.train_batch_size,
            'eval_batch_size': args.eval_batch_size
        }, n_steps=args.n_steps)
    # Train the model
    model_wrapped = TrainableWrapper(model, verbose=False)
    results = model_wrapped.run()
    all_results_metrics = [results.metrics]
    for _ in range(args.n_sampling_seeds-1):
        results_new = model_wrapped.trainable.evaluate()
        results_new = model_wrapped.compute_results_eubo(results_new)
        all_results_metrics.append(results_new.metrics)
    # Save to the list
    dump_results.append({
        'params': {
            'ref_type': ref_type,
            'loss_type': loss_type,
            'integrator_type': integrator_type,
            'model_type': model_type,
            'time_type': time_type
        },
        'metrics': list_of_dict_2_dict_of_list(all_results_metrics)
    })
    # Save the file
    with open(args.results_path + '/' + filename, 'wb') as f:
        pickle.dump({
            'config': config,
            'results': dump_results,
            'true_weight': target.expectations['true_weight'],
            'true_weight_cor': target.expectations['true_weight_cor'],
            'forward_ess_gmm': forward_ess_gmm,
            'forward_ess_ebm': forward_ess_ebm,
            'losses_ebm': {'loss': losses_ebm, 'loss_grad': losses_ebm_grad},
            'ebm_params': ebm_params,
            'gmm_params': {k: v.cpu() for k, v in solver_details_gmm.items()}
        }, f)
