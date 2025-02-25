# Helper functions

# Libraries
from functools import partial
from hydra import initialize, compose
from hydra.utils import instantiate
from sde_sampler.additions.da_ebm import DAEBM
from sde_sampler.additions.drl import DiffusionRecoveryLikelihood
from sde_sampler.additions.ebm_mle import MaximumLikelihoodEBM, smc_sampler, re_sampler
from sde_sampler.additions.ks import compute_sliced_ks
from sde_sampler.additions.mcmc import mala_step, rwmh_step, heuristics_step_size
from sde_sampler.additions.mmd import mmd_median
from sde_sampler.additions.perfect_score_mog import nabla_log_pt
from sde_sampler.distr.gauss import GMM, GMMFull, Gauss, GaussFull
from sde_sampler.eval.sinkhorn import Sinkhorn
from sde_sampler.models.reparam import RemoveReferenceCtrl
from sde_sampler.utils.common import get_timesteps
from sklearn.mixture import GaussianMixture
from tqdm import trange
import torch

solver_types = {
    'dds_orig': 'dds',
    'pis_orig': 'pis',
    'dis_orig': 'dis',
    'cmcd': 'cmcd',
    'vp-ref': 'vp_rds',
    'pbm-ref': 'pbm_rds'
}

model_types = {
    'target_informed_zero_init': 'score',
    'target_informed_unet_zero_init': 'score_unet',
    'target_informed_langevin_init': 'langevin_init',
    'target_informed_lerp_tempering': 'lerp',
    'base_zero_init': 'basic',
    'unet_zero_init': 'basic_unet'
}


def make_target_details(target_name, **kwargs):
    assert target_name in ['two_modes', 'bracket_two_modes', 'two_modes_full', 'many_modes',
                           'rings', 'checkerboard',
                           'phi_four', 'mnist', 'mnist_zero_one', 'cancer', 'credit',
                           'ionosphere', 'sonar'
                           ]
    if target_name in ['two_modes', 'two_modes_full']:
        return {
            'name': target_name,
            'dim': kwargs.get('dim', 5),
            'ill_conditioned': kwargs.get('ill_conditioned', 'not' if target_name == 'target_name' else 'medium'),
            'a': kwargs.get('a', 1.0)
        }
    elif target_name == 'bracket_two_modes':
        return {
            'name': target_name,
            'dim': kwargs.get('dim', 5),
            'a': kwargs.get('a', 0.75),
        }
    elif target_name == 'many_modes':
        return {
            'name': 'many_modes',
            'dim': kwargs.get('dim', 5),
            'n_modes': kwargs.get('n_modes', 4),
            'mixture_weight_factor': kwargs.get('mixture_weight_factor', 3.),
            'var': kwargs.get('var', 0.5)
        }
    elif target_name == 'funnel':
        return {'name': 'funnel'}
    elif target_name == 'rings':
        return {'name': 'rings'}
    elif target_name == 'checkerboard':
        return {'name': 'checkerboard'}
    elif target_name == 'phi_four':
        return {
            'name': 'phi_four',
            'dim': kwargs.get('dim', 100),
            'b': kwargs.get('b', 0.0)
        }
    elif target_name == 'mnist':
        return {'name': 'mnist'}
    elif target_name == 'mnist_zero_one':
        return {'name': 'mnist_zero_one'}
    elif target_name == 'cancer':
        return {'name': 'cancer'}
    elif target_name == 'credit':
        return {'name': 'credit'}
    elif target_name == 'ionosphere':
        return {'name': 'ionosphere'}
    elif target_name == 'sonar':
        return {'name': 'sonar'}
    else:
        raise NotImplementedError('Target {} not supported.')


def make_model(solver_type, ref_type, loss_type, integrator_type, model_type, time_type,
               solver_details, target_details, training_details, optim_details=None, n_steps=100,
               force_base_zero_init=False, use_ema=False, force_vp20=False, force_vp_cosine=False,
               compute_samples_based_metrics=True, force_T_cosine=None):
    # Assertions
    assert solver_type in solver_types
    assert ref_type in ['default', 'gaussian', 'gmm', 'nn']
    assert loss_type in ['kl', 'lv']
    assert integrator_type in ['em', 'ei', 'ddpm_like']
    assert model_type in model_types
    assert time_type in ['uniform', 'snr']
    assert isinstance(solver_details, dict)
    assert isinstance(target_details, dict) and ('name' in target_details)
    assert isinstance(training_details, dict)

    # Exceptions for orig models
    if ('orig' in solver_type) or ('dis' in solver_type) or ('cmcd' in solver_type):
        if not ((model_type == 'base_zero_init') and force_base_zero_init):
            if (solver_type == 'dds_orig') and (
                    model_type not in ['target_informed_zero_init', 'target_informed_unet_zero_init']):
                raise ValueError('Only target_informed_zero_init model is supported.')
            if (solver_type == 'pis_orig') and (
                    model_type not in ['target_informed_zero_init', 'target_informed_unet_zero_init']):
                raise ValueError('Only target_informed_zero_init model is supported.')
            if ('dis' in solver_type) and (model_type == 'base_zero_init'):
                raise ValueError('Model base_zero_init is not supported.')
            if (solver_type == 'cmcd') and (model_type == 'base_zero_init'):
                raise ValueError('Only base_zero_init is supported for CMCD.')
        if not (time_type == 'uniform'):
            raise ValueError('Only uniform time discretisation is supported for orig/cmcd models.')
        if not (integrator_type == 'em'):
            raise ValueError('Can\'t use EI or DDPM-like discretization with orig models.')
        if force_vp20 and (solver_type != 'dis_orig'):
            raise ValueError('Can\'t use vp_20 for orig models other than DIS.')
        if force_vp_cosine:
            raise ValueError('Can\'t use vp_cosine for orig models.')

    # Exception for our models
    if 'ref' in solver_type:
        if model_type == 'target_informed_lerp_tempering':
            raise ValueError('Model target_informed_lerp_tempering is not supported.')
        # Exception for SNR with PBM
        if (solver_type == 'pbm-ref') and (time_type == 'uniform'):
            raise ValueError('PBM schedule is unstable with uniform time discretization.')
        # Exception for use of DDPM
        if (integrator_type == 'ddpm_like') and (time_type == 'uniform'):
            # Note that DDPM + VP Linear works if end = T - 1e-4 in uniform
            raise ValueError('Using the integration scheme from DDPM with uniform times is unstable.')

    # Exceptions for custom VP
    if force_vp20 and force_vp_cosine:
        raise ValueError('Can\'t use vp_20 and vp_cosine at the same time.')
    if (solver_type == 'pbm-ref') and (force_vp20 or force_vp_cosine):
        raise ValueError('Can\'t use vp_20 or vp_cosine with PBM.')

    # Exception for non-ref model
    if ((ref_type != 'default') and ('ref' not in solver_type)) and (solver_type != 'cmcd'):
        raise ValueError('Only ref models can use a non-default ref.')
    if (solver_type == 'cmcd') and (ref_type not in ['default', 'gaussian']):
        raise ValueError('Can\'t use ref other than gaussian for CMCD.')

    # Exception for EI/DDPM with langevin init
    if (model_type == 'target_informed_langevin_init') and (integrator_type in ['ei', 'ddpm_like']):
        raise ValueError('Can\'t use EI or DDPM-like with Langevin score.')

    # Build the model
    with initialize(version_base=None, config_path="../conf/"):
        # Load the generic config
        overrides = [
            "+target=" + target_details['name'],
            "+solver=" + solver_types[solver_type],
            "model@generative_ctrl=" + model_types[model_type],
            "loss.method=" + loss_type,
        ]
        if force_vp20:
            overrides.append('sde=vp_20')
        if force_vp_cosine:
            overrides.append('sde=vp_cos')
        cfg = compose(overrides=overrides)
        # Overwrite the target
        for k, v in target_details.items():
            if k != 'name':
                cfg['target'][k] = v
        # Override the training details
        cfg['use_ema'] = use_ema
        cfg['train_steps'] = training_details['train_steps']
        cfg['train_batch_size'] = training_details['train_batch_size']
        cfg['eval_batch_size'] = training_details['eval_batch_size']
        if solver_type != 'dds_orig':
            cfg['train_timesteps']['steps'] = n_steps
        # Change time if SNR-adapted
        if time_type == 'snr':
            cfg['train_timesteps']['start'] = 1e-4
            cfg['train_timesteps']['end'] = cfg['sde']['terminal_t'] - 1e-4
        # Change initial timestep for VP cosine
        if force_vp_cosine:
            cfg['train_timesteps']['start'] = 1e-3
        # Set EI or EM or DDPM-like
        if ('ref' in solver_type) and (integrator_type == 'ei'):
            cfg['loss']['_target_'] = 'sde_sampler.losses.oc.EIReferenceSDELoss'
        if ('ref' in solver_type) and (integrator_type == 'ddpm_like'):
            cfg['loss']['_target_'] = 'sde_sampler.losses.oc.DDPMLikeReferenceSDELoss'
        if solver_type == 'dds_orig':
            cfg['loss']['sigma'] = solver_details['sigma']
            if force_T_cosine is not None:
                cfg['train_timesteps']['end'] = force_T_cosine
        elif solver_type == 'pis_orig':
            cfg['sde']['diff_coeff'] = solver_details['sigma']
        elif (solver_type == 'dis_orig') or (solver_type == 'dis_discrete'):
            cfg['sde']['scale_diff_coeff'] = solver_details['sigma']
        elif ('ref' in solver_type) and (ref_type == 'default'):
            if 'pbm' in solver_type:
                cfg['sde']['diff_coeff'] = solver_details['sigma']
            if 'vp' in solver_type:
                cfg['sde']['scale_diff_coeff'] = solver_details['sigma']
        # # Use angle encoding for ALDP
        # if target_details['name'] == 'aladip':
        #     cfg['generative_ctrl']['base_model']['use_angle_encoding'] = True
        # Change optim details
        if optim_details is not None:
            for k, v in optim_details.items():
                cfg['optim'][k] = v
    model = instantiate(cfg.solver, cfg)
    model.setup()

    # Add sample based metrics
    if compute_samples_based_metrics:
        model.eval_sample_losses = {
            'sinkhorn': Sinkhorn(),
            'mmd': mmd_median,
            'ks': compute_sliced_ks
        }

    # Change the reference distributions
    if 'ref' in solver_type:
        if ref_type == 'default':
            pass
        elif ref_type == 'gaussian':
            model.change_reference_type(
                ref_type='gaussian',
                mean=solver_details['mean_ref'],
                var=solver_details['var_ref']
            )
        elif ref_type == 'gmm':
            model.change_reference_type(
                ref_type='gmm',
                weights=solver_details['weights_ref'],
                means=solver_details['means_ref'],
                variances=solver_details['variances_ref']
            )
        elif ref_type == 'nn':
            model.change_reference_type(
                ref_type='nn',
                net=solver_details['net'],
                eps=torch.tensor(cfg['train_timesteps']['start'])
            )
    if ('cmcd' in solver_type) and (ref_type == 'gaussian'):
        model.update_prior(mean=solver_details['mean'], var=solver_details['var'])

    # Set the type of time
    if time_type == 'snr':
        model.train_timesteps = partial(get_timesteps, **model.train_timesteps.keywords, sde=model.sde)
        model.eval_timesteps = partial(get_timesteps, **model.eval_timesteps.keywords, sde=model.sde)

    # Remove ref score when using Langevin init
    if (model_type == 'target_informed_langevin_init') and ('ref' in solver_type):
        model.generative_ctrl = RemoveReferenceCtrl(model.generative_ctrl, model.reference_score_t)

    # Return the model
    return model


def mcmc_sample(device, target, x_init, mcmc_type='mala', step_size=1e-3, n_chains_per_mode=4, dataset_length=50000,
                n_warmup_steps=512,
                skip_chain_per_mode=False, target_log_prob_and_grad=None, adapt_step_size=True, shuffle=True,
                verbose=False):
    # Build the target's score
    if (mcmc_type == 'mala') and (target_log_prob_and_grad is None):
        def target_log_prob_and_grad(y):
            y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
            log_prob_y = target.unnorm_log_prob(y_)
            return log_prob_y.flatten(), torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()
    # Sample the target distribution with MCMC
    if skip_chain_per_mode:
        n_chains = x_init.shape[0]
        y_init = x_init.clone()
    else:
        n_chains = x_init.shape[0] * n_chains_per_mode
        y_init = torch.concat([
            x_init[i].unsqueeze(0).expand((n_chains_per_mode, -1)) for i in range(x_init.shape[0])
        ], dim=0)
    n_mcmc_steps = int(dataset_length / n_chains)
    step_size = step_size * torch.ones((n_chains, 1), device=device)
    y = torch.autograd.Variable(y_init.clone(), requires_grad=True)
    if mcmc_type == 'mala':
        target_log_prob_y, target_grad_y = target_log_prob_and_grad(y)
    else:
        target_log_prob_y = target.unnorm_log_prob(y).flatten()
    ys = torch.empty((n_mcmc_steps, *y.shape))
    if verbose:
        r = trange(n_warmup_steps + n_mcmc_steps)
    else:
        r = range(n_warmup_steps + n_mcmc_steps)
    for step_id in r:
        # Run MCMC step
        if mcmc_type == 'mala':
            y, target_log_prob_y, target_grad_y, log_acc = mala_step(
                y=y,
                target_log_prob_y=target_log_prob_y,
                target_grad_y=target_grad_y,
                target_log_prob_and_grad=target_log_prob_and_grad,
                step_size=step_size
            )
        else:
            y, target_log_prob_y, log_acc = rwmh_step(
                y=y,
                target_log_prob_y=target_log_prob_y,
                target_log_prob=target.unnorm_log_prob,
                step_size=step_size
            )
        # Display the acceptance
        if verbose:
            r.set_postfix({
                'acc': torch.minimum(torch.exp(log_acc), torch.ones_like(log_acc)).mean().item(),
                'step_size': '{:.2e}'.format(torch.mean(step_size).item())
            })
        # Adapt the step size
        if adapt_step_size:
            step_size = heuristics_step_size(step_size, log_acc)
        # Save the step
        if step_id >= n_warmup_steps:
            ys[step_id - n_warmup_steps] = y.detach().clone().cpu()
    # Return everything
    ret = ys.view((-1, *x_init.shape[1:]))
    if shuffle:
        return ret[torch.randperm(ret.shape[0])]
    else:
        return ret


def fit_gmm(n_components, dataset, means_init=None, em_type='diag', max_iter=1000):
    # Check many different reg_covar values
    for reg_covar in [1e-6, 5e-5, 1e-5, 5e-4, 1e-4, 5e-3, 1e-3, 5e-2, 1e-2]:
        # Handle exceptions
        try:
            # Fit a GMM model
            dim = dataset.shape[-1]
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type=em_type,
                means_init=means_init.cpu().numpy() if means_init is not None else None,
                reg_covar=reg_covar,
                max_iter=max_iter
            )
            gmm = gmm.fit(dataset.view((-1, dataset.shape[-1])).numpy())
            # Extract the parameters
            weights = torch.from_numpy(gmm.weights_).float()
            means = torch.from_numpy(gmm.means_).float()
            variances = torch.from_numpy(gmm.covariances_).float()
            # Test the parameters
            if len(variances.shape[1:]) == 2:
                GMMFull(dim=dim, loc=means, cov=variances, mixture_weights=weights)
            else:
                GMM(dim=dim, loc=means, scale=variances.sqrt(), mixture_weights=weights)
            # If we reached this place, break
            return weights, means, variances
        except:
            continue
    raise ValueError("Couldn't fit a GMM on this dataset.")


def build_ebm(ebm_type, sde, prior, net, target_acceptance=0.75, use_snr_adapted_disc=False, perc_keep_mcmc=-1.0,
              start_eps=1e-3, end_eps=0.0, n_steps=100, **kwargs):
    if ebm_type == 'drl':
        ebm_class = DiffusionRecoveryLikelihood
    elif ebm_type == 'daebm':
        ebm_class = DAEBM
    elif 'mle' in ebm_type:
        ebm_class = MaximumLikelihoodEBM
    else:
        raise NotImplementedError('EBM type {} not found.'.format(ebm_type))
    return ebm_class(sde=sde, prior=prior, net=net, target_acceptance=target_acceptance,
                     use_snr_adapted_disc=use_snr_adapted_disc,
                     perc_keep_mcmc=perc_keep_mcmc, start_eps=start_eps, end_eps=end_eps, n_steps=n_steps, **kwargs
                     )


def compute_score_mse_ebm_mog(model, is_drl, device, target, means, covs, weights, batch_size=4096):
    # Move to device
    model = model.to(device)
    means = means.to(device)
    covs = covs.to(device)
    weights = weights.to(device)
    # Compute the MSE
    mses = []
    times = model.times.clone().to(device)
    for i, t in enumerate(times[:-1]):
        # Reshape t
        t_ = t * torch.ones((batch_size, 1), device=device)
        # Get samples
        x_t = model.sde.s(t_) * target.sample((batch_size,)).to(device)
        x_t += model.sde.s(t_) * torch.sqrt(model.sde.sigma_sq(t_)) * torch.randn_like(x_t)
        # Compute the optimal score
        score_opt = nabla_log_pt(t_, x_t, model.sde, means=means, covs=covs, weights=weights)
        # Compute the estimated score
        if is_drl:
            score_est = model.net(t_, model.alphas[i] * x_t)
        else:
            score_est = model.net(t_, x_t)
        # Compute the MSE
        mses.append(torch.sum(torch.square(score_opt - score_est), dim=-1).mean().cpu().detach())
    return torch.stack(mses)


class ScoreWithReferenceScore(torch.nn.Module):
    def __init__(self, score_ref, score):
        super().__init__()
        self.score_ref = score_ref
        self.score_ref_simple = lambda t, x: self.score_ref(t, x.unsqueeze(0)).squeeze(0)
        self.score_ref_vec = torch.vmap(self.score_ref_simple)
        self.score = score

    def forward(self, t, x):
        # Compute reference score
        if isinstance(t, float) or (len(t.shape) == 1):
            ref_score = self.score_ref(t, x)
        else:
            ref_score = self.score_ref_vec(t, x)
        return ref_score - self.score(t, x)


def define_tempering_utils(mean, var, target_log_prob, target_score=None):
    # Define the prior
    dim = mean.shape[0]
    if len(var.shape) == 2:
        prior = GaussFull(dim=dim, loc=mean, cov=var)
    else:
        prior = Gauss(dim=dim, loc=mean, scale=var.sqrt())
    # Define the annealing
    if target_score is None:
        def target_log_prob_and_grad(y):
            y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
            log_prob_y = target_log_prob(y_).flatten()
            return log_prob_y, torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

    def log_prob_and_grads(t, x):
        # Compute the target log-prob and score
        if target_score is not None:
            log_prob_t = target_log_prob(x).flatten()
            grad_t = target_score(x)
        else:
            log_prob_t, grad_t = target_log_prob_and_grad(x)
        # Compute the prior log-prob and score
        log_prob_p = prior.log_prob(x).flatten()
        grad_p = prior.score(x)
        # Compute the final log_prob and gradient
        log_prob = t.flatten() * log_prob_p + (1. - t).flatten() * log_prob_t
        grad = t * grad_p + (1. - t) * grad_t
        return log_prob, grad

    return prior, log_prob_and_grads


def run_smc_sampler(mean, var, n_steps, step_size, n_particles, n_mcmc_steps, n_warmup_mcmc_steps,
                    target_log_prob, target_score=None, reweight_threshold=1.0, target_acceptance=0.75, verbose=False):
    prior, log_prob_and_grads = define_tempering_utils(mean, var, target_log_prob, target_score=target_score)
    step_sizes_per_noise = step_size * torch.ones((n_steps, n_particles, 1), device=mean.device)
    times = torch.linspace(0.0, 1.0, n_steps).view((-1, 1, 1)).to(mean.device)
    times = times.repeat((1, n_particles, 1))
    return smc_sampler(
        x_init=prior.sample((n_particles,)),
        times=times,
        log_prob_and_grads=log_prob_and_grads,
        n_warmup_mcmc_steps=n_warmup_mcmc_steps,
        n_mcmc_steps=n_mcmc_steps,
        step_sizes_per_noise=step_sizes_per_noise,
        per_noise_init=False,
        reweight_threshold=reweight_threshold,
        target_acceptance=target_acceptance,
        verbose=verbose
    )[0][0]


def run_re_sampler(mean, var, n_steps, step_size, batch_size, swap_frequency, n_mcmc_steps, n_warmup_mcmc_steps,
                   target_log_prob, target_score=None, target_acceptance=0.75, verbose=False):
    prior, log_prob_and_grads = define_tempering_utils(mean, var, target_log_prob, target_score=target_score)
    step_sizes_per_noise = step_size * torch.ones((n_steps, batch_size, 1), device=mean.device)
    times = torch.linspace(0.0, 1.0, n_steps).view((-1, 1, 1)).to(mean.device)
    times = times.repeat((1, batch_size, 1))
    return re_sampler(
        x_init=prior.sample((batch_size,)),
        times=times,
        log_prob_and_grads=log_prob_and_grads,
        swap_frequency=swap_frequency,
        n_warmup_mcmc_steps=n_warmup_mcmc_steps,
        n_mcmc_steps=n_mcmc_steps,
        step_sizes_per_noise=step_sizes_per_noise,
        per_noise_init=False,
        target_acceptance=target_acceptance,
        verbose=verbose
    )[0][0]
