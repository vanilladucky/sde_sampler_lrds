from __future__ import annotations

from typing import Callable

import torch
from torch.nn import Module

import math

from sde_sampler.eq.sdes import OU
from sde_sampler.utils.autograd import compute_gradx
from sde_sampler.utils.common import clip_and_log

from sde_sampler.distr.gauss import log_prob_gaussian, log_prob_gaussian_full, score_gauss, score_gauss_full
from sde_sampler.distr.gauss import GMM, GMMFull, Gauss, GaussFull


class ClippedCtrl(Module):
    """Base class for clipping a neural network"""

    def __init__(
            self,
            base_model: Module,
            clip_model: float | None = None,
            name: str = "ctrl",
            **kwargs,
    ):
        super().__init__()
        self.base_model = base_model
        self.clip_model = clip_model
        self.name = name

    def clipped_base_model(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        output = clip_and_log(
            self.base_model(t, x),
            max_norm=self.clip_model,
            name=self.name + "_model",
            t=t,
        )
        return output

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.clipped_base_model(t, x)


class RemoveReferenceCtrl(Module):
    """Substracts ref_score from an already defined NN. Only used for Langevin init only
    (i.e., use with CancelDriftCtrl only)"""

    def __init__(self, score, ref_score, use_rescaling=True, sde=None):
        super().__init__()
        assert not (use_rescaling and (sde is not None))
        self.score = score
        self.ref_score = ref_score
        self.use_rescaling = use_rescaling
        self.sde = sde

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ret = self.score(t, x)
        if self.use_rescaling:
            ret -= self.sde.diff(t, x) * self.ref_score(t, x)
        else:
            ret -= self.ref_score(t, x)
        return ret


class ScoreCtrl(ClippedCtrl):
    """Implements the target informed NN

        base_model(t,x) + score_model(t) * target_score(x)

    """

    def __init__(
            self,
            *args,
            target_score: Callable,
            score_model: Module | None = None,
            detach_score: bool = True,
            scale_score: float = 1.0,
            clip_score: float | None = None,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.score_model = score_model
        self.target_score = target_score
        self.detach_score = detach_score
        self.scale_score = scale_score
        self.clip_score = clip_score

    def clipped_target_score(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = clip_and_log(
            self.target_score(x, create_graph=self.detach_score),
            max_norm=self.clip_score,
            name=self.name + "_score",
            t=t,
        )
        assert output.shape == x.shape
        return output

    def clipped_score_model(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        output = clip_and_log(
            self.score_model(t, x),
            max_norm=self.clip_model,
            name=self.name + "_score_model",
            t=t,
        )
        assert output.shape in [(1, 1), (1, x.shape[-1]), x.shape, (x.shape[0], 1)]
        return output

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)
        score = self.scale_score * self.clipped_target_score(t, x)
        if self.score_model is not None:
            score *= self.clipped_score_model(t, x)
        return ctrl + score


class CancelDriftCtrl(ScoreCtrl):
    """Implements Langevin init by substracting the denoising SDE's drift"""

    def __init__(self, *args, sde: OU, langevin_init: bool = True, use_rescaling=True, **kwargs):
        super().__init__(*args, **kwargs)
        if sde.noise_type not in ["diagonal", "scalar"]:
            raise ValueError(f"Invalid sde noise type {sde.noise_type}.")
        self.sde = sde
        self.langevin_init = langevin_init
        self.use_rescaling = use_rescaling

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)

        # Score
        sde_diff = self.sde.diff(t, x)
        sde_drift = self.sde.drift(t, x)
        score = self.scale_score * self.clipped_target_score(t, x)

        if self.score_model is not None:
            score *= self.clipped_score_model(t, x)

        if self.use_rescaling:
            return ctrl + (sde_drift / sde_diff) + 0.5 * sde_diff * score
        else:
            return ctrl + (sde_drift / torch.square(sde_diff)) + 0.5 * score


class LerpCtrl(ScoreCtrl):
    """Implements a constrain on both ends of a neural network"""

    def __init__(
        self,
        *args,
        sde: OU,
        prior_score: Callable,
        hard_constrain: bool = False,
        scale_lerp: float = 1.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if sde.noise_type not in ["diagonal", "scalar"]:
            raise ValueError(f"Invalid sde noise type {sde.noise_type}.")
        self.sde = sde
        self.prior_score = prior_score
        self.hard_constrain = hard_constrain
        self.scale_lerp = scale_lerp

    def clipped_interpolated_score(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        x = x.detach() if self.detach_score else x
        output = self.target_score(x, create_graph=self.detach_score)
        output = torch.lerp(self.prior_score(x), output, t / self.sde.terminal_t)
        output = clip_and_log(
            output,
            max_norm=self.clip_score,
            name=self.name + "_score",
            t=t,
        )
        assert output.shape == x.shape
        return output

    def constrain(self, output, t):
        return 4 * output * (self.sde.terminal_t - t) * t / self.terminal_t**2

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctrl = self.clipped_base_model(t, x)
        if self.hard_constrain:
            ctrl = self.constrain(ctrl, t)

        # Interpolated score
        score = self.scale_score * self.clipped_interpolated_score(t, x)
        if self.score_model is not None:
            score_model = self.clipped_score_model(t, x)
            if self.hard_constrain:
                score_model = self.constrain(score_model, t)
            score *= score_model

        return ctrl + self.sde.diff(t, x) * score


class BetterPotentialCtrl(ClippedCtrl):
    """Advanced class to define a NN as the gradient of another NN. (Used together with EBMs)"""

    def __init__(self, sde, data_mean, data_scalar_var, energy_type='sq_norm', use_gaussian_prior=True,
                 use_s_t_scaling=True, **kwargs):
        """Constructor

        Args:
            * sde (OU): SDE object
            * data_mean (torch.Tensor of shape (dim,)): Mean of the data (for rescaling)
            * data_scalar_var (torch.Tensor of shape (dim,)): Diagonance variance of the data (for rescaling)
            * energy_type (str): Type of energy function (default is 'sq_norm')
                - dot
                        E(t, x) = NN(t,X)^T X
                - sq_norm
                        E(t, X) = norm(NN(t,X))^2
                - residual_sq_norm
                        E(t, X) = norm(NN(t,X) - X)^2
            * use_gaussian_prior (bool): Whether to add a Gaussian term to the energy (default is True)
            * use_s_t_scaling (bool): Use sde.s to anneal the Gaussian prior overtime (default is True)

        """
        super().__init__(**kwargs)
        if (use_gaussian_prior == False) and (energy_type == 'residual_sq_norm'):
            raise ValueError("Can't use residual_sq_norm without gaussian prior.")
        if (use_s_t_scaling and not use_gaussian_prior):
            raise ValueError("Can't use coef_t without gaussian prior.")
        self.sde = sde
        self.energy_type = energy_type
        self.use_gaussian_prior = use_gaussian_prior
        self.use_s_t_scaling = use_s_t_scaling
        self.has_unnorm_log_prob_and_grad = False
        self.has_sample_prior = False
        self.register_buffer("data_mean", data_mean)
        self.register_buffer("data_scalar_var", data_scalar_var)

    def scaling_input(self, t, x, scaling_factor):
        """Input normalization in a Karras fashion"""
        c_i = scaling_factor * self.sde.s(t) * torch.sqrt(self.data_scalar_var + self.sde.sigma_sq(t))
        c_m = scaling_factor * self.sde.s(t) * self.data_mean
        return (x - c_m) / c_i

    def energy(self, t, x, scaling_factor=1.0):
        """Compute the energy"""
        x_scaled = self.scaling_input(t, x, scaling_factor)
        if self.use_gaussian_prior:
            ret = 0.5 * torch.sum(torch.square(x_scaled), dim=-1)
            coef_t = self.sde.s(t)
            if self.energy_type == 'residual_sq_norm':
                coef_t = coef_t.unsqueeze(-1)
        else:
            ret = 0.0
            coef_t = 1.0
        if self.energy_type == 'dot':
            return ret + coef_t * torch.sum(self.base_model(t, x_scaled) * x, dim=-1)
        elif self.energy_type == 'sq_norm':
            return ret + 0.5 * coef_t * torch.sum(torch.square(self.base_model(t, x_scaled)), dim=-1)
        elif self.energy_type == 'residual_sq_norm':
            if self.use_s_t_scaling:
                return 0.5 * torch.sum(torch.square(coef_t * self.base_model(t, x_scaled) - x_scaled), dim=-1)
            else:
                return 0.5 * torch.sum(torch.square(self.base_model(t, x_scaled) - x_scaled), dim=-1)
        else:
            return ret + coef_t * self.base_model(t, x_scaled).sum(dim=-1)

    def unnorm_log_prob(self, t, x, scaling_factor=1.0):
        """Compute the log_prob as the negative energy"""
        return -self.energy(t, x, scaling_factor=scaling_factor)

    def forward(self, t, x, scaling_factor=1.0):
        """Compute the gradient of the energy"""
        with torch.set_grad_enabled(True):
            return compute_gradx(self.unnorm_log_prob, t=t, x=x, scaling_factor=scaling_factor, retain_graph=True)[0]


class GMMTitledPotential(torch.nn.Module):
    """Advanced class to define a NN as the gradient of another NN multiplied with a perfectly noised GMM.
    In implemented,

        E(t, X) = E_{NN}(t, X) * GMM(t, X)
        s(t, X) = -nabla_X E(t,X)

    This classc is used together with EBMs"""

    def __init__(self, base_model, sde, weights, means, variances, t_limit=0.0,
                 use_s_t_scaling=False, tilt_type='dot', use_scaling_factor=False):
        """Constructor

        Args:
            * base_model (torch.nn.Module): Initial NN
            * sde (OU): SDE
            * weights (torch.Tensor of shape (n_modes,)): Weights of the GMM
            * means (torch.Tensor of shape (n_modes, dim)): Means of the GMM
            * variances (torch.Tensor of shape (n_modes, dim) or (n_modes, dim, dim) or tuple): Covariances of the GMM
            * t_limit (float): Threshold time to ignore the annealed 
            * use_s_t_scaling (bool): Use sde.s to anneal the GMM prior overtime (default is True)
            * tilt_type (str): Type of energy function (default is 'sq_norm')
                - dot
                        E(t, x) = NN(t,X)^T X
                - sq_norm
                        E(t, X) = norm(NN(t,X))^2
            * use_scaling_factor (bool): Whether a scaling factor args is passed everywhere (for DRL)

        """
        super().__init__()
        # Store the neural network
        self.tilt_type = tilt_type
        self.base_model = base_model
        self.use_s_t_scaling = use_s_t_scaling
        # Store the SDE
        self.sde = sde
        # Store the GMM params
        self.dim = means.shape[-1]
        self.t_limit = t_limit
        self.is_full_gmm_type = isinstance(variances, tuple) or (len(variances.shape) == 3)
        self.register_buffer("weights", weights)
        self.register_buffer("means", means)
        self.use_full_decomp = isinstance(variances, tuple)
        if isinstance(variances, tuple):
            self.register_buffer("cov_D", variances[0])
            self.register_buffer("cov_P", variances[1])
            self.register_buffer("variances", torch.einsum('...ik,...k,...jk->...ij',
                                                           self.cov_P, self.cov_D, self.cov_P))
        else:
            self.register_buffer("variances", variances)
        # Get the gaussian approxmation
        if self.is_full_gmm_type:
            self.prior_final = GMMFull(dim=self.means.shape[-1], loc=self.means,
                                       cov=self.variances, mixture_weights=self.weights)
        else:
            self.prior_final = GMM(dim=self.means.shape[-1], loc=self.means,
                                   scale=self.variances.sqrt(), mixture_weights=self.weights)
        self.register_buffer("mean_gauss", self.prior_final.distr.mean)
        self.register_buffer("var_gauss", self.prior_final.distr.variance)
        # Vectorize the prior computations
        self.use_scaling_factor = use_scaling_factor
        if self.use_scaling_factor:
            self.prior_log_prob_v = torch.vmap(lambda t, x, scaling_factor: self.prior_log_prob(
                t, x.unsqueeze(0), scaling_factor).squeeze(0))
            self.prior_log_prob_and_grad_v = torch.vmap(
                lambda t, x, scaling_factor: tuple(
                    y.squeeze(0) for y in self.prior_log_prob_and_grad(t, x.unsqueeze(0), scaling_factor)),
                out_dims=(0, 0))
        else:
            self.prior_log_prob_v = torch.vmap(lambda t, x: self.prior_log_prob(t, x.unsqueeze(0)).squeeze(0))
            self.prior_log_prob_and_grad_v = torch.vmap(
                lambda t, x: tuple(y.squeeze(0) for y in self.prior_log_prob_and_grad(t, x.unsqueeze(0))),
                out_dims=(0, 0))
        # Indicate capabilities
        self.has_unnorm_log_prob_and_grad = True
        self.has_sample_prior = True

    def get_gmm_params(self, t, scaling_factor=1.0):
        """Get the parameters of the GMM at time t"""
        if self.use_full_decomp:
            variances_init = (scaling_factor ** 2 * self.cov_D, self.cov_P)
        else:
            variances_init = scaling_factor ** 2 * self.variances
        return self.sde.marginal_gmm_params(
            torch.maximum(t, self.t_limit * torch.ones_like(t)),
            means_init=scaling_factor * self.means,
            variances_init=variances_init,
            weights_init=self.weights
        )

    def sample_prior(self, ts):
        """Sample the noised GMM distribution at times ts"""
        # Sample the prior
        prior_samples = self.prior_final.sample((ts.shape[0],))
        # Noise the samples
        ts_ = torch.maximum(ts, self.t_limit * torch.ones_like(ts))
        loc, var = self.sde.marginal_params(ts_.view((-1, 1)), prior_samples)
        return loc + torch.sqrt(var) * torch.randn_like(loc)

    def prior_log_prob(self, t, x, scaling_factor=1.0, return_grad_utils=False):
        """Compute the likelihood of the noised GMM distribution at time t (NOT VECTORIZED ON TIME)"""
        # Compute the parameters at time t
        weights_t, means_t, variances_t = self.get_gmm_params(t, scaling_factor=scaling_factor)
        if isinstance(variances_t, tuple):
            prec_t, log_det_cov_t = variances_t
        else:
            prec_t, log_det_cov_t = None, None
        weights_t /= weights_t.sum()
        # Compute the log-prob at time t
        if self.is_full_gmm_type:
            if return_grad_utils:
                log_probs, return_precision_times_diff = log_prob_gaussian_full(x, means_t, variances_t,
                                                                                precisions=prec_t,
                                                                                covariances_log_det=log_det_cov_t,
                                                                                return_precision_times_diff=True)
            else:
                log_probs = log_prob_gaussian_full(
                    x, means_t, variances_t, precisions=prec_t, covariances_log_det=log_det_cov_t)
        else:
            log_probs = log_prob_gaussian(x, means_t, variances_t)
        log_probs += torch.log(weights_t.unsqueeze(0))
        log_prob = torch.logsumexp(log_probs, dim=-1)
        if return_grad_utils:
            if self.is_full_gmm_type:
                return log_prob, log_probs, (return_precision_times_diff,)
            else:
                return log_prob, log_probs, (means_t, variances_t)
        else:
            return log_prob

    def prior_log_prob_and_grad(self, t, x, scaling_factor=1.0):
        """Compute the likelihood and score of the noised GMM distribution at time t (NOT VECTORIZED ON TIME)"""
        # Compute the log_prob
        log_prob, log_probs, grad_utils = self.prior_log_prob(
            t, x, scaling_factor=scaling_factor, return_grad_utils=True)
        probs = torch.nn.functional.softmax(log_probs, dim=-1).unsqueeze(-1)
        # Compute the gradient
        if self.is_full_gmm_type:
            grad = -torch.sum(probs * grad_utils[0], dim=1)
        else:
            grad = -torch.sum(probs * (x.unsqueeze(1) - grad_utils[0].unsqueeze(0)) / grad_utils[1].unsqueeze(0), dim=1)
        return log_prob, grad

    def scaling_input(self, t, x, scaling_factor):
        """Input normalization in a Karras fashion"""
        c_i = scaling_factor * self.sde.s(t) * torch.sqrt(self.var_gauss + self.sde.sigma_sq(t))
        c_m = scaling_factor * self.sde.s(t) * self.mean_gauss
        return (x - c_m) / c_i

    def base_energy(self, t, x, scaling_factor=1.0):
        """Energy of the base model"""
        if self.tilt_type == 'dot':
            x_scaled = self.scaling_input(t, x, scaling_factor=scaling_factor)
            return torch.sum(self.base_model(t, x_scaled) * x_scaled, dim=-1)
        elif self.tilt_type == 'sq_norm':
            x_scaled = self.scaling_input(t, x, scaling_factor=scaling_factor)
            return 0.5 * torch.sum(torch.square(self.base_model(t, x_scaled)), dim=-1)
        else:
            return self.base_model(t, self.scaling_input(t, x, scaling_factor=scaling_factor)).sum(dim=-1)

    def base_unnorm_log_prob(self, t, x, scaling_factor=1.0):
        """Log-likelihood of the base model as the negative energy"""
        return -self.base_energy(t, x, scaling_factor=scaling_factor)

    def energy(self, t, x, scaling_factor=1.0):
        """Energy of the model"""
        if self.use_s_t_scaling:
            base_energy_factor = self.sde.s(t).flatten()
        else:
            base_energy_factor = 1.0
        if self.use_scaling_factor:
            return -self.prior_log_prob_v(t, x, scaling_factor) + base_energy_factor * self.base_energy(t, x,
                                                                                                        scaling_factor)
        else:
            return -self.prior_log_prob_v(t, x) + base_energy_factor * self.base_energy(t, x)

    def unnorm_log_prob(self, t, x, scaling_factor=1.0):
        """Log-likelihood of the model as the negative energy"""
        return -self.energy(t, x, scaling_factor=scaling_factor)

    def unnorm_log_prob_and_grad(self, t, x, scaling_factor=1.0, retain_graph=False):
        """Log-likelihood and score of the model"""
        if self.use_s_t_scaling:
            base_energy_factor = self.sde.s(t)
        else:
            base_energy_factor = 1.0
        with torch.set_grad_enabled(True):
            base_grad, base_log_prob = compute_gradx(self.base_unnorm_log_prob, t=t, x=x,
                                                     scaling_factor=scaling_factor, retain_graph=retain_graph)
        if self.use_scaling_factor:
            prior_log_prob, prior_grad = self.prior_log_prob_and_grad_v(t, x, scaling_factor)
        else:
            prior_log_prob, prior_grad = self.prior_log_prob_and_grad_v(t, x)
        log_prob = prior_log_prob
        if self.use_s_t_scaling:
            log_prob += base_energy_factor.flatten() * base_log_prob
        else:
            log_prob += base_log_prob
        grad = prior_grad + base_energy_factor * base_grad
        return log_prob, grad

    def forward(self, t, x, scaling_factor=1.0):
        """Compute the score of the model"""
        if t.shape[0] == 1:
            t = t.flatten() * torch.ones((x.shape[0], 1), device=x.device)
        return self.unnorm_log_prob_and_grad(t, x, scaling_factor=scaling_factor, retain_graph=True)[1]


class GaussTiltedPotential(GMMTitledPotential):
    """Advanced class to define a NN as the gradient of another NN multiplied with a perfectly noised Gaussian.
    In implemented,

        E(t, X) = E_{NN}(t, X) * Gauss(t, X)
        s(t, X) = -nabla_X E(t,X)

    This classc is used together with EBMs"""

    def __init__(self, base_model, sde, mean, variance, t_limit=0.0,
                 tilt_type='dot', use_s_t_scaling=False, use_scaling_factor=False):
        """Constructor

        Args:
            * base_model (torch.nn.Module): Initial NN
            * sde (OU): SDE
            * mean (torch.Tensor of shape (dim,)): Mean of the Gaussian
            * variance (torch.Tensor of shape (dim,) or (dim, dim) or tuple): Covariance of the Gaussian
            * t_limit (float): Threshold time to ignore the annealed 
            * use_s_t_scaling (bool): Use sde.s to anneal the Gaussian prior overtime (default is True)
            * tilt_type (str): Type of energy function (default is 'sq_norm')
                - dot
                        E(t, x) = NN(t,X)^T X
                - sq_norm
                        E(t, X) = norm(NN(t,X))^2
            * use_scaling_factor (bool): Whether a scaling factor args is passed everywhere (for DRL)

        """

        super(GMMTitledPotential, self).__init__()
        # Store the neural network
        self.tilt_type = tilt_type
        self.base_model = base_model
        self.use_s_t_scaling = use_s_t_scaling
        # Store the SDE
        self.sde = sde
        # Store the GMM params
        self.dim = mean.shape[-1]
        self.t_limit = t_limit
        self.is_full_cov_type = isinstance(variance, tuple) or (len(variance.shape) == 2)
        self.register_buffer("mean", mean)
        self.register_buffer("mean_gauss", mean)
        self.use_full_decomp = isinstance(variance, tuple)
        if isinstance(variance, tuple):
            self.register_buffer("cov_D", variance[0])
            self.register_buffer("cov_P", variance[1])
            self.register_buffer("variance", torch.einsum('...ik,...k,...jk->...ij',
                                                          self.cov_P, self.cov_D, self.cov_P))
        else:
            self.register_buffer("variance", variance)
        # Get the gaussian approxmation
        if self.is_full_cov_type:
            self.prior_final = GaussFull(dim=self.mean.shape[-1], loc=self.mean, cov=self.variance)
        else:
            self.prior_final = Gauss(dim=self.mean.shape[-1], loc=self.mean, scale=self.variance.sqrt())
        self.register_buffer("var_gauss", self.prior_final.distr.variance)
        # Vectorize the prior computations
        self.use_scaling_factor = use_scaling_factor
        if self.use_scaling_factor:
            self.prior_log_prob_v = torch.vmap(lambda t, x, scaling_factor: self.prior_log_prob(
                t, x.unsqueeze(0), scaling_factor).squeeze(0))
            self.prior_log_prob_and_grad_v = torch.vmap(
                lambda t, x, scaling_factor: tuple(
                    y.squeeze(0) for y in self.prior_log_prob_and_grad(t, x.unsqueeze(0), scaling_factor)),
                out_dims=(0, 0))
        else:
            self.prior_log_prob_v = torch.vmap(lambda t, x: self.prior_log_prob(t, x.unsqueeze(0)).squeeze(0))
            self.prior_log_prob_and_grad_v = torch.vmap(
                lambda t, x: tuple(y.squeeze(0) for y in self.prior_log_prob_and_grad(t, x.unsqueeze(0))),
                out_dims=(0, 0))
        # Indicate capabilities
        self.has_unnorm_log_prob_and_grad = True
        self.has_sample_prior = True

    def get_gauss_params(self, t, scaling_factor=1.0):
        """Get the parameters of the Gaussian at time t"""
        if self.use_full_decomp:
            variance_init = (scaling_factor ** 2 * self.cov_D, self.cov_P)
        else:
            variance_init = scaling_factor ** 2 * self.variance
        return self.sde.marginal_params(
            torch.maximum(t, self.t_limit * torch.ones_like(t)),
            x_init=scaling_factor * self.mean,
            var_init=variance_init,
        )

    def prior_log_prob(self, t, x, scaling_factor=1.0, return_grad_utils=False):
        """Compute the likelihood of the noised Gaussian distribution at time t (NOT VECTORIZED ON TIME)"""
        # Compute the parameters at time t
        mean_t, variance_t = self.get_gauss_params(t, scaling_factor=scaling_factor)
        if isinstance(variance_t, tuple):
            prec_t, log_det_cov_t = variance_t
        else:
            prec_t, log_det_cov_t = None, None
        # Compute the log-prob at time t
        if self.is_full_cov_type:
            if return_grad_utils:
                log_prob, utils = log_prob_gaussian_full(x, mean_t, variance_t, precisions=prec_t,
                                                         covariances_log_det=log_det_cov_t,
                                                         return_precision_times_diff=True)
                return log_prob.squeeze(-1), utils
            else:
                return log_prob_gaussian_full(x, mean_t, variance_t, precisions=prec_t,
                                              covariances_log_det=log_det_cov_t,
                                              return_precision_times_diff=False).squeeze(-1)
        else:
            log_prob = log_prob_gaussian(x, mean_t, variance_t).squeeze(-1)
            if return_grad_utils:
                return log_prob, (mean_t, variance_t)
            else:
                return log_prob

    def prior_log_prob_and_grad(self, t, x, scaling_factor=1.0):
        """Compute the likelihood and score of the noised Gaussian distribution at time t (NOT VECTORIZED ON TIME)"""
        # Compute the log_prob
        log_prob, utils = self.prior_log_prob(t, x, scaling_factor=scaling_factor,
                                              return_grad_utils=True)
        if self.is_full_cov_type:
            grad = -utils
        else:
            grad = -(x - utils[0]) / utils[1]
        return log_prob, grad.squeeze(1)


class DRLWrapper(torch.nn.Module):
    """Wrapper around a DRL trained EBM to obtain something on the right space"""

    def __init__(self, net, scaling_factors):
        super().__init__()
        self.net = net
        self.scaling_factors = scaling_factors
        self.has_unnorm_log_prob_and_grad = self.net.has_unnorm_log_prob_and_grad
        self.has_sample_prior = False

    def unnorm_log_prob_and_grad(self, t, x, scaling_factor=1.0, retain_graph=False):
        log_prob, grad = self.net.unnorm_log_prob_and_grad(t, scaling_factor * x, scaling_factor=scaling_factor)
        return log_prob, scaling_factor * grad

    def unnorm_log_prob(self, t, x, scaling_factor=1.0):
        return self.net.unnorm_log_prob(t, scaling_factor * x, scaling_factor=scaling_factor)

    def energy(self, t, x, scaling_factor=1.0):
        return self.net.energy(t, scaling_factor * x, scaling_factor=scaling_factor)

    def forward(self, t, x, scaling_factor=1.0):
        return scaling_factor * self.net.forward(t, scaling_factor * x, scaling_factor=scaling_factor)


class EBMAnatomyTrick(torch.nn.Module):
    """Apply the EBM Anatomy trick by dividing the energy by the step size of Langevin"""

    def __init__(self, net, times, steps_sizes):
        super().__init__()
        self.net = net
        self.register_buffer("times", times.flatten())
        self.register_buffer("steps_sizes", steps_sizes.flatten())
        self.has_unnorm_log_prob_and_grad = self.net.has_unnorm_log_prob_and_grad
        self.has_sample_prior = False
        if self.times.shape != self.steps_sizes.shape:
            raise ValueError('Times and step_sizes should have the same (flattened) shape')

    def find_factor(self, t):
        return self.steps_sizes[torch.searchsorted(self.times, t)].flatten()

    def unnorm_log_prob_and_grad(self, t, x, scaling_factor=1.0, retain_graph=False):
        log_prob, grad = self.net.unnorm_log_prob_and_grad(
            t, x, scaling_factor=scaling_factor, retain_graph=retain_graph)
        factor = self.find_factor(t)
        return log_prob / factor, grad / factor.view((-1, *(1,) * (len(x.shape) - 1)))

    def unnorm_log_prob(self, t, x, scaling_factor=1.0):
        return self.net.unnorm_log_prob(t, x, scaling_factor=scaling_factor) / self.find_factor(t)

    def energy(self, t, x, scaling_factor=1.0):
        return self.net.energy(t, x, scaling_factor=scaling_factor) / self.find_factor(t)

    def forward(self, t, x, scaling_factor=1.0):
        factor = self.find_factor(t).view((-1, *(1,) * (len(x.shape) - 1)))
        return self.net(t, x, scaling_factor=scaling_factor) / factor
