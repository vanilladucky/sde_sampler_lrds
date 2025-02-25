from __future__ import annotations

from typing import Callable

import math
import torch
from torch.nn import Module

from sde_sampler.distr.gauss import Gauss, GaussFull, GMM, GMMFull
from sde_sampler.distr.gauss import score_gauss, score_gauss_full, score_mog, score_mog_full
from sde_sampler.utils.common import clip_and_log


class TorchSDE(Module):
    """Generic SDE class"""

    noise_type: str = "diagonal"
    sde_type: str = "ito"

    def __init__(
            self,
            terminal_t: float = 1.0,
    ):
        super().__init__()
        self.register_buffer(
            "terminal_t", torch.tensor(terminal_t, dtype=torch.float), persistent=False
        )

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift"""
        raise NotImplementedError

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion"""
        raise NotImplementedError

    def f(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift with the same shape as x"""
        return self.drift(t, x).expand_as(x)

    def g(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion with the same shape as x"""
        return self.diff(t, x).expand_as(x)


class LangevinSDE(TorchSDE):
    """Classic Langevin SDE"""

    def __init__(
            self,
            target_score: Callable,
            diff_coeff: float = 1.0,
            clip_score: float | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_score = target_score
        self.register_buffer(
            "diff_coeff", torch.tensor(diff_coeff, dtype=torch.float), persistent=False
        )
        self.clip_score = clip_score

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift"""
        drift = self.target_score(x) * self.diff_coeff ** 2 / 2.0
        return clip_and_log(
            drift,
            max_norm=self.clip_score,
            name="score",
            t=t,
        )

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion"""
        return self.diff_coeff


class ControlledLangevinSDE(TorchSDE):
    """Langevin SDE along a tempering path"""

    def __init__(
            self,
            target_score: Callable,
            prior_score: Callable,
            diff_coeff: float = 1.0,
            terminal_t: float = 1.0,
            clip_score: float | None = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.target_score = target_score
        self.prior_score = prior_score
        self.register_buffer(
            "diff_coeff", torch.tensor(diff_coeff, dtype=torch.float), persistent=False
        )
        self.register_buffer(
            "terminal_t", torch.tensor(terminal_t, dtype=torch.float), persistent=False
        )
        self.clip_score = clip_score

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift"""
        drift = self.target_score(x) * (t / self.terminal_t) + self.prior_score(x) * (1. - t / self.terminal_t)
        drift *= 0.5 * self.diff_coeff ** 2
        return clip_and_log(
            drift,
            max_norm=self.clip_score,
            name="tempering_score",
            t=t,
        )

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion"""
        return self.diff_coeff


class OU(TorchSDE):
    """Generic linear SDE (or Ornstein-Uhlenbeck SDE) class

    It implements

        dX_t = drift_coeff_t(t) * X dt + diff_coeff_t(t) dW_t
    """

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Value of diff_coeff_t"""
        raise NotImplementedError

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Value of diff_coef_t"""
        raise NotImplementedError

    def drift_div(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Divergence of drift_coeff_t"""
        return self.drift_coeff_t(t) * x.shape[-1]

    def drift_div_int(
            self, s: torch.Tensor, t: torch.Tensor, x: torch.Tensor
    ) -> torch.Tensor:
        """Integral from s to t of the divergence of drift_coeff_t"""
        return self.int_drift_coeff_t(s, t) * x.shape[-1]

    def drift(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Drift of the SDE defined as drift_coeff_t(t) * x"""
        return self.drift_coeff_t(t) * x

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient"""
        return self.diff_coeff_t(t)

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of drift_coeff_t"""
        raise NotImplementedError

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of diff_coeff_t"""
        raise NotImplementedError

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Value of exp(int_0_t drift_coef_t(u) du)"""
        raise NotImplementedError

    def sigma_sq(self, t: torch.Tensor) -> torch.Tensor:
        """Value of int_0_t diff_coef_t(u) / s(u) du"""
        raise NotImplementedError

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel from s to t (with s < t)

        This means that

            X_t = mean_factor * X_s + sqrt(var_factor) * Z

        where Z ~ N(0, I)
        """
        mean_factor = self.s(t) - self.s(s)
        var_factor = self.s(t) ** 2 * (self.sigma_sq(t) - self.sigma_sq(s))
        return mean_factor, var_factor

    def omega_ddpm(self, t_k, t_k_p_1):
        """Weight betwen t_k and t_k_p_1 (t_k < t_k_p_1) of the VI loss with DDPM-like transition kernels.
        Remark: this general function may be unstable, rather use the specific versions for VP and PBM schemes."""
        T = self.terminal_t
        alpha_k_p_1_m_k, sigma_sq_k_p_1_m_k = self.transition_params(T - t_k_p_1, T - t_k)
        alpha_k_p_1_m_0, sigma_sq_k_p_1_m_0 = self.s(T - t_k), self.s(T - t_k) ** 2 * self.sigma_sq(T - t_k)
        alpha_k_m_0, sigma_sq_k_m_0 = self.s(T - t_k_p_1), self.s(T - t_k_p_1) ** 2 * self.sigma_sq(T - t_k_p_1)
        w_k_sq = (alpha_k_m_0 ** 2 / alpha_k_p_1_m_0 ** 2) * (sigma_sq_k_p_1_m_0 ** 2 / sigma_sq_k_m_0 ** 2)
        var = sigma_sq_k_p_1_m_k * sigma_sq_k_m_0
        var /= sigma_sq_k_p_1_m_k + sigma_sq_k_m_0 * alpha_k_p_1_m_k ** 2
        return w_k_sq * var

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, z=None):
        """DDPM-like denoising transition kernel from t_k to t_k_p_1 (t_k < t_k_p_1) conditioned on x.
        Remark: this general function may be unstable, rather use the specific versions for VP and PBM schemes."""
        T = self.terminal_t
        alpha_k_p_1_m_k, sigma_sq_k_p_1_m_k = self.transition_params(T - t_k_p_1, T - t_k)
        alpha_k_p_1_m_0, sigma_sq_k_p_1_m_0 = self.s(T - t_k), self.s(T - t_k) ** 2 * self.sigma_sq(T - t_k)
        alpha_k_m_0, sigma_sq_k_m_0 = self.s(T - t_k_p_1), self.s(T - t_k_p_1) ** 2 * self.sigma_sq(T - t_k_p_1)
        x_0 = (sigma_sq_k_p_1_m_0 * s + x) / alpha_k_p_1_m_0
        var = sigma_sq_k_p_1_m_k * sigma_sq_k_m_0
        var /= sigma_sq_k_p_1_m_k + sigma_sq_k_m_0 * alpha_k_p_1_m_k ** 2
        mean = var * ((alpha_k_p_1_m_k / sigma_sq_k_p_1_m_k) * x + (alpha_k_m_0 / sigma_sq_k_m_0) * x_0)
        if z is None:
            z = torch.randn_like(x)
        ret = mean + torch.sqrt(var) * z
        return ret, z

    def marginal_params(
            self,
            t: torch.Tensor,
            x_init: torch.Tensor,
            var_init: torch.Tensor | None = None,
            is_mixture: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the parameters of a Gaussian marginal

        Args:
            * t (float): Current time
            * x_init (torch.Tensor of shape (dim,) or (n_modes, dim)): Mean of the Gaussian
            * var_init (torch.Tensor of shape (dim,) or (n_modes, dim) or (dim, dim) or (n_modes, dim, dim)): Variance of the Gaussian
                var_init can also be a tuple which would corespond to a decomposition between a Cholesky matrix and a log-determinant
            * is_mixture (bool): Whether there are multiple means or variances provided

        Returns:
            * loc (torch.Tensor of the same shape as x_init): Mean of the marginal
            * var (torch.Tensor of the same shape as var): Variance of the marginal
        """
        loc = self.s(t) * x_init
        var = self.s(t) ** 2 * self.sigma_sq(t)
        if var_init is not None:
            # Check if the variance is given as a (P,D) decomposition
            if isinstance(var_init, tuple):
                # Compute the precision and covariance
                diag = var_init[0] + self.sigma_sq(t)
                prec = torch.einsum('...ik,...k,...jk->...ij', var_init[1], 1. / diag, var_init[1])
                prec /= self.s(t) ** 2
                log_det = torch.sum(torch.log(diag), dim=-1)
                log_det += 2. * diag.shape[-1] * torch.log(self.s(t))
                var = (prec, log_det)
            else:
                if is_mixture:
                    if len(var_init.shape) == 3:
                        var = var * torch.eye(var_init.shape[-1], device=var.device).unsqueeze(0)
                else:
                    if len(var_init.shape) == 2:
                        var = var * torch.eye(var_init.shape[-1], device=var.device)
                var = var + self.s(t) ** 2 * var_init
        return loc, var

    def marginal_distr(
            self,
            t: torch.Tensor,
            x_init: torch.Tensor,
            var_init: torch.Tensor | None = None,
    ) -> Gauss:
        """Get the marginal distribution object given a target of mean x_init and variance var_init"""
        loc, var = self.marginal_params(t, x_init, var_init=var_init)
        if isinstance(var, tuple):
            return GaussFull(dim=x_init.shape[-1], loc=loc, prec=var[0], domain_tol=None)
        elif len(var.shape) == 2:
            return GaussFull(dim=x_init.shape[-1], loc=loc, cov=var, domain_tol=None)
        else:
            return Gauss(dim=x_init.shape[-1], loc=loc, scale=var.sqrt(), domain_tol=None)

    def marginal_score(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            x_init: torch.Tensor,
            var_init: torch.Tensor | None = None,
    ) -> Gauss:
        """Compute the marginal score at t and x given a target of mean x_init and variance var_init"""
        loc, var = self.marginal_params(t, x_init, var_init=var_init)
        if isinstance(var, tuple):
            return score_gauss_full(x, loc, var, precisions=var[0])
        elif len(var.shape) == 2:
            return score_gauss_full(x, loc, var)
        else:
            return score_gauss(x, loc, var)

    def marginal_gmm_params(
            self,
            t: torch.Tensor,
            means_init: torch.Tensor,
            variances_init: torch.Tensor,
            weights_init: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the parameters of a Gaussian mixture marginal

        Args:
            * t (float): Current time
            * means_init (torch.Tensor of shape (n_modes, dim)): Means
            * variances_init (torch.Tensor of shape (n_modes, dim) or (n_modes, dim, dim)): Variances
                var_init can also be a tuple which would corespond to a decomposition between a Cholesky matrix and a log-determinant
            * weights_init (torch.Tensor of shape (n_modes,)): Weights

        Returns:
            * weights (torch.Tensor of the same shape as weights_init): Weights of the marginal
            * means (torch.Tensor of the same shape as means_init): Mean of the marginal
            * variances (torch.Tensor of the same shape as variances_init): Variance of the marginal
        """
        means, variances = self.marginal_params(t, x_init=means_init, var_init=variances_init, is_mixture=True)
        if weights_init is None:
            weights = torch.ones((means.shape[0],), device=means.device) / means.shape[0]
        else:
            weights = weights_init
        return weights, means, variances

    def marginal_gmm_distr(
            self,
            t: torch.Tensor,
            means_init: torch.Tensor,
            variances_init: torch.Tensor,
            weights_init: torch.Tensor | None = None,
    ) -> GMM:
        """Get the marginal distribution object given a Gaussian mixture target"""
        weights, means, variances = self.marginal_gmm_params(t, means_init=means_init, variances_init=variances_init,
                                                             weights_init=weights_init)
        if isinstance(variances, tuple):
            return GMMFull(dim=means_init.shape[-1], loc=means, prec=variances[0], cov_log_det=variances[1],
                           mixture_weights=weights, domain_tol=None)
        elif len(variances.shape) == 3:
            return GMMFull(dim=means_init.shape[-1], loc=means, cov=variances,
                           mixture_weights=weights, domain_tol=None)
        else:
            return GMM(dim=means_init.shape[-1], loc=means, scale=torch.sqrt(variances),
                       mixture_weights=weights, domain_tol=None)

    def marginal_gmm_score(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
            means_init: torch.Tensor,
            variances_init: torch.Tensor,
            weights_init: torch.Tensor | None = None,
    ) -> Gauss:
        """Compute the marginal score at t and x given a Gaussian mixture target"""
        weights, means, variances = self.marginal_gmm_params(t, means_init=means_init, variances_init=variances_init,
                                                             weights_init=weights_init)
        if isinstance(variances, tuple):
            return score_mog_full(x, weights, means, None, precisions=variances[0], covariances_log_det=variances[1])
        elif len(variances.shape) == 3:
            return score_mog_full(x, weights, means, variances)
        else:
            return score_mog(x, weights, means, variances)

    def log_snr(self, t):
        """Compute the log-SNR at time t"""
        alpha_bar_t = self.s(t)
        sigmas_sq_bar_t = torch.square(alpha_bar_t) * self.sigma_sq(t)
        return torch.log(torch.square(alpha_bar_t) / sigmas_sq_bar_t)


class ConstOU(OU):
    """Special case of linear SDE with constant drift and diffusion.

    It implements

        dX_t = -drift_coeff X dt + diff_coeff dW_t
    """

    def __init__(self, drift_coeff: float = 2.0, diff_coeff: float = 2.0, **kwargs):
        if drift_coeff < 0 or diff_coeff <= 0:
            raise ValueError("Choose non-negative drift_coeff and positive diff_coeff.")
        super().__init__(**kwargs)
        self.register_buffer(
            "drift_coeff",
            torch.tensor(drift_coeff, dtype=torch.float),
            persistent=False,
        )
        self.drift_coeff: torch.Tensor
        self.register_buffer(
            "diff_coeff", torch.tensor(diff_coeff, dtype=torch.float), persistent=False
        )
        self.diff_coeff: torch.Tensor

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Value of drift_coeff_t"""
        return -self.drift_coeff

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Value of diff_coeff_t"""
        return self.diff_coeff

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of drift_coeff_t"""
        dt = t - s
        assert (dt >= 0).all()
        return -self.drift_coeff * dt

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of diff_coeff_t"""
        dt = t - s
        assert (dt >= 0).all()
        return self.diff_coeff ** 2 * dt

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Value of exp(int_0_t drift_coef_t(u) du)"""
        return torch.exp(-self.drift_coeff * t)

    def sigma_sq(self, t: torch.Tensor) -> torch.Tensor:
        """Value of int_0_t diff_coef_t(u) / s(u) du"""
        return -0.5 * self.diff_coeff ** 2 * (1. - torch.exp(2. * self.drift_coeff * t))


class ScaledBM(ConstOU):
    """Brownian motion

    It implements

        dX_t = sigma dW_t

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, drift_coeff=0.0, **kwargs)

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Value of exp(int_0_t drift_coef_t(u) du)"""
        return torch.ones_like(t)

    def sigma_sq(self, t: torch.Tensor) -> torch.Tensor:
        """Value of int_0_t diff_coef_t(u) / s(u) du"""
        return self.diff_coeff ** 2 * t


class VP(OU):
    """Variance Preserving SDE with linear scheduling"""

    def __init__(
            self,
            diff_coeff_sq_min: float = 0.1,
            diff_coeff_sq_max: float = 20.0,
            scale_diff_coeff: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_buffer(
            "scale_diff_coeff",
            torch.tensor(scale_diff_coeff, dtype=torch.float),
            persistent=False,
        )
        self.register_buffer(
            "diff_coeff_sq_min",
            torch.tensor(diff_coeff_sq_min, dtype=torch.float),
            persistent=False,
        )
        self.diff_coeff_sq_min: torch.Tensor
        self.register_buffer(
            "diff_coeff_sq_max",
            torch.tensor(diff_coeff_sq_max, dtype=torch.float),
            persistent=False,
        )
        self.diff_coeff_sq_max: torch.Tensor

    def _diff_coeff_sq_t(self, t: torch.Tensor) -> torch.Tensor:
        return torch.lerp(
            self.diff_coeff_sq_min, self.diff_coeff_sq_max, t / self.terminal_t
        )

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient"""
        return -0.5 * self._diff_coeff_sq_t(t)

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient"""
        return self.scale_diff_coeff * torch.sqrt(self._diff_coeff_sq_t(t))

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of drift_coeff_t"""
        dt = t - s
        assert (dt >= 0).all()
        return (
            -0.25
            * (self._diff_coeff_sq_t(t) + self._diff_coeff_sq_t(s))
            * dt
        )

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of difft_coeff_t"""
        dt = t - s
        assert (dt >= 0).all()
        return (
            0.5
            * self.scale_diff_coeff ** 2
            * (self._diff_coeff_sq_t(t) + self._diff_coeff_sq_t(s))
            * dt
        )

    def alpha_(self, t):
        """Linear scheduling"""
        return self.diff_coeff_sq_min * t + (0.5 * t ** 2 / self.terminal_t) \
            * (self.diff_coeff_sq_max - self.diff_coeff_sq_min)

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel from s to t (with s < t)

        This means that

            X_t = mean_factor * X_s + sqrt(var_factor) * Z

        where Z ~ N(0, I)
        """
        lambda_s_t = 1. - torch.exp(self.alpha_(s) - self.alpha_(t))
        mean_factor = torch.sqrt(1. - lambda_s_t)
        var_factor = self.scale_diff_coeff ** 2 * lambda_s_t
        return mean_factor, var_factor

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Value of exp(int_0_t drift_coef_t(u) du)"""
        return torch.exp(-0.5 * self.alpha_(t))

    def sigma_sq(self, t: torch.Tensor) -> torch.Tensor:
        """Value of int_0_t diff_coef_t(u) / s(u) du"""
        return -self.scale_diff_coeff ** 2 * (1. - (1. / self.s(t) ** 2))

    def omega(self, t_k, t_k_p_1):
        """Weight betwen t_k and t_k_p_1 (t_k < t_k_p_1) of the VI loss with EI transition kernels."""
        return 4. * self.scale_diff_coeff ** 2 * torch.tanh(
            (self.alpha_(self.terminal_t - t_k) - self.alpha_(self.terminal_t - t_k_p_1)) / 4.)

    def lambda_(self, t_k, t_k_p_1):
        """Function lambda used internally (note that t_k < t_k_p_1)"""
        return torch.exp(self.alpha_(self.terminal_t - t_k) - self.alpha_(self.terminal_t - t_k_p_1)) - 1.

    def omega_ddpm(self, t_k, t_k_p_1):
        """Weight betwen t_k and t_k_p_1 (t_k < t_k_p_1) of the VI loss with DDPM-like transition kernels."""
        lambda_k_ = 1. - torch.exp(-self.alpha_(self.terminal_t - t_k))
        lambda_k_p_1_ = 1. - torch.exp(-self.alpha_(self.terminal_t - t_k_p_1))
        return self.scale_diff_coeff ** 2 * (lambda_k_ / lambda_k_p_1_) * self.lambda_(t_k, t_k_p_1)

    def ei_integration_step(self, x, t_k, t_k_p_1, s, z=None):
        """EI denoising transition kernel from t_k to t_k_p_1 (t_k < t_k_p_1) conditioned on x."""
        lambda_k = self.lambda_(t_k, t_k_p_1)
        ret = torch.sqrt(1. + lambda_k) * x + 2. * self.scale_diff_coeff ** 2 * (torch.sqrt(1. + lambda_k) - 1.) * s
        if z is None:
            z = torch.randn_like(ret)
        ret += self.scale_diff_coeff * torch.sqrt(lambda_k) * z
        return ret, z

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, z=None, min_clip=0.5):
        """DDPM-like denoising transition kernel from t_k to t_k_p_1 (t_k < t_k_p_1) conditioned on x."""
        T = self.terminal_t
        lambda_ = self.lambda_(t_k, t_k_p_1)
        lambda__ = 1. - torch.exp(self.alpha_(T - t_k_p_1) - self.alpha_(T - t_k))
        lambda_k = 1. - torch.exp(-self.alpha_(T - t_k))
        lambda_k_p_1 = 1. - torch.exp(-self.alpha_(T - t_k_p_1))
        diff_alpha = (self.alpha_(self.terminal_t - t_k) - self.alpha_(self.terminal_t - t_k_p_1)) / 2.
        # Clip to avoid numerical issues
        var = self.scale_diff_coeff ** 2 * lambda__ * (lambda_k_p_1 / lambda_k)
        mean = torch.sqrt(1. + lambda_) * x + 2. * self.scale_diff_coeff ** 2 * torch.sinh(diff_alpha) * s
        if z is None:
            z = torch.randn_like(x)
        ret = mean + torch.sqrt(var) * z
        return ret, z


class CosineVP(VP):
    """Variance Preserving SDE with cosine scheduling"""

    def __init__(
            self,
            c: float = 0.008,
            scale_diff_coeff: float = 1.0,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.register_buffer(
            "c",
            torch.tensor(c, dtype=torch.float),
            persistent=False,
        )
        self.register_buffer(
            "scale_diff_coeff",
            torch.tensor(scale_diff_coeff, dtype=torch.float),
            persistent=False,
        )

    def _diff_coeff_sq_t(self, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of diff_coeff_t"""
        return torch.pi * torch.tan(0.5 * torch.pi * ((t / self.terminal_t) + self.c) / (1. + self.c)) \
            / (self.terminal_t * (1. + self.c))

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of drift_coeff_t"""
        raise NotImplementedError('int_drift_coeff_t is not yet implemented')

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of diff_coeff_t"""
        raise NotImplementedError('int_diff_coeff_sq_t is not yet implemented')

    def alpha_(self, t):
        """Cosine scheduling"""
        return -2. * torch.log(torch.cos(0.5 * torch.pi * ((t / self.terminal_t) + self.c) / (1. + self.c)))


class PinnedBM(OU):
    """Pinned Brownian Motion"""

    def __init__(self, diff_coeff: float = 2.0, **kwargs):
        if diff_coeff <= 0:
            raise ValueError("Choose positive diff_coeff.")
        super().__init__(**kwargs)
        self.register_buffer(
            "diff_coeff", torch.tensor(diff_coeff, dtype=torch.float), persistent=False
        )
        self.diff_coeff: torch.Tensor

    def drift_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Drift coefficient"""
        return -1. / (self.terminal_t - t)

    def diff_coeff_t(self, t: torch.Tensor) -> torch.Tensor:
        """Diffusion coefficient"""
        return self.diff_coeff

    def int_drift_coeff_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of drift_coeff_t"""
        assert ((t - s) >= 0).all()
        return torch.log(self.terminal_t - t) - torch.log(self.terminal_t - s)

    def int_diff_coeff_sq_t(self, s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Integral from s to t of difft_coeff_t"""
        dt = t - s
        assert (dt >= 0).all()
        return self.diff_coeff ** 2 * dt

    def transition_params(self, s, t):
        """Mean and variance parameters for noising transition kernel from s to t (with s < t)

        This means that

            X_t = mean_factor * X_s + sqrt(var_factor) * Z

        where Z ~ N(0, I)
        """
        mean_factor = (self.terminal_t - t) / (self.terminal_t - s)
        var_factor = mean_factor * (t - s) * self.diff_coeff ** 2
        return mean_factor, var_factor

    def s(self, t: torch.Tensor) -> torch.Tensor:
        """Value of exp(int_0_t drift_coef_t(u) du)"""
        return (self.terminal_t - t) / self.terminal_t

    def sigma_sq(self, t: torch.Tensor) -> torch.Tensor:
        """Value of int_0_t diff_coef_t(u) / s(u) du"""
        return self.diff_coeff ** 2 * self.terminal_t * t / (self.terminal_t - t)

    def omega(self, t_k, t_k_p_1):
        """Weight betwen t_k and t_k_p_1 (t_k < t_k_p_1) of the VI loss with EI transition kernels."""
        return self.diff_coeff ** 2 * (t_k / t_k_p_1) * (t_k_p_1 - t_k)

    def omega_ddpm(self, t_k, t_k_p_1):
        """Weight betwen t_k and t_k_p_1 (t_k < t_k_p_1) of the VI loss with DDPM-like transition kernels."""
        T = self.terminal_t
        return self.diff_coeff ** 2 * ((T - t_k) / (T - t_k_p_1)) * (t_k_p_1 - t_k)

    def ei_integration_step(self, x, t_k, t_k_p_1, s, z=None):
        """EI denoising transition kernel from t_k to t_k_p_1 (t_k < t_k_p_1) conditioned on x."""
        ret = (t_k_p_1 / t_k) * x + self.diff_coeff ** 2 * (t_k_p_1 - t_k) * s
        if z is None:
            z = torch.randn_like(ret)
        # numerical issues may arise from the following term for small times
        var = self.diff_coeff ** 2 * (t_k_p_1 / t_k) * (t_k_p_1 - t_k)
        ret += torch.sqrt(var) * z
        return ret, z

    def ddpm_integration_step(self, x, t_k, t_k_p_1, s, z=None):
        """DDPM-like denoising transition kernel from t_k to t_k_p_1 (t_k < t_k_p_1) conditioned on x."""
        T = self.terminal_t
        var = self.diff_coeff ** 2 * ((T - t_k_p_1) / (T - t_k)) * (t_k_p_1 - t_k)
        # numerical issues may arise from the following term for small times
        mean = (t_k_p_1 / t_k) * x
        mean += self.diff_coeff ** 2 * (t_k_p_1 - t_k) * s
        if z is None:
            z = torch.randn_like(x)
        ret = mean + torch.sqrt(var) * z
        return ret, z


class ControlledSDE(TorchSDE):
    """SDE with a control term"""

    def __init__(
            self,
            sde: OU,
            ctrl: Callable | None,
            **kwargs,
    ):
        super().__init__(terminal_t=sde.terminal_t.item(), **kwargs)
        self.sde = sde
        self.sde_type = self.sde.sde_type
        self.noise_type = self.sde.noise_type
        self.ctrl = ctrl

    def drift(
            self,
            t: torch.Tensor,
            x: torch.Tensor,
    ) -> torch.Tensor:
        """Drift"""
        return self.f_and_g(t, x)[0]

    def diff(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Diffusion"""
        return self.sde.diff(t, x)

    # Minimal speedup by saving one diff evaluation for torchsde
    def f_and_g(
            self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sde_diff = self.sde.diff(t, x)
        sde_drift = self.sde.drift(t, x)
        if self.ctrl is not None:
            sde_drift += sde_diff * self.ctrl(self.terminal_t - t, x)
        return sde_drift, sde_diff.expand_as(x)
