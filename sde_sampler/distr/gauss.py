from __future__ import annotations

import logging
import math
from numbers import Number

import torch
from torch import distributions
from torch.nn.init import trunc_normal_

from .base import Distribution


def gmm_params(name: str = "heart", dim: int = 2):
    """Get the MoG parameters for various predefined distributions."""
    if name == "heart":
        loc = 1.5 * torch.tensor(
            [
                [-0.5, -0.25],
                [0.0, -1],
                [0.5, -0.25],
                [-1.0, 0.5],
                [-0.5, 1.0],
                [0.0, 0.5],
                [0.5, 1.0],
                [1.0, 0.5],
            ]
        )
        factor = 1 / len(loc)

    elif name == "dist":
        loc = torch.tensor(
            [
                [0.0, 0.0],
                [2, 0.0],
                [0.0, 3.0],
                [-4, 0.0],
                [0.0, -5],
            ]
        )
        factor = math.sqrt(0.2)

    elif name in ["fab", "multi"]:
        n_mixes, loc_scaling = (40, 40) if name == "fab" else (80, 80)
        generator = torch.Generator()
        generator.manual_seed(42)
        loc = (torch.rand((n_mixes, 2), generator=generator) - 0.5) * 2 * loc_scaling
        factor = torch.nn.functional.softplus(torch.tensor(1.0, device=loc.device))
    elif name == "grid":
        x_coords = torch.linspace(-5, 5, 3)
        loc = torch.cartesian_prod(x_coords, x_coords)
        factor = math.sqrt(0.3)
    elif name == "circle":
        freq = 2 * torch.pi * torch.arange(1, 9) / 8
        loc = torch.stack([4.0 * freq.cos(), 4.0 * freq.sin()], dim=1)
        factor = math.sqrt(0.3)
    else:
        raise ValueError("Unknown mode for the Gaussian mixture.")

    if dim > 2:
        loc = torch.cat([loc, torch.zeros(8, dim - 2)], dim=1)
    scale = factor * torch.ones_like(loc)
    mixture_weights = torch.ones(loc.shape[0], device=loc.device)
    return loc, scale, mixture_weights


def log_prob_gaussian(x, mean, variance):
    """Compute the log likelihood of a Gaussian distribution with diagonal covariance matrix, evaluated at x.
    By vectorisation, can be used to compute the log-likelihood of a MoG distribution."""
    log_prob = -0.5 * torch.sum(torch.square(x.unsqueeze(1) - mean.unsqueeze(0)) / variance.unsqueeze(0), dim=-1)
    log_prob -= 0.5 * mean.shape[-1] * math.log(2. * math.pi)
    log_prob -= 0.5 * torch.log(variance).sum(dim=-1).unsqueeze(0)
    return log_prob


def log_prob_gaussian_full(x, means, covariances, precisions=None, covariances_log_det=None,
                           return_precision_times_diff=False):
    """Compute the log likelihood of a Gaussian distribution with full covariance matrix, evaluated at x.
    By vectorisation, can be used to compute the log-likelihood of a MoG distribution."""
    diff = x.unsqueeze(1) - means.unsqueeze(0)
    if precisions is None:
        precision_times_diff = torch.linalg.solve(covariances.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
    else:
        precision_times_diff = torch.matmul(precisions.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
    log_prob = -0.5 * torch.sum(diff * precision_times_diff, dim=-1)
    log_prob -= 0.5 * means.shape[-1] * math.log(2. * math.pi)
    if covariances_log_det is None:
        log_prob -= 0.5 * torch.logdet(covariances).unsqueeze(0)
    else:
        log_prob -= 0.5 * covariances_log_det.unsqueeze(0)
    if return_precision_times_diff:
        return log_prob, precision_times_diff
    else:
        return log_prob


def score_mog(x, weights, means, variances):
    """Compute the score of a MoG distribution with diagonal covariance matrices on the components, evaluated at x."""
    # Normalize the weights
    weights /= weights.sum()
    # Compute the individual gaussian probs
    gaussian_probs = torch.nn.functional.softmax(
        torch.log(weights.unsqueeze(0)) + log_prob_gaussian(x, means, variances),
        dim=-1)
    # Compute the final score
    return -torch.sum(
        gaussian_probs.unsqueeze(-1) * (x.unsqueeze(1) - means.unsqueeze(0)) / variances.unsqueeze(0), dim=1)


def score_mog_full(x, weights, means, covariances, precisions=None, covariances_log_det=None):
    """Compute the score of a MoG distribution with full covariance matrices on the components, evaluated at x."""
    # Normalize the weights
    weights /= weights.sum()
    # Compute the individual gaussian probs
    log_probs, precision_times_diff = log_prob_gaussian_full(x, means, covariances,
                                                             precisions=precisions,
                                                             covariances_log_det=covariances_log_det,
                                                             return_precision_times_diff=True)
    gaussian_probs = torch.nn.functional.softmax(torch.log(weights.unsqueeze(0)) + log_probs, dim=-1)
    # Compute the final score
    return -torch.sum(gaussian_probs.unsqueeze(-1) * precision_times_diff, dim=1)


def score_gauss(x, means, variances):
    """Compute the score of a Gaussian distribution with diagonal covariance matrix, evaluated at x."""
    return -(x - means) / variances


def score_gauss_full(x, means, covariances, precisions=None):
    """Compute the score of a Gaussian distribution with full covariance matrix, evaluated at x."""
    diff = x - means.unsqueeze(0)
    if precisions is None:
        return -torch.linalg.solve(covariances.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)
    else:
        return -torch.matmul(precisions.unsqueeze(0), diff.unsqueeze(-1)).squeeze(-1)


class GMM(Distribution):
    def __init__(
            self,
            dim: int = 2,
            loc: torch.Tensor | None = None,
            scale: torch.Tensor | None = None,
            mixture_weights: torch.Tensor | None = None,
            n_reference_samples: int = int(1e7),
            name: str | None = None,
            domain_scale: float = 5,
            domain_tol: float | None = 1e-5,
            **kwargs,
    ):
        """Builds a mixture of Gaussians with diagonal covariance matrices on the components.

        Args:
            * dim (int): Dimension
            * mixture_weights (torch.Tensor of shape (n_modes,)): Weights
            * loc (torch.Tensor of shape (n_modes,dim)): Means
            * scale (torch.Tensor of shape (n_modes, dim)): Square root diagonal of the covariances
            * n_reference_samples (int) : number of reference samples
        """
        super().__init__(
            dim=dim,
            log_norm_const=0.0,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        if name is not None:
            if any(t is not None for t in [loc, scale, mixture_weights]):
                logging.warning(
                    "Ignoring loc, scale, and mixture weights since name is specified."
                )
            loc, scale, mixture_weights = gmm_params(name, dim=dim)

        # Check shapes
        self.n_mixtures = loc.shape[0]
        if not (loc.shape == scale.shape == (self.n_mixtures, self.dim)):
            raise ValueError("Shape missmatch between loc and scale.")
        if mixture_weights is None and self.n_mixtures > 1:
            raise ValueError("Require mixture weights.")
        if not (mixture_weights is None or mixture_weights.shape == (self.n_mixtures,)):
            raise ValueError("Shape missmatch for the mixture weights.")

        # Initialize
        self.register_buffer("loc", loc, persistent=False)
        self.register_buffer("scale", scale, persistent=False)
        self.register_buffer("mixture_weights", mixture_weights, persistent=False)
        self._initialize_distr()

        # Check domain
        if self.domain is None:
            self.set_domain(torch.stack([
                self.distr.mean - domain_scale * self.distr.stddev,
                self.distr.mean + domain_scale * self.distr.stddev
            ], dim=1))
        # if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
        #     raise ValueError("domain does not satisfy tolerance at the boundary.")

    @property
    def stddevs(self) -> torch.Tensor:
        """Returns the square root diagonal of the covariance matrix."""
        return self.distr.variance.sqrt()

    def _initialize_distr(
            self,
    ) -> distributions.MixtureSameFamily | distributions.Independent:
        """Builds the inner torch.distributions.MixtureSameFamily object"""
        if self.mixture_weights is None:
            self.distr = distributions.Independent(
                distributions.Normal(self.loc.squeeze(0), self.scale.squeeze(0)), 1
            )
        else:
            modes = distributions.Independent(
                distributions.Normal(self.loc, self.scale), 1
            )
            mix = distributions.Categorical(self.mixture_weights)
            self.distr = distributions.MixtureSameFamily(mix, modes)

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized log-likelihood of the distribution at x"""
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        """Returns the corresponding torch.distributions.MixtureSameFamily object"""
        if self.mixture_weights is None:
            return distributions.Normal(self.loc[0, dim], self.scale[0, dim])
        modes = distributions.Normal(self.loc[:, dim], self.scale[:, dim])
        mix = distributions.Categorical(self.mixture_weights)
        return distributions.MixtureSameFamily(mix, modes)

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        """Evaluates the log-likelihood of the distribution at x"""
        return self.marginal_distr(dim=dim).log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """Return samples from the distribution."""
        if shape is None:
            shape = tuple()
        return self.distr.sample(torch.Size(shape))

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Evaluates the score of the distribution at x"""
        return score_mog(x, self.mixture_weights, self.loc, torch.square(self.scale))

    def has_entropy(self):
        """Informs if the entropy of the mode weights is computable."""
        return self.n_mixtures > 1

    def compute_mode_count(self, samples):
        """Computes the empirical mode weights."""
        log_prob_per_mode = self.distr.component_distribution.log_prob(samples.unsqueeze(1))
        idx = torch.argmax(log_prob_per_mode, dim=-1)
        counts = torch.FloatTensor([
            (idx == k).sum() for k in range(self.n_mixtures)
        ]).to(log_prob_per_mode.device)
        return counts

    def entropy(self, samples, counts=None):
        """Computes the entropy of the mode weights."""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        entropy = -torch.sum(hist * (torch.log(hist) / math.log(counts.shape[0])))
        return entropy

    def kl_weights(self, samples, counts=None):
        """Computes the KL between the empirical weights and the true ones"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.distr.mixture_distribution.probs.flatten()
        true_hist /= true_hist.sum()
        return torch.sum(true_hist * torch.log(true_hist / hist))

    def tv_weights(self, samples, counts=None):
        """Computes the TV between the empirical weights and the true ones"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.distr.mixture_distribution.probs.flatten()
        true_hist /= true_hist.sum()
        return torch.sum(torch.abs(hist - true_hist))

    def compute_forgotten_modes(self, samples, tol=0.05, counts=None):
        """Compute the number of forgotten modes"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.distr.mixture_distribution.probs.flatten()
        true_hist /= true_hist.sum()
        return torch.sum(hist < tol * true_hist.min()) / self.n_mixtures

    def compute_stats_sampling(self, return_samples=False):
        """Compute various metrics based on samples."""
        # Get the samples
        samples = super().compute_stats_sampling(return_samples=True)
        # Compute entropy metrics
        if self.has_entropy():
            # Compute mode weight metrics
            counts = self.compute_mode_count(samples)
            # Compute the metrics
            self.expectations['emc'] = self.entropy(samples, counts=counts).item()
            self.expectations['kl_weights'] = self.kl_weights(samples, counts=counts).item()
            self.expectations['tv_weights'] = self.tv_weights(samples, counts=counts).item()
            self.expectations['num_forgotten_modes'] = self.compute_forgotten_modes(samples, counts=counts).item()
        if return_samples:
            return samples


class GMMFull(GMM):
    def __init__(
            self,
            dim: int = 2,
            loc: torch.Tensor | None = None,
            cov: torch.Tensor | None = None,
            prec: torch.Tensor | None = None,
            cov_log_det: torch.Tensor | None = None,
            mixture_weights: torch.Tensor | None = None,
            n_reference_samples: int = int(1e7),
            domain_scale: float = 5,
            domain_tol: float | None = 1e-5,
            **kwargs,
    ):
        """Builds a mixture of Gaussians with full covariance matrices on the components

        Args:
            * dim (int): Dimension
            * mixture_weights (torch.Tensor of shape (n_modes,)): Weights
            * loc (torch.Tensor of shape (n_modes,dim)): Means
            * cov (torch.Tensor of shape (n_modes, dim)): Covariances
            * prec (torch.Tensor of shape (n_modes, dim)): Precision matrices
            * cov_log_det (torch.Tensor of shape (n_modes,1)) : Log-determinants of the covariances
            * n_reference_samples (int) : number of reference samples
        """
        super(GMM, self).__init__(
            dim=dim,
            log_norm_const=0.0,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )

        # Check shapes
        self.n_mixtures = loc.shape[0]
        if not loc.shape == (self.n_mixtures, self.dim):
            raise ValueError("Shape missmatch with loc.")
        if (cov is not None) and (not cov.shape == (self.n_mixtures, self.dim, self.dim)):
            raise ValueError("Shape missmatch between loc and cov.")
        if (prec is not None) and (not prec.shape == (self.n_mixtures, self.dim, self.dim)):
            raise ValueError("Shape missmatch between loc and prec.")
        if (cov is None) and (prec is None):
            raise ValueError("Either cov or prec must be set.")
        if mixture_weights is None and self.n_mixtures > 1:
            raise ValueError("Require mixture weights.")
        if not (mixture_weights is None or mixture_weights.shape == (self.n_mixtures,)):
            raise ValueError("Shape missmatch for the mixture weights.")

        # Initialize
        self.register_buffer("loc", loc, persistent=False)
        self.has_cov = cov is not None
        if cov is not None:
            self.register_buffer("cov", cov, persistent=False)
            self.register_buffer("prec", torch.linalg.inv(cov), persistent=False)
            if cov_log_det is not None:
                self.register_buffer("cov_log_det", cov_log_det, persistent=False)
            else:
                self.register_buffer("cov_log_det", torch.logdet(cov), persistent=False)
        if prec is not None:
            self.register_buffer("prec", prec, persistent=False)
            self.register_buffer("cov", torch.linalg.inv(prec), persistent=False)
            if cov_log_det is not None:
                self.register_buffer("cov_log_det", cov_log_det, persistent=False)
            else:
                self.register_buffer("cov_log_det", torch.logdet(self.cov), persistent=False)
        self.register_buffer("mixture_weights", mixture_weights, persistent=False)
        self._initialize_distr()

        # Check domain
        if self.domain is None:
            self.set_domain(torch.stack([
                self.distr.mean - domain_scale * self.distr.stddev,
                self.distr.mean + domain_scale * self.distr.stddev
            ], dim=1))
        # if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
        #     raise ValueError("domain does not satisfy tolerance at the boundary.")

    @property
    def stddevs(self) -> torch.Tensor:
        """Returns the square root diagonal of the covariance matrix."""
        return self.distr.stddev

    def _initialize_distr(
            self,
    ) -> distributions.MixtureSameFamily | distributions.Independent:
        """Builds the inner torch.distributions.MixtureSameFamily object"""
        if self.mixture_weights is None:
            self.distr = distributions.MultivariateNormal(
                loc=self.loc.squeeze(0),
                covariance_matrix=self.cov.squeeze(0)
            )
        else:
            modes = distributions.MultivariateNormal(
                loc=self.loc,
                covariance_matrix=self.cov
            )
            mix = distributions.Categorical(self.mixture_weights)
            self.distr = distributions.MixtureSameFamily(mix, modes)

    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        """Returns the corresponding torch.distributions.MixtureSameFamily object"""
        raise NotImplementedError('Marginal distribution not implemented with full covariance.')

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        """Evaluates the log-likelihood of the distribution at x"""
        raise NotImplementedError('Marginal distribution not implemented with full covariance.')

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Evaluates the score of the distribution at x"""
        return score_mog_full(x, self.mixture_weights, self.loc, self.cov,
                              precisions=self.prec, covariances_log_det=self.cov_log_det)


class TwoModes(GMM):

    def __init__(self, dim=2, a=1.0, centered=False, ill_conditioned='not', **kwargs):
        """Builds a mixture of two unequally weighted Gaussians with diagonal covariances

            p = (2/3) N(-a 1_d, C) + (1/3) N(+a 1_d, C)

        Args:
            * dim (int): Dimension
            * a (float): Half distance between the modes (default is 1.0)
            * centered (bool): Whether to center the distribution (default is False)
            * ill_conditioned (str): Type of conditioning (either 'not', 'medium' or 'hard')
                (default is 'not')
        """
        # Build the parameters
        assert ill_conditioned in ['not', 'medium', 'hard']
        # 1. Mixture weights : 2/3, 1/3
        mixture_weights = torch.FloatTensor([2., 1.])
        # 2. Means : -a, a
        loc = torch.stack([-a * torch.ones((dim,)), a * torch.ones((dim,))])
        if centered:
            loc += (a / 3.) * torch.ones((dim,))
        # 3. Standard deviation
        if ill_conditioned == 'medium':
            scale = torch.sqrt(0.05 * torch.logspace(-1, 0., dim)).unsqueeze(0).expand(2, -1)
        elif ill_conditioned == 'hard':
            scale = torch.sqrt(0.05 * torch.logspace(-2., 0., dim)).unsqueeze(0).expand(2, -1)
        else:
            scale = torch.sqrt(0.05 * torch.ones_like(loc))

        # Call the param constructor
        super().__init__(dim=dim, loc=loc, scale=scale, mixture_weights=mixture_weights, **kwargs)

    def compute_mode_weight(self, samples):
        """Computes the weight on the strongest mode."""
        counts = self.compute_mode_count(samples)
        return 100. * counts[0] / counts.sum()

    def compute_stats_sampling(self, return_samples=False):
        """Compute various metrics based on samples."""
        # Get the samples
        samples = super().compute_stats_sampling(return_samples=True)
        self.expectations['mode_weight'] = self.compute_mode_weight(samples).item()
        if return_samples:
            return samples


class TwoModesFull(GMMFull):

    def __init__(self, dim=2, a=1.0, centered=False, ill_conditioned='medium', rand_factor=5., seed_q=42, **kwargs):
        """Builds a mixture of two unequally weighted Gaussians with full covariances

            p = (2/3) N(-a 1_d, C) + (1/3) N(+a 1_d, C)

        Args:
            * dim (int): Dimension
            * a (float): Half distance between the modes (default is 1.0)
            * centered (bool): Whether to center the distribution (default is False)
            * ill_conditioned (str): Type of conditioning (either 'medium' or 'hard')
                (default is 'medium')
            * rand_factor (float): Uniform spread of the random matrix used to build the covariance via QR
                (default is 5.)
            * seed_q (int): Seed for the sampling of the random matrix (default is 42)
        """
        # Build the parameters
        assert ill_conditioned in ['medium', 'hard']
        # 1. Mixture weights : 2/3, 1/3
        mixture_weights = torch.FloatTensor([2., 1.])
        # 2. Means : -a, a
        loc = torch.stack([-a * torch.ones((dim,)), a * torch.ones((dim,))])
        if centered:
            loc += (a / 3.) * torch.ones((dim,))
        # 3. Build a orthonormal matrix
        generator = torch.Generator()
        generator.manual_seed(seed_q)
        q = torch.linalg.qr(rand_factor * torch.rand((dim, dim), generator=generator), mode='complete').Q
        # 4. Covariance
        if ill_conditioned == 'hard':
            cov = torch.diag(0.05 * torch.logspace(-2., 0., dim))
        else:
            cov = torch.diag(0.05 * torch.logspace(-1, 0., dim))
        cov = torch.matmul(q, torch.matmul(cov, q.T))
        cov = torch.stack([cov, cov.clone()], dim=0)
        # Call the param constructor
        super().__init__(dim=dim, loc=loc, cov=cov, mixture_weights=mixture_weights, **kwargs)

    def compute_mode_weight(self, samples):
        """Computes the weight on the strongest mode."""
        counts = self.compute_mode_count(samples)
        return 100. * counts[0] / counts.sum()

    def compute_stats_sampling(self, return_samples=False):
        """Compute various metrics based on samples."""
        # Get the samples
        samples = super().compute_stats_sampling(return_samples=True)
        self.expectations['mode_weight'] = self.compute_mode_weight(samples).item()
        if return_samples:
            return samples


class BracketTwoModes(GMM):

    def __init__(self, dim=2, a=0.75, equilibrated=False, var_min=0.01, var_max=0.2, **kwargs):
        """Builds a mixture of two unequally weighted Gaussians with diagonal covariances

            p = (2/3) N(-a 1_d, C_1) + (1/3) N(+a 1_d, C_2) where (C_1)_i = (C_2)_(dim-i)

        Args:
            * dim (int): Dimension
            * a (float): Half distance between the modes (default is 1.0)
            * equilibrated (bool): Whether to make the mode weights equal (default is False)
            * var_min (float) : Minimal variance value on C_1 and C_2 (default is 0.01)
            * var_max (float) : Maximal variance value on C_1 and C_2 (default is 0.2)
        """
        # Make the means
        loc = torch.stack([
            -a * torch.ones((dim,)),
            +a * torch.ones((dim,))
        ], dim=0)
        # Make the variances
        variance_diag = torch.linspace(var_min, var_max, dim)
        variances = torch.stack([
            variance_diag, torch.flip(variance_diag, dims=(0,))
        ], dim=0)
        scale = torch.sqrt(variances)
        # Make the weights
        if equilibrated:
            weights = torch.ones((2,)) / 2.
        else:
            weights = torch.FloatTensor([2, 1]) / 2.
        # Call the parent constructor
        super().__init__(dim=dim, loc=loc, scale=scale, mixture_weights=weights, **kwargs)

    def compute_mode_weight(self, samples):
        """Computes the weight on the strongest mode."""
        counts = self.compute_mode_count(samples)
        return 100. * counts[0] / counts.sum()

    def compute_stats_sampling(self, return_samples=False):
        """Compute various metrics based on samples."""
        # Get the samples
        samples = super().compute_stats_sampling(return_samples=True)
        self.expectations['mode_weight'] = self.compute_mode_weight(samples).item()
        if return_samples:
            return samples


class ManyModes(GMM):

    def __init__(self, n_modes=3, dim=2, seed_loc=42, mixture_weight_factor=3., var=0.1, **kwargs):
        """Builds a mixture of n_modes unequally weighted Gaussians with same isotropic covariances,
        with means randomly generated.

        Args:
            * n_modes (int): Number of components (default is 3)
            * dim (int): Dimension
            * seed_loc (float): Seed to generate the means
            * mixture_weight_factor (float): Spread of the mixture weights (default is 3.)
            Note that 1. corresponds to an equilibrated target.
            * var_min (float) : Value of coordinate-wise variance
        """
        # Build the parameters
        generator = torch.Generator()
        generator.manual_seed(seed_loc)
        # 1. Mixture weights: geometrically increasing (base = 1. <-> uniform weights)
        mixture_weights = torch.logspace(0., 1., n_modes, base=mixture_weight_factor)
        # 2. Means : uniformly chosen in [-n_modes, n_modes]^dim
        loc = 2 * n_modes * torch.rand((n_modes, dim), generator=generator) - n_modes
        # 3. Standard deviation: same for each mode
        scale = torch.sqrt(var * torch.ones_like(loc))

        # Call the param constructor
        super().__init__(dim=dim, loc=loc, scale=scale, mixture_weights=mixture_weights, **kwargs)


class Gauss(GMM):
    def __init__(
            self,
            dim: int = 1,
            loc: torch.Tensor | Number = 0.0,
            scale: torch.Tensor | Number = 1.0,
            **kwargs,
    ):
        """Builds a diagonal covariance Gaussian distribution

        Args:
            * dim (int): Dimension
            * loc (torch.Tensor of shape (dim,)): Mean
            * scale (torch.Tensor of shape (dim,)): Square root diagonal of the covariance
        """
        # Setup parameters
        params = {"loc": loc, "scale": scale}
        params = {k: Gauss._prepare_input(p, dim) for k, p in params.items()}
        super().__init__(dim=dim, **params, **kwargs)
        self.stddevs = self.scale.squeeze(0)

    @staticmethod
    def _prepare_input(param: torch.Tensor | Number, dim: int = 1):
        if not isinstance(param, torch.Tensor):
            param = torch.tensor(param, dtype=torch.float)
        param = torch.atleast_2d(param)
        if param.numel() == 1:
            param = param.repeat(1, dim)
        return param

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Evaluates the score of the distribution at x."""
        return score_gauss(x, self.loc, torch.square(self.scale))


class GaussFull(Distribution):
    def __init__(
            self,
            dim: int = 1,
            loc: torch.Tensor | None = None,
            cov: torch.Tensor | None = None,
            prec: torch.Tensor | None = None,
            n_reference_samples: int = int(1e7),
            domain_scale: float = 5,
            domain_tol: float | None = 1e-5,
            **kwargs,
    ):
        """Builds a full covariance Gaussian distribution

        Args:
            * dim (int): Dimension
            * loc (torch.Tensor of shape (dim,)): Mean
            * cov (torch.Tensor of shape (dim,dim)): Covariance
            * prec (torch.Tensor of shape (dim,dim)): Precision matrix (default is None)
        """
        super().__init__(
            dim=dim,
            log_norm_const=0.0,
            n_reference_samples=n_reference_samples,
            **kwargs,
        )
        # Check shape
        if not loc.shape == (self.dim,):
            raise ValueError("Shape missmatch with loc.")
        if (cov is not None) and (not cov.shape == (self.dim, self.dim)):
            raise ValueError("Shape missmatch between loc and cov.")
        if (prec is not None) and (not prec.shape == (self.dim, self.dim)):
            raise ValueError("Shape missmatch between loc and cov.")
        if (cov is None) and (prec is None):
            raise ValueError("Either cov or prec must be set.")
        # Store the parameters
        self.register_buffer("loc", loc, persistent=False)
        if cov is not None:
            self.register_buffer("cov", cov, persistent=False)
            self.register_buffer("prec", torch.linalg.inv(cov), persistent=False)
        if prec is not None:
            self.register_buffer("prec", prec, persistent=False)
            self.register_buffer("cov", torch.linalg.inv(prec), persistent=False)

        # Build the distribution
        self.distr = distributions.MultivariateNormal(
            loc=self.loc,
            covariance_matrix=self.cov
        )
        # Check domain
        if self.domain is None:
            self.set_domain(torch.stack([
                self.distr.mean - domain_scale * self.distr.stddev,
                self.distr.mean + domain_scale * self.distr.stddev
            ], dim=1))
        # if domain_tol is not None and (self.pdf(self.domain.T) > domain_tol).any():
        #     raise ValueError("domain does not satisfy tolerance at the boundary.")

    @property
    def stddevs(self) -> torch.Tensor:
        """Returns the square root diagonal of the covariance matrix."""
        return self.distr.component_distribution.stddev

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized log-likelihood of the distribution at x"""
        log_prob = self.distr.log_prob(x).unsqueeze(-1)
        assert log_prob.shape == (*x.shape[:-1], 1)
        return log_prob

    def marginal_distr(self, dim=0) -> torch.distributions.Distribution:
        """Returns the corresponding torch.distributions.MixtureSameFamily object"""
        raise NotImplementedError('Marginal distribution not implemented with full covariance.')

    def marginal(self, x: torch.Tensor, dim=0) -> torch.Tensor:
        """Evaluates the log-likelihood of the distribution at x"""
        raise NotImplementedError('Marginal distribution not implemented with full covariance.')

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """Returns samples from the distribution"""
        if shape is None:
            shape = tuple()
        return self.distr.sample(torch.Size(shape))

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Evaluates the score of the distribution at x"""
        return score_gauss_full(x, self.loc, self.cov, precisions=self.prec)


class IsotropicGauss(Gauss):
    # Typially used as prior (supports truncation and faster methods)
    def __init__(
            self,
            dim: int = 1,
            loc: float = 0.0,
            scale: float = 1.0,
            truncate_quartile: float | None = None,
            **kwargs,
    ):
        """Builds an isotropic covariance Gaussian distribution

        Args:
            * dim (int): Dimension
            * loc (torch.Tensor of shape (dim,)): Mean
            * scale (torch.Tensor of shape (dim,)): Square root diagonal of the covariance matrix
            * truncate_quartile (float): Value of quartile truncation (default is None)
        """
        super().__init__(
            dim=dim,
            loc=loc,
            scale=scale,
            **kwargs,
        )

        assert torch.allclose(self.loc, self.loc[0, 0])
        assert torch.allclose(self.scale, self.scale[0, 0])

        # Calculate truncation values
        if truncate_quartile is not None:
            quartiles = torch.tensor(
                [truncate_quartile / 2, 1 - truncate_quartile / 2],
                device=self.domain.device,
            )
            truncate_quartile = self.marginal_distr().icdf(quartiles).tolist()
        self.truncate_quartile = truncate_quartile

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized log-likelihood of the distribution at x"""
        var = self.scale[0, 0] ** 2
        norm_const = -0.5 * self.dim * (2.0 * math.pi * var).log()
        sq_sum = torch.sum((x - self.loc[0, 0]) ** 2, dim=-1, keepdim=True)
        return norm_const - 0.5 * sq_sum / var

    def score(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Evaluates the score of the distribution at x"""
        return (self.loc[0, 0] - x) / self.scale[0, 0] ** 2

    def marginal(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Evaluates the log-likelihood of the distribution at x"""
        return self.marginal_distr().log_prob(x).exp()

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """Returns samples from the distribution"""
        if shape is None:
            shape = tuple()
        if self.truncate_quartile is None:
            return self.loc[0, 0] + self.scale[0, 0] * torch.randn(
                *shape, self.dim, device=self.domain.device
            )
        tensor = torch.empty(*shape, self.dim, device=self.domain.device)
        return trunc_normal_(
            tensor,
            mean=self.loc[0, 0],
            std=self.scale[0, 0],
            a=self.truncate_quartile[0],
            b=self.truncate_quartile[1],
        )
