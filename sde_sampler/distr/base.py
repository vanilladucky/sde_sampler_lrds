from __future__ import annotations

import logging
import math
from functools import partial
from pathlib import Path
from typing import Callable
from tqdm import trange

import torch
import torchquad

EXPECTATION_FNS: dict[str, Callable] = {
    "square": lambda x: (x**2).sum(dim=-1, keepdims=True),
    "abs": lambda x: x.abs().sum(dim=-1, keepdims=True),
    "sum": lambda x: x.sum(dim=-1, keepdims=True),
    "square_minus_sum": lambda x: (x**2 - x).sum(dim=-1, keepdims=True),
}
DATA_DIR = Path(__file__).parents[2] / "data"


class Distribution(torch.nn.Module):
    """Base class for probability distributions."""

    def __init__(
        self,
        dim: int,
        log_norm_const: float = None,
        domain: float | torch.Tensor | None = None,
        n_reference_samples: int | None = None,
        grid_points: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_reference_samples = n_reference_samples
        self.grid_points = grid_points
        self.set_domain(domain)

        # Initialize
        self.log_norm_const = log_norm_const
        self.register_buffer("stddevs", None, persistent=False)
        self.expectations = {}

    def has_entropy(self):
        """Informs if the entropy of the mode weights is computable."""
        return False

    def set_domain(self, d: torch.Tensor | float | None = None):
        if d is not None:
            if not isinstance(d, torch.Tensor):
                d = torch.tensor(d, dtype=torch.float)
            if d.ndim == 0:
                d = torch.stack([-d, d], dim=-1)
            if d.ndim == 1:
                d = d.unsqueeze(0)
            if d.shape == (1, 2):
                d = d.repeat(self.dim, 1)
            assert d.shape == (self.dim, 2)
        self.register_buffer("domain", d, persistent=False)

    def compute_stats_sampling(self, return_samples=False):
        """Computes various reference expectation values based on Monte Carlo estimation with true samples."""
        samples = self.sample((self.n_reference_samples,))
        for name, fn in EXPECTATION_FNS.items():
            if name not in self.expectations:
                self.expectations[name] = fn(samples).mean().item()
        if self.stddevs is None:
            self.stddevs = samples.std(dim=0)
        if return_samples:
            return samples

    def compute_stats_integration(self):
        """Computes various reference expectation values based on exact integration."""
        integrate = partial(
            torchquad.Boole().integrate,
            dim=self.dim,
            N=self.grid_points,
            integration_domain=self.domain,
        )

        if self.log_norm_const is None:
            norm_const = integrate(self.unnorm_pdf).item()
            self.log_norm_const = math.log(norm_const)

        for name, fn in EXPECTATION_FNS.items():
            if name not in self.expectations:
                self.expectations[name] = integrate(
                    lambda x: fn(x) * self.pdf(x)
                ).item()

            if self.stddevs is None:
                expectations = integrate(lambda x: x * self.pdf(x)).unsqueeze(0)
                stddevs = integrate(
                    lambda x: (x - expectations) ** 2 * self.pdf(x)
                ).sqrt()
                self.stddevs = torch.atleast_1d(stddevs)

    @torch.no_grad()
    def compute_stats(self):
        """Computes various reference expectation values to assess the statistical estimation"""
        if hasattr(self, "sample") and self.n_reference_samples is not None:
            self.compute_stats_sampling()

        elif self.grid_points is not None and self.domain is not None:
            try:
                with torch.device(self.domain.device):
                    self.compute_stats_integration()
            # the `torch.device` context is not available for PyTorch < 2.0
            except AttributeError:
                device = self.domain.device
                self.to("cpu")
                self.compute_stats_integration()
                self.to(device)
        else:
            logging.warning(
                f"Cannot compute statistics for distribution `%s`",
                self.__class__.__name__,
            )

    def _initialize_distr(self):
        # This can be used to reinitialize distributions, e.g,  when transfering between devices
        pass

    def _apply(self, fn):
        torch.nn.Module._apply(self, fn)
        self._initialize_distr()

    def log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the log-likelihood of the distribution at x"""
        if self.log_norm_const is None:
            raise NotImplementedError
        return self.unnorm_log_prob(x) - self.log_norm_const

    def pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the probability density function of the distribution at x"""
        return self.log_prob(x).exp()

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized log-likelihood of the distribution at x"""
        raise NotImplementedError

    def unnorm_pdf(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized probability density function of the distribution at x"""
        return self.unnorm_log_prob(x).exp()

    def score(self, x: torch.Tensor, create_graph=False) -> torch.Tensor:
        """Evaluates the score of the distribution at x by differentiating the log-likelihood."""
        grad = x.requires_grad
        x.requires_grad_(True)
        with torch.set_grad_enabled(True):
            log_rho = self.unnorm_log_prob(x).sum()
            score = torch.autograd.grad(log_rho, x, create_graph=create_graph)[0]
        x.requires_grad_(grad)
        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.unnorm_log_prob(x)

    # def objective(self, x: torch.Tensor) -> torch.Tensor:
    #    Can be implemented for usage as optimization method

    # def marginal(self, x: np.ndarray | float, dim: int = 0) -> np.ndarray:
    #    Can be implemented for additional metrics

    # def sample(self, shape: tuple | None = None) -> torch.Tensor:
    #     Can be implemented for additional metrics

    # def filter(self, x: torch.Tensor) -> torch.Tensor:
    #     Can be implemented to filter samples

    # def metrics(self, samples: torch.Tensor, *args, **kwargs) -> dict[str, float]
    #     Can be implemented for additional metrics

    # def plots(self, samples: torch.Tensor, *args, **kwargs) -> dict[str, Union[go.Figure, plt.Figure]]
    #     Can be implemented for additional plots


def sample_uniform(domain: torch.Tensor, batchsize: int = 1) -> torch.Tensor:
    """Samples uniformly batchsize samples on the target domain."""
    dim = domain.shape[0]
    diam = domain[:, 1] - domain[:, 0]
    rand = torch.rand(batchsize, dim, device=domain.device)
    return domain[:, 0] + rand * diam


class WrapperDistrNN(Distribution):
    """Base class for multi-level Energy-based models."""

    def __init__(self, dim, net, t):
        super().__init__(dim=dim, log_norm_const=0.0)
        self.net = net
        self.t = t

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized probability density function of the distribution at (t,x)"""
        return self.net.unnorm_log_prob(
            self.t * torch.ones((x.shape[0], 1), device=x.device), x
        )


def run_gdflow(U, grad_U, x_, n_steps, dt, verbose=False, verbose_freq=100):
    x = x_.clone()
    if verbose:
        r = trange(n_steps)
    else:
        r = range(n_steps)
    for t in r:
        x -= dt * grad_U(x)
        if verbose and (t % verbose_freq) == 0:
            r.set_postfix(u='{:.3e}'.format(U(x).sum().item()))
    return x.detach()
