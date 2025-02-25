import torch
import math
from .base import Distribution


class Checkerboard(Distribution):
    """Class for 2D Checkerboard distribution."""

    def __init__(
        self,
        dim: int = 2,
        width: int = 4,
        unequilibrated: bool = True,
        n_reference_samples: int = int(1e5),
        **kwargs,
    ):
        if dim != 2:
            raise ValueError("The checkboard should be two-dimensional.")
        super().__init__(dim=2, log_norm_const=0.0, n_reference_samples=n_reference_samples, **kwargs)

        # Get the coordinates
        self.width = width
        x_min, y_max = self.build_extremal_points()
        x_max = x_min + 2
        y_min = y_max - 2
        self.n_mixtures = x_min.shape[0]

        # Get the middle of each square
        self.loc = torch.stack([
            (x_min + x_max) / 2.,
            (y_min + y_max) / 2.
        ], dim=-1)

        # Build the component distribution
        uniform_dist = torch.distributions.Independent(torch.distributions.Uniform(
            low=torch.stack([x_min, y_min], dim=-1),
            high=torch.stack([x_max, y_max], dim=-1),
            validate_args=False
        ), reinterpreted_batch_ndims=1)
        weights = torch.ones((self.n_mixtures,))
        if unequilibrated:
            weights[torch.arange(self.n_mixtures) % 2 == 0] *= 3
        self.distr = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(weights),
            component_distribution=uniform_dist,
            validate_args=False
        )

        # Build a mask
        self.hist_mask = torch.stack([
            torch.arange(self.width) % 2 == 0,
            torch.arange(self.width) % 2 == 1,
            torch.arange(self.width) % 2 == 0,
            torch.arange(self.width) % 2 == 1,
        ], dim=0)

        # Check domain
        if self.domain is None:
            self.set_domain(torch.FloatTensor([
                [-4, -4 + 2 * self.width],
                [-4, 4]
            ]))

    def build_extremal_points(self):
        """Builds the extremal points of the checkerboard."""
        x_pos, y_pos = [], []
        for y in [4, 0]:
            x_pos_tmp = list(range(-2, -4 + 2 * self.width, 4))
            x_pos += x_pos_tmp
            y_pos += [y] * len(x_pos_tmp)
            x_pos_tmp = list(range(-4, -4 + 2 * self.width, 4))
            x_pos += x_pos_tmp
            y_pos += [y-2] * len(x_pos_tmp)
        return torch.FloatTensor(x_pos), torch.FloatTensor(y_pos)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """Get samples from the checkerboard distribution."""
        if shape is None:
            shape = tuple()
        return self.distr.sample(torch.Size(shape))

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized log-likelihood of the distribution at x"""
        return self.distr.log_prob(x).unsqueeze(-1)

    def score(self, x: torch.Tensor, create_graph=False) -> torch.Tensor:
        """Evaluates the score of the distribution at x"""
        return torch.zeros_like(x)

    def marginal(self, x: torch.Tensor, dim: int = 0) -> torch.Tensor:
        raise NotImplementedError('marginal is not implemented.')

    def has_entropy(self):
        """Informs if the entropy of the mode weights is computable."""
        return True

    def compute_mode_count(self, samples):
        """Computes the empirical mode weights."""
        return torch.histogramdd(samples.cpu(), bins=(self.width, 4),
                                 range=tuple(x.item() for x in self.domain.flatten()))[0].T

    def entropy(self, samples, counts=None):
        """Computes the entropy of the mode weights."""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts[self.hist_mask].flatten()
        hist /= counts.sum()
        entropy = -torch.sum(hist * (torch.log(hist) / math.log(counts.shape[0])))
        return entropy

    def kl_weights(self, samples, counts=None):
        """Computes the KL between the empirical weights and the true ones"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts[self.hist_mask].flatten()
        hist /= counts.sum()
        true_hist = self.distr.mixture_distribution.probs.to(hist.device)
        true_hist /= true_hist.sum()
        return torch.sum(true_hist * torch.log(true_hist / hist))

    def tv_weights(self, samples, counts=None):
        """Computes the TV between the empirical weights and the true ones"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts[self.hist_mask].flatten()
        hist /= counts.sum()
        true_hist = self.distr.mixture_distribution.probs.to(hist.device)
        true_hist /= true_hist.sum()
        return torch.sum(torch.abs(hist - true_hist))

    def compute_forgotten_modes(self, samples, tol=0.05, counts=None):
        """Compute the number of forgotten modes"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts[self.hist_mask].flatten()
        hist /= counts.sum()
        true_hist = self.distr.mixture_distribution.probs.to(hist.device)
        true_hist /= true_hist.sum()
        return torch.sum(hist < tol * true_hist.min()) / self.n_mixtures

    def compute_stats_sampling(self, return_samples=False):
        """Compute various metrics based on samples."""
        samples = super().compute_stats_sampling(return_samples=True)
        counts = self.compute_mode_count(samples)
        self.expectations['emc'] = self.entropy(samples, counts=counts).item()
        self.expectations['kl_weights'] = self.kl_weights(samples, counts=counts).item()
        self.expectations['tv_weights'] = self.tv_weights(samples, counts=counts).item()
        self.expectations['num_forgotten_modes'] = self.compute_forgotten_modes(samples, counts=counts).item()
        if return_samples:
            return samples

    def _apply(self, fn):
        super(Checkerboard, self)._apply(fn)
        self.loc = fn(self.loc)
        self.distr.mixture_distribution.probs = fn(
            self.distr.mixture_distribution.probs)
        self.distr.mixture_distribution.logits = fn(
            self.distr.mixture_distribution.logits)
        self.distr.component_distribution.base_dist.low = fn(
            self.distr.component_distribution.base_dist.low)
        self.distr.component_distribution.base_dist.high = fn(
            self.distr.component_distribution.base_dist.high)
