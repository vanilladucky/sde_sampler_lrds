import torch
import math
from .base import Distribution
from .gauss import score_mog


class PolarTransform(torch.distributions.transforms.Transform):
    """Polar transformation"""

    domain = torch.distributions.constraints.real_vector
    codomain = torch.distributions.constraints.real_vector
    bijective = True
    sign = +1

    def __eq__(self, other):
        return isinstance(other, PolarTransform)

    def _call(self, x):
        return torch.stack([
            x[..., 0] * torch.cos(x[..., 1]),
            x[..., 0] * torch.sin(x[..., 1])
        ], dim=-1)

    def _inverse(self, y):
        x = torch.stack([
            torch.linalg.norm(y, dim=-1),
            torch.atan2(y[..., 1], y[..., 0])
        ], dim=-1)
        x[..., 1] = x[..., 1] + (x[..., 1] < 0).type_as(y) * (2 * torch.pi)
        return x

    def log_abs_det_jacobian(self, x, y):
        return torch.log(x[..., 0])


class Rings(Distribution):
    """Base class for 2D Rings distribution"""

    def __init__(
        self,
        dim: int = 2,
        lower_rad: float = 1.0,
        upper_rad: float = 5.0,
        num_rad: int = 3,
        scale: float = 0.1,
        equilibrated: bool = False,
        n_reference_samples: int = int(1e6),
        domain_tol: float = 5.,
        **kwargs
    ):
        if dim != 2:
            raise ValueError("The rings should be two-dimensional.")
        super().__init__(dim=dim, log_norm_const=0.0, n_reference_samples=n_reference_samples, **kwargs)
        # Make the radius distribution
        self.n_mixtures = num_rad
        self.radiuses = torch.linspace(lower_rad, upper_rad, self.n_mixtures)
        if equilibrated:
            weights = torch.ones((self.n_mixtures,))
        else:
            weights = self.radiuses / self.radiuses.sum()
        self.radius_dist = torch.distributions.MixtureSameFamily(
            mixture_distribution=torch.distributions.Categorical(weights),
            component_distribution=torch.distributions.Normal(loc=self.radiuses, scale=scale)
        )
        # Make the angle distribution
        self.angle_dist = torch.distributions.Uniform(low=0.0, high=2*torch.pi)
        # Make the polar transform
        self.transform = PolarTransform()
        # Set the extreme values
        self.domain_tol = domain_tol
        if self.domain is None:
            self.set_domain(torch.FloatTensor([
                [-upper_rad - self.domain_tol * scale, upper_rad + self.domain_tol * scale],
                [-upper_rad - self.domain_tol * scale, upper_rad + self.domain_tol * scale]
            ]))

    def sample(self, shape):
        """Returns samples from the distribution"""
        r = self.radius_dist.sample(shape)
        theta = self.angle_dist.sample(shape)
        if len(shape) == 0:
            x = torch.FloatTensor([r, theta])
        else:
            x = torch.stack([r, theta], dim=1)
        return self.transform(x)

    def sample_init_points(self, n_points_per_mode):
        """Returns samples from the circular modes"""
        r = self.radius_dist.component_distribution.sample((n_points_per_mode,)).flatten()
        theta = self.angle_dist.sample((r.shape[0],))
        return self.transform(torch.stack([r, theta], dim=1))

    def unnorm_log_prob(self, value):
        """Evaluates the unnormalized log-likelihood at value (by inverting polar transformation)"""
        x = self.transform.inv(value)
        ret = self.radius_dist.log_prob(x[..., 0]) + self.angle_dist.log_prob(x[..., 1]
                                                                              ) - self.transform.log_abs_det_jacobian(x, value)
        return ret.view((-1, 1))

    def score_radius(self, x):
        """Evaluates the score of the radius distribution at x"""
        return score_mog(x, weights=self.radius_dist.mixture_distribution.probs,
                         means=self.radius_dist.component_distribution.loc.unsqueeze(-1),
                         variances=self.radius_dist.component_distribution.variance.unsqueeze(-1))

    def score(self, x, eps=1e-7, **kwargs):
        """Evaluates the score of the distribution at x"""
        norm_x = torch.linalg.norm(x, dim=-1, keepdim=True) + eps
        return x * ((self.score_radius(norm_x) / norm_x) - (1. / torch.square(norm_x)))

    def has_entropy(self):
        """Informs if the entropy of the mode weights is computable"""
        return True

    def compute_mode_count(self, samples):
        """Computes the empirical mode weights"""
        radiuses_sq = torch.square(samples[:, 0]) + torch.square(samples[:, 1])
        distances_to_radius = torch.abs(radiuses_sq.unsqueeze(-1) - torch.square(self.radiuses))
        idx = torch.argmin(distances_to_radius, dim=-1)
        counts = torch.FloatTensor([
            (idx == k).sum() for k in range(self.radiuses.shape[0])
        ]).to(samples.device)
        return counts

    def entropy(self, samples, counts=None):
        """Computes the entropy of the mode weights"""
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
        true_hist = self.radius_dist.mixture_distribution.probs.flatten()
        true_hist /= true_hist.sum()
        return torch.sum(true_hist * torch.log(true_hist / hist))

    def tv_weights(self, samples, counts=None):
        """Computes the TV between the empirical weights and the true ones"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.radius_dist.mixture_distribution.probs.flatten()
        true_hist /= true_hist.sum()
        return torch.sum(torch.abs(hist - true_hist))

    def compute_forgotten_modes(self, samples, tol=0.05, counts=None):
        """Compute the number of forgotten modes"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.radius_dist.mixture_distribution.probs.flatten()
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
        super(Rings, self)._apply(fn)
        self.radiuses = fn(self.radiuses)
        self.radius_dist.mixture_distribution.probs = fn(
            self.radius_dist.mixture_distribution.probs)
        self.radius_dist.mixture_distribution.logits = fn(
            self.radius_dist.mixture_distribution.logits)
        self.radius_dist.component_distribution.loc = fn(
            self.radius_dist.component_distribution.loc)
        self.radius_dist.component_distribution.scale = fn(
            self.radius_dist.component_distribution.scale)
        self.angle_dist.low = fn(self.angle_dist.low)
        self.angle_dist.high = fn(self.angle_dist.high)
