"""
Adapted from https://github.com/fmu2/NICE
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
# from PIL import Image
from torchvision.transforms import Resize
# from torchvision.utils import make_grid

from .base import DATA_DIR, Distribution, run_gdflow


class StandardLogistic(torch.distributions.Distribution):
    def __init__(self):
        super(StandardLogistic, self).__init__(validate_args=False)

    def log_prob(self, x):
        """Computes data log-likelihood.

        Args:
            x: input tensor.
        Returns:
            log-likelihood.
        """
        return -(nn.functional.softplus(x) + nn.functional.softplus(-x))

    def sample(self, size, eps=1e-20):
        """Samples from the distribution.

        Args:
            size: number of samples to generate.
        Returns:
            samples.
        """
        z = torch.distributions.Uniform(eps, 1.0 - eps).sample(size)
        return torch.log(z) - torch.log(1.0 - z)


class Dequant(nn.Module):
    """Class used to handle dequantization."""

    def __init__(self, quants=256.):
        super(Dequant, self).__init__()
        self.quants = quants
        self.log_det = math.log1p(1. / (self.quants - 1.))

    def forward(self, x, reverse=False):
        if reverse:
            x = torch.floor(x * self.quants).clamp(min=0, max=self.quants-1)
            x /= self.quants - 1.
            log_det_J = self.log_det * x.shape[-1]
        else:
            x = x * (self.quants - 1.) + torch.rand_like(x).detach()
            x /= self.quants
            log_det_J = -self.log_det * x.shape[-1]
        return x, log_det_J

# Custom sigmoid transform


def _clipped_sigmoid(x, tiny, eps):
    return torch.clamp(torch.sigmoid(x), min=tiny, max=1. - eps)


class Sigmoid(nn.Module):

    def __init__(self, alpha=1e-5, tiny=1.17549e-38, eps=1.19209e-07):
        super(Sigmoid, self).__init__()
        self.alpha = alpha
        self.log_det_aff = math.log1p(-self.alpha)
        self.tiny = tiny
        self.eps = eps

    def forward(self, x, reverse=False):
        if reverse:
            # Sigmoid transform
            log_det_J = (-x - 2. * torch.nn.functional.softplus(-x)).sum(dim=-1)
            x = _clipped_sigmoid(x, self.tiny, self.eps)
            # Reversing scaling for numerical stability
            log_det_J -= self.log_det_aff * x.shape[-1]
            x = (x - 0.5 * self.alpha) / (1 - self.alpha)
        else:
            # Scaling for numerical stability
            x = x * (1. - self.alpha) + 0.5 * self.alpha  # Scale to prevent boundaries 0 and 1
            log_det_J = self.log_det_aff * x.shape[-1]
            # Inverse sigmoid transform
            x = x.clamp(min=self.tiny, max=1. - self.eps)
            log_det_J -= (torch.log(x) + torch.log1p(-x)).sum(dim=-1)
            x = torch.log(x) - torch.log1p(-x)
        return x, log_det_J


class Coupling(nn.Module):
    def __init__(self, in_out_dim, mid_dim, hidden, mask_config):
        """Initialize a coupling layer.

        Args:
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(nn.Linear(in_out_dim // 2, mid_dim), nn.ReLU())
        self.mid_block = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(mid_dim, mid_dim), nn.ReLU())
                for _ in range(hidden - 1)
            ]
        )
        self.out_block = nn.Linear(mid_dim, in_out_dim // 2)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W // 2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(off)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))


class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.

        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.

        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        if reverse:
            x = x * torch.exp(-self.scale)
            log_det_J = -torch.sum(self.scale)
        else:
            x = x * torch.exp(self.scale)
            log_det_J = torch.sum(self.scale)
        return x, log_det_J


class NiceModel(nn.Module):
    def __init__(self, prior, coupling, in_out_dim, mid_dim, hidden, mask_config,
                 use_dequant=False, use_sigmoid=False, alpha_sigmoid=1e-5):
        """Initialize a NICE.

        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            in_out_dim: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
            use_dequant: whether to use a dequantization transform (default is False)
            use_sigmoid: whether to use a sigmoid transform (default is False)
            alpha_sigmoid: value of alpha for the sigmoid transform (default is 1e-5)
        """
        super(NiceModel, self).__init__()
        self.prior = prior
        self.in_out_dim = in_out_dim

        self.coupling = nn.ModuleList(
            [
                Coupling(
                    in_out_dim=in_out_dim,
                    mid_dim=mid_dim,
                    hidden=hidden,
                    mask_config=(mask_config + i) % 2,
                )
                for i in range(coupling)
            ]
        )
        self.scaling = Scaling(in_out_dim)
        self.use_dequant = use_dequant
        if self.use_dequant:
            self.dequant = Dequant()
        self.use_sigmoid = use_sigmoid
        if self.use_sigmoid:
            self.sigmoid = Sigmoid(alpha=alpha_sigmoid)

    def g(self, z):
        """Transformation g: Z -> X (inverse of f).

        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, reverse=True)
        if self.use_sigmoid:
            x, _ = self.sigmoid(x, reverse=True)
        if self.use_dequant:
            x, _ = self.dequant(x, reverse=True)
        return x

    def f(self, x):
        """Transformation f: X -> Z (inverse of g).

        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        log_det = 0.0
        if self.use_dequant:
            x, log_det_dequant = self.dequant(x)
            log_det += log_det_dequant
        if self.use_sigmoid:
            x, log_det_sigmoid = self.sigmoid(x)
            log_det += log_det_sigmoid
        for i in range(len(self.coupling)):
            x = self.coupling[i](x)
        x, log_det_scaling = self.scaling(x)
        return x, log_det + log_det_scaling

    def log_prob(self, x):
        """Computes data log-likelihood.

        (See Section 3.3 in the NICE paper.)

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x)
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.

        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        device = self.scaling.scale.device
        z = self.prior.sample((size, self.in_out_dim)).to(device)
        return self.g(z)

    def forward(self, x):
        """Forward pass.

        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x)

    def _apply(self, fn):
        new_self = super(NiceModel, self)._apply(fn)
        if isinstance(new_self.prior, torch.distributions.Normal):
            new_self.prior.loc = fn(new_self.prior.loc)
            new_self.prior.scale = fn(new_self.prior.scale)
        return new_self


class Nice(Distribution):
    """NICE trained on resized MNIST."""

    def __init__(
        self,
        model: nn.Module | None = None,
        checkpoint: str = DATA_DIR / "nice.pt",
        mean_data_path: str = DATA_DIR / "mnist_mean.pt",
        sample_chunk_size: int = 10000,
        dim: int = 196,
        log_norm_const: float = 0.0,
        n_reference_samples=int(1e6),
    ):
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
        )
        self.shape = (14, 14)
        if not self.dim == math.prod(self.shape):
            raise ValueError(f"Dimension is {self.dim} but needs to be 196.")
        self.sample_chunk_size = sample_chunk_size
        mean = torch.load(mean_data_path, weights_only=False).reshape((1, 28, 28))
        mean = Resize(size=self.shape, antialias=True)(mean).reshape((1, self.dim))
        self.register_buffer("mean", mean, persistent=False)

        # Model
        self.model = model
        if self.model is None:
            # Load checkpoint and NICE model
            ckpt = torch.load(checkpoint, weights_only=False)
            self.model = NiceModel(
                prior=self.build_prior(ckpt['latent']),
                coupling=ckpt["coupling"],
                in_out_dim=196,
                mid_dim=ckpt["mid_dim"],
                hidden=ckpt["hidden"],
                mask_config=ckpt["mask_config"],
                use_sigmoid=ckpt["use_sigmoid_layer"],
                alpha_sigmoid=ckpt["alpha_sigmoid"]
            )
            if ckpt['skip_centering']:
                self.mean = torch.zeros_like(self.mean)
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.model.eval()
            for p in self.model.parameters():
                p.requires_grad_(False)

    def build_prior(self, latent):
        if latent == "normal":
            return torch.distributions.Normal(torch.tensor(0.0), torch.tensor(1.0))
        elif latent == "logistic":
            return StandardLogistic()
        else:
            raise NotImplementedError('Prior {} not supported.'.format(latent))

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        return self.model.log_prob(x).unsqueeze(-1) + self.log_norm_const

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        if shape is None:
            shape = (1,)
        if len(shape) > 1:
            raise ValueError(f"Can only sample shapes (batch_size, dim).")

        # Chunk to avoid OOM
        size = shape[0]
        iterations, rem_size = divmod(size, self.sample_chunk_size)

        # Collect samples
        with torch.no_grad():
            samples = [
                self.model.sample(self.sample_chunk_size) for _ in range(iterations)
            ]
            if rem_size:
                samples.append(self.model.sample(rem_size))

        # Concatenate samples
        samples = torch.cat(samples)
        assert samples.shape == (size, self.dim)
        return samples

    # def plots(self, samples: torch.Tensor, n_max=64) -> dict[str, Image.Image]:
    #     samples = samples + self.mean
    #     samples = samples.reshape(-1, 1, *self.shape)
    #     grid = make_grid(samples[:n_max], normalize=True)
    #     ndarr = (
    #         grid.mul(255)
    #         .add_(0.5)
    #         .clamp_(0, 255)
    #         .permute(1, 2, 0)
    #         .to("cpu", torch.uint8)
    #         .numpy()
    #     )
    #     im = Image.fromarray(ndarr)
    #     return {"plots/samples": im}


class MixtureNice(Distribution):
    """Class to handle mixture of NICE distributions, individually trained on one digit."""

    def __init__(
        self,
        equilibrated: bool = False,
        normalize: bool = True,
        digits=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
        models: list | None = None,
        checkpoints: list | None = None,
        means_data_path: list | None = None,
        local_minimums: str | None = None,
        sample_chunk_size: int = 10000,
        dim: int = 196,
        log_norm_const: float = 0.0,
        n_reference_samples=2048,
    ):
        # Call the parent constructor
        super().__init__(
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
        )
        # Get the default checkpoints
        self.digits = sorted(tuple(digits))
        self.n_digits = len(self.digits)
        if models is not None:
            models_ = [models[i] for i in self.digits]
        if checkpoints is None:
            checkpoints = [DATA_DIR / "nice_label_{}.pt".format(label) for label in self.digits]
        if means_data_path is None:
            means_data_path = [DATA_DIR / "mnist_mean_label_{}.pt".format(label) for label in self.digits]

        # Load all the nice models
        self.normalize = normalize
        self.nice_dists = [Nice(
            model=models_[i] if models is not None else None,
            checkpoint=checkpoints[i],
            mean_data_path=means_data_path[i],
            sample_chunk_size=sample_chunk_size,
            dim=dim,
            log_norm_const=log_norm_const,
            n_reference_samples=n_reference_samples,
        ) for i in range(self.n_digits)]
        # Make the weigths
        if equilibrated:
            self.register_buffer("mixture_weights", torch.ones((self.n_digits,)) / self.n_digits, persistent=False)
        else:
            weights = torch.ones((self.n_digits,))
            weights[::2] = 3.
            weights /= weights.sum()
            self.register_buffer("mixture_weights", weights, persistent=False)

        # Compute the modes
        if local_minimums is None:
            self.register_buffer('local_minimums', torch.load(
                DATA_DIR / "x_min_nf_mnist.pt", weights_only=False)[self.digits])
        else:
            x_init = torch.cat([self.nice_dists[i].sample((1,)) for i in range(self.n_digits)], dim=0)

            def U(x): return torch.cat([-self.nice_dists[i].log_prob(x[i].unsqueeze(0))
                                        for i in range(self.n_digits)], dim=0).flatten()

            def grad_U(x): return torch.cat([
                -self.nice_dists[i].score(x[i].unsqueeze(0) - self.nice_dists[i].mean) for i in range(self.n_digits)
            ], dim=0)
            x_min = run_gdflow(U, grad_U, x_init, n_steps=10000, dt=1e-4, verbose=True)
            for i in range(self.n_digits):
                x_min[i] += self.nice_dists[i].mean.flatten()
            if self.normalize:
                x_min = 2. * (x_min - 0.5)
            self.register_buffer('local_minimums', x_min)

    def sample(self, shape: tuple | None = None) -> torch.Tensor:
        """Returns samples from the mixture distribution."""
        # Sample the multinomial
        idx = torch.multinomial(self.mixture_weights, num_samples=shape[0], replacement=True)
        # Sample the flows
        ret = torch.empty((shape[0], self.dim), device=idx.device)
        for i in range(self.n_digits):
            mask = (idx == i)
            n_mask = int(mask.sum())
            if mask.sum() > 0:
                ret[mask] = self.nice_dists[i].sample((n_mask,))
                ret[mask] += self.nice_dists[i].mean
        # Normalize
        if self.normalize:
            ret = 2. * (ret - 0.5)
        return ret

    def unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the unnormalized log-likelihood of the mixture distribution at x"""
        # Normalize
        if self.normalize:
            x = (x + 1.) / 2.
        # Evaluate the log-prob
        log_probs = torch.stack([
            self.nice_dists[i].log_prob(x - self.nice_dists[i].mean).unsqueeze(-1) for i in range(self.n_digits)
        ], dim=0).view((self.n_digits, x.shape[0]))
        # Add the weights
        log_probs += torch.log(self.mixture_weights).unsqueeze(-1)
        # Return the log-prob
        ret = torch.logsumexp(log_probs, dim=0)
        if self.normalize:
            ret = ret - x.shape[-1] * math.log(2.0)
        return ret.unsqueeze(-1)

    def score(self, x: torch.Tensor, create_graph=False, return_log_prob=False) -> torch.Tensor:
        """Evaluates the score of the mixture distribution at x"""
        # Normalize
        if self.normalize:
            x = (x + 1.) / 2.
        x_ = torch.autograd.Variable(x.clone(), requires_grad=True)
        # Compute the log_prob and score for each model
        log_probs, scores = [], []
        for i in range(self.n_digits):
            with torch.set_grad_enabled(True):
                x_centered_ = x_ - self.nice_dists[i].mean
                log_prob_ = self.nice_dists[i].log_prob(x_centered_)
                scores.append(torch.autograd.grad(log_prob_.sum(), x_centered_, create_graph=create_graph)[0])
            log_probs.append(log_prob_.detach().flatten())
        log_probs = torch.stack(log_probs, dim=0)
        scores = torch.stack(scores, dim=0)
        # Compute the weights
        grad_weights = torch.nn.functional.softmax(log_probs + torch.log(self.mixture_weights.unsqueeze(-1)), dim=0)
        # Compute the gradient
        grad = torch.sum(grad_weights.unsqueeze(-1) * scores, dim=0)
        if self.normalize:
            grad = grad / 2.0
        if return_log_prob:
            log_prob = torch.logsumexp(log_probs + torch.log(self.mixture_weights.unsqueeze(-1)), dim=0)
            if self.normalize:
                log_prob = log_prob - x.shape[-1] * math.log(2.0)
            return log_prob, grad
        else:
            return grad

    def has_entropy(self):
        """Informs if the entropy of the mode weights is computable"""
        return True

    def get_classes(self, samples):
        """Returns the MNIST classes associated to samples, based on maximum likelihood."""
        if self.normalize:
            def process(x): return (x + 1.) / 2.
        else:
            def process(x): return x
        return torch.stack([
            self.nice_dists[i].log_prob(process(samples) - self.nice_dists[i].mean).unsqueeze(-1) for i in range(self.n_digits)
        ], dim=0).argmax(dim=0).flatten()

    def compute_mode_count(self, samples):
        """Computes the empirical mode weights"""
        labels = self.get_classes(samples)
        return torch.FloatTensor([
            (labels == i).sum() for i in range(self.n_digits)
        ]).to(labels.device)

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
        true_hist = self.mixture_weights.flatten()
        return torch.sum(true_hist * torch.log(true_hist / hist))

    def tv_weights(self, samples, counts=None):
        """Computes the TV between the empirical weights and the true ones"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.mixture_weights.flatten()
        return torch.sum(torch.abs(hist - true_hist))

    def compute_forgotten_modes(self, samples, tol=0.05, counts=None):
        """Compute the number of forgotten modes"""
        if counts is None:
            counts = self.compute_mode_count(samples)
        hist = counts.flatten() / counts.sum()
        true_hist = self.mixture_weights.flatten()
        return torch.sum(hist < tol * true_hist.min()) / self.n_digits

    def compute_mode_weight(self, samples):
        """Computes the weight on the first mode."""
        if self.n_digits == 2:
            counts = self.compute_mode_count(samples)
            return 100. * counts[0] / counts.sum()
        else:
            return 0.0

    def compute_stats_sampling(self, return_samples=False):
        """Compute various metrics based on samples."""
        samples = super().compute_stats_sampling(return_samples=True)
        counts = self.compute_mode_count(samples)
        self.expectations['mode_weight'] = self.compute_mode_weight(samples).item()
        self.expectations['emc'] = self.entropy(samples, counts=counts).item()
        self.expectations['kl_weights'] = self.kl_weights(samples, counts=counts).item()
        self.expectations['tv_weights'] = self.tv_weights(samples, counts=counts).item()
        self.expectations['num_forgotten_modes'] = self.compute_forgotten_modes(samples, counts=counts).item()
        if return_samples:
            return samples

    def _apply(self, fn):
        super(MixtureNice, self)._apply(fn)
        for model in self.nice_dists:
            model._apply(fn)
