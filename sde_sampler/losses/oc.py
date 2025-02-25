from __future__ import annotations

import logging
from typing import Callable

import torch
import math

from sde_sampler.eq.sdes import OU
from sde_sampler.utils.autograd import compute_divx
from sde_sampler.utils.common import Results


class BaseOCLoss:
    """Base class for variational loss on joint distributions"""

    def __init__(
            self,
            generative_ctrl: Callable,
            generative_ctrl_ema: Callable,
            sde: OU | None = None,
            method: str = "kl",
            traj_per_sample: int = 1,
            filter_samples: Callable | None = None,
            max_rnd: float | None = None,
            sde_ctrl_dropout: float | None = None,
            sde_ctrl_noise: float | None = None,
            **kwargs,
    ):
        """Builds the variational loss object.

        Args:
        * generative_ctrl (Callable): neural network used to model the control term in the generative SDE
        * generative_ctrl_ema (Callable): EMA version of generative_ctrl
        * sde (class from sde_sampler.eq.sdes): type of noising process
        * method (str): type of variational loss (KL or LV)
        * traj_per_sample (int): number of trajectories sampled per batch element (default is 1)
        * filter_samples (Callable): filter function over the trajectory samples (default is None)
        * max_rnd (float) : maximum value for the density log-ratio (default is None)
        * sde_ctrl_noise (float): noise magnitude to add to the sde control in LV setting (default is None)
        """
        self.generative_ctrl = generative_ctrl
        self.generative_ctrl_ema = generative_ctrl_ema
        self.sde = sde
        if method not in ["kl", "kl_ito", "lv", "lv_traj"]:
            raise ValueError("Unknown loss method.")
        self.method = method
        if traj_per_sample == 1 and self.method == "lv_traj":
            raise ValueError("Cannot compute variance over a single trajectory.")
        self.traj_per_sample = traj_per_sample

        # Filter
        self.filter_samples = filter_samples
        self.max_rnd = max_rnd

        # SDE controls
        self.sde_ctrl_noise = sde_ctrl_noise
        self.sde_ctrl_dropout = sde_ctrl_dropout
        if self.method in ["kl", "kl_ito"]:
            for attr in ["sde_ctrl_noise", "sde_ctrl_dropout"]:
                if getattr(self, attr) is not None:
                    logging.warning("%s should only be used for the log-variance loss.")

        # Metrics
        self.n_filtered = 0

    def filter(
            self, rnd: torch.Tensor, samples: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Filters the density log-ratio and the obtained samples on the trajectory.

        Args:
        * rnd (torch.Tensor of shape (batch_size,1)): density log-ratio
        * samples (torch.Tensor of shape (batch_size,dim)): samples approximating the target
        """
        mask = True
        if samples is not None and self.filter_samples is not None:
            mask = self.filter_samples(samples)
        if self.max_rnd is None:
            return mask & rnd.isfinite()
        return mask & (rnd < self.max_rnd)

    def generative_and_sde_ctrl(
            self, t: torch.Tensor, x: torch.Tensor, use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluates the generative control at (t,x) -> generative_ctrl, and generates a detached version -> sde_ctrl

        Args:
        * t (torch.Tensor of shape (batch_size,1))
        * x (torch.Tensor of shape (batch_size,dim))
        * use_ema (bool): indicates whether to use the EMA version (default is False)
        """
        if use_ema:
            generative_ctrl = self.generative_ctrl_ema(t, x)
        else:
            generative_ctrl = self.generative_ctrl(t, x)
        sde_ctrl = generative_ctrl.detach()
        if self.sde_ctrl_noise is not None:
            sde_ctrl += self.sde_ctrl_noise * torch.randn_like(sde_ctrl)
        if self.sde_ctrl_dropout is not None:
            mask = torch.rand_like(sde_ctrl) > self.sde_ctrl_dropout
            sde_ctrl[mask] = -(self.sde.drift(t, x) / self.sde.diff(t, x))[mask]
        return generative_ctrl, sde_ctrl

    def compute_loss(
            self, rnd: torch.Tensor, samples: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict]:
        """Computes the variational loss

        Args:
        * rnd (torch.Tensor of shape (batch_size,1)): density log-ratio
        * samples (torch.Tensor of shape (batch_size,dim)): samples approximating the target
        """
        # Compute loss
        mask = self.filter(rnd, samples=samples)
        assert mask.shape == rnd.shape
        if self.method == "lv_traj":
            # compute variance over the trajectories
            rnd = rnd.reshape(self.traj_per_sample, -1, 1)
            mask = mask.reshape(self.traj_per_sample, -1, 1)
            mask = mask.all(dim=0)
            self.n_filtered += self.traj_per_sample * (mask.numel() - mask.sum()).item()
            loss = rnd[:, mask].var(dim=0).mean()
        else:
            self.n_filtered += (mask.numel() - mask.sum()).item()
            if self.method == "lv":
                loss = rnd[mask].var()
            else:
                loss = rnd[mask].mean()

        return loss, {"train/n_filtered_cumulative": self.n_filtered}

    @staticmethod
    def compute_results(
            rnd: torch.Tensor,
            compute_weights: bool = False,
            ts: torch.Tensor | None = None,
            samples: torch.Tensor | None = None,
            xs: torch.Tensor | None = None,
    ):
        """Computes various metrics based on the density log-ratio and the generated samples

         Args:
        * rnd (torch.Tensor of shape (batch_size,1)): density log-ratio
        * compute_weights (bool): indicates whether to compute importance weights from the rnd
        * ts (torch.Tensor of shape (n_steps + 1, batch_size,1): time discretization
        * samples (torch.Tensor of shape (batch_size,dim)): samples approximating the target
        * xs (torch.Tensor of shape (n_steps + 1, batch_size,dim)): samples from the generative trajectories
        """
        metrics = {}
        neg_rnd = -rnd
        metrics["eval/elbo"] = neg_rnd.mean().item()
        if compute_weights:
            # Compute normalized importance weights
            weights = torch.nn.functional.softmax(neg_rnd, dim=0)

            log_norm_const_preds = {
                # "log_norm_const_lb_ito": neg_rnd.mean().item(),
                "log_norm_const_is": (neg_rnd.logsumexp(dim=0) - math.log(len(weights))).item(),
            }
            metrics["eval/lv_loss"] = rnd.var().item()
        else:
            weights = None
            # log_norm_const_preds = {"log_norm_const_lb": neg_rnd.mean().item()}
            log_norm_const_preds = {}
        return Results(
            samples=samples,
            weights=weights,
            log_norm_const_preds=log_norm_const_preds,
            ts=ts,
            xs=xs,
            metrics=metrics,
        )

    def __call__(
            self, ts: torch.Tensor, x: torch.Tensor, *args, **kwargs
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Simulates denoising trajectories starting from x and computes the variational loss.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        """
        raise NotImplementedError

    def eval(self, ts: torch.Tensor, x: torch.Tensor, *args, use_ema=True, **kwargs) -> Results:
        """[TEST] Simulates denoising trajectories starting from x and computes various metrics.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectoy sampling (default is True)
        """
        raise NotImplementedError

    def load_state_dict(self, state_dict: dict):
        self.n_filtered = state_dict["n_filtered"]

    def state_dict(self) -> dict:
        return {"n_filtered": self.n_filtered}


class EMReferenceSDELoss(BaseOCLoss):
    """Basic RDS loss with EM integrator"""

    def __init__(self, *args, reference_ctrl: Callable | None = None,
                 use_rescaling: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        """Builds the variational loss object. Here, generative_ctrl is meant to approximate: nabla log p_t/p_t^ref

        Additional args:
        * reference_ctrl (Callable): reference control term in the generative SDE (corresponds to nabla log p_t^ref)
        * use_rescaling (bool): indicates whether to rescale the generative control
        """
        self.reference_ctrl = reference_ctrl
        self.use_rescaling = use_rescaling

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        rnd = 0.0

        # Final time
        T = ts[-1]

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(T - s, x, use_ema=use_ema)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl = self.generative_ctrl_ema(T - s, x)
                else:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(T - s, x)
            sde_diff = self.sde.diff(T - s, x)
            dt = t - s

            # Rescale if needed
            if not self.use_rescaling:
                generative_ctrl *= sde_diff
                sde_ctrl *= sde_diff

            # Loss increments
            if change_sde_ctrl:
                running_cost = generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl)
                rnd += running_cost.sum(dim=-1, keepdim=True) * dt
            else:
                rnd += 0.5 * (generative_ctrl ** 2).sum(dim=-1, keepdim=True) * dt

            # Euler-Maruyama
            db = torch.randn_like(x) * dt.sqrt()
            drift_ = -self.sde.drift(T - s, x)
            if self.reference_ctrl is not None:
                drift_ += torch.square(sde_diff) * self.reference_ctrl(T - s, x)
            x = x + (drift_ + sde_diff * sde_ctrl) * dt + sde_diff * db

            # Compute ito integral
            rnd += (generative_ctrl * db).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd += reference_log_prob(x).view((-1, 1)) - terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)

        return x, rnd, xs

    def compute_eubo(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
            use_ema: bool = False
    ) -> tuple[torch.Tensor]:
        """Simulates noising trajectories starting from x and computes the corresponding density log-ratio.
        This is based on the EUBO metric proposed in https://arxiv.org/abs/2406.07423.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the target distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

         Returns:
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        """
        # Here, x corresponds to be true samples from the target distribution

        # Terminal cost at data distribution
        rnd = reference_log_prob(x).view((-1, 1)) - terminal_unnorm_log_prob(x)

        # Final time
        T = ts[-1]

        times_s = ts[:-1].flip((0,))
        times_t = ts[1:].flip((0,))
        mean_factors, var_factors = self.sde.transition_params(T - times_t, T - times_s)
        std_factors = var_factors.sqrt()

        # Compute the rnd starting from the end
        for i, (s, t) in enumerate(zip(times_s, times_t)):
            # Noise the samples
            z = torch.randn_like(x)
            x *= mean_factors[i]
            x += std_factors[i] * z

            if use_ema:
                generative_ctrl = self.generative_ctrl_ema(T - s, x)
            else:
                generative_ctrl = self.generative_ctrl(T - s, x)
            ref_ctrl = self.reference_ctrl(T - s, x)
            sde_diff = self.sde.diff(T - s, x)
            dt = t - s

            # Rescale if needed
            if self.use_rescaling:
                generative_ctrl /= sde_diff

            # Loss increments
            running_cost = generative_ctrl * (ref_ctrl + 0.5 * generative_ctrl)
            rnd -= running_cost.sum(dim=-1, keepdim=True) * dt * sde_diff ** 2
            rnd += (generative_ctrl * x).sum(dim=-1, keepdim=True) * (
                1. / mean_factors[i] - 1. + self.sde.drift_coeff_t(T - s) * dt)

            # Compute ito integral
            rnd -= (generative_ctrl * z).sum(dim=-1, keepdim=True) * (std_factors[i] / mean_factors[i])

        assert rnd.shape == (x.shape[0], 1)

        return rnd

    def __call__(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Simulates denoising trajectories starting from x and computes the variational loss.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        """
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            change_sde_ctrl=change_sde_ctrl,
            return_traj=False,
        )

        return self.compute_loss(rnd, samples=samples)

    def eval(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable | None = None,
            compute_weights: bool = True,
            return_traj: bool = True,
            use_ema: bool = True,
    ) -> Results:
        """[TEST] Simulates denoising trajectories starting from x and computes various metrics.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is True)
        """
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            change_sde_ctrl=False,
            return_traj=return_traj,
            use_ema=use_ema
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )


class EIReferenceSDELoss(EMReferenceSDELoss):
    """Basic RDS loss with EI integrator"""

    def __init__(self, *args, reference_ctrl: Callable | None = None, **kwargs):
        """Builds the variational loss object. Here, generative_ctrl is meant to approximate: nabla log p_t/p_t^ref

        Additional args:
        * reference_ctrl (Callable): reference control term in the generative SDE (corresponds to nabla log p_t^ref)
        """
        super().__init__(*args, use_rescaling=False, **kwargs)
        self.reference_ctrl = reference_ctrl
        # NOTE: we do not use any rescaling here

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        rnd = 0.0

        # Final time
        T = ts[-1]

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(T - s, x, use_ema=use_ema)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl = self.generative_ctrl_ema(T - s, x)
                else:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(T - s, x)

            # Loss increments
            if change_sde_ctrl:
                running_cost = generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl)
                rnd += self.sde.omega(s, t) * running_cost.sum(dim=-1, keepdim=True)
            else:
                rnd += 0.5 * self.sde.omega(s, t) * (generative_ctrl ** 2).sum(dim=-1, keepdim=True)

            # Exponential integration
            x, z = self.sde.ei_integration_step(x, s, t, self.reference_ctrl(T - s, x) + sde_ctrl)

            # Compute ito integral
            rnd += torch.sqrt(self.sde.omega(s, t)) * (generative_ctrl * z).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd += reference_log_prob(x).view((-1, 1)) - terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)
        return x, rnd, xs

    def compute_eubo(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
            use_ema: bool = False
    ) -> tuple[torch.Tensor]:
        """Simulates noising trajectories starting from x and computes the corresponding density log-ratio.
        This is based on the EUBO metric proposed in https://arxiv.org/abs/2406.07423.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the target distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

         Returns:
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        """
        # Here, x corresponds to be true samples from the target distribution

        # Terminal cost at data distribution
        rnd = reference_log_prob(x).view((-1, 1)) - terminal_unnorm_log_prob(x)

        # Final time
        T = ts[-1]

        times_s = ts[:-1].flip((0,))
        times_t = ts[1:].flip((0,))
        mean_factors, var_factors = self.sde.transition_params(T - times_t, T - times_s)
        std_factors = var_factors.sqrt()

        # Compute the rnd starting from the end
        for i, (s, t) in enumerate(zip(times_s, times_t)):
            # Noise the samples
            z = torch.randn_like(x)
            x *= mean_factors[i]
            x += std_factors[i] * z

            if use_ema:
                generative_ctrl = self.generative_ctrl_ema(T - s, x)
            else:
                generative_ctrl = self.generative_ctrl(T - s, x)
            ref_ctrl = self.reference_ctrl(T - s, x)

            # Loss increments
            running_cost = generative_ctrl * (ref_ctrl + 0.5 * generative_ctrl)
            rnd -= running_cost.sum(dim=-1, keepdim=True) * self.sde.omega(s, t)

            # Compute ito integral
            rnd -= (generative_ctrl * z).sum(dim=-1, keepdim=True) * torch.sqrt(self.sde.omega(s, t))

        assert rnd.shape == (x.shape[0], 1)

        return rnd


class DDPMLikeReferenceSDELoss(EMReferenceSDELoss):
    """Basic RDS loss with DDPM integrator"""

    def __init__(self, *args, reference_ctrl: Callable | None = None, **kwargs):
        """Builds the variational loss object. Here, generative_ctrl is meant to approximate: nabla log p_t/p_t^ref

        Additional args:
        * reference_ctrl (Callable): reference control term in the generative SDE (corresponds to nabla log p_t^ref)
        """
        super().__init__(*args, use_rescaling=False, **kwargs)
        self.reference_ctrl = reference_ctrl
        # NOTE: we do not use any rescaling here

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference unnormalized log-likelihood
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        rnd = 0.0

        # Final time
        T = ts[-1]

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(T - s, x, use_ema=use_ema)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl = self.generative_ctrl_ema(T - s, x)
                else:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(T - s, x)

            # Loss increments
            if change_sde_ctrl:
                running_cost = generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl)
                rnd += self.sde.omega_ddpm(s, t) * running_cost.sum(dim=-1, keepdim=True)
            else:
                rnd += 0.5 * self.sde.omega_ddpm(s, t) * (generative_ctrl ** 2).sum(dim=-1, keepdim=True)

            # Exponential integration
            x, z = self.sde.ddpm_integration_step(x, s, t, self.reference_ctrl(T - s, x) + sde_ctrl)

            # Compute ito integral
            rnd += torch.sqrt(self.sde.omega_ddpm(s, t)) * (generative_ctrl * z).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd += reference_log_prob(x).view((-1, 1)) - terminal_unnorm_log_prob(x)

        assert rnd.shape == (x.shape[0], 1)
        if return_traj:
            xs = torch.stack(xs)

        return x, rnd, xs


class ControlledLangevinSDELoss(BaseOCLoss):
    """Discrete time CMCD loss as per Appendix D.5 without schedule learning"""

    def __init__(self, *args, use_rescaling: bool = True, **kwargs, ):
        """Builds the variational loss object. Here, generative_ctrl is meant to correct the Annealed Langevin process.

        Additional args:
        * use_rescaling (bool): indicates whether to rescale the generative control
        """
        super().__init__(*args, **kwargs)
        self.use_rescaling = use_rescaling

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            train: bool = True,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood
        * train (bool): indicates whether the simulation is used for training (default is True)
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        if train and self.method in ["kl", "kl_ito"]:
            rnd = 0.0
        else:
            rnd = initial_log_prob(x)
            assert rnd.shape == (x.shape[0], 1)

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl_s, sde_ctrl = self.generative_and_sde_ctrl(s, x, use_ema=use_ema)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl_s = self.generative_ctrl_ema(s, x)
                else:
                    sde_ctrl = generative_ctrl_s = self.generative_ctrl(s, x)
            sde_diff = self.sde.diff_coeff
            # IMPORTANT: here, the volatility is chosen constant
            dt = t - s

            # Rescale if needed
            if not self.use_rescaling:
                generative_ctrl_s *= 0.5 * sde_diff
                sde_ctrl *= 0.5 * sde_diff

            # Euler-Maruyama
            db = dt.sqrt() * torch.randn_like(x)
            drift_s = self.sde.drift(s, x)
            y = x + (drift_s + sde_ctrl * sde_diff) * dt + sde_diff * db

            # Computations on y
            drift_t = self.sde.drift(t, y)
            if use_ema:
                generative_ctrl_t = self.generative_ctrl_ema(t, y)
            else:
                generative_ctrl_t = self.generative_ctrl(t, y)
            # Rescale if needed
            if not self.use_rescaling:
                generative_ctrl_t *= 0.5 * sde_diff

            # Loss increment
            cost = (drift_s + drift_t) / sde_diff + generative_ctrl_s - generative_ctrl_t
            rnd += 0.5 * (cost ** 2).sum(dim=-1, keepdim=True) * dt
            rnd += (cost * (sde_ctrl - generative_ctrl_s)).sum(dim=-1, keepdim=True) * dt

            # Compute ito integral
            rnd += (cost * db).sum(dim=-1, keepdim=True)

            x = y

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd -= terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)
        return x, rnd, xs

    def compute_eubo(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            use_ema: bool = False
    ) -> tuple[torch.Tensor]:
        """Simulates noising trajectories starting from x and computes the corresponding density log-ratio.
       This is based on the EUBO metric proposed in https://arxiv.org/abs/2406.07423.

       Args:
       * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
       * x (torch.Tensor of shape (batch_size, dim)): samples from the target distribution
       * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
       * initial_log_prob (Callable): base log-likelihood
       * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
       * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
       """
        # Here, x corresponds to be true samples from the target distribution

        # Terminal cost at data distribution
        rnd = - terminal_unnorm_log_prob(x)

        times_s = ts[:-1].flip((0,))
        times_t = ts[1:].flip((0,))

        # Compute the rnd starting from the end
        for (s, t) in zip(times_s, times_t):
            # Evaluate
            if use_ema:
                generative_ctrl_t = self.generative_ctrl_ema(t, x)
            else:
                generative_ctrl_t = self.generative_ctrl(t, x)
            sde_diff = self.sde.diff_coeff
            # IMPORTANT: here, the volatility is chosen constant
            dt = t - s

            # Rescale if needed
            if not self.use_rescaling:
                generative_ctrl_t *= 0.5 * sde_diff

            # Euler-Maruyama
            db = dt.sqrt() * torch.randn_like(x)
            drift_t = self.sde.drift(t, x)
            y = x + (drift_t - generative_ctrl_t * sde_diff) * dt + sde_diff * db

            # Computations on y
            drift_s = self.sde.drift(t, y)
            if use_ema:
                generative_ctrl_s = self.generative_ctrl_ema(s, y)
            else:
                generative_ctrl_s = self.generative_ctrl(s, y)
            # Rescale if needed
            if not self.use_rescaling:
                generative_ctrl_s *= 0.5 * sde_diff

            # Loss increment
            cost = (drift_s + drift_t) / sde_diff + generative_ctrl_s - generative_ctrl_t
            rnd -= 0.5 * (cost ** 2).sum(dim=-1, keepdim=True) * dt

            # Compute ito integral
            rnd -= (cost * db).sum(dim=-1, keepdim=True)

            x = y

        rnd += initial_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        return rnd

    def __call__(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None,
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Simulates denoising trajectories starting from x and computes the variational loss.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood
        """
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            change_sde_ctrl=change_sde_ctrl,
            train=True,
            return_traj=False,
        )
        return self.compute_loss(rnd, samples=samples)

    def eval(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            compute_weights: bool = True,
            return_traj: bool = True,
            use_ema: bool = True
    ) -> Results:
        """[TEST] Simulates denoising trajectories starting from x and computes various metrics.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is True)
        """
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            train=False,
            return_traj=return_traj,
            use_ema=use_ema
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )


class DiscreteTimeReversalLossEI(BaseOCLoss):
    """Discrete time DIS loss as per Appendix D.4"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_rescaling = False
        # NOTE: we do not use any rescaling here
    """Builds the variational loss object. Here, generative_ctrl is meant to approximate: nabla log p_t"""

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            train: bool = True,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood
        * train (bool): indicates whether the simulation is used for training (default is True)
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        if train and self.method in ["kl", "kl_ito"]:
            rnd = 0.0
        else:
            rnd = initial_log_prob(x)
            assert rnd.shape == (x.shape[0], 1)

        # Final time
        T = ts[-1]

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(T - s, x, use_ema=use_ema)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl = self.generative_ctrl_ema(T - s, x)
                else:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(T - s, x)

            # Loss increments
            if change_sde_ctrl:
                running_cost = generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl)
                rnd += self.sde.omega(s, t) * running_cost.sum(dim=-1, keepdim=True)
            else:
                rnd += 0.5 * self.sde.omega(s, t) * (generative_ctrl ** 2).sum(dim=-1, keepdim=True)

            # Exponential integration
            x, z = self.sde.ei_integration_step(x, s, t, sde_ctrl)

            # Compute ito integral
            rnd += torch.sqrt(self.sde.omega(s, t)) * (generative_ctrl * z).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd -= terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)
        return x, rnd, xs

    def compute_eubo(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            use_ema: bool = False
    ) -> tuple[torch.Tensor]:
        """Simulates noising trajectories starting from x and computes the corresponding density log-ratio.
       This is based on the EUBO metric proposed in https://arxiv.org/abs/2406.07423.

       Args:
       * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
       * x (torch.Tensor of shape (batch_size, dim)): samples from the target distribution
       * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
       * initial_log_prob (Callable): base log-likelihood
       * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
       * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
       """
        # Here, x corresponds to be true samples from the target distribution

        # Terminal cost at data distribution
        rnd = - terminal_unnorm_log_prob(x)

        # Final time
        T = ts[-1]

        times_s = ts[:-1].flip((0,))
        times_t = ts[1:].flip((0,))
        mean_factors, var_factors = self.sde.transition_params(T - times_t, T - times_s)
        std_factors = var_factors.sqrt()

        # Compute the rnd starting from the end
        for i, (s, t) in enumerate(zip(times_s, times_t)):
            # Noise the samples
            z = torch.randn_like(x)
            x *= mean_factors[i]
            x += std_factors[i] * z

            if use_ema:
                generative_ctrl = self.generative_ctrl_ema(T - s, x)
            else:
                generative_ctrl = self.generative_ctrl(T - s, x)

            # Loss increments
            running_cost = 0.5 * generative_ctrl ** 2
            rnd -= running_cost.sum(dim=-1, keepdim=True) * self.sde.omega(s, t)

            # Compute ito integral
            rnd -= (generative_ctrl * z).sum(dim=-1, keepdim=True) * torch.sqrt(self.sde.omega(s, t))

        rnd += initial_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        return rnd

    def __call__(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None,
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Simulates denoising trajectories starting from x and computes the variational loss.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood
        """
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            change_sde_ctrl=change_sde_ctrl,
            train=True,
            return_traj=False,
        )
        return self.compute_loss(rnd, samples=samples)

    def eval(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            compute_weights: bool = True,
            return_traj: bool = True,
            use_ema: bool = True
    ) -> Results:
        """[TEST] Simulates denoising trajectories starting from x and computes various metrics.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is True)
        """
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            train=False,
            return_traj=return_traj,
            use_ema=use_ema
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )


class TimeReversalLoss(BaseOCLoss):
    """Original DIS loss : see https://github.com/juliusberner/sde_sampler/blob/main/sde_sampler/losses/oc.py"""

    def __init__(
            self,
            *args,
            inference_ctrl: Callable | None = None,
            div_estimator: str | None = None,
            use_rescaling: bool = True,
            **kwargs,
    ):
        """Builds the variational loss object. Here, generative_ctrl is meant to approximate: nabla log p_t

        Args:
        * inference_ctrl (Callable): neural network used to model the control term in the noising SDE (default is None)
        * div_estimator (str): type of divergence estimator
        * use_rescaling (bool): indicates whether to rescale the generative control and the inference control"""
        super().__init__(*args, **kwargs)
        self.inference_ctrl = inference_ctrl
        self.div_estimator = div_estimator
        if not use_rescaling:
            raise ValueError('use_rescaling must be True for TimeReversalLoss.')
        self.use_rescaling = use_rescaling
        if self.div_estimator is not None and self.inference_ctrl is None:
            logging.warning(
                "Without inference control the divergence estimator has no effect."
            )

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            train: bool = True,
            compute_ito_int: bool = False,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood (here Gaussian)
        * train (bool): indicates whether the simulation is used for training (default is True)
        * compute_ito_int (bool): indicates whether to compute the Brownian motion part of the loss (default is False)
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        if train and self.method in ["kl", "kl_ito"]:
            rnd = 0.0
        else:
            rnd = initial_log_prob(x)
            assert rnd.shape == (x.shape[0], 1)

        xs = [x] if return_traj else None
        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(s, x)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(s, x)
                else:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(s, x)
            sde_diff = self.sde.diff(s, x)
            dt = t - s
            # Rescale if needed
            if not self.use_rescaling:
                generative_ctrl *= sde_diff
                sde_ctrl *= sde_diff

            # Loss increments
            if self.inference_ctrl is None:
                gen_plus_inf_ctrl = gen_minus_inf_ctrl = generative_ctrl

            else:
                div_estimator = self.div_estimator if train else None
                div_ctrl, inference_ctrl = compute_divx(
                    self.inference_ctrl,
                    s,
                    x,
                    noise_type=div_estimator,
                    create_graph=train,
                )

                # Rescale if needed
                if not self.use_rescaling:
                    inference_ctrl *= sde_diff
                    div_ctrl *= sde_diff

                # This assumes the diffusion coeff. to be independent of x
                rnd += sde_diff * div_ctrl * dt
                gen_plus_inf_ctrl = generative_ctrl + inference_ctrl
                gen_minus_inf_ctrl = generative_ctrl - inference_ctrl

            if change_sde_ctrl:
                cost = gen_plus_inf_ctrl * (sde_ctrl - 0.5 * gen_minus_inf_ctrl)
                rnd += cost.sum(dim=-1, keepdim=True) * dt
            else:
                rnd += 0.5 * (gen_plus_inf_ctrl ** 2).sum(dim=-1, keepdim=True) * dt

            if not train:
                rnd -= self.sde.drift_div_int(s, t, x)

            # Euler-Maruyama
            db = torch.randn_like(x) * dt.sqrt()
            x = x + (self.sde.drift(s, x) + sde_diff * sde_ctrl) * dt + sde_diff * db

            # Compute ito integral
            if compute_ito_int:
                rnd += (gen_plus_inf_ctrl * db).sum(dim=-1, keepdim=True)

            if return_traj:
                xs.append(x)

        # Terminal cost
        rnd -= terminal_unnorm_log_prob(x)
        assert rnd.shape == (x.shape[0], 1)

        if return_traj:
            xs = torch.stack(xs)
        return x, rnd, xs

    def __call__(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None,
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Simulates denoising trajectories starting from x and computes the variational loss.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood (here Gaussian)
        """
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        compute_ito_int = self.method != "kl"
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            compute_ito_int=compute_ito_int,
            change_sde_ctrl=change_sde_ctrl,
            train=True,
            return_traj=False,
        )
        return self.compute_loss(rnd, samples=samples)

    def eval(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            initial_log_prob: Callable | None = None,
            compute_weights: bool = True,
            return_traj: bool = True,
            use_ema: bool = True
    ) -> Results:
        """[TEST] Simulates denoising trajectories starting from x and computes various metrics.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * initial_log_prob (Callable): base log-likelihood (here Gaussian)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is True)
        """
        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            initial_log_prob=initial_log_prob,
            compute_ito_int=compute_weights,
            train=False,
            return_traj=return_traj,
            use_ema=use_ema
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )


class ExponentialIntegratorSDELoss(BaseOCLoss):
    """Original DDS loss: : see https://github.com/juliusberner/sde_sampler/blob/main/sde_sampler/losses/oc.py"""

    def __init__(self, *args, alpha: float, sigma: float, **kwargs):
        """Builds the variational loss object. Here, generative_ctrl is meant to approximate: nabla log p_t/p_t^ref"""
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.sigma = sigma

    def simulate(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
            compute_ito_int: bool = False,
            change_sde_ctrl: bool = False,
            return_traj: bool = False,
            use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Simulates denoising trajectories starting from x and computes the corresponding density log-ratio.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference log-likelihood (here Gaussian)
        * compute_ito_int (bool): indicates whether to compute the Brownian motion part of the loss (default is False)
        * change_sde_ctrl (bool): indicates whether to detach the generative control term (used with LV, default is False)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is False)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)

        Returns:
        * x (torch.Tensor of shape (batch_size, dim)): samples approximating the target distribution
        * rnd (torch.Tensor of shape (batch_size, 1)): density log-ratio
        * xs (torch.Tensor of shape (n_steps+1, batch_size, 1)): samples from the generative trajectories (default is None)
        """
        # Initial cost
        rnd = 0.0

        xs = [x] if return_traj else None

        # Simulate
        for s, t in zip(ts[:-1], ts[1:]):
            # Evaluate
            if change_sde_ctrl:
                generative_ctrl, sde_ctrl = self.generative_and_sde_ctrl(s, x, use_ema=use_ema)
                running_cost = (
                    generative_ctrl * (sde_ctrl - 0.5 * generative_ctrl)
                ).sum(dim=-1, keepdim=True)
            else:
                if use_ema:
                    sde_ctrl = generative_ctrl = self.generative_ctrl_ema(s, x)
                else:
                    sde_ctrl = generative_ctrl = self.generative_ctrl(s, x)
                running_cost = 0.5 * (generative_ctrl ** 2).sum(dim=-1, keepdim=True)
            dt = t - s

            # Exponential integrator as implemented by Vargas et.al
            beta_k = torch.clip(self.alpha * dt.sqrt(), 0, 1)
            alpha_k = torch.sqrt(1.0 - beta_k ** 2)
            rnd += beta_k ** 2 * self.sigma ** 2 * running_cost
            noise = torch.randn_like(x)
            x = (
                x * alpha_k
                + (beta_k ** 2) * (self.sigma ** 2) * sde_ctrl
                + self.sigma * beta_k * noise
            )

            # Compute ito integral
            if compute_ito_int:
                rnd += (self.sigma * generative_ctrl * noise * beta_k).sum(
                    dim=-1, keepdim=True
                )

            if return_traj:
                xs.append(x)

        # compute reference log prob value based on based in prior
        reference_log_prob_value = reference_log_prob(x).view((-1, 1))
        rnd += reference_log_prob_value - terminal_unnorm_log_prob(x)

        assert rnd.shape == (x.shape[0], 1)  # one loss number for each sample

        if return_traj:
            xs = torch.stack(xs)

        return x, rnd, xs

    def __call__(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable,
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Simulates denoising trajectories starting from x and computes the variational loss.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference log-likelihood (here Gaussian)
        """
        # Repeat initial values
        if self.traj_per_sample != 1:
            x = x.repeat(self.traj_per_sample, 1, 1).reshape(-1, x.shape[-1])

        # Simulate
        compute_ito_int = self.method != "kl"
        change_sde_ctrl = self.method in ["lv", "lv_traj"]
        samples, rnd, _ = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            compute_ito_int=compute_ito_int,
            change_sde_ctrl=change_sde_ctrl,
            return_traj=False,
        )

        return self.compute_loss(rnd, samples=samples)

    def eval(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            terminal_unnorm_log_prob: Callable,
            reference_log_prob: Callable | None = None,
            compute_weights: bool = True,
            return_traj: bool = True,
            use_ema: bool = True,
    ) -> Results:
        """[TEST] Simulates denoising trajectories starting from x and computes various metrics.

        Args:
        * ts (torch.Tensor of shape (n_steps + 1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim)): samples from the base distribution
        * terminal_unnorm_log_prob (Callable): target unnormalized log-likelihood
        * reference_log_prob (Callable): reference log-likelihood (here Gaussian)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is True)
        """

        samples, rnd, xs = self.simulate(
            ts,
            x,
            terminal_unnorm_log_prob=terminal_unnorm_log_prob,
            reference_log_prob=reference_log_prob,
            compute_ito_int=compute_weights,
            change_sde_ctrl=False,
            return_traj=return_traj,
            use_ema=use_ema
        )
        return BaseOCLoss.compute_results(
            rnd, compute_weights=compute_weights, ts=ts, samples=samples, xs=xs
        )
