from __future__ import annotations

import time
from typing import Callable

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.nn import Module

from sde_sampler.distr.base import Distribution, WrapperDistrNN, sample_uniform
from sde_sampler.distr.delta import Delta
from sde_sampler.distr.gauss import Gauss, GaussFull
from sde_sampler.eq.integrator import EulerIntegrator
from sde_sampler.eq.sdes import OU, VP, PinnedBM, ControlledSDE, ControlledLangevinSDE
from sde_sampler.eval.plots import get_plots
from sde_sampler.losses.oc import BaseOCLoss
from sde_sampler.solver.base import Trainable
from sde_sampler.utils.common import Results, clip_and_log


class TrainableDiff(Trainable):
    """Base class for diffusion-based variational samplers"""
    save_attrs = Trainable.save_attrs + ["generative_ctrl"]

    def __init__(self, cfg: DictConfig):
        """Builds the object from the hydra configuration"""
        super().__init__(cfg=cfg)

        # Train
        self.train_batch_size: int = self.cfg.train_batch_size
        self.train_timesteps: Callable = instantiate(self.cfg.train_timesteps)
        self.train_ts = None
        self.clip_target: float | None = self.cfg.get("clip_target")

        # Eval
        self.eubo_available = True
        self.eval_timesteps: Callable = instantiate(self.cfg.eval_timesteps)
        self.eval_ts = None
        self.eval_batch_size: int = self.cfg.eval_batch_size
        self.eval_integrator = EulerIntegrator()

    def setup_models(self, langevin_based=False, skip_prior=False):
        """Sets up the models based on the chosen density path

        Args:
        * langevin_based (bool): indicates whether the density path is taken from Annealed Langevin dynamics.
        If False, the density path is a linear noising diffusion path (default is False)
        * skip_prior (bool): indicates whether to skip the definition of the base distribution (default is False)
        """
        if not skip_prior:
            self.prior: Distribution = instantiate(self.cfg.prior)
        if langevin_based:
            self.sde: ControlledLangevinSDE = instantiate(
                self.cfg.get("sde"),
                prior_score=self.prior.score,
                target_score=self.target.score,
            )
        else:
            self.sde: OU = instantiate(self.cfg.get("sde"))
        self.generative_ctrl: Module = instantiate(
            self.cfg.generative_ctrl,
            sde=self.sde,
            prior_score=self.prior.score,
            target_score=self.target.score,
        )

        # EMA
        if self.use_ema:
            total_ema_updates = self.cfg.train_steps / (self.cfg.train_batch_size * self.cfg.ema_steps)
            alpha = 1.0 - self.cfg.ema_decay
            alpha = min(1.0, alpha / total_ema_updates)
            self.generative_ctrl_ema = torch.optim.swa_utils.AveragedModel(self.generative_ctrl,
                                                                           multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(
                                                                               1. - alpha),
                                                                           device=self.device)
        else:
            self.generative_ctrl_ema = self.generative_ctrl

    def clipped_target_unnorm_log_prob(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the target unnormalized log-likelihood at x with clipping values"""
        output = clip_and_log(
            self.target.unnorm_log_prob(x),
            max_norm=self.clip_target,
            name="target",
        )
        return output

    def _compute_loss(
            self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss starting from base samples x with time discretization ts

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1)
        * x (torch.Tensor of shape (batch_size, dim)
        """
        raise NotImplementedError

    def _compute_results(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            use_ema: bool = True,
            compute_weights: bool = True,
            return_traj: bool = True,
    ) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler starting from x

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        """
        raise NotImplementedError

    def compute_loss(self) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss over a batch"""
        x = self.prior.sample((self.train_batch_size,))
        if self.train_ts is None:
            self.train_ts = self.train_timesteps(device=x.device)
        else:
            self.train_ts = self.train_ts.to(x.device)
        ts = self.train_ts
        return self._compute_loss(ts, x)

    def compute_results(self, use_ema=True) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler"""
        # Sample trajectories
        x = self.prior.sample((self.eval_batch_size,))
        if self.eval_ts is None:
            self.eval_ts = self.eval_timesteps(device=x.device)
        else:
            self.eval_ts = self.eval_ts.to(x.device)
        ts = self.eval_ts

        results = self._compute_results(
            ts,
            x,
            use_ema=use_ema,
            compute_weights=True,
        )
        assert results.xs.shape == (len(ts), *results.samples.shape)

        # Sample w/o ito integral
        start_time = time.time()
        add_results = self._compute_results(
            ts,
            x,
            use_ema=use_ema,
            compute_weights=False,
            return_traj=False,
        )

        # Update results
        results.metrics["eval/sample_time"] = time.time() - start_time
        results.metrics.update(add_results.metrics)
        results.log_norm_const_preds.update(add_results.log_norm_const_preds)

        # Sample trajectories of inference proc
        if (
                self.plot_results
                and hasattr(self, "inference_sde")
                and hasattr(self.target, "sample")
        ):
            x_target = self.target.sample((self.eval_batch_size,))
            xs = self.eval_integrator.integrate(
                sde=self.inference_sde, ts=ts, x_init=x_target, timesteps=ts
            )
            plots = get_plots(
                distr=self.prior,
                samples=xs[-1],
                ts=ts,
                xs=xs,
                marginal_dims=self.eval_marginal_dims,
                domain=self.target.domain,
            )
            results.plots.update({f"{k}_inference": v for k, v in plots.items()})

        return results


class Bridge(TrainableDiff):
    """Class to model General Bridge Sampler (GBS) and DIS solvers.
    In the case of DIS, inference_ctrl is None."""
    save_attrs = TrainableDiff.save_attrs + ["inference_ctrl", "loss"]

    def setup_models(self):
        """Sets up the models"""
        super().setup_models()
        # inference_ctrl models the control term in the noising process
        self.inference_ctrl = self.cfg.get("inference_ctrl")
        self.inference_sde: OU = instantiate(
            self.cfg.sde,
        )
        if self.inference_ctrl is not None:
            self.inference_ctrl: Module = instantiate(
                self.cfg.inference_ctrl,
                sde=self.sde,
                prior_score=self.prior.score,
                target_score=self.target.score,
            )
            self.inference_sde = ControlledSDE(
                sde=self.inference_sde, ctrl=self.inference_ctrl
            )
        elif not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")

        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            generative_ctrl_ema=self.generative_ctrl_ema,
            sde=self.sde,
            inference_ctrl=self.inference_ctrl,
            filter_samples=getattr(self.target, "filter", None),
        )

    def _compute_loss(
        self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss starting from base samples x with time discretization ts

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1)
        * x (torch.Tensor of shape (batch_size, dim)
        """
        return self.loss(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            initial_log_prob=self.prior.log_prob,
        )

    def _compute_results(
        self,
        ts: torch.Tensor,
        x: torch.Tensor,
        use_ema: bool = True,
        compute_weights: bool = True,
        return_traj: bool = True,
    ) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler starting from x

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        """
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            use_ema=use_ema,
            initial_log_prob=self.prior.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class CMCD(TrainableDiff):
    """Class to model CMCD"""
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Enable EUBO
        self.eubo_available = True

    def setup_models(self, skip_prior=False):
        """Sets up the models"""
        super().setup_models(langevin_based=True, skip_prior=skip_prior)
        if not (isinstance(self.prior, Gauss) or isinstance(self.prior, GaussFull)):
            raise ValueError("Can only be used with gaussian prior.")
        self.inference_sde: ControlledLangevinSDE = instantiate(
            self.cfg.sde,
            prior_score=self.prior.score,
            target_score=self.target.score,
        )
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            generative_ctrl_ema=self.generative_ctrl_ema,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

    def update_prior(self, mean, var):
        """Updates the diagonal covariance Gaussian base distribution with parameters mean and var"""
        dim = mean.shape[0]
        if len(var.shape) == 2:
            self.prior = GaussFull(dim=dim, loc=mean, cov=var)
        else:
            self.prior = Gauss(dim=dim, loc=mean, scale=var.sqrt())
        self.setup_models(skip_prior=True)
        self.prior.to(self.device)
        self.sde.to(self.device)
        self.inference_sde.to(self.device)
        self.generative_ctrl.to(self.device)
        self.generative_ctrl_ema.to(self.device)

    def _compute_loss(
            self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss starting from base samples x with time discretization ts

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1)
        * x (torch.Tensor of shape (batch_size, dim)
        """
        return self.loss(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            initial_log_prob=self.prior.log_prob,
        )

    def _compute_results(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            use_ema: bool = True,
            compute_weights: bool = True,
            return_traj: bool = True,
    ) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler starting from x

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        """
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            use_ema=use_ema,
            initial_log_prob=self.prior.log_prob,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class PIS(TrainableDiff):
    """Class to model PIS"""
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable EUBO
        self.eubo_available = False

    def setup_models(self):
        """Sets up the models"""
        super().setup_models()
        if not isinstance(self.prior, Delta):
            raise ValueError("Can only be used with dirac delta prior.")
        self.reference_distr = self.sde.marginal_distr(
            t=self.sde.terminal_t.to(self.device), x_init=self.prior.loc.to(self.device)
        )
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            generative_ctrl_ema=self.generative_ctrl_ema,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

        # Inference SDE
        inference_sde: OU = instantiate(
            self.cfg.sde,
        )
        self.inference_sde = ControlledSDE(sde=inference_sde, ctrl=self.inference_ctrl)

    def inference_ctrl(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the inference control, i.e., the score of the reference Gaussian distribution, at (t,x)"""
        reference_score = self.sde.marginal_score(t=t, x=x, x_init=self.prior.loc)
        return self.sde.diff(t, x) * reference_score.clip(max=1e5)

    def _compute_loss(
            self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss starting from base samples x with time discretization ts

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1)
        * x (torch.Tensor of shape (batch_size, dim)
        """
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            use_ema: bool = True,
            compute_weights: bool = True,
            return_traj: bool = True,
    ) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler starting from x

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        """
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            use_ema=use_ema,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class DDS(TrainableDiff):
    """Class to model DDS"""
    # This implements the basic DDS algorithm
    # with the intended exponential integrator
    # https://arxiv.org/abs/2302.13834
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Disable EUBO
        self.eubo_available = False

    def setup_models(self):
        """Sets up the models"""
        super().setup_models()
        if not isinstance(self.prior, Gauss):
            raise ValueError("Can only be used with Gaussian prior.")

        # prior = reference_distr for terminal loss
        self.reference_distr = self.prior
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            generative_ctrl_ema=self.generative_ctrl_ema,
            sde=self.sde,
            filter_samples=getattr(self.target, "filter", None),
        )

    def _compute_loss(
            self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss starting from base samples x with time discretization ts

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1)
        * x (torch.Tensor of shape (batch_size, dim)
        """
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            use_ema: bool = True,
            compute_weights: bool = True,
            return_traj: bool = True,
    ) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler starting from x

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        """
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            use_ema=use_ema,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )


class RDS(TrainableDiff):
    """Class to model RDS"""
    save_attrs = TrainableDiff.save_attrs + ["loss"]

    def setup_models(self):
        """Sets up the models"""
        super().setup_models()
        self.inference_sde = instantiate(self.cfg.sde)
        self.change_reference_type(ref_type='default')
        self.loss: BaseOCLoss = instantiate(
            self.cfg.loss,
            generative_ctrl=self.generative_ctrl,
            generative_ctrl_ema=self.generative_ctrl_ema,
            sde=self.sde,
            reference_ctrl=self.reference_ctrl,
            filter_samples=getattr(self.target, "filter", None),
        )

    def change_reference_type(self, ref_type='default', net=None, eps=None, mean=None, var=None, means=None,
                              variances=None, weights=None):
        """Defines the reference distribution and the corresponding annealed unnormalized densities and scores

        Args:
        * ref_type (str): type of reference
            - 'default': reference from PIS or DDS obtained from the prior and sde parameters
            - 'gaussian': Gaussian distribution defined by
                * mean (torch.Tensor of shape (dim,))
                * var (torch.Tensor of shape (dim,) or torch.Tensor of shape (dim,dim)) : diagonal or full covariance
            - 'gmm': Gaussian mixture defined by
                * means (torch.Tensor of shape (n_modes, dim))
                * variances (torch.Tensor of shape (n_modes,dim) or torch.Tensor of shape (n_modes, dim,dim)):
                diagonal or full covariances
                * weights (torch.Tensor of shape (dim,)): mixture weights
            - 'net': Energy-based model defined by
                * net (Callable)
                * eps (torch.FloatTensor): time threshold to obtain the reference distribution (close to 0)
            """
        if ref_type == 'default':
            if isinstance(self.sde, VP):
                self.reference_distr_utils = {
                    'x_init': self.prior.loc.to(self.device).flatten(),
                    'var_init': torch.square(self.prior.scale.to(self.device)).flatten()
                }
            elif isinstance(self.sde, PinnedBM):
                self.reference_distr_utils = {
                    'x_init': self.prior.loc.to(self.device).flatten(),
                    'var_init': self.sde.terminal_t * self.sde.diff_coeff ** 2 * torch.ones_like(self.prior.loc).to(
                        self.device).flatten()
                }
            else:
                raise ValueError('Default reference for SDE type {} is not supported.'.format(
                    str(type(self.sde))
                ))
            self.reference_distr = self.sde.marginal_distr(
                t=torch.tensor(0.0).to(self.device), **self.reference_distr_utils)
            self.reference_score_t = lambda t, x: self.sde.marginal_score(t=t, x=x, **self.reference_distr_utils)
        elif ref_type == 'gaussian':
            if isinstance(var, tuple):
                var = tuple([a.to(self.device).float() for a in var])
            else:
                var = var.to(self.device).float()
            self.reference_distr_utils = {
                'x_init': mean.to(self.device).float(),
                'var_init': var
            }
            self.reference_distr = self.sde.marginal_distr(
                t=torch.tensor(0.0).to(self.device), **self.reference_distr_utils)
            self.reference_score_t = lambda t, x: self.sde.marginal_score(t=t, x=x, **self.reference_distr_utils)
        elif ref_type == 'gmm':
            if isinstance(variances, tuple):
                variances = tuple([a.to(self.device).float() for a in variances])
            else:
                variances = variances.to(self.device).float()
            self.reference_distr_utils = {
                'means_init': means.to(self.device).float(),
                'variances_init': variances,
                'weights_init': weights.to(self.device).float()
            }
            self.reference_distr = self.sde.marginal_gmm_distr(
                t=torch.tensor(0.0).to(self.device), **self.reference_distr_utils)
            self.reference_score_t = lambda t, x: self.sde.marginal_gmm_score(
                t=t, x=x, **self.reference_distr_utils)
        elif ref_type == 'nn':
            net_ = net.to(self.device)
            for j, p in enumerate(net_.parameters()):
                p.requires_grad_(False)
            self.reference_distr_utils = {
                'net': net_
            }
            self.reference_distr = WrapperDistrNN(dim=self.prior.loc.shape[-1], net=net_, t=eps.to(self.device))
            self.reference_score_t = lambda t, x: net_(t=t.view((1, 1)).expand((x.shape[0], -1)), x=x)
        else:
            raise NotImplementedError('Reference type {} is unknown.'.format(ref_type))
        self.ref_type = ref_type

    def reference_ctrl(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Evaluates the reference control at (t,x)"""
        return self.reference_score_t(t, x)

    def _compute_loss(
            self, ts: torch.Tensor, x: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """[TRAINING] Computes the variational loss starting from base samples x with time discretization ts

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1)
        * x (torch.Tensor of shape (batch_size, dim)
        """
        return self.loss(
            ts, x, self.clipped_target_unnorm_log_prob, self.reference_distr.log_prob
        )

    def _compute_results(
            self,
            ts: torch.Tensor,
            x: torch.Tensor,
            use_ema: bool = True,
            compute_weights: bool = True,
            return_traj: bool = True,
    ) -> Results:
        """ [TEST] Computes various metrics by simulating from the variational sampler starting from x

        Args:
        * ts (torch.Tensor of shape (n_steps+1, batch_size, 1): time discretization
        * x (torch.Tensor of shape (batch_size, dim): samples from the base distribution
        * use_ema (bool): indicates whether to use EMA in trajectory sampling (default is False)
        * compute_weights (bool): indicates whether to compute importance weights from the rnd (default is True)
        * return_traj (bool): indicates whether to return the full generative trajectory (default is True)
        """
        return self.loss.eval(
            ts,
            x,
            self.clipped_target_unnorm_log_prob,
            self.reference_distr.log_prob,
            use_ema=use_ema,
            compute_weights=compute_weights,
            return_traj=return_traj,
        )

    def state_dict(self, *args, **kwargs):
        """Saves the reference parameters into a dictionary"""
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({'ref_{}'.format(k): v for k, v in self.reference_distr_utils.items()})
        state_dict['ref_type'] = self.ref_type
        return state_dict

    def load_state_dict(self, state_dict, *args, **kwargs):
        """Loads the reference parameters from state_dict:"""
        super().load_state_dict(state_dict, *args, **kwargs)
        if 'ref_type' in state_dict:
            self.ref_type = state_dict['ref_type']
            if self.ref_type == 'default':
                pass
            elif self.ref_type == 'gaussian':
                self.change_reference_type(
                    ref_type='gaussian',
                    mean=state_dict['ref_x_init'],
                    var=state_dict['ref_var_init']
                )
            elif self.ref_type == 'gmm':
                self.change_reference_type(
                    ref_type='gmm',
                    weights=state_dict['ref_weights_init'],
                    means=state_dict['ref_means_init'],
                    variances=state_dict['ref_variances_init']
                )
            elif self.ref_type == 'nn':
                self.change_reference_type(
                    ref_type='nn',
                    net=state_dict['ref_net'],
                    eps=self.train_timesteps()[0]
                )
