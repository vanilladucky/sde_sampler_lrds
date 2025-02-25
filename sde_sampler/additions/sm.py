# Libraries
import torch
from tqdm import tqdm, trange
from sde_sampler.utils.common import get_timesteps


class ScoreMatching(torch.nn.Module):
    """Implements standard score matching"""

    def __init__(self, sde, prior, score_net, t_start, t_end, n_steps=100, antithetic=True, time_type='uniform'):
        super().__init__()
        self.sde = sde
        self.prior = prior
        self.score_net = score_net
        self.score_net_ema = self.score_net
        self.antithetic = antithetic
        self.t_start = t_start
        self.t_end = t_end
        self.n_steps = n_steps
        self.time_type = time_type
        # Build times
        if self.time_type == 'snr_adapted_multinomial':
            self.register_buffer('times', get_timesteps(start=self.t_start, end=self.t_end, steps=self.n_steps,
                                                        sde=self.sde).unsqueeze(-1))
        elif self.time_type == 'uniform_multinomial':
            self.register_buffer('times', get_timesteps(start=self.t_start, end=self.t_end,
                                                        steps=self.n_steps).unsqueeze(-1))
        if 'multinomial' in self.time_type:
            self.register_buffer('time_weights', torch.ones((self.n_steps,)))

    def sample_time(self, batch_size, device):
        if self.time_type == 'multinomial':
            return self.times[torch.multinomial(self.time_weights, batch_size, replacement=True)].to(device)
        else:
            return (self.t_end - self.t_start) * torch.rand((batch_size, 1), device=device) + self.t_start

    def compute_loss(self, ts, xs):
        # Noise xs
        loc, var = self.sde.marginal_params(ts, xs)
        zs = torch.randn_like(xs)
        ys = loc + torch.sqrt(var) * zs
        # Compute the MSE
        # NOTE: This corresponds to the orignal loss from ArXiv:2011.13456
        # with sigma^2_t weighting.
        loss = torch.mean(torch.square(torch.sqrt(var) * self.score_net(ts, ys) + zs), dim=-1)
        # Apply the antithetic trick
        if self.antithetic:
            ys_antithetic = loc - torch.sqrt(var) * zs
            loss += torch.mean(torch.square(torch.sqrt(var) * self.score_net(ts, ys_antithetic) - zs), dim=-1)
            loss /= 2.0
        # Return the loss
        return loss.mean()

    def sample(self, n_samples, device, n_steps=None, return_losses=False, keep_intermediates=False, use_ddpm_kernel=False, verbose=True):
        # Move the score to the right device
        self.score_net_ema = self.score_net_ema.to(device)
        self.score_net_ema.eval()
        # Build the timesteps
        T = self.sde.terminal_t
        if 'multinomial' in self.time_type:
            ts = self.times.clone().to(device)
        else:
            ts = get_timesteps(
                start=self.t_start,
                end=self.t_end,
                steps=n_steps if n_steps is not None else self.n_steps,
                sde=self.sde if 'snr' in self.time_type else None).to(device)
            ts = ts.unsqueeze(-1).to(device)
        # Sample the initial sample
        x = self.prior.sample((n_samples,)).to(device)
        if return_losses:
            t_ones = torch.ones((x.shape[0], 1), device=device)
        # Solve the SDE with EI
        if keep_intermediates:
            traj = [x]
        losses = []
        with torch.no_grad():
            r = list(zip(ts[:-1], ts[1:]))
            if verbose:
                r = tqdm(r)
            for s, t in r:
                if return_losses:
                    loss = self.compute_loss((T - s) * t_ones, x)
                    losses.append(loss.item())
                if use_ddpm_kernel:
                    x, _ = self.sde.ddpm_integration_step(x, s, t, self.score_net_ema(T - s, x))
                else:
                    x, _ = self.sde.ei_integration_step(x, s, t, self.score_net_ema(T - s, x))
                if keep_intermediates:
                    traj.append(x)
        if return_losses:
            if keep_intermediates:
                return traj, losses
            else:
                return x, losses
        else:
            if keep_intermediates:
                return traj
            else:
                return x

    def train(self, device, data, batch_size, n_epochs, lr=3e-4, use_ema=True, ema_decay=0.995, ema_steps=10, verbose=True):
        # Move score net to the device
        self.score_net = self.score_net.to(device)
        self.score_net.train()
        # Build an optimizer
        optimizer = torch.optim.Adam(self.score_net.parameters(), lr=lr)
        # Build the EMA
        if use_ema:
            adjust = batch_size * ema_steps / n_epochs
            alpha = 1.0 - ema_decay
            alpha = min(1.0, alpha * adjust)
            self.score_net_ema = torch.optim.swa_utils.AveragedModel(self.score_net,
                                                                     multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(1.0 - alpha))
        # Build a dataset
        dataset = torch.utils.data.TensorDataset(data.to(device))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
        # Run the optimization
        if verbose:
            r = trange(n_epochs)
        else:
            r = range(n_epochs)
        losses = []
        step_id = 0
        for _ in r:
            for batch in train_loader:
                xs = batch[0]
                ts = self.sample_time(xs.shape[0], device)
                optimizer.zero_grad()
                loss = self.compute_loss(ts, xs)
                loss.backward()
                optimizer.step()
                if use_ema and (step_id % ema_steps == 0):
                    self.score_net_ema.update_parameters(self.score_net)
                losses.append(loss.item())
                if verbose:
                    r.set_postfix(loss=losses[-1])
                step_id += 1
        return losses


class TargetScoreMatching(ScoreMatching):
    """Implements target score matching"""

    def __init__(self, target_score, **kwargs):
        super().__init__(**kwargs)
        self.target_score = target_score

    def compute_loss(self, ts, xs):
        # Noise xs
        s, sigma_sq = self.sde.s(ts), self.sde.sigma_sq(ts)
        zs = torch.randn_like(xs)
        ys = s * xs + s * torch.sqrt(sigma_sq) * zs
        # Compute the loss
        # Note: This isn't the original weighting
        loss = torch.mean(torch.square(s * self.score_net(ts, ys) - self.target_score(xs)), dim=-1)
        # Apply the antithetic trick
        if self.antithetic:
            ys_antithetic = s * xs - s * torch.sqrt(sigma_sq) * zs
            loss += torch.mean(torch.square(s * self.score_net(ts, ys_antithetic) - self.target_score(xs)), dim=-1)
            loss /= 2.0
        # Return the loss
        return loss.mean()


class PerfectScoreMatching(ScoreMatching):
    """Implements perfect score matching"""

    def __init__(self, perfect_score, **kwargs):
        super().__init__(**kwargs)
        self.perfect_score = perfect_score

    def compute_loss(self, ts, xs):
        # Noise xs
        s, sigma_sq = self.sde.s(ts), self.sde.sigma_sq(ts)
        zs = torch.randn_like(xs)
        ys = s * xs + s * torch.sqrt(sigma_sq) * zs
        # Compute the loss
        loss = torch.mean(torch.square(self.score_net(ts, ys) - self.perfect_score(ts, ys)), dim=-1)
        # Apply the antithetic trick
        if self.antithetic:
            ys_antithetic = s * xs - s * torch.sqrt(sigma_sq) * zs
            loss += torch.mean(torch.square(self.score_net(ts, ys_antithetic) -
                                            self.perfect_score(ts, ys_antithetic)), dim=-1)
            loss /= 2.0
        # Return the loss
        return loss.mean()
