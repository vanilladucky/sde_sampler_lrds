# Implementation of Diffusion Recovery Likelihood

# Libraries
import math
import torch
from tqdm import trange
from .mcmc import mala_step
from sde_sampler.utils.common import get_timesteps


def heuristics_step_size(stepsize, mean_log_acceptance, target_acceptance=0.75, factor=1.01, tol=0.05):
    """Heuristic for adaptative step size"""
    if mean_log_acceptance - math.log(target_acceptance) > math.log1p(tol):
        return stepsize * factor
    if math.log(target_acceptance) - mean_log_acceptance > -math.log1p(-tol):
        return stepsize / factor
    return stepsize


class DiffusionRecoveryLikelihood(torch.nn.Module):

    def __init__(self, sde, prior, net, b=2e-2, use_b_adaptation=False, target_acceptance=0.75, use_snr_adapted_disc=False,
                 use_gao_weighting=True, use_bar_weighting=False, use_weighting_on_reg=False, use_var_reduction=False,
                 perc_keep_mcmc=-1.0, start_eps=1e-3, end_eps=0.0, n_steps=100):
        # Call the parent constructor
        super().__init__()
        # Store SDE parameters
        self.sde = sde.cpu()
        self.prior = prior.cpu()
        self.net = net.cpu()
        # Parameters for training
        self.use_b_adaptation = use_b_adaptation
        self.target_acceptance = target_acceptance
        self.use_snr_adapted_disc = use_snr_adapted_disc
        self.use_gao_weighting = use_gao_weighting
        self.use_bar_weighting = use_bar_weighting
        self.use_weighting_on_reg = use_weighting_on_reg
        self.use_var_reduction = use_var_reduction
        self.perc_keep_mcmc = perc_keep_mcmc
        self.keep_some_mcmc = perc_keep_mcmc > 0.0
        # Save the factor b
        self.b = b
        # Build the time discreitzation
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.n_steps = n_steps
        self.build_time_disc()

    def net_energy(self, k, x):
        return self.net.energy(self.times[k], x, scaling_factor=self.alphas[k])

    def build_time_disc(self):
        # Make the intermediate times
        if self.use_snr_adapted_disc:
            self.register_buffer('times', get_timesteps(start=self.start_eps, end=self.sde.terminal_t-self.end_eps,
                                                        steps=self.n_steps, sde=self.sde).unsqueeze(-1))
        else:
            self.register_buffer('times', get_timesteps(start=self.start_eps, end=self.sde.terminal_t-self.end_eps,
                                                        steps=self.n_steps).unsqueeze(-1))
        # Make the alphas and sigmas
        alphas, sigmas_sq = self.sde.transition_params(self.times[:-1], self.times[1:])
        self.register_buffer('alphas', alphas)
        self.register_buffer('sigmas_sq', sigmas_sq)
        # Make the alphas_bar and sigmas_bar
        s = self.sde.s(self.times)
        self.register_buffer('alphas_bar', s)
        self.register_buffer('sigmas_sq_bar', torch.square(s) * self.sde.sigma_sq(self.times))
        # Build the step sizes
        self.register_buffer('step_size', 0.5 * self.b *
                             torch.sqrt(self.sigmas_sq_bar[:-1] / self.sigmas_sq_bar[0]) * self.sigmas_sq)

    def conditional_log_prob(self, k, y_k, x_k_p_1):
        en = self.net_energy(k, y_k)
        return -en - 0.5 * torch.sum(torch.square(x_k_p_1 - y_k) / self.sigmas_sq[k], dim=-1)

    def conditional_log_prob_and_grad(self, k, y_k, x_k_p_1):
        if hasattr(self.net, 'unnorm_log_prob_and_grad'):
            log_prob_net, grad_net = self.net.unnorm_log_prob_and_grad(
                self.times[k], y_k, scaling_factor=self.alphas[k])
            log_prob = log_prob_net - 0.5 * torch.sum(torch.square(x_k_p_1 - y_k) / self.sigmas_sq[k], dim=-1)
            grad = grad_net + ((x_k_p_1 - y_k) / self.sigmas_sq[k])
        else:
            log_prob = self.conditional_log_prob(k, y_k, x_k_p_1)
            grad = torch.autograd.grad(log_prob.sum(), y_k)[0].detach()
        return log_prob, grad

    @torch.no_grad()
    def sample_noise_process_pairs(self, k, x_data):
        z = torch.randn_like(x_data)
        x_k = self.alphas_bar[k] * x_data + torch.sqrt(self.sigmas_sq_bar[k]) * z
        if self.use_var_reduction:
            x_k_p_1 = self.alphas_bar[k+1] * x_data + torch.sqrt(self.sigmas_sq_bar[k+1]) * z
        else:
            x_k_p_1 = self.alphas[k] * x_k + torch.sqrt(self.sigmas_sq[k]) * torch.randn_like(x_k)
        return x_k, x_k_p_1

    def conditional_sample(self, k, x_k_p_1, n_mcmc_steps, return_intermediates=False):
        # Disable gradient with respect to the network
        parameters_states = []
        for p in self.net.parameters():
            parameters_states.append(p.requires_grad)
            p.requires_grad_(False)
        # Make the log-prob and scores
        def log_prob_and_grad(y_k): return self.conditional_log_prob_and_grad(k, y_k, x_k_p_1)
        # Initialize the chain
        y = torch.autograd.Variable(x_k_p_1.clone(), requires_grad=True)
        log_prob_y, grad_y = log_prob_and_grad(y)
        # Run the MCMC algorithm
        if return_intermediates:
            ys = [y.detach().clone()]
        accs = []
        for i in range(n_mcmc_steps):
            y, log_prob_y, grad_y, log_acc = mala_step(y, log_prob_y, grad_y, log_prob_and_grad, self.step_size[k])
            acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
            accs.append(acc)
            if return_intermediates:
                ys.append(y.detach().clone())
            if self.use_b_adaptation:
                self.step_size = heuristics_step_size(self.step_size, float(log_acc.logsumexp(dim=0) - math.log(log_acc.shape[0])),
                                                      target_acceptance=self.target_acceptance)
        # Restore gradients with respect to the network
        for i, p in enumerate(self.net.parameters()):
            p.requires_grad_(parameters_states[i])
        # Return the final MCMC sample
        if return_intermediates:
            return torch.stack(ys), torch.stack(accs, dim=0)
        else:
            return y.detach(), torch.stack(accs, dim=0)

    def train(self, device, data, batch_size, n_epochs, lr=3e-4, reg_val=5e-3, n_mcmc_steps=30, verbose=True):
        # Build an optimizer
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Build a dataset
        dataset = torch.utils.data.TensorDataset(data.to(device))
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        # Run the optimization
        if verbose:
            r = trange(n_epochs)
        else:
            r = range(n_epochs)
        losses = []
        acceptances = []
        for _ in r:
            for batch in train_loader:
                # Get the data
                data = batch[0]
                # Sample the noise levels
                ks = torch.randint(0, self.n_steps-1, (data.shape[0],), device=data.device)
                # Make the noisy versions
                x_k_pos, x_k_p_1_pos = self.sample_noise_process_pairs(ks, data)
                # Get the negative samples
                if self.keep_some_mcmc:
                    ys_k_neg, accs = self.conditional_sample(ks, x_k_p_1_pos, n_mcmc_steps=n_mcmc_steps,
                                                             return_intermediates=True)
                    mcmc_length_kept = int(self.perc_keep_mcmc * ys_k_neg.shape[0])
                    ys_k_neg = ys_k_neg[-mcmc_length_kept:]
                    ks_neg = ks.clone().unsqueeze(0).expand((mcmc_length_kept, -1)).flatten()
                    y_k_neg = ys_k_neg.view((-1, ys_k_neg.shape[-1]))
                    x_k_neg = y_k_neg / self.alphas[ks_neg]
                else:
                    y_k_neg, accs = self.conditional_sample(ks, x_k_p_1_pos, n_mcmc_steps=n_mcmc_steps)
                    x_k_neg = y_k_neg / self.alphas[ks]
                    ks_neg = ks.clone()
                acceptances.append(torch.FloatTensor([accs[:, ks == j].mean() for j in range(self.n_steps-1)]))
                # Reset the gradients
                optimizer.zero_grad()
                # Evaluate the energy
                energy_neg = self.net_energy(ks_neg, x_k_neg * self.alphas[ks_neg])
                energy_pos = self.net_energy(ks, x_k_pos * self.alphas[ks])
                # Compute the loss
                if self.use_gao_weighting:
                    weight = torch.sqrt(self.sigmas_sq[0] / self.sigmas_sq[ks]).flatten()
                elif self.use_bar_weighting:
                    weight = torch.sqrt(self.sigmas_sq_bar[0] / self.sigmas_sq_bar[ks]).flatten()
                else:
                    weight = torch.ones_like(ks)
                if self.keep_some_mcmc:
                    weight_neg = weight.clone().unsqueeze(0).expand((mcmc_length_kept, -1, -1))
                    weight_neg = weight_neg.flatten()
                else:
                    weight_neg = weight.clone()
                loss = torch.mean(weight * energy_pos) - torch.mean(weight_neg * energy_neg)
                if self.use_weighting_on_reg:
                    loss += reg_val * (torch.mean(weight * torch.square(energy_pos)) +
                                       torch.mean(weight_neg * torch.square(energy_neg)))
                else:
                    loss += reg_val * (torch.mean(torch.square(energy_pos)) + torch.mean(torch.square(energy_neg)))
                if verbose:
                    r.set_postfix(loss='{:.2e}'.format(loss.item()), acc='{:.1f}%'.format(100. * accs.mean()))
                losses.append(loss.item())
                # Call the backward on loss
                loss.backward()
                # Perform a gradient step
                optimizer.step()
        return losses, acceptances
