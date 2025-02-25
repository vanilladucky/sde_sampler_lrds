# Implement Diffusion Assisted EBM

# Libraries
import torch
from tqdm import trange
from .ebm_mle import MaximumLikelihoodEBM
from .mcmc import mala_step, heuristics_step_size
from .hacking import list_of_dict_2_dict_of_list

# Perform Langevin-within-Gibbs sampling


def sample_langevin_gibbs_mcmc(k_init, x_init, times, log_probs, log_prob_and_grads, n_warmup_mcmc_steps, n_mcmc_steps, n_local_steps,
                               step_sizes_per_noise, target_acceptance=0.75, store_weights=False, verbose=True):
    # Build the storage
    n_noise_levels = times.shape[0]
    batch_size = x_init.shape[0]
    data_shape = x_init.shape[1:]
    data_shape_ones = (1,) * (len(x_init.shape)-1)
    ks = torch.empty((n_mcmc_steps, batch_size), device=x_init.device, dtype=int)
    ts = torch.empty((n_mcmc_steps, batch_size, *data_shape_ones), device=x_init.device)
    xs = torch.empty((n_mcmc_steps, *x_init.shape), device=x_init.device)
    diagnostics = []
    # Evaluate all the times for each points

    def log_probs_all(y): return log_probs(
        times.unsqueeze(1).expand((-1, batch_size)).reshape((-1, *data_shape_ones)),
        y.unsqueeze(0).expand((n_noise_levels, *(-1,) * len(y.shape))).reshape((-1, *data_shape))
    )
    # Initialize t and x
    k, t, x = k_init.clone(), times[k_init].clone().view((batch_size, *data_shape_ones)), x_init.clone()
    log_prob_x, grad_x = log_prob_and_grads(t, x)
    # Run the sampling
    if verbose:
        r = trange(n_warmup_mcmc_steps+n_mcmc_steps)
    else:
        r = range(n_warmup_mcmc_steps+n_mcmc_steps)
    for step_id in r:
        # Make an empty diag
        diag = {}
        # Compute at all the noise levels
        all_log_prob_x = log_probs_all(x)
        # Reshape everything
        all_log_prob_x = all_log_prob_x.view((n_noise_levels, batch_size))
        # Compute the weights
        with torch.no_grad():
            weights = torch.nn.functional.softmax(all_log_prob_x, dim=0)
        if store_weights:
            diag['weights'] = weights.clone().cpu()
        # Sample the new times
        k = torch.multinomial(weights.T, 1)
        t = times[k].view((batch_size, *data_shape_ones))
        # Make the local log_prob_and_grad
        def cur_log_prob_and_grad(y): return log_prob_and_grads(t, y)
        log_prob_x, grad_x = cur_log_prob_and_grad(x)
        # Do local MCMC steps
        cur_step_size = step_sizes_per_noise[k].view((-1, *data_shape_ones)).clone()
        for _ in range(n_local_steps):
            x, log_prob_x, grad_x, log_acc = mala_step(x, log_prob_x, grad_x, cur_log_prob_and_grad, cur_step_size)
            cur_step_size = heuristics_step_size(cur_step_size, log_acc, target_acceptance=target_acceptance)
        # Store the step size and compute acceptance levels
        acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
        accs = torch.zeros((n_noise_levels,))
        for k_ in range(times.shape[0]):
            mask = k == k_
            if mask.sum() > 0:
                step_sizes_per_noise[k_] = cur_step_size[mask].mean()
                if step_id >= n_warmup_mcmc_steps:
                    accs[k_] = acc[mask.cpu().flatten()].mean()
        diag['local_acc'] = accs
        # Store everything
        if step_id >= n_warmup_mcmc_steps:
            ks[step_id-n_warmup_mcmc_steps] = k.flatten().clone()
            ts[step_id-n_warmup_mcmc_steps] = t.clone()
            xs[step_id-n_warmup_mcmc_steps] = x.clone()
        # Append the diagnostic
        if step_id >= n_warmup_mcmc_steps:
            diagnostics.append(diag)
        # Display the logs
        if verbose:
            r.set_postfix(local_acc=acc.mean().cpu().item())
    # Pack the diagnostics together
    diagnostics = list_of_dict_2_dict_of_list(diagnostics)
    diagnostics = {metric_name: torch.stack(v) for metric_name, v in diagnostics.items()}
    # Return everything
    return ks, ts, xs, step_sizes_per_noise, diagnostics


class DAEBM(MaximumLikelihoodEBM):

    def __init__(self, sde, prior, net, step_size=1e-3, target_acceptance=0.75, perc_keep_mcmc=-1.0,
                 persistent_size=8192, store_weights=False, use_snr_adapted_disc=False, start_eps=1e-3, end_eps=0.0, n_steps=100):
        # Call the parent constructor
        super(MaximumLikelihoodEBM, self).__init__()
        # Store SDE parameters
        self.sde = sde.cpu()
        self.prior = prior
        self.prior.cpu()
        self.net = net.cpu()
        # Build a persistent buffer
        self.persistent_size = persistent_size
        self.register_buffer('persistent_k', (n_steps-1) * torch.ones((persistent_size,), dtype=int))
        self.register_buffer('persistent_x', self.prior.sample((persistent_size,)))
        # Sampling details
        self.step_size = step_size
        # Parameters for training
        self.store_weights = store_weights
        self.target_acceptance = target_acceptance
        self.use_snr_adapted_disc = use_snr_adapted_disc
        self.perc_keep_mcmc = perc_keep_mcmc
        self.keep_some_mcmc = perc_keep_mcmc > 0.0
        # Build the time discreitzation
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.n_steps = n_steps
        self.build_time_disc()

    def log_probs(self, t, y):
        return self.net.unnorm_log_prob(t, y)

    def train(self, device, data, batch_size, n_epochs, lr=3e-4, initial_n_warmup_mcmc_steps=1024, n_warmup_mcmc_steps=0,
              n_mcmc_steps=32, n_local_steps=8, n_accumulation_steps=1, verbose=True):
        # Build an optimizer
        optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Build a dataset
        dataset = torch.utils.data.TensorDataset(data.to(device))
        n_mcmc_steps_kept = int(self.perc_keep_mcmc * n_mcmc_steps)
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        n_batches = len(train_loader)
        # Get the initial step sizes
        self.step_sizes_per_noise = self.step_size * torch.ones((self.times.shape[0], 1), device=device)
        # Run the optimization
        if verbose:
            r = trange(n_epochs)
        else:
            r = range(n_epochs)
        losses = []
        diagnostics = []
        for epoch_id in r:
            for batch_id, batch in enumerate(train_loader):
                # Get the data
                is_very_first_batch = (epoch_id == 0) and (batch_id == 0)
                data = batch[0]
                data_shape = data.shape[1:]
                data_shape_ones = (1,) * len(data_shape)
                # Get the positive samples
                ks_pos = torch.randint(0, self.times.shape[0], (batch_size,), device=device)
                ts_pos = self.times[ks_pos].view((-1, *data_shape_ones))
                xs_pos = self.sde.s(ts_pos) * data
                xs_pos += self.sde.s(ts_pos) * self.sde.sigma_sq(ts_pos).sqrt() * torch.randn_like(xs_pos)
                # Only perform negative sampling on some batches
                if (batch_id % n_accumulation_steps == 0):
                    # Disable gradient with respect to the network
                    parameters_states = []
                    for p in self.net.parameters():
                        parameters_states.append(p.requires_grad)
                        p.requires_grad_(False)
                    # Get samples from the persistent state
                    idx = torch.randperm(self.persistent_size)[:batch_size]
                    ks_neg = self.persistent_k[idx]
                    xs_neg = self.persistent_x[idx]
                    # Run the Langevin-within-Gibbs sampler
                    ks_neg, ts_neg, xs_neg, self.step_sizes_per_noise, diags = sample_langevin_gibbs_mcmc(
                        k_init=ks_neg,
                        x_init=xs_neg,
                        times=self.times.flatten(),
                        log_probs=self.log_probs,
                        log_prob_and_grads=self.log_prob_and_grads,
                        n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps if is_very_first_batch else n_warmup_mcmc_steps,
                        n_mcmc_steps=n_mcmc_steps,
                        n_local_steps=n_local_steps,
                        step_sizes_per_noise=self.step_sizes_per_noise,
                        target_acceptance=self.target_acceptance,
                        store_weights=self.store_weights,
                        verbose=False
                    )
                    diagnostics.append(diags)
                    # Put back into the persistent state
                    self.persistent_k[idx] = ks_neg[-1].clone()
                    self.persistent_x[idx] = xs_neg[-1].clone()
                    # Restore gradients with respect to the network
                    for i, p in enumerate(self.net.parameters()):
                        p.requires_grad_(parameters_states[i])
                    # Select the MCMC samples
                    if self.keep_some_mcmc:
                        ts_neg = ts_neg[-n_mcmc_steps_kept:]
                        xs_neg = xs_neg[-n_mcmc_steps_kept:]
                    else:
                        ts_neg = ts_neg[-1]
                        xs_neg = xs_neg[-1]
                    ts_neg = ts_neg.reshape((-1, *data_shape_ones))
                    xs_neg = xs_neg.reshape((-1, *data_shape))
                # Evaluate the energy
                en_pos = self.net.energy(ts_pos, xs_pos)
                if (batch_id % n_accumulation_steps == 0):
                    en_neg = self.net.energy(ts_neg, xs_neg)
                # Compute the loss
                loss = en_pos.mean() - en_neg.mean()
                if batch_id >= n_batches - (n_batches % n_accumulation_steps):
                    loss /= (n_batches % n_accumulation_steps)
                else:
                    loss /= n_accumulation_steps
                # Display the loss on progress bar
                if verbose:
                    r.set_postfix(
                        loss=float(loss.cpu().item()),
                        local_acc=diags['local_acc'].mean().item()
                    )
                # Store the loss
                losses.append(loss.item())
                # Call the backward on loss
                loss.backward(retain_graph=n_accumulation_steps > 1)
                if ((batch_id + 1) % n_accumulation_steps == 0) or (batch_id + 1 == n_batches):
                    # Perform a gradient step
                    optimizer.step()
                    # Reset the gradients
                    optimizer.zero_grad()
        return torch.FloatTensor(losses), diagnostics
