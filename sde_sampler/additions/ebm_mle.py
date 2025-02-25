# Implement direct maximum likelihood training of annealed EBMs

# Libraries
import torch
import pickle
from tqdm import trange
from .mcmc import mala_step, precond_mala_step, ula_step, precond_ula_step, heuristics_step_size
from ..utils.common import get_timesteps


def smc_sampler(x_init, times, log_prob_and_grads, n_warmup_mcmc_steps, n_mcmc_steps, step_sizes_per_noise, per_noise_init=False,
                reweight_threshold=1.0, use_pdds_weights=False, sde=None, target_acceptance=0.75, precond_matrix_per_noise=None,
                precond_matrix_chol_per_noise=None, use_ula=False, verbose=True):
    """SMC sampler. 

    It implements Annealed Langevin, Sequential Monte Carlo and Particle Denoising Diffusion Sampler.

    Args:
        * x_init (torch.Tensor of shape (n_levels, batch_size, *data_shape) if per_noise_init and (batch_size, *data_shape) otherwise): Initial samples
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (batch_size, *data_shape_ones) and x of shape (batch_size, *data_shape)): Intermediate distributions
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_size, *data_shape_ones)): Step size for each noise level
        * per_noise_init (bool): Whether x_init contains per noise initialization (default is False)
        * reweight_threshold (float): ESS threshold for reweighting (default is 1.0)
        * use_pdds_weights (bool): Transition and reweight with PDDS weights (default is False)
        * sde (sde object): SDE for PDDS (only used if use_pdds_weights is True)
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * precond_matrix_per_noise (torch.Tensor of shape (n_levels, *data_shape, *data_shape)): Preconditioning matrices
        * precond_matrix_chol_per_noise (torch.Tensor of shape (n_levels, *data_shape, *data_shape)): Cholesky decomposition of the preconditioning matrices
        * use_ula (bool): Skip the Metropolis-Hastings check in local steps (default is False)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, batch_size, *data_shape)): Samples at each noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_size, *data_shape_ones)): Updated step size for each noise level
        * diags (dict): Dictionnary of diagnostics
    """
    # Assertions
    if per_noise_init and (reweight_threshold > 0.0):
        raise ValueError("Can't use per_noise_init in SMC mode.")
    if (sde is None) and use_pdds_weights:
        raise ValueError("Can't use PDDS weights without the SDE object.")
    # Parse the initial point shape
    if per_noise_init:
        batch_size = x_init.shape[1]
        data_shape = x_init.shape[2:]
    else:
        batch_size = x_init.shape[0]
        data_shape = x_init.shape[1:]
    # Check if the MALA steps should be pre-conditionned
    use_precond = (precond_matrix_per_noise is not None) and (precond_matrix_chol_per_noise is not None)
    # Initialize the storage for the weights, gradiens and points
    if reweight_threshold > 0.0:
        log_weights = torch.zeros((batch_size,), device=x_init.device)
        log_prob_x_prev = torch.empty((batch_size,), device=x_init.device)
        x_prev = torch.empty_like(x_init)
        grad_x_prev = torch.empty_like(x_init)
        ess_logs = torch.ones((times.shape[0],))
    # Initialize the sample buffer
    samples = torch.empty((times.shape[0], n_mcmc_steps, batch_size, *data_shape), device=x_init.device)
    if not use_ula:
        mean_accs = torch.empty((times.shape[0],))
    x = x_init.clone()
    if verbose:
        r = trange(times.shape[0]-1, -1, -1)
    else:
        r = range(times.shape[0]-1, -1, -1)
    for time_id in r:
        # Set the first point
        if per_noise_init:
            x = x_init[time_id].clone()
        else:
            x = x.clone()
        # Define the current log-prob and score

        def cur_log_prob_and_grad(y):
            return log_prob_and_grads(times[time_id], y)
        # Select the current step size and pre-conditioning
        cur_step_size = step_sizes_per_noise[time_id]
        if use_precond:
            cur_precond_matrix = precond_matrix_per_noise[time_id]
            cur_precond_matrix_col = precond_matrix_chol_per_noise[time_id]
        # Compute the inital values
        log_prob_x, grad_x = cur_log_prob_and_grad(x)
        if use_precond:
            precond_grad_x = torch.matmul(cur_precond_matrix, grad_x.unsqueeze(-1)).squeeze(-1)
        # Move the particles with the SDE reverse kernel
        if use_pdds_weights and (time_id != times.shape[0]-1):
            # Perform the transition
            x, z = sde.ei_integration_step(x_prev, sde.terminal_t - times[time_id+1],
                                           sde.terminal_t - times[time_id], grad_x_prev)
            # Compute the transition probability
            log_prob_transition_backward = -0.5 * torch.sum(torch.square(z), dim=-1)
            mean_factor_forward, var_factor_forward = sde.transition_params(times[time_id], times[time_id+1])
            log_prob_transition_forward = -0.5 * \
                torch.sum(torch.square(mean_factor_forward * x - x_prev) / var_factor_forward, dim=-1)
            # Update the log_prob_x and grad_x
            log_prob_x, grad_x = cur_log_prob_and_grad(x)
            if use_precond:
                precond_grad_x = torch.matmul(cur_precond_matrix, grad_x.unsqueeze(-1)).squeeze(-1)
        # Compute the weights
        if (reweight_threshold > 0.0) and (time_id != times.shape[0]-1):
            # Compute the importance weights
            if use_pdds_weights:
                log_weights = log_prob_x - log_prob_x_prev
                log_weights += log_prob_transition_forward - log_prob_transition_backward
            else:
                log_weights += log_prob_x - log_prob_x_prev
            weights = torch.nn.functional.softmax(log_weights, dim=0)
            # Compute the ESS
            ess = (1.0 / torch.sum(torch.square(weights))) / batch_size
            ess_logs[time_id] = ess.cpu().clone()
            # Ressample the particles
            if ess < reweight_threshold:
                idx = torch.multinomial(weights, batch_size, replacement=True)
                x = x[idx]
                log_prob_x = log_prob_x[idx]
                grad_x = grad_x[idx]
                if use_precond:
                    precond_grad_x = precond_grad_x[idx]
                log_weights.zero_()
        # Warmup the MALA sampler
        for step_id in range(n_warmup_mcmc_steps):
            if use_ula:
                # Perform the ULA step
                if use_precond:
                    x, log_prob_x, grad_x, precond_grad_x = precond_ula_step(x, log_prob_x, grad_x, precond_grad_x,
                                                                             cur_log_prob_and_grad, cur_step_size, cur_precond_matrix, cur_precond_matrix_col)
                else:
                    x, log_prob_x, grad_x = ula_step(x, log_prob_x, grad_x, cur_log_prob_and_grad, cur_step_size)

            else:
                # Perform the MALA step
                if use_precond:
                    x, log_prob_x, grad_x, precond_grad_x, log_acc = precond_mala_step(x, log_prob_x, grad_x, precond_grad_x,
                                                                                       cur_log_prob_and_grad, cur_step_size, cur_precond_matrix, cur_precond_matrix_col)
                else:
                    x, log_prob_x, grad_x, log_acc = mala_step(
                        x, log_prob_x, grad_x, cur_log_prob_and_grad, cur_step_size)
                if target_acceptance > 0.0:
                    cur_step_size = heuristics_step_size(cur_step_size, log_acc, target_acceptance=target_acceptance)
        # Run the MALA sampler
        if not use_ula:
            sum_acc = 0.0
        for step_id in range(n_mcmc_steps):
            # Perform the MCMC step
            if use_ula:
                # Perform the ULA step
                if use_precond:
                    x, log_prob_x, grad_x, precond_grad_x = precond_ula_step(x, log_prob_x, grad_x, precond_grad_x,
                                                                             cur_log_prob_and_grad, cur_step_size, cur_precond_matrix, cur_precond_matrix_col)
                else:
                    x, log_prob_x, grad_x = ula_step(x, log_prob_x, grad_x, cur_log_prob_and_grad, cur_step_size)

            else:
                # Perform the MALA step
                if use_precond:
                    x, log_prob_x, grad_x, precond_grad_x, log_acc = precond_mala_step(x, log_prob_x, grad_x, precond_grad_x,
                                                                                       cur_log_prob_and_grad, cur_step_size, cur_precond_matrix, cur_precond_matrix_col)
                else:
                    x, log_prob_x, grad_x, log_acc = mala_step(
                        x, log_prob_x, grad_x, cur_log_prob_and_grad, cur_step_size)
                # Store the acceptance and update the step size
                acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
                sum_acc += acc
                if target_acceptance > 0.0:
                    cur_step_size = heuristics_step_size(cur_step_size, log_acc, target_acceptance=target_acceptance)
            # Store the sample
            samples[time_id, step_id] = x.clone()
        # Store the mean acceptance and the step size
        if not use_ula:
            mean_accs[time_id] = (sum_acc / n_mcmc_steps).mean()
        step_sizes_per_noise[time_id] = cur_step_size.clone()
        # Display the logs
        if verbose:
            logs = {}
            if not use_ula:
                logs['local_acc'] = float(mean_accs[time_id].cpu())
            if reweight_threshold > 0.0:
                logs['ess'] = float(ess_logs[time_id])
            r.set_postfix(**logs)
        # Rename the log-prob
        x_prev = x.clone()
        grad_x_prev = grad_x.clone()
        log_prob_x_prev = log_prob_x.clone()
    # Return everything
    diags = {}
    if not use_ula:
        diags['local_acc'] = mean_accs
    if reweight_threshold > 0.0:
        diags['ess'] = ess_logs
    return samples, step_sizes_per_noise, diags


def make_re_pairings(num_noise_levels, device=None):
    """Make the pairings for replica exchange

    Args:
        * num_noise_levels (int): Number of noise levels
        * device (torch.device): Torch device to use (default is None)

    Returns:
        * pairings (list of torch.Tensor): List of two index arrays provided the noise levels
            to jump to in the even and in the odd case
    """

    arr = torch.arange(num_noise_levels, device=device)
    # Even pass
    mask_a = (arr % 2 == 0) & (arr + 1 < num_noise_levels)
    a = torch.stack([arr[mask_a], arr[mask_a] + 1], dim=-1)
    # Odd pass
    mask_b = (arr % 2 == 1) & (arr + 1 < num_noise_levels)
    b = torch.stack([arr[mask_b], arr[mask_b] + 1], dim=-1)
    # Return everything
    return [a, b]


def re_step(x, log_prob_x, grad_x, log_prob_and_grads, times, idx_i, idx_j, batch_size, data_shape, data_shape_ones):
    """Make a step of Replica-Exchange

    Args:
        * x (torch.Tensor of shape (n_levels, batch_size, *data_shape)): Input samples (at each noise level)
        * log_prob_x (torch.Tensor of shape (n_levels, batch_size,)): Log-prob of the input samples at each noise level
        * grad_x (torch.Tensor of shape (n_levels, batch_size, *data_shape)): Score of the input samples at each noise level
        * log_prob_and_grads (function taking t of shape (K, batch_size, *data_shape_ones) and x of shape (K, batch_size, *data_shape)): Intermediate distributions
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Intermediate times
        * idx_i, idx_j (torch.Tensor of shape (n_pairs,)): Indexes of the different pairs
        * batch_size (int): Batch size
        * data_shape (tuple of int): Shape of the data
        * data_shape_ones (tuple of 1): (1,) * len(data_shape)

    Returns:
        * x (torch.Tensor of shape (n_levels * batch_size, *data_shape)): Input samples (at each noise level)
        * log_prob_x (torch.Tensor of shape (n_levels * batch_size,)): Log-prob of the input samples at each noise level
        * grad_x (torch.Tensor of shape (n_levels * batch_size, *data_shape)): Score of the input samples at each noise level
        * re_acc (float): Mean swap acceptance
    """
    # Get the length og the pair
    pair_length = idx_i.shape[0]
    # Compute log probabilities
    p_i_i, p_j_j = log_prob_x[idx_i], log_prob_x[idx_j]
    grad_i_i, grad_j_j = grad_x[idx_i], grad_x[idx_j]
    with torch.no_grad():
        p_i_j, grad_i_j = log_prob_and_grads(times[idx_i], x[idx_j])
    with torch.no_grad():
        p_j_i, grad_j_i = log_prob_and_grads(times[idx_j], x[idx_i])
    # Compute the log acceptance ratio
    log_acc = (p_i_j + p_j_i) - (p_i_i + p_j_j)
    # Get the mask
    # Equivalent to torch.log(torch.rand_like(log_acc)) < log_acc
    mask_matrix = torch.rand_like(log_acc).log_().lt_(log_acc).bool()
    re_acc = mask_matrix.float().mean()
    # Select the indexes
    x_prev = x.clone()
    log_prob_x[idx_i] = torch.where(mask_matrix, p_i_j, p_i_i)
    log_prob_x[idx_j] = torch.where(mask_matrix, p_j_i, p_j_j)
    mask_matrix = mask_matrix.view((pair_length, batch_size, *data_shape_ones))
    x[idx_i] = torch.where(mask_matrix, x_prev[idx_j], x_prev[idx_i])
    x[idx_j] = torch.where(mask_matrix, x_prev[idx_i], x_prev[idx_j])
    grad_x[idx_i] = torch.where(mask_matrix, grad_i_j, grad_i_i)
    grad_x[idx_j] = torch.where(mask_matrix, grad_j_i, grad_j_j)
    return x, log_prob_x, grad_x, re_acc


def re_sampler(x_init, times, log_prob_and_grads, swap_frequency, n_warmup_mcmc_steps, n_mcmc_steps, step_sizes_per_noise,
               per_noise_init=False, target_acceptance=0.75, precond_matrix_per_noise=None, precond_matrix_chol_per_noise=None,
               use_ula=False, verbose=True):
    """RE sampler

    Args:
        * x_init (torch.Tensor of shape (n_levels, batch_size, *data_shape) if per_noise_init and (batch_size, *data_shape) otherwise): Initial samples
        * times (torch.Tensor of shape (n_levels, batch_size, *data_shape_ones)): Value of the intermediate noise levels
        * log_prob_and_grads (function taking t of shape (batch_size, *data_shape_ones) and x of shape (batch_size, *data_shape)): Intermediate distributions
        * swap_frequency (int): Number of MCMC steps between each local step
        * n_warmup_mcmc_steps (int): Number of initial warmup steps for each noise level
        * n_mcmc_steps (int): Total number of MCMC steps (Swap + Local)
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_size, *data_shape_ones)): Step size for each noise level
        * per_noise_init (bool): Whether x_init contains per noise initialization (default is False)
        * target_acceptance (float): Target acceptance rate for local steps (default is 0.75)
        * precond_matrix_per_noise (torch.Tensor of shape (n_levels, batch_size, *data_shape, *data_shape)): Preconditioning matrices
        * precond_matrix_chol_per_noise (torch.Tensor of shape (n_levels, batch_size, *data_shape, *data_shape)): Cholesky decomposition of the preconditioning matrices
        * use_ula (bool): Skip the Metropolis-Hastings check in local steps (default is False)
        * verbose (bool): Whether to display a progress bar (default is True)

    Returns:
        * samples (torch.Tensor of shape (n_levels, n_mcmc_steps, batch_size, *data_shape)): Samples at each noise level
        * step_sizes_per_noise (torch.Tensor of size (n_levels, batch_sizes, *data_shape_ones)): Updated step size for each noise level
        * diags (dict): Dictionnary of diagnostics
    """
    # Get the shapes
    if per_noise_init:
        batch_size = x_init.shape[1]
        data_shape = x_init.shape[2:]
    else:
        batch_size = x_init.shape[0]
        data_shape = x_init.shape[1:]
    data_shape_ones = (1,) * len(data_shape)
    # Initalize the storage
    samples = torch.empty((times.shape[0], n_mcmc_steps, batch_size, *data_shape), device=x_init.device)
    if not use_ula:
        mean_local_accs = torch.zeros((times.shape[0],))
    mean_swap_acc = 0.0
    # Define the scores and log_probs
    time_ones = times.reshape((-1, *data_shape_ones))

    def local_log_prob_and_grads(y):
        return log_prob_and_grads(time_ones, y)

    def log_prob_and_grads_batched(t, y):
        log_prob, grad = log_prob_and_grads(t.view((-1, *data_shape_ones)), y.view((-1, *data_shape)))
        return log_prob.view(y.shape[:2]), grad.view(y.shape)
    # Get the initial point
    if per_noise_init:
        x = x_init.clone()
    else:
        x = x_init.unsqueeze(0).repeat((times.shape[0], 1, *data_shape_ones))
    x = x.view((-1, *data_shape))
    step_sizes_per_noise = step_sizes_per_noise.view((-1, *data_shape_ones))
    # Check if the MALA steps should be pre-conditionned
    use_precond = (precond_matrix_per_noise is not None) and (precond_matrix_chol_per_noise is not None)
    if use_precond:
        precond_matrix_per_noise = precond_matrix_per_noise.view((-1, *precond_matrix_per_noise.shape[2:]))
        precond_matrix_chol_per_noise = precond_matrix_chol_per_noise.view(
            (-1, *precond_matrix_chol_per_noise.shape[2:]))
    # Initialize log_prob_x and grad_x
    log_prob_x, grad_x = local_log_prob_and_grads(x)
    if use_precond:
        precond_grad_x = torch.matmul(precond_matrix_per_noise, grad_x.unsqueeze(-1)).squeeze(-1)
    # Make the pairings
    pairs = make_re_pairings(times.shape[0], x_init.device)
    # Run the algorithm
    if verbose:
        r = trange(n_warmup_mcmc_steps+n_mcmc_steps)
    else:
        r = range(n_warmup_mcmc_steps+n_mcmc_steps)
    for step_id in r:
        if step_id % swap_frequency == 0:
            # Select the pairs
            swap_id = int(step_id // swap_frequency) % 2
            # Reshape the data into (n_levels, batch_size, ...)
            x = x.view((-1, batch_size, *data_shape))
            grad_x = grad_x.view((-1, batch_size, *data_shape))
            log_prob_x = log_prob_x.view((-1, batch_size))
            # Perform the step
            x, log_prob_x, grad_x, re_acc = re_step(x=x, log_prob_x=log_prob_x, grad_x=grad_x,
                                                    log_prob_and_grads=log_prob_and_grads_batched, times=times, idx_i=pairs[
                                                        swap_id][:, 0], idx_j=pairs[swap_id][:, 1],
                                                    batch_size=batch_size, data_shape=data_shape, data_shape_ones=data_shape_ones)
            # Reshape the data into (n_levels * batch_size, ....)
            x = x.view((-1, *data_shape))
            grad_x = grad_x.view((-1, *data_shape))
            if use_precond:
                precond_grad_x = torch.matmul(precond_matrix_per_noise, grad_x.unsqueeze(-1)).squeeze(-1)
            log_prob_x = log_prob_x.flatten()
            mean_swap_acc = re_acc
        else:
            # Perform the local step
            if use_ula:
                # Perform the ULA step
                if use_precond:
                    x, log_prob_x, grad_x, precond_grad_x = precond_ula_step(
                        x, log_prob_x, grad_x, precond_grad_x, local_log_prob_and_grads, step_sizes_per_noise,
                        precond_matrix_per_noise, precond_matrix_chol_per_noise
                    )
                else:
                    x, log_prob_x, grad_x = ula_step(
                        x, log_prob_x, grad_x, local_log_prob_and_grads, step_sizes_per_noise)
            else:
                # Perform the MALA step
                if use_precond:
                    x, log_prob_x, grad_x, precond_grad_x, log_acc = precond_mala_step(
                        x, log_prob_x, grad_x, precond_grad_x, local_log_prob_and_grads, step_sizes_per_noise,
                        precond_matrix_per_noise, precond_matrix_chol_per_noise
                    )
                else:
                    x, log_prob_x, grad_x, log_acc = mala_step(
                        x, log_prob_x, grad_x, local_log_prob_and_grads, step_sizes_per_noise)
                # Adapt the step-size
                if target_acceptance > 0.0:
                    step_sizes_per_noise = heuristics_step_size(
                        step_sizes_per_noise, log_acc, target_acceptance=target_acceptance)
                # Log the acceptance
                acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
                mean_local_accs = acc.view((-1, batch_size)).mean(dim=-1)
        # Store the samples
        if step_id >= n_warmup_mcmc_steps:
            samples[:, step_id-n_warmup_mcmc_steps] = x.reshape((-1, batch_size, *data_shape)).clone()
        if verbose:
            diags = {'swap_acc': float(mean_swap_acc.cpu())}
            if not use_ula:
                diags['local_acc'] = float(mean_local_accs.mean().cpu())
            r.set_postfix(**diags)
    # Return the samples
    final_diags = {'swap_acc': mean_swap_acc}
    if not use_ula:
        final_diags['local_acc'] = mean_local_accs
    return samples, step_sizes_per_noise, final_diags


class MaximumLikelihoodEBM(torch.nn.Module):

    def __init__(self, sde, prior, net, sampler_type, step_sizes_per_noise=1e-3, precond_matrix_per_noise=None, precond_matrix_chol_per_noise=None, use_ula=False,
                 reweight_threshold=1.0, swap_frequency=16, target_acceptance=0.75, perc_keep_mcmc=-1.0, use_snr_adapted_disc=False, start_eps=1e-3, end_eps=0.0, n_steps=100):
        # Call the parent constructor
        super().__init__()
        # Store SDE parameters
        self.sde = sde
        self.prior = prior
        self.prior.cpu()
        self.net = net.cpu()
        # Store the sampler type
        self.sampler_type = sampler_type
        self.reweight_threshold = reweight_threshold
        self.swap_frequency = swap_frequency
        self.step_sizes_per_noise = step_sizes_per_noise
        self.precond_matrix_per_noise = precond_matrix_per_noise
        self.precond_matrix_chol_per_noise = precond_matrix_chol_per_noise
        self.use_precond = (precond_matrix_per_noise is not None) and (precond_matrix_chol_per_noise is not None)
        self.use_ula = use_ula
        # Parameters for training
        self.target_acceptance = target_acceptance
        self.use_snr_adapted_disc = use_snr_adapted_disc
        self.perc_keep_mcmc = perc_keep_mcmc
        # Build the time discretization
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.n_steps = n_steps
        self.build_time_disc()

    def build_time_disc(self):
        # Make the intermediate times
        if self.use_snr_adapted_disc:
            self.register_buffer('times', get_timesteps(start=self.start_eps, end=self.sde.terminal_t-self.end_eps,
                                                        steps=self.n_steps, sde=self.sde).unsqueeze(-1))
        else:
            self.register_buffer('times', get_timesteps(start=self.start_eps, end=self.sde.terminal_t-self.end_eps,
                                                        steps=self.n_steps).unsqueeze(-1))

    def log_prob_and_grads(self, t, y):
        if self.net.has_unnorm_log_prob_and_grad:
            return self.net.unnorm_log_prob_and_grad(t, y)
        else:
            y_ = torch.autograd.Variable(y.clone(), requires_grad=True)
            log_prob_y = self.net.unnorm_log_prob(t, y_)
            return log_prob_y, torch.autograd.grad(log_prob_y.sum(), y_)[0].detach()

    def sample_model(self, batch_size, times_expanded, is_very_first_negative_sampling, initial_n_warmup_mcmc_steps, n_warmup_mcmc_steps, n_mcmc_steps, x_init_persistant):
        # Disable gradient with respect to the network
        parameters_states = []
        for p in self.net.parameters():
            parameters_states.append(p.requires_grad)
            p.requires_grad_(False)
        # Get the negative samples
        if self.sampler_type == 'annealed_mcmc':
            x_init = self.prior.sample((batch_size,))
            xs_neg, self.step_sizes_per_noise, diags = smc_sampler(
                x_init=x_init,
                times=times_expanded,
                log_prob_and_grads=self.log_prob_and_grads,
                n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps if is_very_first_negative_sampling else n_warmup_mcmc_steps,
                n_mcmc_steps=n_mcmc_steps,
                step_sizes_per_noise=self.step_sizes_per_noise,
                reweight_threshold=0.0,
                target_acceptance=self.target_acceptance,
                precond_matrix_per_noise=self.precond_matrix_per_noise if self.use_precond else None,
                precond_matrix_chol_per_noise=self.precond_matrix_chol_per_noise if self.use_precond else None,
                use_ula=self.use_ula,
                verbose=False
            )
        elif self.sampler_type == 'smc':
            x_init = self.prior.sample((batch_size,))
            xs_neg, self.step_sizes_per_noise, diags = smc_sampler(
                x_init=x_init,
                times=times_expanded,
                log_prob_and_grads=self.log_prob_and_grads,
                n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps if is_very_first_negative_sampling else n_warmup_mcmc_steps,
                n_mcmc_steps=n_mcmc_steps,
                step_sizes_per_noise=self.step_sizes_per_noise,
                reweight_threshold=self.reweight_threshold,
                target_acceptance=self.target_acceptance,
                precond_matrix_per_noise=self.precond_matrix_per_noise if self.use_precond else None,
                precond_matrix_chol_per_noise=self.precond_matrix_chol_per_noise if self.use_precond else None,
                use_ula=self.use_ula,
                verbose=False
            )
        elif self.sampler_type == 'smc_pdds':
            xs_neg, self.step_sizes_per_noise, diags = smc_sampler(
                x_init=x_init,
                times=times_expanded,
                log_prob_and_grads=self.log_prob_and_grads,
                n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps if is_very_first_negative_sampling else n_warmup_mcmc_steps,
                n_mcmc_steps=n_mcmc_steps,
                step_sizes_per_noise=self.step_sizes_per_noise,
                reweight_threshold=self.reweight_threshold,
                use_pdds_weights=True,
                sde=self.sde,
                target_acceptance=self.target_acceptance,
                precond_matrix_per_noise=self.precond_matrix_per_noise if self.use_precond else None,
                precond_matrix_chol_per_noise=self.precond_matrix_chol_per_noise if self.use_precond else None,
                use_ula=self.use_ula,
                verbose=False
            )
        elif self.sampler_type == 'replica_exchange':
            xs_neg, self.step_sizes_per_noise, diags = re_sampler(
                x_init=x_init_persistant,
                times=times_expanded,
                log_prob_and_grads=self.log_prob_and_grads,
                swap_frequency=self.swap_frequency,
                n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps if is_very_first_negative_sampling else n_warmup_mcmc_steps,
                n_mcmc_steps=n_mcmc_steps,
                step_sizes_per_noise=self.step_sizes_per_noise,
                per_noise_init=True,
                target_acceptance=self.target_acceptance,
                precond_matrix_per_noise=self.precond_matrix_per_noise if self.use_precond else None,
                precond_matrix_chol_per_noise=self.precond_matrix_chol_per_noise if self.use_precond else None,
                use_ula=self.use_ula,
                verbose=False
            )
        else:
            raise NotImplementedError('Sampler {} not found.'.format(self.sampler_type))
        # Restore gradients with respect to the network
        for i, p in enumerate(self.net.parameters()):
            p.requires_grad_(parameters_states[i])
        # Return the samples and diags
        return xs_neg, diags

    def cd_sampling(self, times_expanded, n_warmup_mcmc_steps, n_mcmc_steps, xs_pos):
        # Setup the parallel log_prob_and_grad
        data_shape_ones = (1,) * (len(xs_pos.shape) - 1)
        time_ones = times_expanded.reshape((-1, *data_shape_ones))

        def log_prob_and_grads_(z):
            return self.log_prob_and_grads(time_ones, z)
        # Setup the storage
        xs_neg = torch.empty((n_mcmc_steps, *xs_pos.shape), device=xs_pos.device)
        x = xs_pos.clone()
        log_prob_x, grad_x = log_prob_and_grads_(x)
        if self.use_precond:
            precond_grad_x = torch.matmul(self.precond_matrix_per_noise, grad_x.unsqueeze(-1)).squeeze(-1)
        for step_id in range(n_warmup_mcmc_steps+n_mcmc_steps):
            if self.use_ula:
                # Perform the ULA step
                if self.use_precond:
                    x, log_prob_x, grad_x, precond_grad_x = precond_ula_step(
                        x, log_prob_x, grad_x, precond_grad_x, log_prob_and_grads_, self.step_sizes_per_noise,
                        self.precond_matrix_per_noise, self.precond_matrix_chol_per_noise
                    )
                else:
                    x, log_prob_x, grad_x = ula_step(
                        x, log_prob_x, grad_x, log_prob_and_grads_, self.step_sizes_per_noise)
            else:
                # Perform the MALA step
                if self.use_precond:
                    x, log_prob_x, grad_x, precond_grad_x, log_acc = precond_mala_step(
                        x, log_prob_x, grad_x, precond_grad_x, log_prob_and_grads_, self.step_sizes_per_noise,
                        self.precond_matrix_per_noise, self.precond_matrix_chol_per_noise
                    )
                else:
                    x, log_prob_x, grad_x, log_acc = mala_step(
                        x, log_prob_x, grad_x, log_prob_and_grads_, self.step_sizes_per_noise)
                # Adapt the step-size
                if self.target_acceptance > 0.0:
                    self.step_sizes_per_noise = heuristics_step_size(
                        self.step_sizes_per_noise, log_acc, target_acceptance=self.target_acceptance)
                # Log the acceptance
                acc = torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
                mean_local_accs = acc.mean().cpu()
            # Store the samples
            if step_id >= n_warmup_mcmc_steps:
                xs_neg[step_id-n_warmup_mcmc_steps] = x.clone()
        # Build the diagnostics
        if not self.use_ula:
            diags = {'local_acc': mean_local_accs}
        else:
            diags = {}
        return xs_neg, diags

    def save_logs(self, losses, losses_grad, diagnostics, use_ema, epoch_id, ckpt_filepath_maker):
        torch.save({k: v.cpu() for k, v in self.net.state_dict().items()}, ckpt_filepath_maker(epoch_id))
        if use_ema:
            torch.save({k: v.cpu() for k, v in self.ema_net.state_dict().items()},
                       ckpt_filepath_maker(epoch_id, suffix='ema'))
        torch.save(torch.FloatTensor(losses), ckpt_filepath_maker(epoch_id, suffix='losses'))
        if losses_grad is not None:
            torch.save(torch.FloatTensor(losses_grad), ckpt_filepath_maker(epoch_id, suffix='losses_grad'))
        with open(ckpt_filepath_maker(epoch_id, suffix='diag'), 'wb') as f:
            pickle.dump(diagnostics, f)

    def train(self, data, batch_size, n_epochs, reweight_loss=False, lr=3e-4, decay=0.0, clip_val=1.0, initial_n_warmup_mcmc_steps=1024,
              n_mcmc_steps=32, n_accumulation_steps=1, reg_val=0.0, use_ema=False, ema_decay=0.995, ema_steps=10, ckpt_interval=-1,
              ckpt_filepath_maker=None, verbose=True):
        # Can't use n_accumulation_steps > 0 if sampler_type is CD
        if (n_accumulation_steps != 1) and (self.sampler_type == 'cd'):
            raise ValueError('Can\'t use n_accumulation_steps != 1 if sampler_type is CD.')
        # Can't scale loss with CD for now
        if reweight_loss and (self.sampler_type == 'cd'):
            raise NotImplementedError('Loss scaling not implemented with CD.')
        # Build an optimizer
        if decay > 0:
            optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=decay)
        else:
            optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        # Compute the number of kept MCMC steps
        if self.perc_keep_mcmc > 0:
            n_warmup_mcmc_steps = int((1. - self.perc_keep_mcmc) * n_mcmc_steps)
            n_mcmc_steps_kept = int(self.perc_keep_mcmc * n_mcmc_steps)
        else:
            n_warmup_mcmc_steps = n_mcmc_steps - 1
            n_mcmc_steps_kept = 1
        # Build a dataset
        dataset = torch.utils.data.TensorDataset(data)
        if self.sampler_type == 'cd':
            effective_batch_size = batch_size
        else:
            effective_batch_size = min(batch_size * n_mcmc_steps_kept, data.shape[0])
        train_loader = torch.utils.data.DataLoader(
            dataset, batch_size=effective_batch_size, shuffle=True, drop_last=True)
        n_batches = len(train_loader)
        # Make the EMA model
        if use_ema:
            adjust = n_accumulation_steps * self.times.shape[0] * effective_batch_size * ema_steps / n_epochs
            alpha = 1.0 - ema_decay
            alpha = min(1.0, alpha * adjust)
            self.ema_net = torch.optim.swa_utils.AveragedModel(self.net,
                                                               multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(1. - alpha))
        # Compute the times
        data_shape = data.shape[1:]
        data_shape_ones = (1,) * (len(data.shape) - 1)
        ts = self.times.reshape((-1, 1, *data_shape_ones)).repeat((1, effective_batch_size, *data_shape_ones))
        if self.sampler_type == 'cd':
            batch_size_neg = min(batch_size * n_mcmc_steps_kept, data.shape[0])
            ts_neg = self.times.reshape((-1, 1, *data_shape_ones)).repeat((1, batch_size_neg, *data_shape_ones))
            ts_neg = ts_neg.view((-1, *data_shape_ones))
        times_expanded = self.times.reshape((-1, 1, *data_shape_ones)).repeat((1, batch_size, *data_shape_ones))
        mean_factor_ts = self.sde.s(ts)
        std_factor_ts = mean_factor_ts * torch.sqrt(self.sde.sigma_sq(ts))
        ts = ts.view((-1, *data_shape_ones))
        # Get the initial step sizes
        if isinstance(self.step_sizes_per_noise, float) or \
                (isinstance(self.step_sizes_per_noise, torch.Tensor) and (len(self.step_sizes_per_noise.shape) == 0)):
            self.step_sizes_per_noise = self.step_sizes_per_noise * \
                torch.ones((self.times.shape[0], batch_size, *data_shape_ones), device=self.times.device)
        elif len(self.step_sizes_per_noise.shape) > 0:
            self.step_sizes_per_noise = self.step_sizes_per_noise.flatten().unsqueeze(1).repeat((1, batch_size)).to(self.times.device)
            self.step_sizes_per_noise = self.step_sizes_per_noise.view(
                (self.times.shape[0], batch_size, *data_shape_ones))
        else:
            raise ValueError('Wrong shape of step_sizes_per_noise.')
        if self.sampler_type == 'cd':
            self.step_sizes_per_noise = self.step_sizes_per_noise.view((-1, *data_shape_ones))
        # Setup the preconditioning matrices
        if self.use_precond and ((self.sampler_type == 'replica_exchange') or (self.sampler_type == 'cd')):
            self.precond_matrix_per_noise = self.precond_matrix_per_noise.unsqueeze(1)
            self.precond_matrix_per_noise = self.precond_matrix_per_noise.repeat(
                (1, batch_size, *data_shape_ones, *data_shape_ones))
            if self.sampler_type == 'cd':
                self.precond_matrix_per_noise = self.precond_matrix_per_noise.view((-1, *data_shape, *data_shape))
            self.precond_matrix_chol_per_noise = self.precond_matrix_chol_per_noise.unsqueeze(1)
            self.precond_matrix_chol_per_noise = self.precond_matrix_chol_per_noise.repeat(
                (1, batch_size, *data_shape_ones, *data_shape_ones))
            if self.sampler_type == 'cd':
                self.precond_matrix_chol_per_noise = self.precond_matrix_chol_per_noise.view(
                    (-1, *data_shape, *data_shape))
        # Get a persistent state for replica exchange
        if self.sampler_type == 'replica_exchange':
            if self.net.has_sample_prior:
                times_ = self.times.clone().reshape((-1, 1)).repeat((1, batch_size)).view((-1, 1))
                x_init_persistant = self.net.sample_prior(times_)
                x_init_persistant = x_init_persistant.view((self.times.shape[0], batch_size, -1))
            else:
                x_init_persistant = self.prior.sample((self.times.shape[0], batch_size,))
        else:
            x_init_persistant = None
        # Run the optimization
        if verbose:
            r = trange(n_epochs)
        else:
            r = range(n_epochs)
        losses = []
        if clip_val > 0:
            losses_grad = []
        diagnostics = []
        is_very_first_negative_sampling = True
        global_step = 0
        for epoch_id in r:
            for batch_id, batch in enumerate(train_loader):
                # Get the data
                is_very_first_batch = (epoch_id == 0) and (batch_id == 0)
                data_samples = batch[0]
                data_shape = data.shape[1:]
                # Get the positive samples
                xs_pos = mean_factor_ts * data_samples.unsqueeze(0) + std_factor_ts * torch.randn_like(data_samples)
                xs_pos = xs_pos.view((-1, *data_shape))
                # Only perform negative sampling on some batches
                if is_very_first_batch and self.net.has_sample_prior:
                    if self.sampler_type == 'cd':
                        xs_neg = self.net.sample_prior(ts_neg)
                    else:
                        xs_neg = self.net.sample_prior(ts)
                    diags = {}
                elif (batch_id % n_accumulation_steps == 0):
                    # Sample the model
                    if self.sampler_type == 'cd':
                        # Sample with Langevin from the positive samples
                        xs_neg, diags = self.cd_sampling(
                            times_expanded=times_expanded,
                            n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps if is_very_first_negative_sampling else n_warmup_mcmc_steps,
                            n_mcmc_steps=n_mcmc_steps_kept,
                            xs_pos=xs_pos
                        )
                        xs_neg = xs_neg.detach()
                    else:
                        # Sample with annealed MCMC sampler
                        xs_neg, diags = self.sample_model(
                            batch_size=batch_size,
                            times_expanded=times_expanded,
                            is_very_first_negative_sampling=is_very_first_negative_sampling,
                            initial_n_warmup_mcmc_steps=initial_n_warmup_mcmc_steps,
                            n_warmup_mcmc_steps=n_warmup_mcmc_steps,
                            n_mcmc_steps=n_mcmc_steps_kept,
                            x_init_persistant=x_init_persistant
                        )
                        xs_neg = xs_neg.detach()
                        # Update the persistent state
                        if x_init_persistant is not None:
                            x_init_persistant = xs_neg[:, -1]
                    # Set the first negative sampling as done
                    is_very_first_negative_sampling = False
                    # Append the diagnostics
                    diagnostics.append(diags)
                    # Only keep some MCMC
                    xs_neg = xs_neg.reshape((-1, *data_shape))
                # Evaluate the energy
                en_pos = self.net.energy(ts, xs_pos).flatten()
                if (batch_id % n_accumulation_steps == 0):
                    if self.sampler_type == 'cd':
                        en_neg = self.net.energy(ts_neg, xs_neg).flatten()
                    else:
                        en_neg = self.net.energy(ts, xs_neg).flatten()
                # Compute the loss scaling weights
                if reweight_loss:
                    loss_scale = 1.0 / self.sde.sigma_sq(ts).flatten()
                else:
                    loss_scale = 1.0
                # Compute the loss
                if self.sampler_type == 'cd':
                    loss = en_pos.mean() - en_neg.mean()
                else:
                    loss = (loss_scale * (en_pos - en_neg)).mean()
                # Regularize the loss
                if reg_val > 0:
                    loss += reg_val * (torch.square(en_pos).mean() + torch.square(en_neg).mean())
                # Anticipate the loss' averaging
                if batch_id >= n_batches - (n_batches % n_accumulation_steps):
                    loss /= (n_batches % n_accumulation_steps)
                else:
                    loss /= n_accumulation_steps
                # Exit is the loss is NaN
                if torch.isnan(loss).any():
                    self.save_logs(losses, losses_grad if clip_val > 0 else None,
                                   diagnostics, use_ema, epoch_id, ckpt_filepath_maker)
                    raise RuntimeError('NaN loss detected.')
                if torch.mean(loss).abs().item() > 1e9:
                    self.save_logs(losses, losses_grad if clip_val > 0 else None,
                                   diagnostics, use_ema, epoch_id, ckpt_filepath_maker)
                    raise RuntimeError('Training diverged (loss = {:.2e}).'.format(torch.mean(loss).item()))
                # Store the loss
                losses.append(loss.item())
                # Call the backward on loss
                loss.backward(retain_graph=n_accumulation_steps > 1)
                # Log the loss
                infos = {'loss': losses[-1]}
                # Clip the gradient
                if clip_val > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.net.parameters(), clip_val)
                    losses_grad.append(grad_norm.item())
                    infos['norm_grad_loss'] = losses_grad[-1]
                # Display the loss on progress bar
                if verbose:
                    r.set_postfix(
                        **infos,
                        **{k: float(v.cpu().mean().item()) for k, v in diags.items()}
                    )
                if ((batch_id + 1) % n_accumulation_steps == 0) or (batch_id + 1 == n_batches):
                    # Perform a gradient step
                    optimizer.step()
                    global_step += 1
                    # Do the EMA step
                    if use_ema and ((global_step % ema_steps) == 0):
                        self.ema_net.update_parameters(self.net)
                    # Reset the gradients
                    optimizer.zero_grad()
            # Save everything
            if (ckpt_interval > 0) and ((epoch_id % ckpt_interval) == 0):
                self.save_logs(losses, losses_grad if clip_val > 0 else None,
                               diagnostics, use_ema, epoch_id, ckpt_filepath_maker)
        if clip_val > 0:
            return torch.FloatTensor(losses), torch.FloatTensor(losses_grad), diagnostics
        else:
            return torch.FloatTensor(losses), diagnostics

    def _apply(self, fn):
        new_self = super(MaximumLikelihoodEBM, self)._apply(fn)
        new_self.sde = new_self.sde._apply(fn)
        new_self.prior._apply(fn)
        new_self.net = new_self.net._apply(fn)
        return new_self
