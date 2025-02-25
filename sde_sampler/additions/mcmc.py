# MCMC sampling utils

# Libraries
import torch
import math


def sample_multivariate_normal_diag(batch_size, mean, variance):
    """Sample according to multivariate normal with diagonal matrix"""
    z = torch.randn((batch_size, *mean.shape[1:]), device=mean.device)
    if isinstance(variance, torch.Tensor):
        return torch.sqrt(variance) * z + mean
    else:
        return math.sqrt(variance) * z + mean


def log_prob_multivariate_normal_diag(samples, mean, variance, sum_indexes):
    """Evaluate the log density of multivariate normal with diagonal matrix

    WARNING: Single sample along a batch size

    The multiplicative factor
            - 0.5 * dim * torch.log(2.0 * torch.pi * variance)
    was removed.
    """
    ret = -0.5 * torch.sum(torch.square(samples - mean), dim=sum_indexes)
    if isinstance(variance, torch.Tensor) and len(variance.shape) > 0:
        ret /= variance.flatten()
    else:
        ret /= variance
    return ret


# def heuristics_step_size(
#     stepsize, mean_acceptance, target_acceptance=0.75, factor=1.01, tol=0.05
# ):
#     """Heuristic for adaptative step size in a vectorized fashion"""
#     stepsize = torch.where(
#         (mean_acceptance - target_acceptance > tol).view(
#             (-1, *(1,) * (len(stepsize.shape) - 1))
#         ),
#         stepsize * factor,
#         stepsize,
#     )
#     stepsize = torch.where(
#         (target_acceptance - mean_acceptance > tol).view(
#             (-1, *(1,) * (len(stepsize.shape) - 1))
#         ),
#         stepsize / factor,
#         stepsize,
#     )
#     return stepsize

def heuristics_step_size(
    stepsize, mean_log_acceptance, target_acceptance=0.75, factor=1.01, tol=0.05
):
    """Heuristic for adaptative step size in a vectorized fashion"""
    stepsize = torch.where(
        (mean_log_acceptance - math.log(target_acceptance) > math.log1p(tol)).view(
            (-1, *(1,) * (len(stepsize.shape) - 1))
        ),
        stepsize * factor,
        stepsize,
    )
    stepsize = torch.where(
        (math.log(target_acceptance) - mean_log_acceptance > -math.log1p(-tol)).view(
            (-1, *(1,) * (len(stepsize.shape) - 1))
        ),
        stepsize / factor,
        stepsize,
    )
    return stepsize


def mala_step(
    y,
    target_log_prob_y,
    target_grad_y,
    target_log_prob_and_grad,
    step_size,
):
    """Step of the Metropolis-adjusted Langevin algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_grad_y (torch.Tensor of shape (batch_size, dim)): Target's score at y
            target_log_prob_and_grad (function): Target log-likelihood and score
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            target_grad_y_next (torch.Tensor of shape (batch_size, dim)): Target's score at y_next
            step_size_next (torch.Tensor of shape (batch_size,)): Updated step size for each chain
            acc (float): Mean acceptance of the step
    """

    # Sample the proposal
    y_prop = sample_multivariate_normal_diag(
        batch_size=y.shape[0],
        mean=y + step_size * target_grad_y,
        variance=2.0 * step_size,
    )
    # Compute log-densities at the proposal
    target_log_prob_y_prop, target_grad_y_prop = target_log_prob_and_grad(y_prop)
    # Compute the MH ratio
    with torch.no_grad():
        joint_prop = target_log_prob_y_prop - log_prob_multivariate_normal_diag(
            y_prop,
            mean=y + step_size * target_grad_y,
            variance=2.0 * step_size,
            sum_indexes=-1,
        )
        joint_orig = target_log_prob_y - log_prob_multivariate_normal_diag(
            y,
            mean=y_prop + step_size * target_grad_y_prop,
            variance=2.0 * step_size,
            sum_indexes=-1,
        )
    # Acceptance step
    log_acc = joint_prop - joint_orig
    mask = torch.log(torch.rand_like(target_log_prob_y_prop, device=y.device)) < log_acc
    y.data[mask] = y_prop[mask]
    target_log_prob_y.data[mask] = target_log_prob_y_prop[mask]
    target_grad_y.data[mask] = target_grad_y_prop[mask]
    # Return everything
    return (
        y,
        target_log_prob_y,
        target_grad_y,
        # torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
        log_acc
    )


def precond_mala_step(y, target_log_prob_y, target_grad_y, precond_grad_y, target_log_prob_and_grad, step_size,
                      precond_matrix, precond_matrix_chol):
    """Step of the Preconditionned Metropolis-adjusted Langevin algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_grad_y (torch.Tensor of shape (batch_size, dim)): Target's score at y
            precond_grad_y (torch.Tensor of shape (batch_size, dim)): Preconditionned target's score at y
            target_log_prob_and_grad (function): Target log-likelihood and score
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain
            precond_matrix (torch.Tensor of shape (batch_size, dim, dim)): Pre-conditioning matrix
            precond_matrix_chol (torch.Tensor of shape (batch_size, dim, dim)): Cholesky decompositon of pre-conditioning matrix

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            target_grad_y_next (torch.Tensor of shape (batch_size, dim)): Target's score at y_next
            step_size_next (torch.Tensor of shape (batch_size,)): Updated step size for each chain
            acc (float): Mean acceptance of the step
    """
    # Draw a sample from the proposal
    with torch.no_grad():
        y_prop = y + step_size * precond_grad_y
        y_prop += torch.sqrt(2. * step_size) * torch.matmul(precond_matrix_chol,
                                                            torch.randn((*y.shape, 1), device=y.device)).squeeze(-1)
    # Compute log-densities at the proposal
    target_log_prob_y_prop, target_grad_y_prop = target_log_prob_and_grad(y_prop)
    precond_grad_y_prop = torch.matmul(precond_matrix, target_grad_y_prop.unsqueeze(-1)).squeeze(-1)
    # Compute the MH ratio
    # Optimization from https://arxiv.org/pdf/2305.14442 (Proposition 1)
    with torch.no_grad():
        log_acc = target_log_prob_y_prop - target_log_prob_y
        log_acc += 0.5 * torch.sum((y - y_prop - 0.5 * step_size * precond_grad_y_prop) * target_grad_y_prop, dim=-1)
        log_acc -= 0.5 * torch.sum((y_prop - y - 0.5 * step_size * precond_grad_y) * target_grad_y, dim=-1)
    # Perform the acceptance step
    mask = torch.log(torch.rand_like(target_log_prob_y_prop, device=y.device)) < log_acc
    y.data[mask] = y_prop[mask]
    target_log_prob_y.data[mask] = target_log_prob_y_prop[mask]
    target_grad_y.data[mask] = target_grad_y_prop[mask]
    precond_grad_y.data[mask] = precond_grad_y_prop[mask]
    # Return everything
    return (
        y,
        target_log_prob_y,
        target_grad_y,
        precond_grad_y,
        # torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
        log_acc
    )


def ula_step(
    y,
    target_log_prob_y,
    target_grad_y,
    target_log_prob_and_grad,
    step_size,
):
    """Step of the Unadjusted Langevin algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_grad_y (torch.Tensor of shape (batch_size, dim)): Target's score at y
            target_log_prob_and_grad (function): Target log-likelihood and score
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            target_grad_y_next (torch.Tensor of shape (batch_size, dim)): Target's score at y_next
    """

    # Sample the proposal
    with torch.no_grad():
        y_prop = sample_multivariate_normal_diag(
            batch_size=y.shape[0],
            mean=y + step_size * target_grad_y,
            variance=2.0 * step_size,
        )
    # Compute log-densities at the proposal
    target_log_prob_y_prop, target_grad_y_prop = target_log_prob_and_grad(y_prop)
    # Return everything
    return y_prop, target_log_prob_y_prop, target_grad_y_prop


def precond_ula_step(y, target_log_prob_y, target_grad_y, precond_grad_y, target_log_prob_and_grad, step_size,
                     precond_matrix, precond_matrix_chol):
    """Step of the Preconditionned Unadjusted Langevin algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_grad_y (torch.Tensor of shape (batch_size, dim)): Target's score at y
            precond_grad_y (torch.Tensor of shape (batch_size, dim)): Preconditionned target's score at y
            target_log_prob_and_grad (function): Target log-likelihood and score
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain
            precond_matrix (torch.Tensor of shape (batch_size, dim, dim)): Pre-conditioning matrix
            precond_matrix_chol (torch.Tensor of shape (batch_size, dim, dim)): Cholesky decompositon of pre-conditioning matrix

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            target_grad_y_next (torch.Tensor of shape (batch_size, dim)): Target's score at y_next
    """
    # Draw a sample from the proposal
    with torch.no_grad():
        y_prop = y + step_size * precond_grad_y
        y_prop += torch.sqrt(2. * step_size) * torch.matmul(precond_matrix_chol,
                                                            torch.randn((*y.shape, 1), device=y.device)).squeeze(-1)
    # Compute log-densities at the proposal
    target_log_prob_y_prop, target_grad_y_prop = target_log_prob_and_grad(y_prop)
    precond_grad_y_prop = torch.matmul(precond_matrix, target_grad_y_prop.unsqueeze(-1)).squeeze(-1)
    # Return everything
    return y_prop, target_log_prob_y_prop, target_grad_y_prop, precond_grad_y_prop


@torch.no_grad()
def rwmh_step(
    y,
    target_log_prob_y,
    target_log_prob,
    step_size,
):
    """Step of the Random-Walk Metropolis Hasting algorithm

    Args:
            y (torch.Tensor of shape (batch_size, dim)): Previous sample
            target_log_prob_y (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y
            target_log_prob (function): Target log-likelihood
            step_size (torch.Tensor of shape (batch_size,)): Step size for each chain

    Returns:
            y_next (torch.Tensor of shape (batch_size, dim)): Next sample
            target_log_prob_y_next (torch.Tensor of shape (batch_size,)): Target's log-likelihood at y_next
            step_size_next (torch.Tensor of shape (batch_size,)): Updated step size for each chain
            acc (float): Mean acceptance of the step
    """

    # Make a proposal
    with torch.no_grad():
        y_prop = y + step_size * torch.randn_like(y, device=y.device)
    # Compute the Metropolis-Hastings ratio
    target_log_prob_y_prop = target_log_prob(y_prop).flatten()
    with torch.no_grad():
        log_acc = target_log_prob_y_prop - target_log_prob_y
    mask = torch.log(torch.rand((y.shape[0],), device=y.device)) < log_acc
    y.data[mask] = y_prop[mask]
    target_log_prob_y.data[mask] = target_log_prob_y_prop[mask]
    # Return everything
    return (
        y,
        target_log_prob_y,
        # torch.exp(torch.minimum(torch.zeros_like(log_acc), log_acc))
        log_acc
    )
