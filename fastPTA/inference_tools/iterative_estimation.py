# Global imports
import jax
import jax.numpy as jnp

import fastPTA.inference_tools.signal_covariance as sc

# Local imports
from fastPTA.utils import compute_inverse


@jax.jit
def get_update_estimate_diagonal(
    parameters, data, gamma_IJ_lm, frequencies, S_f
):
    """
    Compute parameter updates using iterative estimation with diagonal frequency
    approximation.

    This function implements an iterative parameter estimation algorithm that
    updates parameters based on the Fisher information matrix and the gradient
    of the log-likelihood. It uses the diagonal frequency approximation where
    cross-frequency correlations are ignored.

    Parameters:
    -----------
    parameters : Array
        Current parameter estimates (lm coefficients).
    data : Array
        Data tensor with shape (F, N, N) where F is the number of frequencies
        and N is the number of pulsars.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    frequencies : Array
        Array containing the frequencies.
    S_f : Array
        Array containing the spectral density values at each frequency.

    Returns:
    --------
    Tuple containing:
    - Array: Parameter updates to be added to the current parameters.
    - Array: Inverse Fisher matrix, which provides uncertainty estimates.
    """

    # Compute the covariance and its derivative
    C = sc.get_covariance_diagonal(parameters, gamma_IJ_lm, frequencies, S_f)
    dC = sc.get_dcovariance_diagonal(parameters, gamma_IJ_lm, frequencies, S_f)

    # Compute the inverse of the covariance matrix
    C_inv = compute_inverse(C)

    # Contract the derivative of the covariance matrix with the inverse of
    # the covariance matrix
    C_inv_dC = jnp.einsum("fij,afjk->afik", C_inv, dC)

    F = jnp.einsum("afij,bfji->ab", C_inv_dC, C_inv_dC)

    delta = jnp.einsum("fij,fjk->fik", C_inv, data - C)

    d_term = jnp.einsum("afij,fji->a", C_inv_dC, delta)

    F_inv = compute_inverse(F)

    res = jnp.einsum("ab,b->a", F_inv, d_term).real

    return res, F_inv


# A function to update the iterative estimate
@jax.jit
def get_update_estimate_full(parameters, data, gamma_IJ_lm, C_ff):
    """
    Compute parameter updates using iterative estimation with full frequency
    correlations.

    This function implements an iterative parameter estimation algorithm that
    updates parameters based on the Fisher information matrix and the gradient
    of the log-likelihood. It accounts for the full frequency-frequency
    correlation structure in the data.

    Parameters:
    -----------
    parameters : Array
        Current parameter estimates (lm coefficients).
    data : Array
        Data tensor with shape (F, F, N, N) where F is the number of frequencies
        and N is the number of pulsars.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    C_ff : Array
        2D array containing the frequency-frequency correlation matrix.
        It has shape (F, F) where F is the number of frequencies.

    Returns:
    --------
    Tuple containing:
    - Array: Parameter updates to be added to the current parameters.
    - Array: Inverse Fisher matrix, which provides uncertainty estimates.
    """

    # Compute the covariance and its derivative
    C = sc.get_covariance_full(parameters, gamma_IJ_lm, C_ff)
    dC = sc.get_dcovariance_full(parameters, gamma_IJ_lm, C_ff)

    # Compute the inverse of the covariance matrix
    C_inv = sc.get_inverse_covariance_full(parameters, gamma_IJ_lm, C_ff)

    # Contract the derivative of the covariance matrix with the inverse of
    # the covariance matrix
    C_inv_dC = jnp.einsum("fgij,agljk->aflik", C_inv, dC)

    F = jnp.einsum("afgij,bgfji->ab", C_inv_dC, C_inv_dC)

    delta = jnp.einsum("fgij,gljk->flik", C_inv, data - C)

    d_term = jnp.einsum("afgij,gfji->a", C_inv_dC, delta)

    F_inv = compute_inverse(F)

    res = jnp.einsum("ab,b->a", F_inv, d_term).real

    return res, F_inv


def iterative_estimation(
    update_function, theta, D_IJ, gamma_IJ_lm, ff, S_f, i_max=100
):
    """
    Perform iterative estimation of parameters.

    This function implements an iterative parameter estimation algorithm that
    updates parameters based on the Fisher information matrix and the gradient
    of the log-likelihood.

    Parameters:
    -----------
    update_function : Callable
        Function to update parameter estimates.
    theta : Array
        Initial parameter estimates.
    D_IJ : Array
        Data tensor with shape (..., N, N) where ... is the frequency part
        (can be 1 or 2d) and N is the number of pulsars.
    gamma_IJ_lm : Array
        3D array containing the response tensor for all pulsar pairs
        decomposed into spherical harmonics. It has shape (L, N, N) where
        L is the number of spherical harmonic coefficients and N is the
        number of pulsars.
    ff : Array
        Frequency array, can be 1 or 2d.
    S_f : Array
        Array containing the signal power spectral density for each frequency
        can be 1 or 2d.
    i_max : int
        Maximum number of iterations.

    Returns:
    --------
    Tuple containing:
    - Array: Final parameter estimates.
    - Array: Uncertainties in the parameter estimates.
    - bool: Convergence flag.
    """

    # Initialize iteration variables
    i = 0
    delta_theta = 1000
    uncertainties = 1e-10

    # Iterate until convergence or maximum iterations reached
    while i < i_max and jnp.max(jnp.abs(delta_theta / uncertainties) > 1e-2):

        # Get the update parameters
        delta_theta, F_inv = update_function(theta, D_IJ, gamma_IJ_lm, ff, S_f)

        # Compute uncertainties
        uncertainties = jnp.sqrt(jnp.diag(F_inv))

        # Update current parameter estimates
        theta += delta_theta

        # Update counter
        i += 1

    # Check for convergence
    converged = True if i < i_max else False

    return theta, uncertainties, converged
