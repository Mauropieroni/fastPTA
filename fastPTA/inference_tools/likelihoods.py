# Global imports
import jax
import jax.numpy as jnp

# Local imports
import fastPTA.utils as ut
from fastPTA.inference_tools import signal_covariance as sc
from fastPTA.angular_decomposition import spherical_harmonics as sph


@jax.jit
def log_likelihood_full(parameters, data, gamma_IJ_lm, C_ff):
    """
    Compute the full log-likelihood for the data using a Kronecker product
    structure for the covariance matrix.

    Parameters:
    -----------
    parameters : Array
        Array of power spectrum parameters for the anisotropic signal.
    data : Array
        4D array containing the observed data with shape
        (n_frequencies, n_frequencies, n_pulsars, n_pulsars).
    gamma_IJ_lm : Array
        3D array of spherical harmonics correlations with shape
        (n_coeffs, n_pulsars, n_pulsars), where n_coeffs is the
        number of spherical harmonic coefficients.
    C_ff : Array
        2D array representing the frequency-frequency covariance
        with shape (n_frequencies, n_frequencies).

    Returns:
    --------
    float
        The negative log-likelihood value.
    """

    C_IJ = jnp.einsum("p,pij->ij", parameters, gamma_IJ_lm)

    C_inv = sc.get_inverse_covariance_full(parameters, gamma_IJ_lm, C_ff)

    logdet = ut.logdet_kronecker_product(C_ff, C_IJ)

    data_term = jnp.einsum("mnIJ,nmJI->", C_inv, data)

    return -(logdet + data_term)


# @jax.jit
def log_posterior_full(parameters, Nside, l_max, data, gamma_IJ_lm, C_ff):
    """
    Compute the full log-posterior probability for anisotropic signal.

    This function applies a physical prior that the power spectrum must be
    positive in all pixels of the reconstructed map.

    Parameters:
    -----------
    parameters : Array
        Array of power spectrum parameters for the anisotropic signal.
    Nside : int
        HEALPix Nside parameter controlling the map resolution.
    l_max : int
        Maximum multipole order for the spherical harmonics decomposition.
    data : Array
        4D array containing the observed data with shape
        (n_frequencies, n_frequencies, n_pulsars, n_pulsars).
    gamma_IJ_lm : Array
        3D array of spherical harmonics correlations with shape
        (n_coeffs, n_pulsars, n_pulsars), where n_coeffs is the
        number of spherical harmonic coefficients.
    C_ff : Array
        2D array representing the frequency-frequency covariance
        with shape (n_frequencies, n_frequencies).

    Returns:
    --------
    float
        The log-posterior value, or -infinity if the prior is violated.
    """
    Pk = sph.get_map_from_real_clms(parameters, Nside, l_max=l_max)
    lp = jnp.min(Pk)
    lp = jnp.where(lp < 0, -jnp.inf, lp)

    log_lik = log_likelihood_full(parameters, data, gamma_IJ_lm, C_ff)

    return lp + log_lik


@jax.jit
def prepare_log_likelihood(data, response_IJ, strain_omega):
    """
    Precompute the quantities needed to evaluate the Whittle log-likelihood for
    any signal_value at O(N) cost per frequency instead of O(N^3) using Woodbury
    and matrix-determinant lemmas. The idea is to use

        response_IJ v = lambda * strain_omega v,

    i.e. the eigendecomposition of

        strain_omega^-1/2 response_IJ strain_omega^-1/2 = Q^-1 diag(lambda) Q.
        Q = strain_omega^-1/2 @ eigenvectors

    With this

        C^-1 = Q (I + signal_value * lambda)^-1 Q^T
        logdet(C) = logdet(strain_omega) + sum(log(1 + signal_value * lambda))

    so log_likelihood only needs eigenvalues, logdet(strain_omega), and the
    data projected onto Q, all independent of signal_value.

    Parameters:
    -----------
    data : Array
        Array containing the observed data.
    response_IJ : Array
        Array containing response function.
    strain_omega : Array
        Array containing (diagonal) strain noise.

    Returns:
    --------
    eigenvalues : Array
        Generalized eigenvalues of response_IJ w.r.t. strain_omega, per
        frequency.
    noise_logdet : Array
        Log determinant of strain_omega, per frequency.
    data_eigenbasis : Array
        Data projected onto the generalized eigenbasis Q, per frequency.

    """

    # strain_omega is diagonal, so its diagonal fully describes it
    noise = jnp.diagonal(strain_omega, axis1=-2, axis2=-1)
    inv_sqrt_noise = 1.0 / jnp.sqrt(noise)

    # Whiten response_IJ by the noise so it can be diagonalized in one shot
    whitened_response = (
        inv_sqrt_noise[..., :, None]
        * response_IJ
        * inv_sqrt_noise[..., None, :]
    )

    eigenvalues, eigenvectors = jnp.linalg.eigh(whitened_response)

    # Q simultaneously diagonalizes response_IJ and strain_omega
    Q = inv_sqrt_noise[..., :, None] * eigenvectors

    data_eigenbasis = jnp.einsum("fni,fnm,fmi->fi", Q, data, Q)

    noise_logdet = jnp.sum(jnp.log(noise), axis=-1)

    return eigenvalues, noise_logdet, data_eigenbasis


@jax.jit
def log_likelihood(signal_value, eigenvalues, noise_logdet, data_eigenbasis):
    """
    Compute the logarithm of the likelihood assujming a Whittle likelihood.

    Parameters:
    -----------
    signal_value : Array
        Array containing the signal evaluated in all frequency bins.
    eigenvalues : Array
        Generalized eigenvalues of response_IJ w.r.t. strain_omega, as
        returned by prepare_log_likelihood.
    noise_logdet : Array
        Log determinant of strain_omega, as returned by
        prepare_log_likelihood.
    data_eigenbasis : Array
        Data projected onto the generalized eigenbasis, as returned by
        prepare_log_likelihood.

    Returns:
    --------
    float
        Logarithm of the likelihood.

    """

    # 1 + signal_value * eigenvalues, diagonal of C in the eigenbasis
    denominator = 1.0 + signal_value[..., None] * eigenvalues

    # Log determinant, from the matrix-determinant lemma
    logdet = noise_logdet + jnp.sum(jnp.log(denominator), axis=-1)

    # data term, from the Woodbury identity
    data_term = jnp.abs(jnp.sum(data_eigenbasis / denominator, axis=-1))

    # return the likelihood
    return -jnp.sum(logdet + data_term)


def log_posterior(
    signal_parameters,
    frequency,
    signal_model,
    eigenvalues,
    noise_logdet,
    data_eigenbasis,
    priors,
):
    """
    Compute the logarithm of the posterior probability summing log likelihood
    and prior.

    Parameters:
    -----------
    signal_parameters : Array
        Array containing parameters of the signal model.
    frequency : Array
        Array containing frequency bins.
    signal_model : signal_model object
        Object containing the signal model and its derivatives
    eigenvalues : Array
        Generalized eigenvalues of response_IJ w.r.t. strain_omega, as
        returned by prepare_log_likelihood.
    noise_logdet : Array
        Log determinant of strain_omega, as returned by
        prepare_log_likelihood.
    data_eigenbasis : Array
        Data projected onto the generalized eigenbasis, as returned by
        prepare_log_likelihood.
    priors : prior object
        Object containing the prior probability density functions.

    Returns:
    --------
    float
        Logarithm of the posterior probability.

    """

    # Evaluate the log prior
    lp = priors.evaluate_log_priors(
        dict(zip(signal_model.parameter_names, signal_parameters))
    )

    # If the prior is not finite, return -inf
    if not jnp.isfinite(lp):
        return -jnp.inf

    # Evaluate the signal model
    signal_value = signal_model.template(frequency, signal_parameters)

    # Evaluate the log likelihood
    log_lik = log_likelihood(
        signal_value, eigenvalues, noise_logdet, data_eigenbasis
    )

    # Return log prior + log likelihood
    return lp + log_lik
