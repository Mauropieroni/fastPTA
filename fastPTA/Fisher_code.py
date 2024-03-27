# Setting the path to this file
import os, sys

file_path = os.path.dirname(__file__)
if file_path:
    file_path += "/"

sys.path.append(os.path.join(file_path, "../fastPTA/"))

# Local
from utils import *
from signals import SMBBH_parameters, get_model
from get_tensors import get_tensors


@jit
def get_SNR_integrand(signal_tensor, c_inverse):
    """
    Compute the integrand for the Signal-to-Noise Ratio (SNR) (for some set of
    frequency bins).

    Parameters:
    -----------
    signal_tensor : numpy.ndarray or jax.numpy.ndarray
        3D array containing signal data, assumed to have shape (N, M, M), where
        N is the number of frequency bins and M is the number of pulsars.
    c_inverse : numpy.ndarray or jax.numpy.ndarray
        Inverse covariance matrix, assumed to have shape (N, M, M), where
        N is the number of frequency bins and M is the number of pulsars.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the integrand of Signal-to-Noise Ratio (SNR)
        (for each frequency bin).

    """

    # Builnding the C matrix
    c_bar_SNR = jnp.einsum("ijk,ikl->ijl", c_inverse, signal_tensor)

    # Contracting the 2 matrixes
    return jnp.einsum("ijk,ikj->i", c_bar_SNR, c_bar_SNR)


@jit
def get_fisher_integrand(dsignal_tensor, c_inverse):
    """
    Compute the integrand used to compute the Fisher Information Matrix (for
    some set of frequency bins).

    Parameters:
    -----------
    dsignal_tensor : numpy.ndarray or jax.numpy.ndarray
        4D array containing the derivative of the signal data with respect to
        the model parameters. This function assumes dsignal_tensor to have
        shape(P, N, M, M), where P is the number of parameters, N is the number
        of frequency bins, and M is the number of pulsars.

    c_inverse : numpy.ndarray or jax.numpy.ndarray
        Inverse covariance matrix. This function assumes c_inverse to have
        shape (N, M, M), where P is the number of parameters, N is the number
        of frequency bins, and M is the number of pulsars.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the integrand to compute the Fisher Information Matrix
        for each combination of parameters.

    """

    # Building the C matrix for the fisher
    c_bar = jnp.einsum("ijk,aikl->aijl", c_inverse, dsignal_tensor)

    # Contracting the 2 matrixes
    return jnp.einsum("aijk,bikj->abi", c_bar, c_bar)


@jit
def get_integrands(
    signal,
    dsignal,
    response_IJ,
    noise_tensor,
    HD_functions_IJ,
):
    """
    Compute integrands for Signal-to-Noise Ratio (SNR) and Fisher Information
    Matrix given some signal data, derivatives and other quantities that
    characterize the pulsar configuration used for the analysis

    Parameters:
    -----------
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing signal data.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, N)
        where P is the number of parameters, N is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        3D array containing  the response for all frequencies and pulsar pairs.
        Assumed to have shape (N, M, M), where N is the number of frequency
        bins and M is the number of pulsars.
    noise_tensor : numpy.ndarray or jax.numpy.ndarray
        3D array containing  the noise for all frequencies and pulsar pairs.
        Assumed to have shape (N, M, M), where N is the number of frequency
        bins and M is the number of pulsars.
    HD_functions_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array containing the HD functions for all parameters, frequencies
        and pulsar pairs. Assumed to have shape (P, N, M, M), where P is the
        number of HD coefficients, N is the number of frequency bins and M is
        the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - SNR_integrand: numpy.ndarray or jax.numpy.ndarray
        the integrand for the Signal-to-Noise Ratio (SNR) computation.
    - effective_noise: numpy.ndarray or jax.numpy.ndarray
        the effective noise as a function of frequency.
    - fisher_integrand: numpy.ndarray or jax.numpy.ndarray
        the integrand for the Fisher Information Matrix computation.
    """

    # Assemble the signal tensor
    signal_tensor = response_IJ * signal[:, None, None]

    # Build the covariance
    covariance = signal_tensor + noise_tensor

    # Invert the covariance
    c_inverse = compute_inverse(covariance)

    # This is the SNR integrand
    SNR_integrand = get_SNR_integrand(signal_tensor, c_inverse)

    # Build the effective noise
    effective_noise = jnp.sqrt(signal**2 / SNR_integrand)

    # Assemble the tensor with signal derivativesz
    dsignal_tensor = dsignal[..., None, None] * response_IJ[None, ...]

    # Append HD functions
    dsignal_tensor = jnp.concatenate(
        (
            dsignal_tensor,
            signal[..., None, None] * HD_functions_IJ,
        ),
        axis=0,
    )

    # Get the fisher integrand
    fisher_integrand = get_fisher_integrand(dsignal_tensor, c_inverse)

    return SNR_integrand, effective_noise, fisher_integrand


# @jit
def get_integrands_lm(
    signal_lm,
    signal,
    dsignal,
    response_IJ,
    noise_tensor,
):
    """
    Compute integrands for Signal-to-Noise Ratio (SNR) and Fisher Information
    Matrix given some signal data, derivatives and other quantities that
    characterize the pulsar configuration used for the analysis

    Parameters:
    -----------
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing signal data.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, N)
        where P is the number of parameters, N is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        $D array containing  the response for all frequencies and pulsar pairs.
        Assumed to have shape (lm, N, M, M), where lm are the spherical
        harmonics coefficients N is the number of frequency bins and M is the
        number of pulsars.
    noise_tensor : numpy.ndarray or jax.numpy.ndarray
        3D array containing  the noise for all frequencies and pulsar pairs.
        Assumed to have shape (N, M, M), where N is the number of frequency
        bins and M is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - SNR_integrand: numpy.ndarray or jax.numpy.ndarray
        the integrand for the Signal-to-Noise Ratio (SNR) computation.
    - effective_noise: numpy.ndarray or jax.numpy.ndarray
        the effective noise as a function of frequency.
    - fisher_integrand: numpy.ndarray or jax.numpy.ndarray
        the integrand for the Fisher Information Matrix computation.
    """

    signal_lm_f = signal_lm[:, None] * signal[None, :]

    # Assemble the signal tensor
    signal_tensor = jnp.sum(response_IJ * signal_lm_f[..., None, None], axis=0)

    # Build the covariance
    covariance = signal_tensor + noise_tensor

    # Invert the covariance
    c_inverse = compute_inverse(covariance)

    # This is the SNR integrand
    SNR_integrand = get_SNR_integrand(signal_tensor, c_inverse)

    # Build the effective noise
    effective_noise = jnp.sqrt(signal**2 / SNR_integrand)

    dsignal_lm_f = signal_lm[None, :, None] * dsignal[:, None, :]

    # Assemble the tensor with signal derivatives
    dsignal_tensor1 = jnp.sum(
        response_IJ[None, ...] * dsignal_lm_f[..., None, None], axis=1
    )

    # derivatives of signal_lm_f in parameters
    signal_lm_f_no_monopole = signal_lm_f[1:]
    delta = jnp.eye(len(signal_lm_f_no_monopole))
    dsignal_lm_f2 = delta[..., None] * signal[None, None, ...]

    dsignal_tensor2 = jnp.einsum(
        "ijkl,aij->ajkl", response_IJ[1:], dsignal_lm_f2
    )

    # derivatives of signal_lm_f in parameters
    # dsignal_tensor2 = jnp.einsum("ijkl,aij->ajkl", response_IJ, dsignal_lm_f)

    # Append HD functions
    dsignal_tensor = jnp.concatenate((dsignal_tensor1, dsignal_tensor2), axis=0)

    # Get the fisher integrand
    fisher_integrand = get_fisher_integrand(dsignal_tensor, c_inverse)

    return SNR_integrand, effective_noise, fisher_integrand


def compute_fisher(
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_label="power_law",
    signal_parameters=SMBBH_parameters,
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
):
    """
    Compute Fisher Information and related quantities. Keyword arguments
    for get_tensors and generate_pulsars_catalog can be provided via
    get_tensors_kwargs and generate_catalog_kwargs.

    Parameters:
    -----------
    T_obs_yrs : float, optional
        Observation time in years
        default is 10.33
    n_frequencies : int, optional
        Number of frequency bins
        default is 30
    signal_label : str, optional
        Label indicating the type of signal model to use
        default is "power_law".
    signal_parameters : dict, optional
        Dictionary containing parameters for the signal model
        default is SMBBH_parameters.
    get_tensors_kwargs : dict
        Additional keyword arguments for the get_tensors function.
    generate_catalog_kwargs : dict
        Additional keyword arguments for the generate_catalog function.

    Returns:
    --------
    Tuple containing:
    - frequency: numpy.ndarray or jax.numpy.ndarray
        frequency bins.
    - signal: numpy.ndarray or jax.numpy.ndarray
        the computed signal.
    - HD_functions_IJ : numpy.ndarray or jax.numpy.ndarray
        Hellings and Downs correlations functions projected onto Legendre
        polynomials or binned intervals.
    - HD_coefficients : numpy.ndarray or jax.numpy.ndarray
        Legendre coefficients for Hellings and Downs correlations values up to
        the given order.
    - effective_noise: numpy.ndarray or jax.numpy.ndarray
        effective noise.
    - SNR: float
        Signal-to-Noise Ratio (SNR) value.
    - fisher: numpy.ndarray or jax.numpy.ndarray
        Fisher Information Matrix.

    """

    if "anisotropies" in get_tensors_kwargs.keys():
        anisotropies = get_tensors_kwargs["anisotropies"]
    else:
        anisotropies = False

    if "lm_order" in get_tensors_kwargs.keys():
        lm_order = get_tensors_kwargs["lm_order"]
    else:
        lm_order = False

    # Setting the frequency vector from the observation time
    T_tot = T_obs_yrs * yr
    fmin = 1 / T_tot
    frequency = fmin * (1 + jnp.arange(n_frequencies))

    # Get the functions for the signal and its derivatives
    model = get_model(signal_label)
    signal_model = model["signal_model"]
    dsignal_model = model["dsignal_model"]

    # Computing the signal
    signal = signal_model(frequency, signal_parameters)

    # Building the signal derivatives
    dsignal = jnp.array(
        [
            dsignal_model(i, frequency, signal_parameters)
            for i in range(0, len(signal_parameters))
        ]
    )

    # Gets all the ingredients to compute the fisher
    strain_omega, response_IJ, HD_functions_IJ, HD_coefficients = get_tensors(
        frequency, **get_tensors_kwargs, **generate_catalog_kwargs
    )

    if anisotropies:
        signal_lm = np.zeros((1 + lm_order) ** 2)
        signal_lm[0] = 1.0

        # Computes the fisher
        SNR_integrand, effective_noise, fisher_integrand = get_integrands_lm(
            signal_lm,
            signal,
            dsignal,
            response_IJ,
            strain_omega,
        )

    else:
        # Computes the fisher
        SNR_integrand, effective_noise, fisher_integrand = get_integrands(
            signal,
            dsignal,
            response_IJ,
            strain_omega,
            HD_functions_IJ,
        )

    # Compute SNR and Fisher integrals
    SNR = jnp.sqrt(T_tot * simps(SNR_integrand, x=frequency, axis=-1))
    fisher = T_tot * simps(fisher_integrand, x=frequency, axis=-1)

    return (
        frequency,
        signal,
        HD_functions_IJ,
        HD_coefficients,
        effective_noise,
        SNR,
        fisher,
    )
