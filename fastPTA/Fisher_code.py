# Global
import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut
from fastPTA.signal_templates.signal_utils import SMBBH_parameters
from fastPTA.signals import get_signal_model
from fastPTA.get_tensors import get_tensors
from fastPTA.inference_tools import signal_covariance as sc


# Set the device
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


# Default value for signal_lm
default_signal_lm = jnp.array([1.0 / jnp.sqrt(4 * jnp.pi)])
power_law_model = get_signal_model("power_law")


@jax.jit
def get_SNR_integrand(signal_tensor, c_inverse):
    """
    Compute the integrand for the Signal-to-Noise Ratio (SNR) (for some set of
    frequency bins).

    Parameters:
    -----------
    signal_tensor : Array
        3D array containing signal data, assumed to have shape (F, N, N), where
        F is the number of frequency bins and N is the number of pulsars.
    c_inverse : Array
        3D array representing the inverse covariance matrix, assumed to have
        shape (F, N, N), where F is the number of frequency bins and N is the
        number of pulsars.

    Returns:
    --------
    Array
        Array containing the integrand of Signal-to-Noise Ratio (SNR)
        (for each frequency bin).

    """

    # Builnding the C matrix
    c_bar_SNR = jnp.einsum("ijk,ikl->ijl", c_inverse, signal_tensor)

    # Contracting the 2 matrixes
    return jnp.einsum("ijk,ikj->i", c_bar_SNR, c_bar_SNR)


@jax.jit
def get_fisher_integrand(dsignal_tensor, c_inverse):
    """
    Compute the integrand used to compute the Fisher Information Matrix (for
    some set of frequency bins).

    Parameters:
    -----------
    dsignal_tensor : Array
        4D array containing the derivative of the signal data with respect to
        the model parameters. This function assumes dsignal_tensor to have
        shape (P, F, N, N), where P is the number of parameters, F is the number
        of frequency bins, and N is the number of pulsars.
    c_inverse : Array
        3D array representing the inverse covariance matrix, assumed to have
        shape (F, N, N), where F is the number of frequency bins and N is the
        number of pulsars.

    Returns:
    --------
    Array
        3D array containing the integrand to compute the Fisher Information
        Matrix for each combination of parameters. It has shape (N, N, F),
        where N is the number of pulsars and F is the number of frequency bins.

    """

    # Building the C matrix for the fisher
    c_bar = jnp.einsum("ijk,aikl->aijl", c_inverse, dsignal_tensor)

    # Contracting the 2 matrixes
    return jnp.einsum("aijk,bikj->abi", c_bar, c_bar)


@jax.jit
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
    signal : Array
        Array containing signal data.
    dsignal : Array
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : Array
        3D array containing  the response for all frequencies and pulsar pairs.
        Assumed to have shape (F, N, N), where F is the number of frequency bins
        and N is the number of pulsars.
    noise_tensor : Array
        3D array containing  the noise for all frequencies and pulsar pairs.
        Assumed to have shape (F, N, N), where F is the number of frequency bins
        and N is the number of pulsars.
    HD_functions_IJ : Array
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where HD_order
        is the maximum order of Legendre polynomials / bins, F is the number of
        frequencies,and N is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - SNR_integrand: Array
        the integrand for the Signal-to-Noise Ratio (SNR) computation.
    - effective_noise: Array
        the effective noise as a function of frequency.
    - fisher_integrand: Array
        3D array containing the integrand to compute the Fisher Information
        Matrix for each combination of parameters. It has shape (N, N, F),
        where N is the number of pulsars and F is the number of frequency bins.

    """

    # Assemble the signal tensor
    signal_tensor = jnp.einsum("i,ijk->ijk", signal, response_IJ)

    # Build the covariance
    covariance = signal_tensor + noise_tensor

    # Invert the covariance
    c_inverse = ut.compute_inverse(covariance)

    # This is the SNR integrand
    SNR_integrand = get_SNR_integrand(signal_tensor, c_inverse)

    # Build the effective noise
    effective_noise = jnp.sqrt(signal**2 / SNR_integrand)

    # Assemble the tensor with signal derivatives
    dsignal_tensor = jnp.einsum("ai,ijk->aijk", dsignal, response_IJ)

    # # Assemble the HD part
    HD_tensor = jnp.einsum("i,...ijk->...ijk", signal, HD_functions_IJ)

    # Append HD functions
    dsignal_tensor = jnp.concatenate(
        (
            dsignal_tensor,
            HD_tensor,
        ),
        axis=0,
    )

    # Get the fisher integrand
    fisher_integrand = get_fisher_integrand(dsignal_tensor, c_inverse)

    return SNR_integrand, effective_noise, fisher_integrand


@jax.jit
def get_integrands_lm(
    signal_lm,
    signal,
    dsignal,
    response_IJ,
    noise_tensor,
    HD_functions_IJ,
    lm_basis_idx,
):
    """
    Compute integrands for Signal-to-Noise Ratio (SNR) and Fisher Information
    Matrix given some signal data, derivatives and other quantities that
    characterize the pulsar configuration used for the analysis

    Parameters:
    -----------
    signal_lm: Array
        Array containing the lm coefficients.
    signal : Array
        Array containing signal data.
    dsignal : Array
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, N)
        where P is the number of parameters, N is the number of frequency bins
    response_IJ : Array
        4D array containing  the response for all frequencies and pulsar pairs.
        Assumed to have shape (lm, N, M, M), where lm are the spherical
        harmonics coefficients N is the number of frequency bins and M is the
        number of pulsars.
    noise_tensor : Array
        3D array containing  the noise for all frequencies and pulsar pairs.
        Assumed to have shape (N, M, M), where N is the number of frequency
        bins and M is the number of pulsars.
    HD_functions_IJ : Array
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where HD_order
        is the maximum order of Legendre polynomials / bins, F is the number of
        frequencies,and N is the number of pulsars.
    lm_basis_idx: int
        Index indicating the basis to use for the anisotropy decomposition

    Returns:
    --------
    Tuple containing:
    - SNR_integrand: Array
        the integrand for the Signal-to-Noise Ratio (SNR) computation.
    - effective_noise: Array
        the effective noise as a function of frequency.
    - fisher_integrand: Array
        3D array wite the integrand for the Fisher Information Matrix
        computation.

    """

    (
        signal_tensor,
        dsignal_tensor_frequency_shape,
        dsignal_tensor_anisotropies,
    ) = sc.get_signal_dsignal_tensors_lm(
        lm_basis_idx, signal_lm, signal, dsignal, response_IJ
    )

    # Build the covariance
    covariance = signal_tensor + noise_tensor

    # Invert the covariance
    c_inverse = ut.compute_inverse(covariance)

    # This is the SNR integrand
    SNR_integrand = get_SNR_integrand(signal_tensor, c_inverse)

    # Build the effective noise
    effective_noise = jnp.sqrt(signal**2 / SNR_integrand)

    # Assemble the HD coefficients part, we multiply the monopole, which is
    # given by signal and the HD functions
    dsignal_tensor_HD = jnp.einsum("f,...fab->...fab", signal, HD_functions_IJ)

    # Concatenate the three tensors along the parameter axis
    dsignal_tensor = jnp.concatenate(
        (
            dsignal_tensor_frequency_shape,
            dsignal_tensor_HD,
            dsignal_tensor_anisotropies,
        ),
        axis=0,
    )

    # Get the fisher integrand
    fisher_integrand = get_fisher_integrand(dsignal_tensor, c_inverse)

    # Return all the relevant quantities
    return SNR_integrand, effective_noise, fisher_integrand


def compute_fisher(
    T_obs_yrs=10.33,
    n_frequencies=30,
    signal_model=power_law_model,
    signal_parameters=SMBBH_parameters,
    signal_lm=default_signal_lm,
    get_tensors_kwargs={},
    generate_catalog_kwargs={},
):
    """
    Compute Fisher Information and related quantities. Keyword arguments for
    get_tensors and generate_pulsars_catalog can be provided via
    get_tensors_kwargs and generate_catalog_kwargs.

    Parameters:
    -----------
    T_obs_yrs : float, optional
        Observation time in years
        default is 10.33
    n_frequencies : int, optional
        Number of frequency bins
        default is 30
    signal_model : signal_model object, optional
        Object containing the signal model and its derivatives
        Default is a power_law model
    signal_parameters : dict, optional
        Dictionary containing parameters for the signal model
        default is SMBBH_parameters.
    signal_lm : Array, optional
        Array containing the lm coefficients for anisotropic signals.
        Default is default_signal_lm (monopole only).
    get_tensors_kwargs : dict
        Additional keyword arguments for the get_tensors function.
    generate_catalog_kwargs : dict
        Additional keyword arguments for the generate_catalog function.

    Returns:
    --------
    Tuple containing:
    - frequency: Array
        frequency bins.
    - signal: Array
        the computed signal.
    - HD_functions_IJ : Array
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where HD_order
        is the maximum order of Legendre polynomials / bins, F is the number of
        frequencies,and N is the number of pulsars.
    - HD_coefficients : Array
        Legendre coefficients for Hellings and Downs correlations values up to
        the given HD_order.
    - effective_noise: Array
        effective noise.
    - SNR: float
        Signal-to-Noise Ratio (SNR) value.
    - fisher: Array
        2D array with the Fisher Information Matrix.

    """

    if "anisotropies" in get_tensors_kwargs.keys():
        anisotropies = get_tensors_kwargs["anisotropies"]
        if "lm_basis" in get_tensors_kwargs.keys():
            lm_basis = get_tensors_kwargs["lm_basis"]
        else:
            lm_basis = "spherical_harmonics_basis"
    else:
        anisotropies = False
        lm_basis = "spherical_harmonics_basis"

    lm_basis_idx = sc.lm_basis_map[lm_basis]

    # Setting the frequency vector from the observation time
    frequency = (1.0 + jnp.arange(n_frequencies)) / (T_obs_yrs * ut.yr)

    # Compute the signal
    signal = signal_model.template(frequency, signal_parameters)

    # Building the signal derivatives we transpose to have shape (P, F) where P
    # is the number of parameters and F is the number of frequencies
    dsignal = signal_model.gradient(frequency, signal_parameters).T

    # Gets all the ingredients to compute the fisher
    strain_omega, response_IJ, HD_functions_IJ, HD_coefficients = get_tensors(
        frequency, **get_tensors_kwargs, **generate_catalog_kwargs
    )

    if anisotropies:
        # Computes the fisher
        SNR_integrand, effective_noise, fisher_integrand = get_integrands_lm(
            signal_lm,
            signal,
            dsignal,
            response_IJ,
            strain_omega,
            HD_functions_IJ,
            lm_basis_idx,
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
    SNR = jnp.sqrt(jnp.sum(SNR_integrand, axis=-1))
    fisher = jnp.sum(fisher_integrand, axis=-1)

    return (
        frequency,
        signal,
        HD_functions_IJ,
        HD_coefficients,
        effective_noise,
        SNR,
        fisher,
    )
