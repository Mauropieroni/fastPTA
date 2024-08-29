# Global
import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut
from fastPTA.signals import SMBBH_parameters, get_model
from fastPTA.get_tensors import get_tensors

# Set some global parameters for jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Default value for signal_lm
default_signal_lm = jnp.array([1.0 / jnp.sqrt(4 * jnp.pi)])


@jax.jit
def get_SNR_integrand(signal_tensor, c_inverse):
    """
    Compute the integrand for the Signal-to-Noise Ratio (SNR) (for some set of
    frequency bins).

    Parameters:
    -----------
    signal_tensor : numpy.ndarray or jax.numpy.ndarray
        3D array containing signal data, assumed to have shape (F, N, N), where
        F is the number of frequency bins and N is the number of pulsars.
    c_inverse : numpy.ndarray or jax.numpy.ndarray
        3D array representing the inverse covariance matrix, assumed to have
        shape (F, N, N), where F is the number of frequency bins and N is the
        number of pulsars.

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


@jax.jit
def get_fisher_integrand(dsignal_tensor, c_inverse):
    """
    Compute the integrand used to compute the Fisher Information Matrix (for
    some set of frequency bins).

    Parameters:
    -----------
    dsignal_tensor : numpy.ndarray or jax.numpy.ndarray
        4D array containing the derivative of the signal data with respect to
        the model parameters. This function assumes dsignal_tensor to have
        shape (P, F, N, N), where P is the number of parameters, F is the number
        of frequency bins, and N is the number of pulsars.
    c_inverse : numpy.ndarray or jax.numpy.ndarray
        3D array representing the inverse covariance matrix, assumed to have
        shape (F, N, N), where F is the number of frequency bins and N is the
        number of pulsars.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
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
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing signal data.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        3D array containing  the response for all frequencies and pulsar pairs.
        Assumed to have shape (F, N, N), where F is the number of frequency bins
        and N is the number of pulsars.
    noise_tensor : numpy.ndarray or jax.numpy.ndarray
        3D array containing  the noise for all frequencies and pulsar pairs.
        Assumed to have shape (F, N, N), where F is the number of frequency bins
        and N is the number of pulsars.
    HD_functions_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where HD_order
        is the maximum order of Legendre polynomials / bins, F is the number of
        frequencies,and N is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - SNR_integrand: numpy.ndarray or jax.numpy.ndarray
        the integrand for the Signal-to-Noise Ratio (SNR) computation.
    - effective_noise: numpy.ndarray or jax.numpy.ndarray
        the effective noise as a function of frequency.
    - fisher_integrand: numpy.ndarray or jax.numpy.ndarray
        3D array containing the integrand to compute the Fisher Information
        Matrix for each combination of parameters. It has shape (N, N, F),
        where N is the number of pulsars and F is the number of frequency bins.

    """

    # Assemble the signal tensor
    signal_tensor = response_IJ * signal[:, None, None]

    # Build the covariance
    covariance = signal_tensor + noise_tensor

    # Invert the covariance
    c_inverse = ut.compute_inverse(covariance)

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


@jax.jit
def get_signal_dsignal_tensors_lm_spherical_harmonics_basis(
    signal_lm, signal, dsignal, response_IJ
):
    """
    Compute the signal and its derivatives in the spherical harmonics basis

    Parameters:
    -----------
    signal_lm: numpy.ndarray or jax.numpy.ndarray
        Array containing the lm coefficients.
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing the signal evaluated at all frequencies.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, F, N, N), where lm is the number of coefficients for
        the anisotropy decomposition (spherical harmonics or sqrt basis), F is
        the number of frequencies, N is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - signal_tensor: numpy.ndarray or jax.numpy.ndarray
        the signal tensor in the spherical harmonics basis.
    - dsignal_tensor_frequency_shape: numpy.ndarray or jax.numpy.ndarray
        the derivative of the signal tensor with respect to the signal
        parameters in frequency space.

    """

    # Assemble the signal tensor, signal_lm are the lm coefficients of the
    # signal and signal is the signal in frequency space (with len = F)
    signal_lm_f = signal_lm[:, None] * signal[None, :]

    # Assemble the signal tensor, the response_IJ tensor has shape
    # (lm, N, pulsars, pulsars)
    signal_tensor = jnp.sum(response_IJ * signal_lm_f[..., None, None], axis=0)

    # Derivatives of signal_lm_f in parameters dsignal is the derivative of the
    # signal in frequency space shape (P, N)
    dsignal_lm_f = signal_lm[None, :, None] * dsignal[:, None, :]

    # Assemble the tensor with signal derivatives with respect to the signal
    # parameters, we thus sum over the lm coefficients
    dsignal_tensor_frequency_shape = jnp.sum(
        response_IJ[None, ...] * dsignal_lm_f[..., None, None], axis=1
    )

    # Just build a kronecher delta in (P, lm) space
    delta = jnp.eye(len(signal_lm_f))

    # Derivatives of signal_lm_f with respect to the lm coefficients
    dsignal_lm_f_anisotropies = delta[..., None] * signal[None, None, ...]

    # Assemble the tensor with signal derivatives with respect to the lm
    # coefficients by contracting dsignal_lm_f3 with the response
    dsignal_tensor_anisotropies = jnp.einsum(
        "ijkl,aij->ajkl", response_IJ, dsignal_lm_f_anisotropies
    )

    return (
        signal_tensor,
        dsignal_tensor_frequency_shape,
        dsignal_tensor_anisotropies,
    )


@jax.jit
def get_signal_dsignal_tensors_lm_sqrt_basis(
    signal_lm, signal, dsignal, response_IJ
):
    """
    Compute the signal and its derivatives in the sqrt basis

    Parameters:
    -----------
    signal_lm: numpy.ndarray or jax.numpy.ndarray
        Array containing the lm coefficients.
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing the signal evaluated at all frequencies.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, N, pulsars, pulsars), where lm is the number of
        coefficients for the anisotropy decomposition (spherical harmonics or
        sqrt basis), N is the number of frequencies, and pulsars is the number
        of pulsars.

    Returns:
    --------
    Tuple containing:
    - signal_tensor: numpy.ndarray or jax.numpy.ndarray
        the signal tensor in the sqrt basis.
    - dsignal_tensor_frequency_shape: numpy.ndarray or jax.numpy.ndarray
        the derivative of the signal tensor with respect to the signal
        parameters in frequency space.
    - dsignal_tensor_anisotropies: numpy.ndarray or jax.numpy.ndarray
        the derivative of the signal tensor with respect to the lm coefficients

    """

    # Assemble the signal tensor, signal_lm are the lm coefficients of the
    # signal and signal is the signal in frequency space (with len = F)
    # response_IJ tensor has shape (lm, lm, F, N, N)
    signal_tensor = signal[:, None, None] * jnp.einsum(
        "abcde,a,b->cde", response_IJ, signal_lm, signal_lm
    )

    # Assemble the tensor with signal derivatives with respect to the signal
    # parameters, we thus sum over the lm coefficients and multiply by dsignal
    # with axes introduced for pulsars
    dsignal_tensor_frequency_shape = (
        dsignal[..., None, None]
        * jnp.einsum("abcde,a,b->cde", response_IJ, signal_lm, signal_lm)[
            None, ...
        ]
    )

    # Assemble the tensor with signal derivatives with respect to the lm
    # coefficients. Note that signal_lm enters quadratically in signal_tensor
    dsignal_tensor_anisotropies = (
        2.0
        * signal[None, :, None, None]
        * jnp.einsum("abcde,b->acde", response_IJ, signal_lm)
    )

    return (
        signal_tensor,
        dsignal_tensor_frequency_shape,
        dsignal_tensor_anisotropies,
    )


lm_basis_list = [
    get_signal_dsignal_tensors_lm_spherical_harmonics_basis,
    # get_signal_dsignal_tensors_lm_sqrt_basis,
]


lm_basis_map = {
    "spherical_harmonics_basis": 0,
    # "sqrt_basis": 1,
}


@jax.jit
def get_signal_dsignal_tensors_lm(
    lm_basis_idx, signal_lm, signal, dsignal, response_IJ
):
    """
    Compute the signal and its derivatives in the spherical harmonics basis

    Parameters:
    -----------
    lm_basis_idx: int
        Index indicating the basis to use for the anisotropy decomposition
    signal_lm: numpy.ndarray or jax.numpy.ndarray
        Array containing the lm coefficients.
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing the signal evaluated at all frequencies.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, F)
        where P is the number of parameters, F is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array containing response tensor for all the pulsars pairs.
        It has shape (lm, F, N, N), where lm is the number of coefficients for
        the anisotropy decomposition (spherical harmonics or sqrt basis), F is
        the number of frequencies, N is the number of pulsars.

    Returns:
    --------
    Tuple containing:
    - signal_tensor: numpy.ndarray or jax.numpy.ndarray
        the signal tensor in the chosen basis.
    - dsignal_tensor_frequency_shape: numpy.ndarray or jax.numpy.ndarray
        the derivative of the signal tensor with respect to the signal
        parameters in frequency space.
    - dsignal_tensor_anisotropies: numpy.ndarray or jax.numpy.ndarray
        the derivative of the signal tensor with respect to the lm coefficients

    """

    return jax.lax.switch(
        lm_basis_idx, lm_basis_list, signal_lm, signal, dsignal, response_IJ
    )


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
    signal_lm: numpy.ndarray or jax.numpy.ndarray
        Array containing the lm coefficients.
    signal : numpy.ndarray or jax.numpy.ndarray
        Array containing signal data.
    dsignal : numpy.ndarray or jax.numpy.ndarray
        2D array containing derivative of the signal data with respect to the
        signal parameters. This function assumes dsignal to have shape (P, N)
        where P is the number of parameters, N is the number of frequency bins
    response_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array containing  the response for all frequencies and pulsar pairs.
        Assumed to have shape (lm, N, M, M), where lm are the spherical
        harmonics coefficients N is the number of frequency bins and M is the
        number of pulsars.
    noise_tensor : numpy.ndarray or jax.numpy.ndarray
        3D array containing  the noise for all frequencies and pulsar pairs.
        Assumed to have shape (N, M, M), where N is the number of frequency
        bins and M is the number of pulsars.
    HD_functions_IJ : numpy.ndarray or jax.numpy.ndarray
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where HD_order
        is the maximum order of Legendre polynomials / bins, F is the number of
        frequencies,and N is the number of pulsars.
    lm_basis_idx: int
        Index indicating the basis to use for the anisotropy decomposition

    Returns:
    --------
    Tuple containing:
    - SNR_integrand: numpy.ndarray or jax.numpy.ndarray
        the integrand for the Signal-to-Noise Ratio (SNR) computation.
    - effective_noise: numpy.ndarray or jax.numpy.ndarray
        the effective noise as a function of frequency.
    - fisher_integrand: numpy.ndarray or jax.numpy.ndarray
        3D array wite the integrand for the Fisher Information Matrix
        computation.

    """

    (
        signal_tensor,
        dsignal_tensor_frequency_shape,
        dsignal_tensor_anisotropies,
    ) = get_signal_dsignal_tensors_lm(
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
    dsignal_tensor_HD = signal[:, None, None] * HD_functions_IJ

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
    signal_label="power_law",
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
        4D array with the Legendre or binned projection of the Hellings and
        Downs correlations. The shape is (HD_order + 1, F, N, N), where HD_order
        is the maximum order of Legendre polynomials / bins, F is the number of
        frequencies,and N is the number of pulsars.
    - HD_coefficients : numpy.ndarray or jax.numpy.ndarray
        Legendre coefficients for Hellings and Downs correlations values up to
        the given HD_order.
    - effective_noise: numpy.ndarray or jax.numpy.ndarray
        effective noise.
    - SNR: float
        Signal-to-Noise Ratio (SNR) value.
    - fisher: numpy.ndarray or jax.numpy.ndarray
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

    lm_basis_idx = lm_basis_map[lm_basis]

    # Setting the frequency vector from the observation time
    frequency = (1.0 + jnp.arange(n_frequencies)) / (T_obs_yrs * ut.yr)

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
