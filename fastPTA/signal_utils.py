# Global
import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy.interpolate import RegularGridInterpolator

# Local
import fastPTA.utils as ut


# Set the device
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

# Enable 64-bit precision
jax.config.update("jax_enable_x64", True)


SIGWB_prefactor_data = np.loadtxt(
    ut.path_to_defaults + "SIGWB_prefactor_data.txt"
)

# This is building an interpolator for the SIGWB prefactor
SIGWB_prefactor_interpolator = RegularGridInterpolator(
    [SIGWB_prefactor_data[:, 0]], SIGWB_prefactor_data[:, 1]
)


del SIGWB_prefactor_data


def SIGWB_prefactor(frequency):
    """
    Returns the prefactor appearing in Eqs. 9 - 10 of xxxxx as a function of
    frequency using the interpolated values from the data file.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the prefactor for the SIGWB spectrum.
    """

    # The data are in log scale so we log the frequency and then exp the result
    return 10 ** SIGWB_prefactor_interpolator(jnp.log10(frequency))


def get_gradient(npars, function, frequency, parameters, *args, **kwargs):
    """
    Derivative of the power law spectrum.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the power law spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the power law spectrum.
        The shape of the array is (len(frequency), npars).
    """

    return jnp.array(
        [
            function(i, frequency, parameters, *args, **kwargs)
            for i in range(npars)
        ]
    ).T


def flat(frequency, parameters):
    """
    Returns a flat spectrum.

    The only parameter of this template is the log of the amplitude, so if the
    vector of parameters is longer, it will use only the first parameter.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the flat spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed flat spectrum.
    """

    return 10.0 ** parameters[0] * frequency**0


def d1flat(index, frequency, parameters):
    """
    Derivative of the flat spectrum.

    The only parameter of this template is the log of the amplitude. If index
    is > 0 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the flat spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the flat spectrum with
            respect to the specified parameter.
    """

    # compute the model
    model = flat(frequency, parameters)

    if index == 0:
        # derivative of the log model w.r.t the log amplitude
        dlog_model = jnp.log(10)

    else:
        # raise an error if the index is not valid
        raise ValueError("Cannot use that for this signal")

    # return the derivative multiplying the log derivative by the model
    return model * dlog_model


def power_law(frequency, parameters, pivot=ut.f_yr):
    """
    Returns a power law spectrum.

    The parameters of this template are the log of the amplitude and the tilt
    of the power law.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the power law spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed power law spectrum.
    """

    # unpack parameters
    log_amplitude, tilt = parameters

    # return the power law spectrum
    return 10**log_amplitude * (frequency / pivot) ** tilt


def d1power_law(index, frequency, parameters, pivot=ut.f_yr):
    """
    Derivative of the power law spectrum.

    The parameters of this template are the log of the amplitude and the tilt
    of the power law. If index is > 1 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the power law spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the power law spectrum with
        respect to the specified parameter.
    """

    # compute the model
    model = power_law(frequency, parameters, pivot=pivot)

    if index == 0:
        # derivative of the log of the model w.r.t the log amplitude
        dlog_model = jnp.log(10)

    elif index == 1:
        # derivative of the log of the model w.r.t the tilt
        dlog_model = jnp.log(frequency / pivot)

    else:
        # raise an error if the index is not valid
        raise ValueError("Cannot use that for this signal")

    # return the derivative multiplying the log derivative by the model
    return model * dlog_model


def lognormal(frequency, parameters):
    """
    Returns a lognormal spectrum.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal spectrum.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the lognormal spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed lognormal spectrum.
    """

    # unpack parameters
    log_amplitude, log_width, log_pivot = parameters

    # compute the exponent
    to_exp = -0.5 * (jnp.log(frequency / 10**log_pivot) / 10**log_width) ** 2

    # return the lognormal spectrum
    return 10**log_amplitude * jnp.exp(to_exp)


def d1lognormal(index, frequency, parameters):
    """
    Derivative of the lognormal spectrum.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal spectrum.
    If index is > 2 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the lognormal spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the lognormal spectrum with
        respect to the specified parameter.
    """

    # unpack parameters
    _, log_width, log_pivot = parameters

    # compute the model
    model = lognormal(frequency, parameters)

    if index == 0:
        # derivative of the log of the model w.r.t the log amplitude
        dlog_model = jnp.log(10)

    elif index == 1:
        # derivative of the log of the model w.r.t the log width
        numerator = jnp.log(10) * jnp.log(frequency / 10**log_pivot) ** 2
        denominator = 10 ** (2 * log_width)
        dlog_model = numerator / denominator

    elif index == 2:
        # derivative of the log of the model w.r.t the log pivot frequency
        numerator = jnp.log(10) * jnp.log(frequency / 10**log_pivot)
        denominator = 10 ** (2 * log_width)
        dlog_model = numerator / denominator

    else:
        # raise an error if the index is not valid
        raise ValueError("Cannot use that for this signal")

    # return the derivative multiplying the log derivative by the model
    return model * dlog_model


def broken_power_law(frequency, parameters, smoothing=1.5):
    """
    Returns a broken power law spectrum (BPL).

    The parameters of this template are the log of the amplitude, the log of the
    pivot frequency (in Hz), the tilt of the power law at low frequencies, and
    the tilt of the power law at high frequencies.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the broken power law spectrum.
    smoothing: float, optional
        Some parameter controlling how smooth the transition is in the BPL
    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed broken power law spectrum.
    """

    # unpack parameters
    log_amplitude, log_pivot, a, b = parameters

    # rescale the frequency to the pivot frequency
    x = frequency / 10**log_pivot

    # compute the numerator and denominator
    numerator = (jnp.abs(a) + jnp.abs(b)) ** smoothing
    denominator = (
        jnp.abs(b) * x ** (-a / smoothing) + jnp.abs(a) * x ** (b / smoothing)
    ) ** smoothing

    # return the BPL spectrum
    return 10**log_amplitude * numerator / denominator


def d1broken_power_law(index, frequency, parameters, smoothing=1.5):
    """
    Derivative of the BPL spectrum.

    The parameters of this template are the log of the amplitude, the log of the
    pivot frequency (in Hz), the tilt of the power law at low frequencies, and
    the tilt of the power law at high frequencies. If index is > 3 it will raise
    an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the BPL spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the BPL spectrum with
        respect to the specified parameter.
    """

    # unpack parameters
    log_amplitude, log_pivot, a, b = parameters

    # compute the model
    model = broken_power_law(frequency, parameters)

    if index != 0:
        # rescale the frequency to the pivot frequency
        x = frequency / 10**log_pivot

    if index == 0:
        # derivative of the log of the model w.r.t the log amplitude
        dlog_model = jnp.log(10)

    elif index == 1:
        # derivative of the log of the model w.r.t the log pivot frequency
        dlog_model = (
            jnp.log(10)
            * a
            * b
            * (-1 + x ** ((a + b) / smoothing))
            / (b + a * x ** ((a + b) / smoothing))
        )

    elif index == 2:
        # derivative of the log of the model w.r.t the tilt at low frequencies
        dlog_model = (
            -((b * smoothing) / (a + b))
            + (b * (smoothing + a * jnp.log(x)))
            / (b + a * x ** ((a + b) / smoothing))
        ) / a

    elif index == 3:
        # derivative of the log of the model w.r.t the tilt at high frequencies
        dlog_model = (
            -a
            * (
                smoothing
                - x ** ((a + b) / smoothing)
                * (smoothing - (a + b) * jnp.log(x))
            )
            / ((a + b) * (b + a * x ** ((a + b) / smoothing)))
        )

    else:
        # raise an error if the index is not valid
        raise ValueError("Cannot use that for this signal")

    # return the derivative multiplying the log derivative by the model
    return model * dlog_model


def SMBH_and_flat(frequency, parameters):
    """
    Returns a spectrum combining the SMBH and flat models.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude of the flat spectrum.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH and flat spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed SMBH and flat spectra.
    """

    # compute the SMBH and flat spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    flat_spectrum = flat(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + flat_spectrum


def d1SMBH_and_flat(index, frequency, parameters):
    """
    Derivative of the SMBH + flat spectrum.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude of the flat spectrum.
    If index is > 2 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter w.r.t which the derivative is computed.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH + flat spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the SMBH + flat spectra
        w.r.t the specified parameter.
    """

    if index < 2:
        # compute the derivative of the SMBH spectrum
        dmodel = d1power_law(index, frequency, parameters[:2])

    else:
        # compute the derivative of the flat spectrum
        dmodel = d1flat(index - 2, frequency, parameters[2:])

    return dmodel


def SMBH_and_lognormal(frequency, parameters):
    """
    Returns a spectrum combining the SMBH and lognormal models.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the width,
    and the log of the pivot frequency (in Hz) of the lognormal spectrum. If
    index is > 2 it will raise an error.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH and lognormal spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed SMBH and lognormal spectra.
    """

    # compute the SMBH and lognormal spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    lognormal_spectrum = lognormal(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + lognormal_spectrum


def d1SMBH_and_lognormal(index, frequency, parameters):
    """
    Derivative of the SMBH + lognormal spectrum.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the width,
    and the log of the pivot frequency (in Hz) of the lognormal spectrum. If
    index is > 4 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH + lognormal spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the SMBH + lognormal
        spectra w.r.t the specified parameter.
    """

    if index < 2:
        # compute the derivative of the SMBH spectrum
        dmodel = d1power_law(index, frequency, parameters[:2])

    else:
        # compute the derivative of the lognormal spectrum
        dmodel = d1lognormal(index - 2, frequency, parameters[2:])

    # return the derivative
    return dmodel


def SMBH_and_broken_power_law(frequency, parameters):
    """
    Returns a spectrum combining the SMBH and BPL models.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the pivot
    frequency (in Hz), the tilt of the power law at low frequencies, and the
    tilt of the power law at high frequencies.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH and lognormal spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the sum of the computed SMBH and BPL spectra.
    """

    # compute the SMBH and BPL spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    broken_power_law_spectrum = broken_power_law(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + broken_power_law_spectrum


def d1SMBH_and_broken_power_law(index, frequency, parameters):
    """
    Derivative of the SMBH + BPL spectrum.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the pivot
    frequency (in Hz), the tilt of the power law at low frequencies, and the
    tilt of the power law at high frequencies. If index is > 6 it will raise an
    error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH + BPL spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the SMBH + BPL
        spectra w.r.t the specified parameter.
    """

    if index < 2:
        # compute the derivative of the SMBH spectrum
        dmodel = d1power_law(index, frequency, parameters[:2])

    else:
        # compute the derivative of the BPL spectrum
        dmodel = d1broken_power_law(index - 2, frequency, parameters[2:])

    # return the derivative
    return dmodel


def SIGW_broad_approximated(frequency, parameters):
    """
    Returns the analytical approximation of the SIGW for a broad a lognormal
    scalar spectrum (originally proposed in 2005.12306) see eq. 9 of xxxxx.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal scalar
    spectrum.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the lognormal scalar spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the analytical approximation for the SIGW.
    """

    # unpack parameters
    log_amplitude, log_width, log_pivot = parameters

    # rescale the frequency to the pivot frequency
    x = frequency / (10**log_pivot)

    # get the width
    width = 10**log_width

    # compute the k parameter
    k = x * jnp.exp((3 / 2) * width**2)

    # compute the three terms
    term1 = (
        (4 / (5 * jnp.sqrt(np.pi)))
        * x**3
        * (1 / width)
        * jnp.exp((9 * width**2) / 4)
    ) * (
        (jnp.log(k) ** 2 + (1 / 2) * width**2)
        * jax.scipy.special.erfc(
            (1 / width) * (jnp.log(k) + (1 / 2) * jnp.log(3 / 2))
        )
        - (width / (jnp.sqrt(np.pi)))
        * jnp.exp(-((jnp.log(k) + (1 / 2) * jnp.log(3 / 2)) ** 2) / (width**2))
        * (jnp.log(k) - (1 / 2) * jnp.log(3 / 2))
    )

    term2 = (
        (0.0659 / (width**2))
        * x**2
        * jnp.exp(width**2)
        * jnp.exp(
            -((jnp.log(x) + width**2 - (1 / 2) * jnp.log(4 / 3)) ** 2)
            / (width**2)
        )
    )
    term3 = (
        (1 / 3)
        * jnp.sqrt(2 / np.pi)
        * x ** (-4)
        * (1 / width)
        * jnp.exp(8 * width**2)
        * jnp.exp(-(jnp.log(x) ** 2) / (2 * width**2))
        * jax.scipy.special.erfc(
            (4 * width**2 - jnp.log(x / 4)) / (jnp.sqrt(2) * width)
        )
    )

    # return the SIGW spectrum
    return (
        SIGWB_prefactor(frequency)
        * (10**log_amplitude) ** 2
        * (term1 + term2 + term3)
    )


def power_law_SIGW_broad_approximated(frequency, parameters):
    """
    Returns a spectrum combining the power law and the analytical approximation
    of the SIGW for a broad a lognormal scalar spectrum (originally proposed in
    2005.12306) see eq. 9 of xxxxx.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal scalar
    spectrum.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH and lognormal spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed SMBH and lognormal spectra.
    """

    # compute the power law and SIGW spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    SIGW_spectrum = SIGW_broad_approximated(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + SIGW_spectrum
