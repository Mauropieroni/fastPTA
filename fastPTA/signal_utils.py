# Global
import jax
import jax.numpy as jnp
import numpy as np

from jax.scipy.interpolate import RegularGridInterpolator

# Local
import fastPTA.utils as ut

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])

jax.config.update("jax_enable_x64", True)


cgx = np.loadtxt(ut.path_to_defaults + "fvals.txt")
cgy = np.loadtxt(ut.path_to_defaults + "cgvals.txt")


cg_interpolator = RegularGridInterpolator([cgx], cgy)
del cgx, cgy


def cg(frequency):
    return 10 ** cg_interpolator(jnp.log10(frequency))


def flat(frequency, parameters):
    """
    Generate a flat spectrum.

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

    return 10 ** parameters[0] * frequency**0


def d1flat(index, frequency, parameters):
    """
    Derivative of the flat spectrum.

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

    if index == 0:
        return flat(frequency, parameters) * jnp.log(10)

    else:
        raise ValueError("Cannot use that for this signal")


def power_law(frequency, parameters, pivot=ut.f_yr):
    """
    Generate a power law spectrum.

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

    return 10**log_amplitude * (frequency / pivot) ** tilt


def d1power_law(index, frequency, parameters, pivot=ut.f_yr):
    """
    Derivative of the power law spectrum.

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

    model = power_law(frequency, parameters, pivot=pivot)

    if index == 0:
        dlog_model = jnp.log(10)

    elif index == 1:
        dlog_model = jnp.log(frequency / pivot)

    else:
        raise ValueError("Cannot use that for this signal")

    return model * dlog_model


def lognormal(frequency, parameters):
    """
    Generate a lognormal spectrum.

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

    return 10**log_amplitude * jnp.exp(
        -0.5 * (jnp.log(frequency / (10**log_pivot)) / 10**log_width) ** 2
    )


def d1lognormal(index, frequency, parameters):
    """
    Derivative of the lognormal spectrum.

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
    log_amplitude, log_width, log_pivot = parameters

    model = lognormal(frequency, parameters)

    if index == 0:
        dlog_model = jnp.log(10)

    elif index == 1:
        dlog_model = (
            jnp.log(10)
            / 10 ** (2 * log_width)
            * (jnp.log(frequency / 10**log_pivot)) ** 2
        )

    elif index == 2:
        dlog_model = (
            jnp.log(10)
            / 10 ** (2 * log_width)
            * (jnp.log(frequency / 10**log_pivot))
        )

    else:
        raise ValueError("Cannot use that for this signal")

    return model * dlog_model


def SMBH_and_flat(frequency, parameters):
    """
    Generate a spectrum combining the SMBH and flat models.

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

    return power_law(frequency, parameters[:2]) + flat(
        frequency, parameters[2:]
    )


def d1SMBH_and_flat(index, frequency, parameters):
    """
    Derivative of the SMBH + flat spectrum.

    Parameters:
    -----------
    index : int
        Index of the parameter with respect to which the derivative is computed.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the SMBH + flat spectra.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the SMBH + flat
        spectra with respect to the specified parameter.
    """

    if index < 2:
        return d1power_law(index, frequency, parameters[:2])

    else:
        return d1flat(index - 2, frequency, parameters[2:])


def SMBH_and_lognormal(frequency, parameters):
    """
    Generate a spectrum combining the SMBH and lognormal models.

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

    return power_law(frequency, parameters[:2]) + lognormal(
        frequency, parameters[2:]
    )


def d1SMBH_and_lognormal(index, frequency, parameters):
    """
    Derivative of the SMBH + lognormal spectrum.

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
        spectra with respect to the specified parameter.
    """

    if index < 2:
        return d1power_law(index, frequency, parameters[:2])

    else:
        return d1lognormal(index - 2, frequency, parameters[2:])


def broken_power_law(frequency, parameters, smoothing=1.5):
    """
    Generate a broken power law spectrum.

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
    alpha, gamma, a, b = parameters

    x = frequency / 10**gamma

    return (
        10**alpha
        * (jnp.abs(a) + jnp.abs(b)) ** smoothing
        / (
            (
                jnp.abs(b) * x ** (-a / smoothing)
                + jnp.abs(a) * x ** (b / smoothing)
            )
            ** smoothing
        )
    )


def d1broken_power_law(index, frequency, parameters, smoothing=1.5):
    """
    Derivative of the BPL spectrum.

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
    alpha, gamma, a, b = parameters

    model = broken_power_law(frequency, parameters)

    if index != 0:
        x = frequency / 10**gamma

    if index == 0:
        dlog_model = jnp.log(10)

    elif index == 1:
        dlog_model = (
            jnp.log(10)
            * a
            * b
            * (-1 + x ** ((a + b) / smoothing))
            / (b + a * x ** ((a + b) / smoothing))
        )

    elif index == 2:
        dlog_model = (
            -((b * smoothing) / (a + b))
            + (b * (smoothing + a * jnp.log(x)))
            / (b + a * x ** ((a + b) / smoothing))
        ) / a

    elif index == 3:
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
        raise ValueError("Cannot use that for this signal")

    return model * dlog_model


def SMBH_and_broken_power_law(frequency, parameters):
    """
    Generate a spectrum combining the SMBH and BPL models.

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

    return power_law(frequency, parameters[:2]) + broken_power_law(
        frequency, parameters[2:]
    )


def d1SMBH_and_broken_power_law(index, frequency, parameters):
    """
    Derivative of the SMBH + BPL spectrum.

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
        spectra with respect to the specified parameter.
    """

    if index < 2:
        return d1power_law(index, frequency, parameters[:2])

    else:
        return d1broken_power_law(index - 2, frequency, parameters[2:])


def tanh(frequency, parameters, pivot=ut.f_yr):
    """
    Generate a tanh spectrum.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the tanh spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed tanh spectrum.
    """

    # unpack parameters
    log_amplitude, tilt = parameters

    return 10**log_amplitude * (1 + jnp.tanh(frequency / pivot)) ** tilt


# def d1tanh(index, frequency, parameters, pivot=ut.f_yr):
#     """
#     Derivative of the tanh spectrum.

#     Parameters:
#     -----------
#     index : int
#         Index of the parameter with respect to which the derivative is computed.
#     frequency : numpy.ndarray or jax.numpy.ndarray
#         Array containing frequency bins.
#     parameters : numpy.ndarray or jax.numpy.ndarray
#         Array containing parameters for the tanh spectrum.

#     Returns:
#     --------
#     numpy.ndarray or jax.numpy.ndarray
#         Array containing the computed derivative of the tanh spectrum with
#         respect to the specified parameter.
#     """

#     # unpack parameters
#     log_amplitude, tilt = parameters

#     dlog_model = []

#     def function_tanh(frequency, log_amplitude, tilt):
#         return (10**log_amplitude) * (1 + jnp.tanh(frequency / pivot)) ** tilt

#     if index == 0:
#         for i in range(len(frequency)):
#             dlog_model.append(
#                 grad(function_tanh, argnums=1)(
#                     frequency[i], log_amplitude, tilt
#                 )
#             )

#     elif index == 1:
#         for i in range(len(frequency)):
#             dlog_model.append(
#                 grad(function_tanh, argnums=2)(
#                     frequency[i], log_amplitude, tilt
#                 )
#             )

#     else:
#         raise ValueError("Cannot use that for this signal")

#     return dlog_model


def SIGW(frequency, parameters):
    """
    Generate a spectrum for SIGW.

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

    x = frequency / (10**log_pivot)
    width = 10**log_width
    k = x * jnp.exp((3 / 2) * width**2)

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

    return cg(frequency) * (10**log_amplitude) ** 2 * (term1 + term2 + term3)


# # The following one is if you want ot compute the derivative numerically
# def d1SIGW(index, frequency, parameters):
#     """
#     Derivative of the tanh spectrum.

#     Parameters:
#     -----------
#     index : int
#         Index of the parameter with respect to which the derivative is computed.
#     frequency : numpy.ndarray or jax.numpy.ndarray
#         Array containing frequency bins.
#     parameters : numpy.ndarray or jax.numpy.ndarray
#         Array containing parameters for the tanh spectrum.

#     Returns:
#     --------
#     numpy.ndarray or jax.numpy.ndarray
#         Array containing the computed derivative of the tanh spectrum with
#         respect to the specified parameter.
#     """

#     # unpack parameters
#     log_amplitude, log_width, log_pivot = parameters

#     def function_SIGW(log_amplitude, log_width, log_pivot):
#         return SIGW(frequency, (log_amplitude, log_width, log_pivot))

#     if index < 3:
#         dlog_model = jacfwd(function_SIGW, argnums=index)(
#             log_amplitude, log_width, log_pivot
#         )

#     else:
#         raise ValueError("Cannot use that for this signal")

#     return dlog_model


def power_law_SIGW(frequency, parameters):
    """
    Generate a spectrum combining the SIGW and flat models.

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

    return power_law(frequency, parameters[:2]) + SIGW(
        frequency, parameters[2:]
    )


# def d1power_law_SIGW(index, frequency, parameters):
#     """
#     Derivative of the SIGW + flat spectrum.

#     Parameters:
#     -----------
#     index : int
#         Index of the parameter to differentiate.
#     frequency : numpy.ndarray or jax.numpy.ndarray
#         Array containing frequency bins.
#     parameters : numpy.ndarray or jax.numpy.ndarray
#         Array containing parameters for the SMBH + flat spectra.

#     Returns:
#     --------
#     numpy.ndarray or jax.numpy.ndarray
#         Array containing the computed derivative of the SMBH + flat
#         spectra with respect to the specified parameter.
#     """

#     if index < 2:
#         return dpower_law(index, frequency, parameters[:2])

#     else:
#         return dSIGW(index - 2, frequency, parameters[2:])


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
