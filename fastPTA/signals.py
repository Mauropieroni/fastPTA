# Global
import jax
import jax.numpy as jnp

import numpy as np

# Local
import fastPTA.utils as ut


jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])
from jax import grad, jacfwd
from scipy import special


# Current SMBBH SGWB log_amplitude best-fit
SMBBH_log_amplitude = -7.1995
SMBBH_tilt = 2

# Current SMBBH SGWB parameters
SMBBH_parameters = jnp.array([SMBBH_log_amplitude, SMBBH_tilt])

# A value for a flat spectrum
CGW_flat_parameters = jnp.array([-7.0])

# Some values for a LN spectrum
LN_log_amplitude = -6.45167492
LN_log_width = -0.91240383
LN_log_pivot = -7.50455732
CGW_LN_parameters = jnp.array([LN_log_amplitude, LN_log_width, LN_log_pivot])

# Some values for a BPL spectrum
BPL_log_amplitude = -5.8
BPL_log_width = -8.0
BPL_tilt_1 = 3
BPL_tilt_2 = 1.5
CGW_BPL_parameters = jnp.array(
    [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, BPL_tilt_2]
)

### Some values for a Tanh spectrum
Tanh_log_amplitude = -8.0
Tanh_tilt = 8.0
CGW_Tanh_parameters = jnp.array([Tanh_log_amplitude, Tanh_tilt])

### Some values for a SIGW spectrum
SIGW_log_amplitude = -1.7
SIGW_log_width = np.log10(0.5)
SIGW_log_pivot = -7.8
CGW_SIGW_parameters = jnp.array(
    [SIGW_log_amplitude, SIGW_log_width, SIGW_log_pivot]
)


cgx = np.loadtxt(ut.path_to_defaults + "fvals.txt")
cgy = np.loadtxt(ut.path_to_defaults + "cgvals.txt")


def cg(f):
    return 10 ** jnp.interp(np.log10(f), cgx, cgy)


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


def dflat(index, frequency, parameters):
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


def dpower_law(index, frequency, parameters, pivot=ut.f_yr):
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

    # unpack parameters
    log_amplitude, tilt = parameters

    model = power_law(frequency, parameters)

    if index == 0:
        dlog_model = jnp.log(10)

    elif index == 1:
        dlog_model = jnp.log(frequency / pivot)

    else:
        raise ValueError("Cannot use that for this signal")

    return model * dlog_model


## Check analytical and numerical derivative
## To use grad() we need a function of the parameters

# def function_powerlaw(log_amplitude, tilt, pivot):
#     return 10**log_amplitude * (frequency_point / pivot) ** tilt

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# for i in range(len(X)):
#     Y.append(dpower_law(0, frequency_point, [X[i], SMBBH_tilt], ut.f_yr))
#     Ynum.append(grad(function_powerlaw, argnums=0)(X[i], SMBBH_tilt, ut.f_yr))
#     Z.append(dpower_law(1, frequency_point, [SMBBH_log_amplitude, X[i]], ut.f_yr))
#     Znum.append(grad(function_powerlaw, argnums=1)(SMBBH_log_amplitude, X[i], ut.f_yr))
# plt.loglog(X, Y, label='$LogAmplitude - analytic$')
# plt.loglog(X, Ynum, color='yellow', linestyle='dashed', label='$LogAmplitude - numeric$')
# plt.loglog(X, Z, label='$Tilt - analytic$')
# plt.loglog(X,Znum, color='cyan', linestyle='dashed', label='$Tilt - numeric$')
# plt.legend(loc='center left', bbox_to_anchor=(.1, .8), prop={'size': 10})
# plt.show()


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


def dlognormal(index, frequency, parameters):
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


## Check analytical and numerical derivative
## To use grad() we need a function of the parameters

# def function_lognormal(log_amplitude, log_width, log_pivot):
#     return 10**log_amplitude * jnp.exp(
#         -0.5 * (jnp.log(frequency_point / (10**log_pivot)) / 10**log_width) ** 2
#     )

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# W = []
# Wnum = []
# for i in range(len(X)):
#     Y.append(dlognormal(0, frequency_point, [X[i], LN_log_width, LN_log_pivot]))
#     Ynum.append(grad(function_lognormal, argnums=0)(X[i], LN_log_width, LN_log_pivot))
#     Z.append(dlognormal(1, frequency_point, [LN_log_amplitude, X[i], LN_log_pivot]))
#     Znum.append(grad(function_lognormal, argnums=1)(LN_log_amplitude, X[i], LN_log_pivot))
#     W.append(dlognormal(2, frequency_point, [LN_log_amplitude, LN_log_width, -X[i]]))
#     Wnum.append(grad(function_lognormal, argnums=2)(LN_log_amplitude, LN_log_width, -X[i]))
# plt.loglog(X,Y, label="$LogAmplitude - Analytic$")
# plt.loglog(X,Ynum, color='yellow', linestyle='dashed', label="$LogAmplitude - Numeric$")
# plt.loglog(X,Z, label="$LogWidth - Analytic$")
# plt.loglog(X, Znum, color='cyan', linestyle='dashed', label="$LogWidth - Numeric$")
# plt.legend(loc='center left', bbox_to_anchor=(.1, 0.2), prop={'size': 10})
# plt.show()
# plt.plot(-X, W, label="$LogPivot - Analytic$")
# plt.plot(-X, Wnum, color='yellow', linestyle='dashed', label="$LogPivot - Numeric$")
# plt.legend(loc='center left', bbox_to_anchor=(.5, 0.2), prop={'size': 10})
# plt.show()


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


def dSMBH_and_flat(index, frequency, parameters):
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
        return dpower_law(index, frequency, parameters[:2])

    else:
        return dflat(index - 2, frequency, parameters[2:])


# Check analytical and numerical derivative
## Already done for the two separate functions


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


def dSMBH_and_lognormal(index, frequency, parameters):
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
        return dpower_law(index, frequency, parameters[:2])

    else:
        return dlognormal(index - 2, frequency, parameters[2:])


# Check analytical and numerical derivative
## Already done for the two separate functions


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


def dbroken_power_law(index, frequency, parameters, smoothing=1.5):
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

    # elif index == 4:
    #     dlog_model = (
    #         jnp.log(a + b)
    #         + (a * b * (-1 + x ** ((a + b) / smoothing)) * jnp.log(x))
    #         / (smoothing * (b + a * x ** ((a + b) / smoothing)))
    #         - jnp.log(b / (x) ** (a / smoothing) + a * (x) ** (b / smoothing))
    #     )

    else:
        raise ValueError("Cannot use that for this signal")

    return model * dlog_model


# Check analytical and numerical derivative
# To use grad() we need a function of the parameters

# def function_bpl(alpha, gamma, a, b):

#     smoothing = 1.5
#     x = frequency_point / 10**gamma

#     return (
#         10**alpha
#         * (jnp.abs(a) + jnp.abs(b)) ** smoothing
#         / ((
#             jnp.abs(b) * x ** (-a / smoothing)
#             + jnp.abs(a) * x ** (b / smoothing)
#         )
#         ** smoothing)
#     )

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# W = []
# Wnum = []
# K = []
# Knum = []
# for i in range(len(X)):
#     Y.append(dbroken_power_law(0, frequency_point, [X[i], BPL_log_width, BPL_tilt_1, BPL_tilt_2]))
#     Ynum.append(grad(function_bpl, argnums=0)(X[i], BPL_log_width, BPL_tilt_1, BPL_tilt_2))
#     Z.append(dbroken_power_law(1, frequency_point, [BPL_log_amplitude, X[i], BPL_tilt_1, BPL_tilt_2]))
#     Znum.append(grad(function_bpl, argnums=1)(BPL_log_amplitude, X[i], BPL_tilt_1, BPL_tilt_2))
#     W.append(dbroken_power_law(2, frequency_point, [BPL_log_amplitude, BPL_log_width, X[i], BPL_tilt_2]))
#     Wnum.append(grad(function_bpl, argnums=2)(BPL_log_amplitude, BPL_log_width, X[i], BPL_tilt_2))
#     K.append(dbroken_power_law(3, frequency_point, [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, X[i]]))
#     Knum.append(grad(function_bpl, argnums=3)(BPL_log_amplitude, BPL_log_width, BPL_tilt_1, X[i]))
# plt.plot(X,Y, label="$LogAmplitude - Analytic$")
# plt.plot(X,Ynum, color='yellow', linestyle='dashed', label="$LogAmplitude - Numeric$")
# plt.legend(loc='center left', bbox_to_anchor=(.1, 0.2), prop={'size': 10})
# plt.show()
# plt.plot(X,Z, label="$LogWidth - Analytic$")
# plt.plot(X, Znum, color='cyan', linestyle='dashed', label="$LogWidth - Numeric$")
# plt.legend(loc='center left', bbox_to_anchor=(.1, 0.2), prop={'size': 10})
# plt.show()
# plt.plot(X,W, label="$Tilt_1 - Analytic$")
# plt.plot(X, Wnum, color='yellow', linestyle='dashed', label="$Tilt_1 - Numeric$")
# plt.plot(X,K, label="$Tilt_2 - Analytic$")
# plt.plot(X, Knum, color='blue', linestyle='dashed', label="$Tilt_2 - Numeric$")
# plt.legend(loc='center left', bbox_to_anchor=(.1, 0.2), prop={'size': 10})
# plt.show()


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


def dSMBH_and_broken_power_law(index, frequency, parameters):
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
        return dpower_law(index, frequency, parameters[:2])

    else:
        return dbroken_power_law(index - 2, frequency, parameters[2:])


# Check analytical and numerical derivative
## Already done for the two separate functions


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

    ### unpack parameters
    log_amplitude, tilt = parameters

    return (10**log_amplitude) * (1 + jnp.tanh(frequency / pivot)) ** tilt


# def dtanh(index, frequency, parameters, pivot=ut.f_yr):
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

#     ### unpack parameters
#     log_amplitude, tilt = parameters

#     model = tanh(frequency, parameters, pivot=ut.f_yr)

#     if index == 0:
#         dlog_model = jnp.log(10)

#     elif index == 1:
#         dlog_model = (
#             jnp.log(1 + jnp.tanh(frequency/pivot))
#         )

#     else:
#         raise ValueError("Cannot use that for this signal")

#     return model * dlog_model

# def function_tanh(frequency, log_amplitude, tilt):
#     return (10**log_amplitude) * (1+jnp.tanh(frequency/ut.f_yr))**tilt

# The following one is if you want ot compute the derivative numerically


def dtanh(index, frequency, parameters, pivot=ut.f_yr):
    """
    Derivative of the tanh spectrum.

    Parameters:
    -----------
    index : int
        Index of the parameter with respect to which the derivative is computed.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the tanh spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the tanh spectrum with
        respect to the specified parameter.
    """

    ### unpack parameters
    log_amplitude, tilt = parameters

    model = tanh(frequency, parameters, pivot=ut.f_yr)

    dlog_model = []

    def function_tanh(frequency, log_amplitude, tilt):
        return (10**log_amplitude) * (1 + jnp.tanh(frequency / ut.f_yr)) ** tilt

    if index == 0:
        for i in range(len(frequency)):
            dlog_model.append(
                grad(function_tanh, argnums=1)(
                    frequency[i], log_amplitude, tilt
                )
            )

    elif index == 1:
        for i in range(len(frequency)):
            dlog_model.append(
                grad(function_tanh, argnums=2)(
                    frequency[i], log_amplitude, tilt
                )
            )

    else:
        raise ValueError("Cannot use that for this signal")

    return dlog_model


# Check analytical and numerical derivative
# To use grad() we need a function of the parameters

# def function_tanh(log_amplitude, tilt):
#     return (10**log_amplitude) * (1+jnp.tanh(frequency_point/ut.f_yr))**tilt

# X = np.arange(1, 10, 0.01)
# Y = []
# Ynum = []
# Z = []
# Znum = []
# for i in range(len(X)):
#     Y.append(dtanh(0, frequency_point, [X[i], Tanh_tilt], ut.f_yr))
#     Ynum.append(grad(function_tanh, argnums=0)(X[i], Tanh_tilt))
#     Z.append(dtanh(1, frequency_point, [Tanh_log_amplitude, X[i]], ut.f_yr))
#     Znum.append(grad(function_tanh, argnums=1)(Tanh_log_amplitude, X[i]))
# plt.loglog(X, Y, label='$LogAmplitude - analytic$')
# plt.loglog(X, Ynum, color='yellow', linestyle='dashed', label='$LogAmplitude - numeric$')
# plt.loglog(X, Z, label='$Tilt - analytic$')
# plt.loglog(X,Znum, color='cyan', linestyle='dashed', label='$Tilt - numeric$')
# plt.legend(loc='center left', bbox_to_anchor=(.05, .85), prop={'size': 10})
# plt.show()


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

    ### unpack parameters
    log_amplitude, log_width, log_pivot = parameters

    x = frequency / (10**log_pivot)
    width = 10**log_width
    k = x * jnp.exp((3 / 2) * width**2)

    return (
        cg(frequency)
        * (10**log_amplitude) ** 2
        * (
            (
                (4 / (5 * jnp.sqrt(np.pi)))
                * x**3
                * (1 / width)
                * jnp.exp((9 * width**2) / 4)
            )
            * (
                (jnp.log(k) ** 2 + (1 / 2) * width**2)
                * jax.scipy.special.erfc(
                    (1 / width) * (jnp.log(k) + (1 / 2) * jnp.log(3 / 2))
                )
                - (width / (jnp.sqrt(np.pi)))
                * jnp.exp(
                    -((jnp.log(k) + (1 / 2) * jnp.log(3 / 2)) ** 2) / (width**2)
                )
                * (jnp.log(k) - (1 / 2) * jnp.log(3 / 2))
            )
            + (0.0659 / (width**2))
            * x**2
            * jnp.exp(width**2)
            * jnp.exp(
                -((jnp.log(x) + width**2 - (1 / 2) * jnp.log(4 / 3)) ** 2)
                / (width**2)
            )
            + (1 / 3)
            * jnp.sqrt(2 / np.pi)
            * x ** (-4)
            * (1 / width)
            * jnp.exp(8 * width**2)
            * jnp.exp(-(jnp.log(x) ** 2) / (2 * width**2))
            * jax.scipy.special.erfc(
                (4 * width**2 - jnp.log(x / 4)) / (jnp.sqrt(2) * width)
            )
        )
    )


# The following one is if you want ot compute the derivative numerically
def dSIGW(index, frequency, parameters):
    """
    Derivative of the tanh spectrum.

    Parameters:
    -----------
    index : int
        Index of the parameter with respect to which the derivative is computed.
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.
    parameters : numpy.ndarray or jax.numpy.ndarray
        Array containing parameters for the tanh spectrum.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the computed derivative of the tanh spectrum with
        respect to the specified parameter.
    """

    ### unpack parameters
    log_amplitude, log_width, log_pivot = parameters

    def function_SIGW(log_amplitude, log_width, log_pivot):
        return SIGW(frequency, (log_amplitude, log_width, log_pivot))

    if index < 3:
        dlog_model = jacfwd(function_SIGW, argnums=index)(
            log_amplitude, log_width, log_pivot
        )

    else:
        raise ValueError("Cannot use that for this signal")

    return dlog_model


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


def dpower_law_SIGW(index, frequency, parameters):
    """
    Derivative of the SIGW + flat spectrum.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
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
        return dpower_law(index, frequency, parameters[:2])

    else:
        return dSIGW(index - 2, frequency, parameters[2:])


# Check analytical and numerical derivative
## Already done for the two separate functions


def get_model(signal_label):
    """
    Retrieve signal and derivative models based on the specified label.

    Parameters:
    -----------
    signal_label : str
        Label indicating the type of signal model.

    Returns:
    --------
    dict
        Dictionary containing the signal model and its derivative model.

    Notes:
    ------
    Supported signal labels are:
        - "flat":
            Flat signal model.
        - "power_law":
            Power law signal model.
        - "lognormal":
            Log-normal signal model.
        - "power_law_flat":
            Signal model combining SMBH and flat spectrum.
        - "power_law_lognormal":
            Signal model combining SMBH and log-normal spectrum.
        - "bpl":
            Broken power law signal model.
        - "power_law_broken_power_law":
            Signal model combining SMBH and broken power law spectrum.
        - "tanh":
            Signal model with a Tanh.
        - "SIGW":
            Signal model for scalar induce GW
        - "power_law_SIGW";
            Signal model combining SMBH and SIGW

    """

    if signal_label == "flat":
        signal = {"signal_model": flat, "dsignal_model": dflat}

    elif signal_label == "power_law":
        signal = {"signal_model": power_law, "dsignal_model": dpower_law}

    elif signal_label == "lognormal":
        signal = {"signal_model": lognormal, "dsignal_model": dlognormal}

    elif signal_label == "power_law_flat":
        signal = {
            "signal_model": SMBH_and_flat,
            "dsignal_model": dSMBH_and_flat,
        }

    elif signal_label == "power_law_lognormal":
        signal = {
            "signal_model": SMBH_and_lognormal,
            "dsignal_model": dSMBH_and_lognormal,
        }

    elif signal_label == "bpl":
        signal = {
            "signal_model": broken_power_law,
            "dsignal_model": dbroken_power_law,
        }

    elif signal_label == "power_law_broken_power_law":
        signal = {
            "signal_model": SMBH_and_broken_power_law,
            "dsignal_model": dSMBH_and_broken_power_law,
        }

    elif signal_label == "tanh":
        signal = {"signal_model": tanh, "dsignal_model": dtanh}

    elif signal_label == "SIGW":
        signal = {"signal_model": SIGW, "dsignal_model": dSIGW}

    elif signal_label == "power_law_SIGW":
        signal = {
            "signal_model": power_law_SIGW,
            "dsignal_model": dpower_law_SIGW,
        }

    else:
        raise ValueError("Cannot use", signal_label)

    return signal
