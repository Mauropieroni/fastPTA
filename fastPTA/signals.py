# Global
import jax
import jax.numpy as jnp

# Local
import fastPTA.utils as ut


jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])


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
BPL_log_width = -7.3
BPL_tilt_1 = 3
BPL_tilt_2 = 1.5
CGW_BPL_parameters = jnp.array(
    [BPL_log_amplitude, BPL_log_width, BPL_tilt_1, BPL_tilt_2]
)


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
            jnp.abs(b) * x ** (-a / smoothing)
            + jnp.abs(a) * x ** (b / smoothing)
        )
        ** smoothing
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
                * (smoothing + (a + b) * jnp.log(x))
            )
            / ((a + b) * (b + a * x ** ((a + b) / smoothing)))
        )

    elif index == 4:
        dlog_model = (
            jnp.log(a + b)
            + (a * b * (-1 + x ** ((a + b) / smoothing)) * jnp.log(x))
            / (smoothing * (b + a * x ** ((a + b) / smoothing)))
            - jnp.log(b / (x) ** (a / smoothing) + a * (x) ** (b / smoothing))
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
        return dflat(index - 2, frequency, parameters[2:])


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
        - "power_law":
            Power law signal model.
        - "lognormal":
            Log-normal signal model.
        - "power_law_flat":
            Signal model combining SMBH and flat spectrum.
        - "power_law_lognormal":
            Signal model combining SMBH and log-normal spectrum.
        - "power_law_broken_power_law":
            Signal model combining SMBH and broken power law spectrum.
    """

    if signal_label == "power_law":
        signal = {"signal_model": power_law, "dsignal_model": dpower_law}

    elif signal_label == "flat":
        signal = {"signal_model": flat, "dsignal_model": dflat}

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

    elif signal_label == "power_law_broken_power_law":
        signal = {
            "signal_model": SMBH_and_broken_power_law,
            "dsignal_model": dSMBH_and_broken_power_law,
        }

    else:
        raise ValueError("Cannot use", signal_label)

    return signal
