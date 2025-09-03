# Global
import jax
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut
from fastPTA.signal_templates.power_law_template import power_law, d1power_law
from fastPTA.signal_templates.broken_power_law_template import (
    broken_power_law,
    d1broken_power_law,
)


@partial(jax.jit, static_argnums=(2,))
def SMBH_broken_power_law(frequency, parameters, smoothing=1.5):
    """
    Returns a spectrum combining the SMBH and BPL models.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the pivot
    frequency (in Hz), the tilt of the power law at low frequencies, and the
    tilt of the power law at high frequencies.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the SMBH and BPL spectra.
    smoothing: float, optional
        Parameter controlling how smooth the transition is in the BPL.
        Default is 1.5.

    Returns:
    --------
    Array
        Array containing the sum of the computed SMBH and BPL spectra.
    """
    # compute the SMBH and BPL spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    broken_power_law_spectrum = broken_power_law(
        frequency, parameters[2:], smoothing=smoothing
    )

    # return the combined spectrum
    return power_law_spectrum + broken_power_law_spectrum


@partial(jax.jit, static_argnums=(0, 3))
def d1SMBH_broken_power_law(index, frequency, parameters, smoothing=1.5):
    """
    Derivative of the SMBH + BPL spectrum.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the pivot
    frequency (in Hz), the tilt of the power law at low frequencies, and the
    tilt of the power law at high frequencies. If index is > 5 it will raise an
    error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the SMBH + BPL spectra.
    smoothing: float, optional
        Parameter controlling how smooth the transition is in the BPL.
        Default is 1.5.

    Returns:
    --------
    Array
        Array containing the computed derivative of the SMBH + BPL
        spectra w.r.t the specified parameter.
    """
    if index < 2:
        # compute the derivative of the SMBH spectrum
        dmodel = d1power_law(index, frequency, parameters[:2])
    else:
        # compute the derivative of the BPL spectrum
        dmodel = d1broken_power_law(
            index - 2, frequency, parameters[2:], smoothing=smoothing
        )

    # return the derivative
    return dmodel


# Initialize the signal model
SMBH_bpl_model = s_ut.Signal_model(
    "SMBH_broken_power_law",
    SMBH_broken_power_law,
    dtemplate=d1SMBH_broken_power_law,
    parameter_names=[
        "log_amplitude_PL",
        "tilt_PL",
        "log_amplitude_BPL",
        "log_pivot_BPL",
        "low_tilt_BPL",
        "high_tilt_BPL",
    ],
    parameter_labels=[
        r"$\log_{10} A_{\rm PL}$",
        r"$n_{\rm T}$",
        r"$\log_{10} A_{\rm BPL}$",
        r"$\log_{10} f_{\rm BPL}$",
        r"$\alpha_{\rm BPL}$",
        r"$\beta_{\rm BPL}$",
    ],
)
