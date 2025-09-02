# Global
import jax
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut
from fastPTA.signal_templates.power_law_template import power_law, d1power_law
from fastPTA.signal_templates.lognormal_template import lognormal, d1lognormal


@jax.jit
def SMBH_lognormal(frequency, parameters):
    """
    Returns a spectrum combining the SMBH and lognormal models.

    The parameters of this template are the log of the amplitude and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the width,
    and the log of the pivot frequency (in Hz) of the lognormal spectrum.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the SMBH and lognormal spectra.

    Returns:
    --------
    Array
        Array containing the computed SMBH and lognormal spectra.
    """
    # compute the SMBH and lognormal spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    lognormal_spectrum = lognormal(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + lognormal_spectrum


@partial(jax.jit, static_argnums=(0,))
def d1SMBH_lognormal(index, frequency, parameters):
    """
    Derivative of the SMBH + lognormal spectrum.

    The parameters of this template are the log of the amplitude and tilt of
    the power law for SMBHs plus the log of the amplitude, the log of the width,
    and the log of the pivot frequency (in Hz) of the lognormal spectrum.
    If index is > 4 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the SMBH + lognormal spectra.

    Returns:
    --------
    Array
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


# Initialize the signal model
SMBH_lognormal_model = s_ut.Signal_model(
    "SMBH_lognormal",
    SMBH_lognormal,
    dtemplate=d1SMBH_lognormal,
    parameter_names=[
        "log_amplitude_PL",
        "tilt_PL",
        "log_amplitude_LN",
        "log_width_LN",
        "log_pivot_LN",
    ],
    parameter_labels=[
        r"$\log_{10} A_{\rm PL}$",
        r"$n_{\rm T}$",
        r"$\log_{10} A_{\rm LN}$",
        r"$\log_{10} \sigma_{\rm LN}$",
        r"$\log_{10} f_{p, \rm LN}$",
    ],
)
