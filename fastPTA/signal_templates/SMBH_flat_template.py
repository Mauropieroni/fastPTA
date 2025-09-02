# Global
import jax
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut
from fastPTA.signal_templates.power_law_template import power_law, d1power_law
from fastPTA.signal_templates.flat_template import flat, d1flat


@partial(jax.jit, static_argnums=())
def SMBH_flat(frequency, parameters):
    """
    Returns a spectrum combining the SMBH and flat models.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude of the flat spectrum.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the SMBH and flat spectra.

    Returns:
    --------
    Array
        Array containing the computed SMBH and flat spectra.
    """
    # compute the SMBH and flat spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    flat_spectrum = flat(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + flat_spectrum


@partial(jax.jit, static_argnums=(0,))
def d1SMBH_flat(index, frequency, parameters):
    """
    Derivative of the SMBH + flat spectrum.

    The parameters of this template are the log of the amplitude, and tilt of
    the power law for SMBHs plus the log of the amplitude of the flat spectrum.
    If index is > 2 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter w.r.t which the derivative is computed.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the SMBH + flat spectra.

    Returns:
    --------
    Array
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


# Initialize the signal model
SMBH_flat_model = s_ut.Signal_model(
    "SMBH_flat",
    SMBH_flat,
    dtemplate=d1SMBH_flat,
    parameter_names=["log_amplitude_PL", "tilt_PL", "log_amplitude_flat"],
    parameter_labels=[
        r"$\log_{10} A_{\rm PL}$",
        r"$n_{\rm T}$",
        r"$\log_{10} A_{\rm flat}$",
    ],
)
