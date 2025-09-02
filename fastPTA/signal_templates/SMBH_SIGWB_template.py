# Global
import jax
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut
from fastPTA.signal_templates.power_law_template import power_law
from fastPTA.signal_templates.SIGWB_template import SIGWB


@partial(jax.jit, static_argnums=())
def SMBH_SIGWB_model(frequency, parameters):
    """
    Returns a spectrum combining the power law and the analytical approximation
    of the SIGW for a broad a lognormal scalar spectrum (originally proposed in
    2005.12306) see eq. 9 of 2503.10805.

    The parameters of this template are the log of the amplitude and tilt of the
    power law plus the log of the amplitude, the log of the width, and the log
    of the pivot frequency (in Hz) of the lognormal scalar spectrum.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the power law and SIGW spectra.

    Returns:
    --------
    Array
        Array containing the computed power law and SIGW spectra.
    """
    # compute the power law and SIGW spectra
    power_law_spectrum = power_law(frequency, parameters[:2])
    SIGW_spectrum = SIGWB(frequency, parameters[2:])

    # return the combined spectrum
    return power_law_spectrum + SIGW_spectrum


# Initialize the signal model
SMBH_SIGWB_model = s_ut.Signal_model(
    "SMBH_SIGWB_model",
    SMBH_SIGWB_model,
    dtemplate=None,
    parameter_names=[
        "log_amplitude_PL",
        "tilt_PL",
        "log_amplitude_SIGW",
        "log_width_SIGW",
        "log_pivot_SIGW",
    ],
    parameter_labels=[
        r"$\log_{10} A_{\rm PL}$",
        r"$n_{\rm T}$",
        r"$\log_{10} A_{\mathcal{P}}$",
        r"$\log_{10} \sigma$",
        r"$\log_{10} f_p$",
    ],
)
