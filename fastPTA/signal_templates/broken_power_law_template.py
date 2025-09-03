# Global
import jax
import jax.numpy as jnp
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut


@partial(jax.jit, static_argnums=(2,))
def broken_power_law(frequency, parameters, smoothing=1.5):
    """
    Returns a broken power law spectrum (BPL).

    The parameters of this template are the log of the amplitude, the log of the
    pivot frequency (in Hz), the tilt of the power law at low frequencies, and
    the tilt of the power law at high frequencies.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the broken power law spectrum.
    smoothing: float, optional
        Some parameter controlling how smooth the transition is in the BPL

    Returns:
    --------
    Array
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


@partial(jax.jit, static_argnums=(0, 3))
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
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the BPL spectrum.
    smoothing: float, optional
        Some parameter controlling how smooth the transition is in the BPL

    Returns:
    --------
    Array
        Array containing the computed derivative of the BPL spectrum with
        respect to the specified parameter.
    """

    # unpack parameters
    _, log_pivot, a, b = parameters

    # compute the model
    model = broken_power_law(frequency, parameters, smoothing=smoothing)

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


# Initialize the signal model
bpl_model = s_ut.Signal_model(
    "broken_power_law",
    broken_power_law,
    dtemplate=d1broken_power_law,
    parameter_names=["log_amplitude", "log_pivot", "low_tilt", "high_tilt"],
    parameter_labels=[
        r"$\log_{10} A_{BPL}$",
        r"$\log_{10} f_{BPL}$",
        r"$\alpha_{BPL}$",
        r"$\beta_{BPL}$",
    ],
)
