# Global
import jax
import jax.numpy as jnp
from functools import partial


# Local
from fastPTA import utils as ut
from fastPTA.signal_templates import signal_utils as s_ut


@partial(jax.jit, static_argnums=(2,))
def power_law(frequency, parameters, pivot=ut.f_yr):
    """
    Returns a power law spectrum.

    The parameters of this template are the log of the amplitude and the tilt
    of the power law.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the power law spectrum.

    Returns:
    --------
    Array
        Array containing the computed power law spectrum.

    """

    # unpack parameters
    log_amplitude, tilt = parameters

    # return the power law spectrum
    return 10**log_amplitude * (frequency / pivot) ** tilt


@partial(jax.jit, static_argnums=(0, 3))
def d1power_law(index, frequency, parameters, pivot=ut.f_yr):
    """
    Derivative of the power law spectrum.

    The parameters of this template are the log of the amplitude and the tilt
    of the power law. If index is > 1 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the power law spectrum.

    Returns:
    --------
    Array
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


# Initialize the signal model
power_law_model = s_ut.Signal_model(
    "power_law",
    power_law,
    dtemplate=d1power_law,
    parameter_names=["log_amplitude", "tilt"],
    parameter_labels=[r"$\alpha_{\rm PL}$", r"$n_{\rm T}$"],
)
