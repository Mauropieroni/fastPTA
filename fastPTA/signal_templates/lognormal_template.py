# Global
import jax
import jax.numpy as jnp
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut


@jax.jit
def lognormal(frequency, parameters):
    """
    Returns a lognormal spectrum.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal spectrum.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the lognormal spectrum.

    Returns:
    --------
    Array
        Array containing the computed lognormal spectrum.
    """
    # unpack parameters
    log_amplitude, log_width, log_pivot = parameters

    # compute the exponent
    to_exp = -0.5 * (jnp.log(frequency / 10**log_pivot) / 10**log_width) ** 2

    # return the lognormal spectrum
    return 10**log_amplitude * jnp.exp(to_exp)


@partial(jax.jit, static_argnums=(0,))
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
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the lognormal spectrum.

    Returns:
    --------
    Array
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


# Initialize the signal model
lognormal_model = s_ut.Signal_model(
    "lognormal",
    lognormal,
    dtemplate=d1lognormal,
    parameter_names=["log_amplitude", "log_width", "log_pivot"],
    parameter_labels=[
        r"$\log_{10} A_{\rm LN}$",
        r"$\log_{10} \sigma_{\rm LN}$",
        r"$\log_{10} f_{p, \rm LN}$",
    ],
)
