# Global
import jax
import jax.numpy as jnp
from functools import partial


# Local
from fastPTA.signal_templates import signal_utils as s_ut


@jax.jit
def flat(frequency, parameters):
    """
    Returns a flat spectrum.

    The only parameter of this template is the log of the amplitude, so if the
    vector of parameters is longer, it will use only the first parameter.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the flat spectrum.

    Returns:
    --------
    Array
        Array containing the computed flat spectrum.

    """

    return 10.0 ** parameters[0] * frequency**0


@partial(jax.jit, static_argnums=(0,))
def d1flat(index, frequency, parameters):
    """
    Derivative of the flat spectrum.

    The only parameter of this template is the log of the amplitude. If index
    is > 0 it will raise an error.

    Parameters:
    -----------
    index : int
        Index of the parameter to differentiate.
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the flat spectrum.

    Returns:
    --------
    Array
        Array containing the computed derivative of the flat spectrum with
            respect to the specified parameter.

    """

    # compute the model
    model = flat(frequency, parameters)

    if index == 0:
        # derivative of the log model w.r.t the log amplitude
        dlog_model = jnp.log(10)

    else:
        # raise an error if the index is not valid
        raise ValueError("Cannot use that for this signal")

    # return the derivative multiplying the log derivative by the model
    return model * dlog_model


# Initialize the signal model
flat_model = s_ut.Signal_model(
    "flat",
    flat,
    dtemplate=d1flat,
    parameter_names=["log_amplitude"],
    parameter_labels=[r"$\alpha$"],
)
