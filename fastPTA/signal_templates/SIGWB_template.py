# Global
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.interpolate import RegularGridInterpolator

# Local
from fastPTA import utils as ut
from fastPTA.signal_templates import signal_utils as s_ut

SIGWB_prefactor_data = np.loadtxt(
    ut.path_to_defaults + "SIGWB_prefactor_data.txt"
)

# This is building an interpolator for the SIGWB prefactor
SIGWB_prefactor_interpolator = RegularGridInterpolator(
    [SIGWB_prefactor_data[:, 0]], SIGWB_prefactor_data[:, 1]
)


del SIGWB_prefactor_data


def SIGWB_prefactor(frequency):
    """
    Returns the prefactor appearing in Eqs. 9 - 10 of xxxxx as a function of
    frequency using the interpolated values from the data file.

    Parameters:
    -----------
    frequency : numpy.ndarray or jax.numpy.ndarray
        Array containing frequency bins.

    Returns:
    --------
    numpy.ndarray or jax.numpy.ndarray
        Array containing the prefactor for the SIGWB spectrum.
    """

    # The data are in log scale so we log the frequency and then exp the result
    return 10 ** SIGWB_prefactor_interpolator(jnp.log10(frequency))


@partial(jax.jit, static_argnums=())
def SIGWB(frequency, parameters):
    """
    Returns the analytical approximation of the SIGW for a broad a lognormal
    scalar spectrum (originally proposed in 2005.12306) see eq. 9 of 2503.10805.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal scalar
    spectrum.

    Parameters:
    -----------
    frequency : Array
        Array containing frequency bins.
    parameters : Array
        Array containing parameters for the lognormal scalar spectrum.

    Returns:
    --------
    Array
        Array containing the analytical approximation for the SIGW.
    """

    # unpack parameters
    log_amplitude, log_width, log_pivot = parameters

    # rescale the frequency to the pivot frequency
    x = frequency / (10**log_pivot)

    # get the width
    width = 10**log_width

    # compute the k parameter
    k = x * jnp.exp((3 / 2) * width**2)

    # compute the three terms
    term1 = (
        (4 / (5 * jnp.sqrt(np.pi)))
        * x**3
        * (1 / width)
        * jnp.exp((9 * width**2) / 4)
    ) * (
        (jnp.log(k) ** 2 + (1 / 2) * width**2)
        * jax.scipy.special.erfc(
            (1 / width) * (jnp.log(k) + (1 / 2) * jnp.log(3 / 2))
        )
        - (width / (jnp.sqrt(np.pi)))
        * jnp.exp(-((jnp.log(k) + (1 / 2) * jnp.log(3 / 2)) ** 2) / (width**2))
        * (jnp.log(k) - (1 / 2) * jnp.log(3 / 2))
    )

    term2 = (
        (0.0659 / (width**2))
        * x**2
        * jnp.exp(width**2)
        * jnp.exp(
            -((jnp.log(x) + width**2 - (1 / 2) * jnp.log(4 / 3)) ** 2)
            / (width**2)
        )
    )
    term3 = (
        (1 / 3)
        * jnp.sqrt(2 / np.pi)
        * x ** (-4)
        * (1 / width)
        * jnp.exp(8 * width**2)
        * jnp.exp(-(jnp.log(x) ** 2) / (2 * width**2))
        * jax.scipy.special.erfc(
            (4 * width**2 - jnp.log(x / 4)) / (jnp.sqrt(2) * width)
        )
    )

    # return the SIGW spectrum
    return (
        SIGWB_prefactor(frequency)
        * (10**log_amplitude) ** 2
        * (term1 + term2 + term3)
    )


# Initialize the signal model
SIGWB_model = s_ut.Signal_model(
    "SIGWB",
    SIGWB,
    parameter_names=["log_amplitude", "log_width", "log_pivot"],
    parameter_labels=[
        r"$\log_{10} A_{\mathcal{P}}$",
        r"$\log_{10} \sigma$",
        r"$\log_{10} f_p$",
    ],
)
