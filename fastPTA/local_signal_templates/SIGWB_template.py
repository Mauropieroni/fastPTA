r"""
Locally defined template: SIGW from a broad lognormal scalar spectrum.

Three-parameter model for the GWB induced by scalar perturbations
(originally proposed in 2005.12306), using the analytical approximation
of eq. 9 of 2503.10805. Not part of the ``gwb_templates`` catalogue since
it depends on a fastPTA-specific prefactor data file, so it lives here and
is picked up by ``fastPTA.signals`` automatically.
"""

# Global imports
from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator


from gwb_templates import AnalyticTemplate

# Local imports
from fastPTA import utils as ut


class SIGWB(AnalyticTemplate):
    r"""
    Returns the analytical approximation of the SIGW for a broad a lognormal
    scalar spectrum (originally proposed in 2005.12306) see eq. 9 of 2503.10805.

    The parameters of this template are the log of the amplitude, the log of the
    width, and the log of the pivot frequency (in Hz) of the lognormal scalar
    spectrum.

    Free Parameters:
    -----------
    "log_amplitude", "log_width", ""

    log_amplitude
        Base-10 logarithm of the amplitude at the pivot frequency.
    log_width
        Base-10 logarithm of the width of the lognormal bump
    log_pivot
        Base-10 logarithm of pivot frequency of the lognormal bump

    """

    def __init__(
        self,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
        path_to_prefactor_data: str = ut.path_to_defaults
        + "SIGWB_prefactor_data.txt",
    ) -> None:

        default_labels = {
            "log_amplitude": r"$\log_{10} A_{\mathcal{P}}$",
            "log_width": r"$\log_{10} \sigma$",
            "log_pivot": r"$\log_{10} f_p$",
        }
        default_priors = {
            "log_amplitude": {"min": -3.0, "max": 0.0},
            "log_width": {"min": -2.0, "max": 1.0},
            "log_pivot": {"min": -9.0, "max": -7.0},
        }

        self.SIGWB_prefactor_interpolator = self.set_prefactor_interpolator(
            path_to_prefactor_data
        )

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "SIGW (broad lognormal)"
            ),
            parameter_labels=(
                parameter_labels
                if parameter_labels is not None
                else default_labels
            ),
            prior_by_param=(
                prior_by_param if prior_by_param is not None else default_priors
            ),
        )

    def set_prefactor_interpolator(self, path_to_prefactor_data):

        SIGWB_prefactor_data = np.loadtxt(path_to_prefactor_data)

        # This is building an interpolator for the SIGWB prefactor
        SIGWB_prefactor_interpolator = RegularGridInterpolator(
            [SIGWB_prefactor_data[:, 0]], SIGWB_prefactor_data[:, 1]
        )

        del SIGWB_prefactor_data

        return SIGWB_prefactor_interpolator

    def SIGWB_prefactor(self, frequency):
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

        # The data are in log scale: log the frequency and then exp the result
        return 10 ** self.SIGWB_prefactor_interpolator(jnp.log10(frequency))

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_width: jax.Array,
        log_pivot: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the running power-law spectrum at ``frequency``.

        Args:
            frequency: Frequency value(s) in Hz.
            log_amplitude: :math:`\log_{10}` amplitude at the pivot.
            log_width: :math:`\log_{10}` width of the bump.
            log_pivot: :math:`\log_{10}` pivot frequency of the bump.

        Returns:
            Spectrum :math:`\Omega_{\mathrm{GW}} h^2(f)` at each input
            frequency.
        """

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
            * jnp.exp(
                -((jnp.log(k) + (1 / 2) * jnp.log(3 / 2)) ** 2) / (width**2)
            )
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
            self.SIGWB_prefactor(frequency)
            * (10**log_amplitude) ** 2
            * (term1 + term2 + term3)
        )
