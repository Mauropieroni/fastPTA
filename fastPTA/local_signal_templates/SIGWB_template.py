r"""
Example of a locally defined template: power law with running tilt.

Three-parameter model for a GWB:

.. math::

    \Omega_{\mathrm{GW}} h^2(f) = 10^{\alpha_{\mathrm{PL}}}
        \left(\frac{f}{f_{\mathrm{pivot}}}\right)^{n_T
        + \frac{\alpha_T}{2} \ln(f / f_{\mathrm{pivot}})}

This file doubles as the reference for adding new local templates: copy
it, rename the class, and adjust ``omega_gw_h2`` (free parameters are
the positional arguments after ``frequency``), the default labels, and
the default priors. Nothing else is needed — the class is picked up by
``gwb_lisa.templates`` automatically.
"""

# Global imports
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar
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

        default_labels = {}
        default_priors = {}

        self.SIGWB_prefactor_interpolator = self.set_prefactor_interpolator(
            path_to_prefactor_data
        )

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "Power Law with Running"
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

        # The data are in log scale so we log the frequency and then exp the result
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
