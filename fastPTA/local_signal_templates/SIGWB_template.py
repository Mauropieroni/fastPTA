r"""
Locally defined template: SIGW from a broad lognormal scalar spectrum.

Three-parameter model for the GWB induced by scalar perturbations with a
broad lognormal peak. The analytical approximation of the spectral shape
was derived in 2005.12306 (see also 2302.07901); the form implemented
here, including the tabulated prefactor, follows eqs. 8-9 of 2503.10805.
Not yet part of the ``gwb_templates`` catalogue, pending upstream review, so it
lives here and is picked up by ``fastPTA.signals`` automatically.
"""

# Global imports
from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any, ClassVar
import numpy as np

import jax
import jax.numpy as jnp
from jax.scipy.interpolate import RegularGridInterpolator


from gwb_templates import AnalyticTemplate

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_DEFAULT_DATA_FILENAME = "SIGWB_prefactor_data.txt"


def _build_prefactor_interpolator(filename: str) -> RegularGridInterpolator:
    """Load the precomputed prefactor table and build a 1D interpolator."""

    path = os.path.join(_DATA_DIR, filename)
    data = np.loadtxt(path)

    return RegularGridInterpolator([data[:, 0]], data[:, 1])


class SIGWB(AnalyticTemplate):
    r"""
    Analytical approximation of the SIGW background sourced by a broad
    lognormal scalar spectrum (originally proposed in 2005.12306, see also 
    2302.07901).

    Free parameters
    ---------------
    log_amplitude 
        :math:`\log_{10} A_{\mathcal{P}}`, amplitude of the curvature power 
        spectrum.
    log_width
        :math:`\log_{10}` width of the lognormal bump.
    log_pivot
        :math:`\log_{10}` pivot frequency (in Hz) of the lognormal bump.
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = (
        r"""
@article{Pi:2020otn,
    author = "Pi, Shi and Sasaki, Misao",
    title = "{Gravitational Waves Induced by Scalar Perturbations with a
        Lognormal Peak}",
    eprint = "2005.12306",
    archivePrefix = "arXiv",
    primaryClass = "gr-qc",
    reportNumber = "YITP-20-75, YITP-75, IPMU20-0054",
    doi = "10.1088/1475-7516/2020/09/037",
    journal = "JCAP",
    volume = "09",
    pages = "037",
    year = "2020"
}
""",
        r"""
@article{Dandoy:2023jot,
    author = "Dandoy, Virgile and Domcke, Valerie and Rompineve, Fabrizio",
    title = "{Search for scalar induced gravitational waves in the
        international pulsar timing array data release 2 and NANOgrav
        12.5 years datasets}",
    eprint = "2302.07901",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "CERN-TH-2023-027",
    doi = "10.21468/SciPostPhysCore.6.3.060",
    journal = "SciPost Phys. Core",
    volume = "6",
    pages = "060",
    year = "2023"
}
""",
        r"""
@article{Cecchini:2025oks,
    author = "Cecchini, Chiara and Franciolini, Gabriele and Pieroni, Mauro",
    title = "{Forecasting constraints on scalar-induced gravitational waves with
        future pulsar timing array observations}",
    eprint = "2503.10805",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.CO",
    reportNumber = "CERN-TH-2025-045",
    doi = "10.1103/nxx5-gx7d",
    journal = "Phys. Rev. D",
    volume = "111",
    number = "12",
    pages = "123536",
    year = "2025"
}
""",

    )

    def __init__(
        self,
        data_filename: str = _DEFAULT_DATA_FILENAME,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:

        default_labels = {
            "log_amplitude": r"$\log_{10} A_{\mathcal{P}}$",
            "log_width": r"$\log_{10} \Delta$",
            "log_pivot": r"$\log_{10} f_p$",
        }
        default_priors = {
            "log_amplitude": {"min": -3.0, "max": 0.0},
            "log_width": {"min": -2.0, "max": 1.0},
            "log_pivot": {"min": -9.0, "max": -7.0},
        }

        self.SIGWB_prefactor_interpolator = _build_prefactor_interpolator(
            data_filename
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

    def SIGWB_prefactor(self, frequency: jax.Array) -> jax.Array:
        r"""
        Frequency-dependent prefactor of the SIGW spectrum.

        Interpolates the tabulated prefactor entering eqs. 8-9 of
        2503.10805, which encodes the transfer of the GW energy density
        from horizon re-entry to today, including the change in the
        number of relativistic degrees of freedom across the QCD
        crossover.

        Args:
            frequency: Frequency value(s) in Hz.

        Returns:
            Prefactor evaluated at each input frequency.
        """

        # The table is stored as log10(f) vs log10(prefactor): interpolate
        # in log10 of the frequency, then undo the log10 on the output.
        return 10 ** self.SIGWB_prefactor_interpolator(jnp.log10(frequency))

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude: jax.Array,
        log_width: jax.Array,
        log_pivot: jax.Array,
    ) -> jax.Array:
        r"""
        Evaluate the SIGW spectrum sourced by a broad lognormal scalar
        spectrum at ``frequency``.

        Args:
            frequency: Frequency value(s) in Hz.
            log_amplitude: :math:`\log_{10} A_{\mathcal{P}}`, amplitude of
                the curvature power spectrum.
            log_width: :math:`\log_{10} \Delta`, width of the lognormal
                peak in the curvature spectrum.
            log_pivot: :math:`\log_{10} (f_p / \mathrm{Hz})`, peak
                frequency of the lognormal.

        Returns:
            Spectrum :math:`\Omega_{\mathrm{GW}} h^2(f)` at each input
            frequency.
        """

        x = frequency / (10**log_pivot)

        width = 10**log_width

        k = x * jnp.exp((3 / 2) * width**2)

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

        return (
            self.SIGWB_prefactor(frequency)
            * (10**log_amplitude) ** 2
            * (term1 + term2 + term3)
        )
    