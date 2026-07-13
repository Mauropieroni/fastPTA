r"""
Locally defined template: SMBH binary power law plus a SIGW spectrum.

Sum of a power law (fixed to the PTA pivot frequency ``fastPTA.utils.f_yr``,
matching the usual SMBH background convention) and the local
:class:`~fastPTA.local_signal_templates.SIGWB_template.SIGWB` template. The
two sub-templates have disjoint parameters, so the spectrum and its Jacobian
are simply the sum / concatenation of the two pieces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import AnalyticTemplate

from fastPTA import utils as ut
from fastPTA.local_signal_templates.SIGWB_template import SIGWB


class SMBHSIGWB(AnalyticTemplate):
    r"""
    SMBH power law + SIGW spectrum.

    Free parameters
    ---------------
    log_amplitude_PL, tilt_PL
        Amplitude and tilt of the SMBH power law (pivot fixed at
        ``fastPTA.utils.f_yr``).
    log_amplitude_SIGW, log_width_SIGW, log_pivot_SIGW
        Amplitude, width and pivot frequency of the lognormal scalar
        spectrum sourcing the SIGW background (see
        :class:`~fastPTA.local_signal_templates.SIGWB_template.SIGWB`).
    """

    bibtex_entries: ClassVar[tuple[str, ...]] = ()

    def __init__(
        self,
        *,
        model_name: str | None = None,
        model_label: str | None = None,
        parameter_labels: Mapping[str, str] | None = None,
        prior_by_param: Mapping[str, Any] | None = None,
    ) -> None:
        default_labels = {
            "log_amplitude_PL": r"$\log_{10} A_{\rm PL}$",
            "tilt_PL": r"$n_{\rm T}$",
            "log_amplitude_SIGW": r"$\log_{10} A_{\mathcal{P}}$",
            "log_width_SIGW": r"$\log_{10} \sigma$",
            "log_pivot_SIGW": r"$\log_{10} f_p$",
        }
        default_priors = {
            "log_amplitude_PL": {"min": -20.0, "max": -5.0},
            "tilt_PL": {"min": -10.0, "max": 10.0},
            "log_amplitude_SIGW": {"min": -3.0, "max": 0.0},
            "log_width_SIGW": {"min": -2.0, "max": 1.0},
            "log_pivot_SIGW": {"min": -9.0, "max": -7.0},
        }

        self._power_law = PowerLaw(pivot=ut.f_yr)
        self._sigwb = SIGWB()

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "SMBH Power Law + SIGW"
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

    def omega_gw_h2(
        self,
        frequency: jax.Array,
        log_amplitude_PL: jax.Array,
        tilt_PL: jax.Array,
        log_amplitude_SIGW: jax.Array,
        log_width_SIGW: jax.Array,
        log_pivot_SIGW: jax.Array,
    ) -> jax.Array:
        r"""Evaluate the combined SMBH power law + SIGW spectrum."""
        power_law_spectrum = self._power_law.omega_gw_h2(
            frequency, log_amplitude_PL, tilt_PL
        )
        sigwb_spectrum = self._sigwb.omega_gw_h2(
            frequency, log_amplitude_SIGW, log_width_SIGW, log_pivot_SIGW
        )
        return power_law_spectrum + sigwb_spectrum

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian: the two sub-templates have disjoint parameters,
        so it is the concatenation of their own Jacobians (the SIGW piece
        falls back to its own autodiff gradient internally).
        """
        grad_pl = self._power_law.grad_theta_omega_gw_h2(frequency, theta[:2])
        grad_sigwb = self._sigwb.grad_theta_omega_gw_h2(frequency, theta[2:])
        return jnp.concatenate([grad_pl, grad_sigwb], axis=-1)
