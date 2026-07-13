r"""
Locally defined template: SMBH binary power law plus a broken power law.

Sum of a power law (fixed to the PTA pivot frequency ``fastPTA.utils.f_yr``,
matching the usual SMBH background convention) and a fixed-smoothness broken
power law. The two sub-templates have disjoint parameters, so the spectrum
and its Jacobian are simply the sum / concatenation of the two pieces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.broken_power_law_fixed_smoothness import (
    BrokenPowerLawFixedSmoothness,
)
from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import AnalyticTemplate

from fastPTA import utils as ut


class SMBHBrokenPowerLaw(AnalyticTemplate):
    r"""
    SMBH power law + broken power law.

    Free parameters
    ---------------
    log_amplitude_PL, tilt_PL
        Amplitude and tilt of the SMBH power law (pivot fixed at
        ``fastPTA.utils.f_yr``).
    log_amplitude_BPL, log_pivot_BPL, tilt_1_BPL, tilt_2_BPL
        Amplitude, break frequency and low-/high-frequency tilts of the
        broken power law (see
        :class:`gwb_templates.generic_templates.broken_power_law_fixed_smoothness.BrokenPowerLawFixedSmoothness`).
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
            "log_amplitude_BPL": r"$\log_{10} A_{\rm BPL}$",
            "log_pivot_BPL": r"$\log_{10} f_{\rm BPL}$",
            "tilt_1_BPL": r"$n_{1, \rm BPL}$",
            "tilt_2_BPL": r"$n_{2, \rm BPL}$",
        }
        default_priors = {
            "log_amplitude_PL": {"min": -20.0, "max": -5.0},
            "tilt_PL": {"min": -10.0, "max": 10.0},
            "log_amplitude_BPL": {"min": -20.0, "max": -5.0},
            "log_pivot_BPL": {"min": -9.0, "max": -7.0},
            "tilt_1_BPL": {"min": -10.0, "max": 10.0},
            "tilt_2_BPL": {"min": -10.0, "max": 10.0},
        }

        self._power_law = PowerLaw(pivot=ut.f_yr)
        self._bpl = BrokenPowerLawFixedSmoothness()

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "SMBH Power Law + Broken Power Law"
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
        log_amplitude_BPL: jax.Array,
        log_pivot_BPL: jax.Array,
        tilt_1_BPL: jax.Array,
        tilt_2_BPL: jax.Array,
    ) -> jax.Array:
        r"""Evaluate the combined SMBH power law + broken power law spectrum."""
        power_law_spectrum = self._power_law.omega_gw_h2(
            frequency, log_amplitude_PL, tilt_PL
        )
        bpl_spectrum = self._bpl.omega_gw_h2(
            frequency,
            log_amplitude_BPL,
            log_pivot_BPL,
            tilt_1_BPL,
            tilt_2_BPL,
        )
        return power_law_spectrum + bpl_spectrum

    def _grad_theta_omega_gw_h2_analytical(
        self,
        frequency: jax.Array,
        theta: jax.Array,
    ) -> jax.Array:
        r"""
        Analytic Jacobian: the two sub-templates have disjoint parameters,
        so it is the concatenation of their own Jacobians.
        """
        grad_pl = self._power_law.grad_theta_omega_gw_h2(frequency, theta[:2])
        grad_bpl = self._bpl.grad_theta_omega_gw_h2(frequency, theta[2:])
        return jnp.concatenate([grad_pl, grad_bpl], axis=-1)
