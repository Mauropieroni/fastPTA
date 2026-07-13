r"""
Locally defined template: SMBH binary power law plus a flat spectrum.

Sum of a power law (fixed to the PTA pivot frequency ``fastPTA.utils.f_yr``,
matching the usual SMBH background convention) and a frequency-independent
amplitude. The two sub-templates have disjoint parameters, so the spectrum
and its Jacobian are simply the sum / concatenation of the two pieces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.amplitude import Amplitude
from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import AnalyticTemplate

from fastPTA import utils as ut


class SMBHFlat(AnalyticTemplate):
    r"""
    SMBH power law + flat spectrum.

    Free parameters
    ---------------
    log_amplitude_PL, tilt_PL
        Amplitude and tilt of the SMBH power law (pivot fixed at
        ``fastPTA.utils.f_yr``).
    log_amplitude_flat
        :math:`\log_{10}` amplitude of the flat spectrum.
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
            "log_amplitude_flat": r"$\log_{10} A_{\rm flat}$",
        }
        default_priors = {
            "log_amplitude_PL": {"min": -20.0, "max": -5.0},
            "tilt_PL": {"min": -10.0, "max": 10.0},
            "log_amplitude_flat": {"min": -20.0, "max": -5.0},
        }

        self._power_law = PowerLaw(pivot=ut.f_yr)
        self._flat = Amplitude()

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "SMBH Power Law + Flat"
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
        log_amplitude_flat: jax.Array,
    ) -> jax.Array:
        r"""Evaluate the combined SMBH power law + flat spectrum."""
        power_law_spectrum = self._power_law.omega_gw_h2(
            frequency, log_amplitude_PL, tilt_PL
        )
        flat_spectrum = self._flat.omega_gw_h2(frequency, log_amplitude_flat)
        return power_law_spectrum + flat_spectrum

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
        grad_flat = self._flat.grad_theta_omega_gw_h2(frequency, theta[2:])
        return jnp.concatenate([grad_pl, grad_flat], axis=-1)
