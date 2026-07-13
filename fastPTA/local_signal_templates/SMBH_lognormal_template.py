r"""
Locally defined template: SMBH binary power law plus a lognormal bump.

Sum of a power law (fixed to the PTA pivot frequency ``fastPTA.utils.f_yr``,
matching the usual SMBH background convention) and a lognormal bump. The two
sub-templates have disjoint parameters, so the spectrum and its Jacobian are
simply the sum / concatenation of the two pieces.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import jax
import jax.numpy as jnp

from gwb_templates.generic_templates.lognormal_bump import LognormalBump
from gwb_templates.generic_templates.power_law import PowerLaw
from gwb_templates.template import AnalyticTemplate

from fastPTA import utils as ut


class SMBHLognormal(AnalyticTemplate):
    r"""
    SMBH power law + lognormal bump.

    Free parameters
    ---------------
    log_amplitude_PL, tilt_PL
        Amplitude and tilt of the SMBH power law (pivot fixed at
        ``fastPTA.utils.f_yr``).
    log_amplitude_LN, log_pivot_LN, log_width_LN
        Amplitude, peak frequency and width of the lognormal bump (see
        :class:`gwb_templates.generic_templates.lognormal_bump.LognormalBump`).
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
            "log_amplitude_LN": r"$\log_{10} A_{\rm LN}$",
            "log_pivot_LN": r"$\log_{10} f_{p, \rm LN}$",
            "log_width_LN": r"$\log_{10} \sigma_{\rm LN}$",
        }
        default_priors = {
            "log_amplitude_PL": {"min": -20.0, "max": -5.0},
            "tilt_PL": {"min": -10.0, "max": 10.0},
            "log_amplitude_LN": {"min": -20.0, "max": -5.0},
            "log_pivot_LN": {"min": -9.0, "max": -7.0},
            "log_width_LN": {"min": -2.0, "max": 1.0},
        }

        self._power_law = PowerLaw(pivot=ut.f_yr)
        self._lognormal = LognormalBump()

        super().__init__(
            model_name=model_name,
            model_label=(
                model_label
                if model_label is not None
                else "SMBH Power Law + Lognormal"
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
        log_amplitude_LN: jax.Array,
        log_pivot_LN: jax.Array,
        log_width_LN: jax.Array,
    ) -> jax.Array:
        r"""Evaluate the combined SMBH power law + lognormal bump spectrum."""
        power_law_spectrum = self._power_law.omega_gw_h2(
            frequency, log_amplitude_PL, tilt_PL
        )
        lognormal_spectrum = self._lognormal.omega_gw_h2(
            frequency, log_amplitude_LN, log_pivot_LN, log_width_LN
        )
        return power_law_spectrum + lognormal_spectrum

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
        grad_ln = self._lognormal.grad_theta_omega_gw_h2(frequency, theta[2:])
        return jnp.concatenate([grad_pl, grad_ln], axis=-1)
