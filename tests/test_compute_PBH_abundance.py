import unittest
import numpy as np

import jax
import jax.numpy as jnp

# Local imports
from fastPTA import compute_PBH_Abundance as cpa
import utils as tu


# Already existing test data
f_PBH_lognormal_data = np.load(tu.f_PBH_lognormal_data_path)
find_A_PBH_lognormal_data = np.load(tu.find_A_PBH_lognormal_data_path)


class Test_Abundance_Extended(unittest.TestCase):
    """Extended tests for the compute_PBH_Abundance module."""

    def test_k_of_T_MeV(self):
        """
        Test function for k_of_T_MeV
        """

        T_MeV = np.geomspace(4, 7.3e3, 100)

        k_MeV = cpa.k_of_T_MeV(T_MeV)
        TT_MeV = cpa.T_of_k(k_MeV)

        self.assertAlmostEqual(
            np.sum(np.abs(T_MeV / TT_MeV - 1)), 0.0, delta=1e-5
        )

    def test_hubble_mass_of_T_MeV(self):
        """
        Test function for k_of_T_MeV
        """

        T_MeV = np.geomspace(4, 7.3e3, 100)

        HM = cpa.hubble_mass_of_T_MeV(T_MeV)

        TT_MeV = cpa.T_of_M_H(HM)

        self.assertAlmostEqual(
            np.sum(np.abs(T_MeV / TT_MeV - 1)), 0.0, delta=1e-5
        )

    def test_f_PBH(self):
        """
        Test function for f_PBH_NL_QCD_lognormal
        """

        for v in f_PBH_lognormal_data["data"]:
            self.assertAlmostEqual(
                cpa.f_PBH_NL_QCD_lognormal(*v[:3]) / v[-1] - 1, 0.0, delta=1e-3
            )

    def test_find_A(self):
        """
        Test function for find_A_NL_QCD
        """

        for v in find_A_PBH_lognormal_data["data"]:
            self.assertAlmostEqual(
                cpa.find_A_NL_QCD(*v[:3]) / v[-1] - 1, 0.0, delta=1e-5
            )

    def test_window(self):
        """
        Test the window function with expected values.
        """
        # Test cases with pre-calculated expected values
        test_cases = [
            (0.1, 1.0, 0.9990003570767014),
            (1.0, 1.0, 0.9035060368192702),
            (10.0, 1.0, 0.023540082539625463),
        ]

        for k, r_max, expected in test_cases:
            result = float(cpa.window(k, r_max))
            self.assertAlmostEqual(
                result,
                expected,
                places=5,
                msg=f"Window function failed for k={k}, r_max={r_max}",
            )

    def test_transfer_function(self):
        """
        Test the transfer function with expected values.
        """
        # Test cases with pre-calculated expected values
        test_cases = [
            (0.1, 1.0, 0.9996667063466697),
            (1.0, 1.0, 0.9670610517788172),
            (10.0, 1.0, -0.0861665581638823),
        ]

        for k, r_max, expected in test_cases:
            result = float(cpa.transfer_function(k, r_max))
            self.assertAlmostEqual(
                result,
                expected,
                places=5,
                msg=f"Transfer function failed for k={k}, r_max={r_max}",
            )

    def test_lognormal_spectrum(self):
        """
        Test the lognormal_spectrum function with expected values.
        """
        # Test cases with pre-calculated expected values
        test_cases = [
            (0.1, 0.05, 0.1, 0.2, 7.361654518995322e-12),
            (1.0, 0.05, 0.1, 0.2, 1.1283370232016228e-57),
            (10.0, 0.05, 0.1, 0.2, 0.0),
        ]

        for k, amplitude, delta, ks, expected in test_cases:
            result = float(cpa.lognormal_spectrum(k, amplitude, delta, ks))
            self.assertAlmostEqual(
                result,
                expected,
                places=10,
                msg=(
                    f"Lognormal spectrum failed for k={k}, A={amplitude}, "
                    f"delta={delta}, ks={ks}"
                ),
            )

    def test_P_G(self):
        """
        Test the P_G function with expected values.
        """
        # Test cases with pre-calculated expected values
        test_cases = [
            (0.0, 0.5, 0.7978845608028654),
            (0.5, 0.5, 0.48394144903828673),
            (1.0, 0.5, 0.10798193302637613),
        ]

        for cal_C_G, sigma_c, expected in test_cases:
            result = float(cpa.P_G(cal_C_G, sigma_c))
            self.assertAlmostEqual(
                result,
                expected,
                places=5,
                msg=(
                    f"P_G function failed for cal_C_G={cal_C_G}, "
                    f"sigma_c={sigma_c}"
                ),
            )

    def test_integrand_spectrum(self):
        """
        Test the integrand_spectrum function with expected values.
        """
        # These are computed for specific inputs and depend on window and
        # transfer functions
        test_cases = [
            (0.1, 1.0, 1.0, 9.973365690278532e-05),
            (1.0, 1.0, 1.0, 0.7634311957207981),
            (10.0, 1.0, 1.0, 0.041142763025916046),
        ]

        for k, r_max, Delta_sqr, expected in test_cases:
            result = float(cpa.integrand_spectrum(k, r_max, Delta_sqr))
            self.assertAlmostEqual(
                result,
                expected,
                places=5,
                msg=(
                    f"integrand_spectrum failed for k={k}, r_max={r_max}, "
                    f"Delta_sqr={Delta_sqr}"
                ),
            )

    # Note: The following functions depend on interpolators and are not
    # easily testable in this way:
    # - k_of_T_MeV: Uses relativistic_dofs and entropy_dofs interpolators
    # - hubble_mass_of_T_MeV: Uses relativistic_dofs and entropy_dofs
    # - M_H_of_k: Depends on k_of_T_MeV
    # - compute_sigma_c_NL_QCD: Requires array input for k_vec
    # - integrand_beta: Uses phi_QCD interpolator
    # - compute_beta_NL_C_QCD: Depends on compute_sigma_c_NL_QCD


class Test_f_PBH_Interpolator(unittest.TestCase):
    """Tests for the build_f_PBH_interpolator / get_PBH_abundance_from_
    interpolator fast path used by Priors to speed up the PBH abundance
    check (see compute_PBH_Abundance.py)."""

    amplitude_bounds = (-3.5, -1.5)
    width_bounds = (-1.8, 0.8)
    pivot_bounds = (-9.0, -7.0)

    @classmethod
    def setUpClass(cls):
        # staticmethod: a plain function assigned as a class attribute is
        # bound as a method (self gets passed as its first arg) when
        # accessed via self.approx otherwise.
        cls.approx = staticmethod(
            cpa.build_f_PBH_interpolator(
                cls.amplitude_bounds,
                cls.width_bounds,
                cls.pivot_bounds,
                n_grid=20,
                verbose=False,
            )
        )

    def test_matches_exact_at_grid_point(self):
        """
        The interpolator should closely reproduce the exact calculation
        it was built from, at an interior grid point (no interpolation
        error at a knot itself, up to the floor/log10 round trip).
        """

        log_amplitude, log_width, log_pivot = -1.7, -0.3, -8.25
        ks = 10.0**log_pivot * 2.0 * np.pi / 9.7156e-15

        exact = cpa.f_PBH_NL_QCD_lognormal(
            10.0**log_amplitude, 10.0**log_width, ks
        )
        approx = self.approx(log_amplitude, log_width, log_pivot)

        self.assertAlmostEqual(
            float(np.log10(max(float(exact), 1e-30))),
            float(np.log10(max(float(approx), 1e-30))),
            delta=0.2,
        )

    def test_pbh_exceeds_bound_matches_generic_check(self):
        """
        get_PBH_abundance_from_interpolator's fused pbh_exceeds_bound
        fast path (a single jitted dispatch, see compute_PBH_Abundance.py)
        must agree with the generic get_PBH_abundance(...) > 1.0 or
        isnan(...) check it replaces in Priors.evaluate_log_priors.
        """

        parameter_names = ["a", "b", "c"]
        pbh_names = ("a", "b", "c")
        priors_dictionary = {
            "a": {"uniform": {"loc": self.amplitude_bounds[0], "scale": 2.0}},
            "b": {"uniform": {"loc": self.width_bounds[0], "scale": 2.6}},
            "c": {"uniform": {"loc": self.pivot_bounds[0], "scale": 2.0}},
        }

        get_pbh_abundance = cpa.get_PBH_abundance_from_interpolator(
            parameter_names, pbh_names, priors_dictionary, n_grid=20
        )

        for point in [(-1.7, -0.3, -8.25), (-1.6, -1.8, -8.0)]:
            abundance = float(get_pbh_abundance(list(point)))
            generic_exceeds = abundance > 1.0 or np.isnan(abundance)
            fused_exceeds = bool(
                get_pbh_abundance.pbh_exceeds_bound(list(point))
            )

            self.assertEqual(generic_exceeds, fused_exceeds)

    def test_gradient_is_never_nan(self):
        """
        Regression test for future use in a gradient-based sampler (e.g.
        blackjax): jax.grad through the interpolator, and through the
        jnp.where(exceeds, -jnp.inf, ...) gate pattern Priors uses it
        with, must never be nan -- including under gross extrapolation
        far outside the grid, where the raw interpolated value can itself
        be inf. See the note on pbh_exceeds_bound for why this holds (the
        excluded branch must stay a bare -jnp.inf constant).
        """

        def gate(a, b, c):
            v = self.approx(a, b, c)
            exceeds = (v > 1.0) | jnp.isnan(v)
            stand_in_log_density = -(a**2) - (b**2) - (c**2)
            return jnp.where(exceeds, -jnp.inf, stand_in_log_density)

        points = [
            (-1.7, -0.3, -8.25),  # interior, valid
            (-1.6, -1.8, -8.0),  # interior, likely excluded
            (100.0, 100.0, 100.0),  # gross extrapolation, v -> inf
            (-500.0, -500.0, -500.0),  # gross extrapolation, other side
        ]

        for point in points:
            grads = jax.grad(gate, argnums=(0, 1, 2))(*point)

            self.assertFalse(
                any(bool(jnp.isnan(g)) for g in grads),
                msg=f"nan gradient at {point}: {grads}",
            )


if __name__ == "__main__":
    unittest.main()
