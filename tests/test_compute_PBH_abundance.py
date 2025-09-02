import unittest
import numpy as np

# Local imports
from fastPTA import compute_PBH_Abundance as cpa
import utils as tu


# Already existing test data
f_PBH_lognormal_data = np.load(tu.f_PBH_lognormal_data_path)
find_A_PBH_lognormal_data = np.load(tu.find_A_PBH_lognormal_data_path)


class Test_Abundance_Extended(unittest.TestCase):
    """Extended tests for the compute_PBH_Abundance module."""

    def test_f_PBH(self):
        """
        Test function for f_PBH_NL_QCD_lognormal
        """
        for v in f_PBH_lognormal_data["data"]:
            self.assertAlmostEqual(
                cpa.f_PBH_NL_QCD_lognormal(*v[:3]),
                v[-1],
                places=5,
            )

    def test_find_A(self):
        """
        Test function for find_A_NL_QCD
        """
        for v in find_A_PBH_lognormal_data["data"]:
            self.assertAlmostEqual(cpa.find_A_NL_QCD(*v[:3]), v[-1], places=3)

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


if __name__ == "__main__":
    unittest.main()
