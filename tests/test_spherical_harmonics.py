# Global
import unittest

import healpy as hp
import numpy as np

# Local
import utils as tu

from fastPTA.angular_decomposition import spherical_harmonics as sph

nside = 64
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))


class TestGetTensors(unittest.TestCase):

    def test_get_sort_indexes(self):
        """
        Test the function to get the (l,m) pairs sorted correctly
        """
        data = np.loadtxt(tu.lm_indexes)
        inds = sph.get_sort_indexes(5)

        self.assertTrue(np.allclose(inds[2][inds[-1]], data[:, 0]))
        self.assertTrue(np.allclose(inds[3][inds[-1]], data[:, 1]))

    def test_complex_to_real(self):
        """
        Test the function to go from complex to real spherical harmonics
        coefficients assuming complex are sorted according to the healpy scheme
        """
        l_max = 2
        n_coefficients = sph.get_n_coefficients_complex(l_max)

        complex_vals = np.random.normal(
            0.0, 1.0, n_coefficients
        ) + 1j * np.random.normal(0.0, 1.0, n_coefficients)

        test_real_vals = np.array(
            [
                complex_vals[0].real,
                -np.sqrt(2) * complex_vals[3].imag,
                complex_vals[1].real,
                -np.sqrt(2) * complex_vals[3].real,
                np.sqrt(2) * complex_vals[5].imag,
                -np.sqrt(2) * complex_vals[4].imag,
                complex_vals[2].real,
                -np.sqrt(2) * complex_vals[4].real,
                np.sqrt(2) * complex_vals[5].real,
            ]
        )

        real_vals = sph.complex_to_real_conversion(complex_vals)

        self.assertTrue(np.allclose(real_vals, test_real_vals))

    def test_complex_to_real_to_complex(self):
        """
        Check that starting from complex coefficients the operations commute
        """
        l_max = 2
        n_coefficients = sph.get_n_coefficients_complex(l_max)
        m_grid = np.array([0, 0, 0, 1, 1, 2], dtype=int)

        complex_vals = np.random.normal(
            0.0, 1.0, n_coefficients
        ) + 1j * np.random.normal(0.0, 1.0, n_coefficients)

        complex_vals[m_grid == 0] = complex_vals[m_grid == 0].real

        real_vals = sph.complex_to_real_conversion(complex_vals)

        test_complex_vals = sph.real_to_complex_conversion(real_vals)

        self.assertTrue(np.allclose(complex_vals, test_complex_vals))

    def test_get_real_spherical_harmonics(self, l_max=5):
        """
        Test the function to get the spherical harmonics
        """
        sp_harm = sph.get_real_spherical_harmonics(l_max, theta, phi)

        c = 0
        for ell in range(l_max + 1):
            for m in range(-ell, ell + 1):
                sp = sph.sph_harm_y(ell, np.abs(m), theta, phi)

                if m == 0:
                    sp = sp.real
                elif m > 0:
                    sp = np.sqrt(2.0) * (-1.0) ** m * sp.real
                elif m < 0:
                    sp = np.sqrt(2.0) * (-1.0) ** m * sp.imag
                else:
                    raise ValueError("Nope")

                # Checks that (with the correct normalization) the scalar
                # product is 1 withing 3 decimal places
                self.assertAlmostEqual(
                    np.abs(np.mean(4 * np.pi * sp_harm[c] * sp) - 1.0),
                    0.0,
                    places=3,
                )

                c += 1

    def test_spherical_harmonics_multipoles(self, l_max=3):
        """
        Test the spherical harmonics projection function
        """
        inds = sph.get_sort_indexes(l_max)
        ll = inds[2][inds[-1]]
        mm = inds[3][inds[-1]]

        for ind in range(len(ll)):
            YY = np.array(sph.sph_harm_y(ll[ind], np.abs(mm[ind]), theta, phi))

            if mm[ind] < 0:
                # The m < 0 is the complex conjugate of the m > 0 so need a -1
                YY = -np.sqrt(2.0) * (-1.0) ** mm[ind] * YY.imag
            elif mm[ind] > 0:
                # For the m > 0 we just need the real part of Y
                YY = np.sqrt(2.0) * (-1.0) ** mm[ind] * YY.real
            else:
                # For the m = 0 we just need the real part of Y
                YY = YY.real

            res_YY = sph.spherical_harmonics_projection(YY, l_max)
            self.assertAlmostEqual(np.abs(res_YY[ind] - 1.0), 0.0, places=4)

    def test_spherical_harmonics_correlations(self):
        """
        Test the spherical harmonics projection function for pulsars pulsar
        matrix
        """
        Y00 = np.array([[sph.sph_harm_y(0, 0, theta, phi).real]])
        Y1m1 = np.array(
            [[np.sqrt(2.0) * sph.sph_harm_y(1, -1, theta, phi).imag]]
        )
        Y10 = np.array([[sph.sph_harm_y(1, 0, theta, phi).real]])
        Y1p1 = np.array(
            [[-np.sqrt(2.0) * sph.sph_harm_y(1, 1, theta, phi).real]]
        )

        Y2m2 = np.array(
            [[np.sqrt(2.0) * sph.sph_harm_y(2, -2, theta, phi).imag]]
        )
        Y2m1 = np.array(
            [[np.sqrt(2.0) * sph.sph_harm_y(2, -1, theta, phi).imag]]
        )
        Y20 = np.array([[sph.sph_harm_y(2, 0, theta, phi).real]])
        Y2p1 = np.array(
            [[-np.sqrt(2.0) * sph.sph_harm_y(2, 1, theta, phi).real]]
        )
        Y2p2 = np.array(
            [[np.sqrt(2.0) * sph.sph_harm_y(2, 2, theta, phi).real]]
        )

        Y3m3 = np.array(
            [[np.sqrt(2.0) * sph.sph_harm_y(3, -3, theta, phi).imag]]
        )
        Y3p3 = np.array(
            [[-np.sqrt(2.0) * sph.sph_harm_y(3, 3, theta, phi).real]]
        )

        res_Y00 = sph.project_correlation_spherical_harmonics(Y00, 1)
        res_Y1m1 = sph.project_correlation_spherical_harmonics(Y1m1, 1)
        res_Y10 = sph.project_correlation_spherical_harmonics(Y10, 1)
        res_Y1p1 = sph.project_correlation_spherical_harmonics(Y1p1, 1)

        res_Y2m2 = sph.project_correlation_spherical_harmonics(Y2m2, 2)
        res_Y2m1 = sph.project_correlation_spherical_harmonics(Y2m1, 2)
        res_Y20 = sph.project_correlation_spherical_harmonics(Y20, 2)
        res_Y2p1 = sph.project_correlation_spherical_harmonics(Y2p1, 2)
        res_Y2p2 = sph.project_correlation_spherical_harmonics(Y2p2, 2)

        res_Y3m3 = sph.project_correlation_spherical_harmonics(Y3m3, 3)
        res_Y3p3 = sph.project_correlation_spherical_harmonics(Y3p3, 3)

        self.assertAlmostEqual(np.abs(res_Y00[0] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y1m1[1] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y10[2] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y1p1[3] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y2m2[4] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y2m1[5] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y20[6] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y2p1[7] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y2p2[8] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y3m3[9] - 1.0), 0.0, delta=1e-4)
        self.assertAlmostEqual(np.abs(res_Y3p3[15] - 1.0), 0.0, delta=1e-4)

    def test_get_map_from_real_clms(self, nside=8, l_max=5):
        """
        Test the function returning the map from the real spherical harmonics
        coefficients
        """
        n_coeffs = sph.get_n_coefficients_real(l_max)

        for i in range(n_coeffs):
            clms = np.zeros(n_coeffs)
            clms[i] += 1.0
            map_from_clms = sph.get_map_from_real_clms(clms, nside)
            cclms = sph.complex_to_real_conversion(
                hp.map2alm(map_from_clms, lmax=l_max)
            )

            self.assertTrue(np.allclose(clms, cclms, rtol=1e-7, atol=1e-7))

    def test_get_CL_from_real_clm(self):
        """
        Test the get_CL_from_real_clm function from spherical_harmonics module.
        Tests both shape and specific values.
        """
        # Test Case 1: Simple input with known values
        l_max = 2
        n_coeffs = sph.get_n_coefficients_real(l_max)

        # Create an array where each coefficient has value equal to its
        # index + 1 i.e., [1, 2, 3, 4, 5, 6, 7, 8, 9]
        clm_real = np.arange(1, n_coeffs + 1, dtype=float)

        # Calculate CL values
        CL = sph.get_CL_from_real_clm(clm_real)

        # Test shape
        self.assertEqual(CL.shape, (l_max + 1,))

        # For l=0, there's only one coefficient (index 0), value = 1
        # CL[0] = 1^2 = 1
        self.assertAlmostEqual(CL[0], 1.0)

        # For l=1, there are 3 coefficients (indices 1, 2, 3), values = 2, 3, 4
        # CL[1] = mean([2^2, 3^2, 4^2]) = mean([4, 9, 16]) = 29/3
        self.assertAlmostEqual(CL[1], 29 / 3)

        # For l=2, there are 5 coefficients (indices 4, 5, 6, 7, 8),
        # values = 5, 6, 7, 8, 9
        # CL[2] = mean([5^2, 6^2, 7^2, 8^2, 9^2])
        # CL[2] = mean([25, 36, 49, 64, 81]) = 255/5 = 51
        self.assertAlmostEqual(CL[2], 51.0)

        # Test Case 2: Multi-dimensional input
        n_dims = 3
        clm_real_multidim = np.tile(clm_real[:, np.newaxis], (1, n_dims))
        # Add some variation to make each dimension different
        for i in range(n_dims):
            clm_real_multidim[:, i] *= i + 1

        CL_multidim = sph.get_CL_from_real_clm(clm_real_multidim)

        # Test shape
        self.assertEqual(CL_multidim.shape, (l_max + 1, n_dims))

        # Test values for each dimension
        for i in range(n_dims):
            factor = i + 1
            self.assertAlmostEqual(CL_multidim[0, i], 1.0 * factor**2)
            self.assertAlmostEqual(CL_multidim[1, i], 29 / 3 * factor**2)
            self.assertAlmostEqual(CL_multidim[2, i], 51.0 * factor**2)

        # Test Case 3: Complex pattern
        # Create an array with a specific pattern for more comprehensive testing
        clm_real_pattern = np.ones(n_coeffs)
        # Set all coefficients with even indices to 2.0
        clm_real_pattern[::2] = 2.0

        CL_pattern = sph.get_CL_from_real_clm(clm_real_pattern)

        # For l=0, there's 1 coefficient with value 2.0
        self.assertAlmostEqual(CL_pattern[0], 4.0)  # 2.0^2

        # For l=1, there are 3 coefficients: [1.0, 2.0, 1.0]
        self.assertAlmostEqual(
            CL_pattern[1], (1.0**2 + 2.0**2 + 1.0**2) / 3
        )  # (1 + 4 + 1)/3 = 2

        # For l=2, there are 5 coefficients: [2.0, 1.0, 2.0, 1.0, 2.0]
        self.assertAlmostEqual(
            CL_pattern[2], (2.0**2 + 1.0**2 + 2.0**2 + 1.0**2 + 2.0**2) / 5
        )  # (4 + 1 + 4 + 1 + 4)/5 = 14/5

    def test_get_dCL_from_real_clm(self):
        """
        Test the get_dCL_from_real_clm function from spherical_harmonics module.
        Tests both shape and specific values.
        """
        # Test Case 1: Simple input with known values
        l_max = 2
        n_coeffs = sph.get_n_coefficients_real(l_max)

        # Create coefficient arrays
        clm_real = np.ones(n_coeffs)
        dclm_real = 0.1 * np.ones(n_coeffs)

        # Calculate dCL values
        dCL = sph.get_dCL_from_real_clm(clm_real, dclm_real)

        # Test shape
        self.assertEqual(dCL.shape, (l_max + 1,))

        # For all l values, we expect dCL[l] = 2 * mean(|clm_real * dclm_real|)
        # Since all clm_real = 1 and all dclm_real = 0.1,
        # we expect dCL[l] = 2 * 0.1 = 0.2
        for ell in range(l_max + 1):
            self.assertAlmostEqual(dCL[ell], 0.2)

        # Test Case 2: Varying coefficients
        clm_real = np.arange(1, n_coeffs + 1, dtype=float)
        dclm_real = 0.1 * clm_real

        dCL = sph.get_dCL_from_real_clm(clm_real, dclm_real)

        # For l=0, there's only one coefficient with value 1 and uncertainty 0.1
        # dCL[0] = 2 * |1 * 0.1| = 0.2
        self.assertAlmostEqual(dCL[0], 0.2)

        # For l=1, there are 3 coefficients with values [2, 3, 4] and
        # uncertainties [0.2, 0.3, 0.4]
        # dCL[1] = 2 * mean(|[2*0.2, 3*0.3, 4*0.4]|) =
        # dCL[1] = 2 * mean([0.4, 0.9, 1.6]) = 2 * 0.966... = 1.933...
        self.assertAlmostEqual(dCL[1], 2 * np.mean([0.4, 0.9, 1.6]))

        # For l=2, there are 5 coefficients with values [5, 6, 7, 8, 9] and
        # uncertainties [0.5, 0.6, 0.7, 0.8, 0.9]
        # dCL[2] = 2 * mean(|[5*0.5, 6*0.6, 7*0.7, 8*0.8, 9*0.9]|) =
        # dCL[2] = 2 * mean([2.5, 3.6, 4.9, 6.4, 8.1]) = 2 * 5.1 = 10.2
        self.assertAlmostEqual(dCL[2], 2 * np.mean([2.5, 3.6, 4.9, 6.4, 8.1]))

        # Test Case 3: Multi-dimensional input
        n_dims = 2
        clm_real_multidim = np.tile(clm_real[:, np.newaxis], (1, n_dims))
        dclm_real_multidim = np.tile(dclm_real[:, np.newaxis], (1, n_dims))

        # Make second dimension different
        clm_real_multidim[:, 1] *= 2
        dclm_real_multidim[:, 1] *= 2

        dCL_multidim = sph.get_dCL_from_real_clm(
            clm_real_multidim, dclm_real_multidim
        )

        # Test shape
        self.assertEqual(dCL_multidim.shape, (l_max + 1, n_dims))

        # Test first dimension values (same as Test Case 2)
        self.assertAlmostEqual(dCL_multidim[0, 0], 0.2)
        self.assertAlmostEqual(dCL_multidim[1, 0], 2 * np.mean([0.4, 0.9, 1.6]))
        self.assertAlmostEqual(
            dCL_multidim[2, 0], 2 * np.mean([2.5, 3.6, 4.9, 6.4, 8.1])
        )

        # Test second dimension values (*4  due to doubling both clm and dclm)
        self.assertAlmostEqual(dCL_multidim[0, 1], 0.8)  # 2 * |2 * 0.2| = 0.8
        self.assertAlmostEqual(
            dCL_multidim[1, 1], 2 * np.mean([1.6, 3.6, 6.4])
        )  # Quadruple of first dimension
        self.assertAlmostEqual(
            dCL_multidim[2, 1], 2 * np.mean([10.0, 14.4, 19.6, 25.6, 32.4])
        )  # Quadruple of first dimension

        # Test Case 4: Test with negative values
        clm_real_neg = np.arange(
            -4, 5, dtype=float
        )  # [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        dclm_real_neg = 0.1 * np.abs(clm_real_neg)

        dCL_neg = sph.get_dCL_from_real_clm(clm_real_neg, dclm_real_neg)

        # Compute expected values manually
        l0_val = 2 * np.abs(-4 * 0.4)  # 2 * 1.6 = 3.2
        l1_val = 2 * np.mean(
            np.abs([-3 * 0.3, -2 * 0.2, -1 * 0.1])
        )  # 2 * mean([0.9, 0.4, 0.1]) = 2 * 0.466... = 0.933...
        l2_val = 2 * np.mean(
            np.abs([0 * 0.0, 1 * 0.1, 2 * 0.2, 3 * 0.3, 4 * 0.4])
        )  # 2 * mean([0, 0.1, 0.4, 0.9, 1.6]) = 2 * 0.6 = 1.2

        self.assertAlmostEqual(dCL_neg[0], l0_val)
        self.assertAlmostEqual(dCL_neg[1], l1_val)
        self.assertAlmostEqual(dCL_neg[2], l2_val)


if __name__ == "__main__":
    unittest.main(verbosity=2)
