# Global
import unittest

import numpy as np
import healpy as hp

# Local
import utils as tu
from fastPTA import utils as ut


nside = 64
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, np.arange(npix))


class TestGetTensors(unittest.TestCase):

    def test_get_sort_indexes(self):
        """
        Test the function to get the (l,m) pairs sorted correctly

        """

        data = np.loadtxt(tu.lm_indexes)
        inds = ut.get_sort_indexes(5)

        self.assertTrue(np.allclose(inds[2][inds[-1]], data[:, 0]))
        self.assertTrue(np.allclose(inds[3][inds[-1]], data[:, 1]))

    def test_complex_to_real(self):
        """
        Test the function to go from complex to real spherical harmonics
        coefficients assuming complex are sorted according to the healpy scheme

        """

        l_max = 2
        n_coefficients = ut.get_n_coefficients_complex(l_max)

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

        real_vals = ut.complex_to_real_conversion(complex_vals)

        self.assertTrue(np.allclose(real_vals, test_real_vals))

    def test_complex_to_real_to_complex(self):
        """
        Check that starting from complex coefficients the operations commute

        """

        l_max = 2
        n_coefficients = ut.get_n_coefficients_complex(l_max)
        m_grid = np.array([0, 0, 0, 1, 1, 2], dtype=int)

        complex_vals = np.random.normal(
            0.0, 1.0, n_coefficients
        ) + 1j * np.random.normal(0.0, 1.0, n_coefficients)

        complex_vals[m_grid == 0] = complex_vals[m_grid == 0].real

        real_vals = ut.complex_to_real_conversion(complex_vals)

        test_complex_vals = ut.real_to_complex_conversion(real_vals)

        self.assertTrue(np.allclose(complex_vals, test_complex_vals))

    def test_get_real_spherical_harmonics(self, l_max=5):
        """
        Test the function to get the spherical harmonics

        """

        sp_harm = ut.get_real_spherical_harmonics(l_max, theta, phi)

        c = 0
        for ell in range(l_max + 1):
            for m in range(-ell, ell + 1):
                sp = ut.sph_harm_y(ell, np.abs(m), theta, phi)

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

        inds = ut.get_sort_indexes(l_max)
        ll = inds[2][inds[-1]]
        mm = inds[3][inds[-1]]

        for ind in range(len(ll)):
            YY = np.array(ut.sph_harm_y(ll[ind], np.abs(mm[ind]), theta, phi))

            if mm[ind] < 0:
                # The m < 0 is the complex conjugate of the m > 0 so need a -1
                YY = -np.sqrt(2.0) * (-1.0) ** mm[ind] * YY.imag
            elif mm[ind] > 0:
                # For the m > 0 we just need the real part of Y
                YY = np.sqrt(2.0) * (-1.0) ** mm[ind] * YY.real
            else:
                # For the m = 0 we just need the real part of Y
                YY = YY.real

            res_YY = ut.spherical_harmonics_projection(YY, l_max)
            self.assertAlmostEqual(np.abs(res_YY[ind] - 1.0), 0.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
