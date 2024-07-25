# Global
import unittest

import numpy as np
import healpy as hp
from scipy.special import sph_harm

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
        Test the function to go from complex to real spherical harmonics coefficients

        """

        l_max = 2
        n_tot = np.sum(1 + np.arange(l_max + 1))

        m_grid = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 2.0], dtype=int)
        m_positive = m_grid[m_grid > 0.0]

        vals = np.random.normal(0.0, 1.0, n_tot) + 1j * np.random.normal(
            0.0, 1.0, n_tot
        )

        test_vals = np.array(
            [
                vals[0].real,
                -np.sqrt(2) * vals[2].imag,
                vals[1].real,
                -np.sqrt(2) * vals[2].real,
                np.sqrt(2) * vals[5].imag,
                -np.sqrt(2) * vals[4].imag,
                vals[3].real,
                -np.sqrt(2) * vals[4].real,
                np.sqrt(2) * vals[5].real,
            ]
        )

        result = ut.complex_to_real_conversion(vals, l_max, m_grid, m_positive)

        self.assertTrue(np.allclose(result, test_vals))

    def test_get_spherical_harmonics(self):
        """
        Test the function to get the spherical harmonics

        """

        sp_harm = ut.get_spherical_harmonics(5, theta, phi)

        c = 0
        for ell in range(6):
            for m in range(-ell, ell + 1):
                sp = sph_harm(np.abs(m), ell, phi, theta)

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
