# Global imports
import unittest

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

# Local imports
from fastPTA.data import generate_data as gd

# Enable 64-bit precision for JAX for consistency with the module
jax.config.update("jax_enable_x64", True)

# Test parameters
NSIDE = 4  # Small Nside for testing
NUM_PULSARS = 5
FREQUENCIES = jnp.geomspace(1e-9, 1e-7, 10)  # 10 frequency bins
TSPAN = 10 * 365.25 * 24 * 3600  # 10 years in seconds


class TestGenerateData(unittest.TestCase):
    """Test class for generate_data.py"""

    def test_generate_gaussian(self):
        """
        Test that generate_gaussian produces correct shape and
        statistical properties
        """
        # Test scalar inputs
        mean = 0.0
        sigma = 1.0
        sample = gd.generate_gaussian(mean, sigma)
        self.assertTrue(np.iscomplex(sample))

        # Test array inputs with size parameter
        size = (100,)
        samples = gd.generate_gaussian(mean, sigma, size=size)
        self.assertEqual(samples.shape, size)

        # Test statistical properties (mean and standard deviation)
        # Use a larger sample size for better statistics
        size = (10000,)
        samples = gd.generate_gaussian(mean, sigma, size=size)
        # Mean should be close to 0 for both real and imaginary parts
        self.assertLess(np.abs(np.mean(samples.real)), 0.1)
        self.assertLess(np.abs(np.mean(samples.imag)), 0.1)
        # Standard deviation should be close to 1/sqrt(2) for both parts
        self.assertLess(np.abs(np.std(samples.real) - 1 / np.sqrt(2)), 0.1)
        self.assertLess(np.abs(np.std(samples.imag) - 1 / np.sqrt(2)), 0.1)

    def test_generate_pulsar_sky_and_kpixels(self):
        """Test the generation of pulsar sky positions and k-pixels"""
        # Call the function with test parameters
        p_vec, cos_IJ, distance, theta_k, phi_k = (
            gd.generate_pulsar_sky_and_kpixels(NUM_PULSARS, NSIDE)
        )

        # Check shapes and types
        self.assertEqual(p_vec.shape, (NUM_PULSARS, 3))
        self.assertEqual(cos_IJ.shape, (NUM_PULSARS, NUM_PULSARS))
        self.assertEqual(distance.shape, (NUM_PULSARS,))

        # Check that p_vec contains unit vectors
        for i in range(NUM_PULSARS):
            norm = jnp.sqrt(jnp.sum(p_vec[i] ** 2))
            self.assertAlmostEqual(float(norm), 1.0, places=5)

        # Check that cos_IJ is symmetric
        np.testing.assert_array_almost_equal(cos_IJ, cos_IJ.T)

        # Check that diagonal elements of cos_IJ are 1
        for i in range(NUM_PULSARS):
            self.assertAlmostEqual(float(cos_IJ[i, i]), 1.0, places=5)

        # Check that cos_IJ values are between -1 and 1
        self.assertTrue(jnp.all((cos_IJ >= -1.0) & (cos_IJ <= 1.0)))

        # Check that distances are positive
        self.assertTrue(jnp.all(distance > 0))

        # Check that number of k-pixels is correct
        npix = hp.nside2npix(NSIDE)
        self.assertEqual(len(theta_k), npix)
        self.assertEqual(len(phi_k), npix)

    def test_generate_hpc_polarization_pixel_frequency(self):
        """Test the generation of GW signal in pixel and frequency space"""
        # Create a test spectrum (constant for simplicity)
        npix = hp.nside2npix(NSIDE)
        nfreq = len(FREQUENCIES)
        H_p_ff = jnp.ones((npix, nfreq)) * 1e-30  # Small constant value

        # Generate the GW signal
        h_tilde = gd.generate_hpc_polarization_pixel_frequency(H_p_ff)

        # Check shape (2 polarizations, npix pixels, nfreq frequencies)
        self.assertEqual(h_tilde.shape, (2, npix, nfreq))

        # Check that the result is complex
        self.assertTrue(np.iscomplex(h_tilde[0, 0, 0]))

    def test_generate_D_IJ(self):
        """Test the generation of pulsar-pulsar correlation matrix D_IJ"""
        # Create a test spectrum (constant for simplicity)
        npix = hp.nside2npix(NSIDE)
        nfreq = len(FREQUENCIES)
        H_p_fi = jnp.ones((npix, nfreq)) * 1e-30  # Small constant value

        # Generate the correlation matrix
        _, zeta_IJ, D_IJ = gd.generate_D_IJ(
            NSIDE, NUM_PULSARS, FREQUENCIES, H_p_fi
        )

        # Check shapes
        self.assertEqual(zeta_IJ.shape, (NUM_PULSARS, NUM_PULSARS))
        self.assertEqual(D_IJ.shape, (nfreq, NUM_PULSARS, NUM_PULSARS))

        # Check that D_IJ is Hermitian (conjugate symmetric) for each frequency
        for i in range(nfreq):
            np.testing.assert_array_almost_equal(
                D_IJ[i], D_IJ[i].conj().T, decimal=5
            )

        # Check that zeta_IJ is symmetric
        np.testing.assert_array_almost_equal(zeta_IJ, zeta_IJ.T, decimal=5)

    def test_generate_D_IJ_fi(self):
        """Test the generation of D_IJ with frequency integration"""
        # Create test frequencies and spectrum
        fi = FREQUENCIES[:5]  # External frequencies (shorter)
        ff = FREQUENCIES  # Internal frequencies (full)
        npix = hp.nside2npix(NSIDE)
        H_p_ff = jnp.ones((npix, len(ff))) * 1e-30  # Small constant value

        # Generate the correlation matrix
        _, zeta_IJ, D_IJ = gd.generate_D_IJ_fi(
            NSIDE, NUM_PULSARS, TSPAN, fi, ff, H_p_ff
        )

        # Check shapes
        self.assertEqual(zeta_IJ.shape, (NUM_PULSARS, NUM_PULSARS))
        self.assertEqual(D_IJ.shape, (len(fi), NUM_PULSARS, NUM_PULSARS))

        # Check that D_IJ is Hermitian (conjugate symmetric) for each frequency
        for i in range(len(fi)):
            np.testing.assert_array_almost_equal(
                D_IJ[i], D_IJ[i].conj().T, decimal=5
            )

        # Check that zeta_IJ is symmetric
        np.testing.assert_array_almost_equal(zeta_IJ, zeta_IJ.T, decimal=5)

    def test_generate_D_IJ_fifj(self):
        """Test the generation of D_IJ with full frequency integration"""
        # Create test frequencies and spectrum
        fi = FREQUENCIES[:5]  # External frequencies (shorter)
        ff = FREQUENCIES  # Internal frequencies (full)
        npix = hp.nside2npix(NSIDE)
        H_p_ff = jnp.ones((npix, len(ff))) * 1e-30  # Small constant value

        # Generate the correlation matrix
        _, zeta_IJ, D_IJ = gd.generate_D_IJ_fifj(
            NSIDE, NUM_PULSARS, TSPAN, fi, ff, H_p_ff
        )

        # Check shapes
        self.assertEqual(zeta_IJ.shape, (NUM_PULSARS, NUM_PULSARS))
        # This should be a 3D tensor with shape (fi, fj, Np, Np)
        self.assertEqual(
            D_IJ.shape, (len(fi), len(fi), NUM_PULSARS, NUM_PULSARS)
        )

        # Check that D_IJ is Hermitian for each frequency pair
        for i in range(len(fi)):
            for j in range(len(fi)):
                np.testing.assert_array_almost_equal(
                    D_IJ[i, j], D_IJ[i, j].conj().T, decimal=5
                )

        # Check that zeta_IJ is symmetric
        np.testing.assert_array_almost_equal(zeta_IJ, zeta_IJ.T, decimal=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
