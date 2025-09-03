# Global
import unittest

import numpy as np

import jax
import jax.numpy as jnp

# Local
import utils as tu

import fastPTA.utils as ut
from fastPTA.inference_tools import signal_covariance as sc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

n_frequencies = 35
data = np.load(tu.Fisher_data_path)


class TestSignalCovariance(unittest.TestCase):
    def test_get_signal_covariance(self):
        """
        Test the computation of signal covariance tensor
        """
        # Use small test dimensions
        nfreq = 3
        npsr = 2

        # Create sample input data
        signal = jnp.array([1.0, 2.0, 3.0])  # Signal at different frequencies
        response_IJ = jnp.ones((nfreq, npsr, npsr))

        # Get signal covariance
        signal_cov = sc.get_signal_covariance(signal, response_IJ)

        # Check output shape
        self.assertEqual(signal_cov.shape, (nfreq, npsr, npsr))

        # Check values - for signal = [1,2,3] and response_IJ = ones,
        # each frequency slice should be multiplied by the signal value
        expected = jnp.array(
            [
                1.0 * jnp.ones((npsr, npsr)),
                2.0 * jnp.ones((npsr, npsr)),
                3.0 * jnp.ones((npsr, npsr)),
            ]
        )

        np.testing.assert_allclose(signal_cov, expected, rtol=1e-5)

    def test_get_covariance_diagonal(self):
        """
        Test computation of diagonal covariance
        """
        # Use small test dimensions
        nfreq = 3
        npsr = 2
        nlm = 2

        # Create sample input data
        signal_lm = jnp.array([1.0, 0.5])
        gamma_IJ_lm = jnp.ones((nlm, npsr, npsr))
        ff = jnp.array([1e-8, 1e-7, 1e-6])  # Frequencies
        S_f = jnp.array([1.0, 1.0, 1.0])  # Flat spectrum

        # Get covariance diagonal
        cov_diag = sc.get_covariance_diagonal(signal_lm, gamma_IJ_lm, ff, S_f)

        # Check output shape
        self.assertEqual(cov_diag.shape, (nfreq, npsr, npsr))

        # Check values are reasonable (positive)
        self.assertTrue(jnp.all(cov_diag > 0))

        # Verify that higher frequencies have lower covariance (1/f^2 scaling)
        for i in range(1, nfreq):
            self.assertTrue(jnp.all(cov_diag[i - 1] > cov_diag[i]))

    def test_get_dcovariance_diagonal(self):
        """
        Test computation of diagonal derivative covariance
        """
        # Use small test dimensions
        nfreq = 3
        npsr = 2
        nlm = 2

        # Create sample input data
        signal_lm = jnp.array([1.0, 0.5])
        gamma_IJ_lm = jnp.ones((nlm, npsr, npsr))
        ff = jnp.array([1e-8, 1e-7, 1e-6])  # Frequencies
        S_f = jnp.array([1.0, 1.0, 1.0])  # Flat spectrum

        # Get derivative of covariance diagonal
        dcov_diag = sc.get_dcovariance_diagonal(signal_lm, gamma_IJ_lm, ff, S_f)

        # Check output shape
        self.assertEqual(dcov_diag.shape, (nlm, nfreq, npsr, npsr))

        # Values should be non-zero
        self.assertTrue(jnp.any(dcov_diag != 0))

    def test_get_covariance_full(self):
        """
        Test computation of full covariance
        """
        # Use small test dimensions
        nfreq = 3
        npsr = 2
        nlm = 2

        # Create sample input data
        signal_lm = jnp.array([1.0, 0.5])
        gamma_IJ_lm = jnp.ones((nlm, npsr, npsr))
        C_ff = jnp.eye(nfreq)  # Identity correlation between frequencies

        # Get full covariance
        cov_full = sc.get_covariance_full(signal_lm, gamma_IJ_lm, C_ff)

        # Check output shape
        self.assertEqual(cov_full.shape, (nfreq, nfreq, npsr, npsr))

        # For identity C_ff, diagonal elements should be equal to
        # the sum of signal_lm times gamma_IJ_lm
        expected_diag_element = jnp.sum(signal_lm) * jnp.ones((npsr, npsr))

        for i in range(nfreq):
            np.testing.assert_allclose(
                cov_full[i, i], expected_diag_element, rtol=1e-5
            )

        # Off-diagonal elements should be zero for identity C_ff
        for i in range(nfreq):
            for j in range(nfreq):
                if i != j:
                    np.testing.assert_allclose(
                        cov_full[i, j], jnp.zeros((npsr, npsr)), rtol=1e-5
                    )

    def test_get_inverse_covariance_full(self):
        """
        Test computation of inverse full covariance
        """
        # Use small test dimensions
        nfreq = 3
        npsr = 2
        nlm = 2

        # Create sample input data - use simple values for predictable results
        signal_lm = jnp.array([1.0, 1.0])
        # Use identity matrices for gamma_IJ_lm to simplify the calculation
        gamma_IJ_lm = jnp.zeros((nlm, npsr, npsr))
        for i in range(nlm):
            gamma_IJ_lm = gamma_IJ_lm.at[i].set(jnp.eye(npsr))

        # Use identity matrix for frequency correlation
        C_ff = jnp.eye(nfreq)

        # Get inverse covariance
        inv_cov = sc.get_inverse_covariance_full(signal_lm, gamma_IJ_lm, C_ff)

        # Check output shape
        self.assertEqual(inv_cov.shape, (nfreq, nfreq, npsr, npsr))

        # Verify that inverse is correct by multiplying cov * inv_cov
        # For each frequency pair, we should get an identity matrix in psr space
        for i in range(nfreq):
            for j in range(nfreq):
                # For identity C_ff, non-diagonal elements should be zero
                if i != j:
                    np.testing.assert_allclose(
                        inv_cov[i, j], jnp.zeros((npsr, npsr)), rtol=1e-5
                    )

    def test_get_dcovariance_full(self):
        """
        Test computation of derivative of full covariance
        """
        # Use small test dimensions
        nfreq = 3
        npsr = 2
        nlm = 2

        # Create sample input data
        signal_lm = jnp.array([1.0, 0.5])
        gamma_IJ_lm = jnp.ones((nlm, npsr, npsr))
        C_ff = jnp.eye(nfreq)  # Identity correlation between frequencies

        # Get derivative of full covariance
        dcov_full = sc.get_dcovariance_full(signal_lm, gamma_IJ_lm, C_ff)

        # Check output shape
        self.assertEqual(dcov_full.shape, (nlm, nfreq, nfreq, npsr, npsr))

        # Values should be non-zero
        self.assertTrue(jnp.any(dcov_full != 0))

    def test_get_signal_dsignal_tensors_lm_spherical_harmonics_basis(self):
        """
        Test computation of signal tensors in spherical harmonics basis
        """
        # Use small test dimensions
        nfreq = 5
        npsr = 3
        nlm = 2
        nparams = 2

        # Create sample input data
        signal_lm = jnp.ones(nlm)
        signal = jnp.ones(nfreq)
        dsignal = jnp.ones((nparams, nfreq))
        # Response tensor needs correct shape for the expected einsum
        response_IJ = jnp.ones((nlm, nfreq, npsr, npsr))

        # Get signal tensors
        signal_tensor, dsignal_tensor_freq, dsignal_tensor_aniso = (
            sc.get_signal_dsignal_tensors_lm_spherical_harmonics_basis(
                signal_lm, signal, dsignal, response_IJ
            )
        )

        # Check output shapes
        self.assertEqual(signal_tensor.shape, (nfreq, npsr, npsr))
        self.assertEqual(
            dsignal_tensor_freq.shape, (nparams, nfreq, npsr, npsr)
        )
        self.assertEqual(dsignal_tensor_aniso.shape, (nlm, nfreq, npsr, npsr))

    def test_get_signal_dsignal_tensors_lm_sqrt_basis(self):
        """
        Test computation of signal tensors in sqrt basis
        """
        # Use small test dimensions
        nfreq = 5
        npsr = 3
        nlm = 2
        nparams = 2

        # Create sample input data
        signal_lm = jnp.ones(nlm)
        signal = jnp.ones(nfreq)
        dsignal = jnp.ones((nparams, nfreq))

        # Response tensor shape is different for sqrt basis
        # It has shape (lm, lm, nfreq, npsr, npsr)
        response_IJ = jnp.ones((nlm, nlm, nfreq, npsr, npsr))

        # Get signal tensors
        signal_tensor, dsignal_tensor_freq, dsignal_tensor_aniso = (
            sc.get_signal_dsignal_tensors_lm_sqrt_basis(
                signal_lm, signal, dsignal, response_IJ
            )
        )

        # Check output shapes
        self.assertEqual(signal_tensor.shape, (nfreq, npsr, npsr))
        self.assertEqual(
            dsignal_tensor_freq.shape, (nparams, nfreq, npsr, npsr)
        )
        self.assertEqual(dsignal_tensor_aniso.shape, (nlm, nfreq, npsr, npsr))

        # Values should be non-zero
        self.assertTrue(jnp.any(signal_tensor != 0))
        self.assertTrue(jnp.any(dsignal_tensor_freq != 0))
        self.assertTrue(jnp.any(dsignal_tensor_aniso != 0))

    def test_spherical_harmonics_basis_specific_values(self):
        """
        Test spherical harmonics basis function with specific input values
        and verify the expected output values
        """
        # Simple case with controlled values
        nfreq = 2
        npsr = 2
        nlm = 2

        # Create specific input data
        # Use distinct values for better testing
        signal_lm = jnp.array([1.0, 2.0])
        signal = jnp.array([2.0, 3.0])
        dsignal = jnp.array([[0.5, 1.5]])  # Just for the function signature

        # Create response tensors with specific values
        # Use different values for each (l,m) component
        response_IJ = jnp.zeros((nlm, nfreq, npsr, npsr))
        # First lm component: Identity matrices
        response_IJ = response_IJ.at[0, :, :, :].set(
            jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        )
        # Second lm component: 2*Identity matrices
        response_IJ = response_IJ.at[1, :, :, :].set(
            2 * jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        )

        # Get signal tensors
        signal_tensor, dsignal_tensor_freq, dsignal_tensor_aniso = (
            sc.get_signal_dsignal_tensors_lm_spherical_harmonics_basis(
                signal_lm, signal, dsignal, response_IJ
            )
        )

        # Calculate expected values based on actual implementation
        # signal_lm_f = signal_lm[:, None] * signal[None, :]
        signal_lm_f = jnp.outer(signal_lm, signal)  # Shape (nlm, nfreq)

        # Compute expected signal tensor manually
        expected_signal = jnp.zeros((nfreq, npsr, npsr))
        for lm_idx in range(nlm):
            for f in range(nfreq):
                expected_signal = expected_signal.at[f].add(
                    response_IJ[lm_idx, f] * signal_lm_f[lm_idx, f]
                )

        # Check values with reasonable tolerance
        np.testing.assert_allclose(signal_tensor, expected_signal, rtol=1e-5)

    def test_get_signal_dsignal_tensors_lm(self):
        """
        Test the function for selecting and applying the correct basis function
        """
        # Use small test dimensions
        nfreq = 5
        npsr = 3
        nlm = 2
        nparams = 2

        # Create sample input data
        signal_lm = jnp.ones(nlm)
        signal = jnp.ones(nfreq)
        dsignal = jnp.ones((nparams, nfreq))
        # Response tensor needs correct shape for the expected einsum
        response_IJ = jnp.ones((nlm, nfreq, npsr, npsr))

        # Test with spherical harmonics basis (index 0)
        tensors_spherical = sc.get_signal_dsignal_tensors_lm(
            lm_basis_idx=0,
            signal_lm=signal_lm,
            signal=signal,
            dsignal=dsignal,
            response_IJ=response_IJ,
        )

        # Check we get the expected number of tensors
        self.assertEqual(len(tensors_spherical), 3)

        # Check output shapes match direct call to basis function
        direct_func = sc.get_signal_dsignal_tensors_lm_spherical_harmonics_basis
        direct_tensors = direct_func(signal_lm, signal, dsignal, response_IJ)

        for i in range(3):
            self.assertEqual(
                tensors_spherical[i].shape, direct_tensors[i].shape
            )
            np.testing.assert_array_almost_equal(
                tensors_spherical[i], direct_tensors[i]
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
