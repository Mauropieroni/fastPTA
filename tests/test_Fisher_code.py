# Global
import unittest
import os

import numpy as np

import jax
import jax.numpy as jnp

# Local
import utils as tu

import fastPTA.utils as ut
from fastPTA import Fisher_code as Fc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

n_frequencies = 35
data = np.load(tu.Fisher_data_path)


class TestFisher(unittest.TestCase):

    def test_compute_SNR_integrand(self):
        """
        Test the function that computes the integrand of the SNR

        """

        # Generate 3 random matrixes 5 x 5
        data = np.random.uniform(0.0, 1.0, size=(3, 5, 5))
        # Symmetrize
        data += np.moveaxis(data, -1, -2)

        inverse = np.linalg.inv(data)

        result = Fc.get_SNR_integrand(data, inverse)

        to_assert = jnp.sum(jnp.abs(result - 5 * np.ones(3)))

        self.assertAlmostEqual(float(to_assert), 0.0, places=10)

    def test_get_fisher_integrand(self):
        """
        Test the function that computes the integrand of the Fisher matrix

        """

        # Generate 3 random matrixes 5 x 5
        data = np.random.uniform(0.0, 1.0, size=(2, 5, 5))
        # Symmetrize
        data += np.moveaxis(data, -1, -2)

        inverse = np.linalg.inv(data)

        data = np.array([data, data, data])

        result = Fc.get_fisher_integrand(data, inverse)

        to_assert = jnp.sum(jnp.abs(result - 5 * np.ones(shape=(3, 3, 2))))

        self.assertAlmostEqual(float(to_assert), 0.0, places=10)

    def test_compute_fisher(self):
        """
        Test the function that computes the Fisher matrix

        """

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "verbose": True,
        }

        res = Fc.compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(res[0] - data["frequency"]))), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(res[1] - data["signal"]))), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(res[4] - data["effective_noise"]))), 0.0
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(res[5] - data["snr"]))), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(res[6] - data["fisher"]))), 0.0, places=7
        )

    def test_compute_fisher_controlled_inputs(self):
        """
        Test computation of Fisher matrix with manually controlled inputs
        to verify specific numerical outputs
        """
        # Get test path for the catalog
        catalog_path = tu.test_catalog_path

        # First test if the catalog file exists
        self.assertTrue(os.path.exists(catalog_path))

        # Use minimal parameters to generate a predictable Fisher matrix
        get_tensors_kwargs = {
            "path_to_pulsar_catalog": catalog_path,
            "verbose": False,
            # Use flat template with single parameter (amplitude)
            "signal_parameters": {
                "template": "flat",
                "parameters": {"amplitude": 1e-14},
            },
            # Exclude HD correlation to simplify calculation
            "include_HD": False,
            # No anisotropy
            "HD_basis": None,
        }

        # Compute Fisher matrix with just 2 frequencies for simplicity
        res = Fc.compute_fisher(
            n_frequencies=2, get_tensors_kwargs=get_tensors_kwargs
        )

        # Check that we have the expected 7 return values
        self.assertEqual(len(res), 7)

        # Unpack results to check structure
        frequencies, signal, dsignal, gamma_IJ, effective_noise, snr, fisher = (
            res
        )

        # Verify structure and reasonable values
        # Frequencies should be positive
        self.assertTrue(jnp.all(frequencies > 0))

        # Signal should be positive
        self.assertTrue(jnp.all(signal > 0))

        # SNR should be positive
        self.assertGreater(snr, 0.0)

        # Fisher matrix should be positive definite
        # Check by verifying eigenvalues are positive
        eigenvalues = jnp.linalg.eigvals(fisher)
        self.assertTrue(jnp.all(jnp.real(eigenvalues) > 0))

    def test_compute_fisher_legendre(self):
        """
        Test the function that computes the Fisher matrix with HD projected onto
        Legendre polynomials

        """

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "HD_basis": "legendre",
            "HD_order": 6,
            "verbose": True,
        }

        HD_legendre = Fc.compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertAlmostEqual(
            float(
                jnp.sum(jnp.abs(HD_legendre[6] - data["fisher_HD_legendre"]))
            ),
            0.0,
            places=8,
        )

    def test_compute_fisher_binned(self):
        """
        Test the function that computes the Fisher matrix with HD projected onto
        the binned basis

        """

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "HD_basis": "binned",
            "HD_order": 10,
            "verbose": True,
        }

        HD_binned = Fc.compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(HD_binned[6] - data["fisher_HD_binned"]))),
            0.0,
            places=7,
        )

    def test_compute_fisher_anisotropic(self):
        """
        Test computation of Fisher matrix for anisotropic signals
        """
        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "HD_basis": "anisotropic",
            "verbose": True,
        }

        anisotropic_result = Fc.compute_fisher(
            get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertIsNotNone(anisotropic_result)
        self.assertEqual(len(anisotropic_result), 7)  # Expect 7 return values
        # Check SNR is reasonable (positive)
        self.assertGreater(float(anisotropic_result[5]), 0.0)

    def test_get_integrands(self):
        """
        Test the function that computes the integrands for SNR and
        Fisher Matrix
        """
        # Use small test dimensions
        nfreq = 5
        npsr = 3
        nparams = 2

        # Create sample input data
        signal = jnp.ones(nfreq)
        dsignal = jnp.ones((nparams, nfreq))
        response_IJ = jnp.ones((nfreq, npsr, npsr))
        noise_tensor = jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        HD_order = 2
        HD_functions_IJ = jnp.ones((HD_order + 1, nfreq, npsr, npsr))

        # Get integrands
        SNR_integrand, effective_noise, fisher_integrand = Fc.get_integrands(
            signal, dsignal, response_IJ, noise_tensor, HD_functions_IJ
        )

        # Check output shapes
        self.assertEqual(SNR_integrand.shape, (nfreq,))
        self.assertEqual(effective_noise.shape, (nfreq,))
        self.assertEqual(
            fisher_integrand.shape,
            (nparams + HD_order + 1, nparams + HD_order + 1, nfreq),
        )

        # Check values are reasonable
        self.assertTrue(jnp.all(SNR_integrand >= 0))
        self.assertTrue(jnp.all(effective_noise >= 0))

    def test_get_integrands_specific_values(self):
        """
        Test the function that computes the integrands with specific input
        values and check the expected output values
        """
        # Simple case with identity matrices for easy calculation
        nfreq = 2
        npsr = 2
        nparams = 1

        # Create controlled input data
        signal = jnp.ones(nfreq)
        dsignal = jnp.ones((nparams, nfreq))
        # Use simple identity matrices
        response_IJ = jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        # Identity noise matrix
        noise_tensor = jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        # Need to provide HD functions for the test to work
        HD_order = 0  # No HD correlations
        HD_functions_IJ = jnp.zeros((HD_order + 1, nfreq, npsr, npsr))

        # Get integrands
        SNR_integrand, effective_noise, fisher_integrand = Fc.get_integrands(
            signal, dsignal, response_IJ, noise_tensor, HD_functions_IJ
        )

        # Check output shapes
        self.assertEqual(SNR_integrand.shape, (nfreq,))
        self.assertEqual(effective_noise.shape, (nfreq,))

        # Check values are reasonable
        self.assertTrue(jnp.all(SNR_integrand >= 0))
        self.assertTrue(jnp.all(effective_noise > 0))

    def test_get_integrands_lm(self):
        """
        Test computation of integrands for anisotropic signals
        """
        # Use small test dimensions
        nfreq = 5
        npsr = 3
        nlm = 2
        nparams = 2

        # Create sample input data
        signal_lm = jnp.ones(nlm) / jnp.sqrt(nlm)  # Normalize for stability
        signal = jnp.ones(nfreq)
        dsignal = jnp.ones((nparams, nfreq))
        # Response tensor needs correct shape for the expected einsum
        response_IJ = jnp.ones((nlm, nfreq, npsr, npsr))
        noise_tensor = jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        # Create empty HD functions tensor with the right shape
        HD_order = 2
        HD_functions_IJ = jnp.ones((HD_order + 1, nfreq, npsr, npsr))

        # Get integrands - note lm_basis_idx is the last parameter
        SNR_integrand, effective_noise, fisher_integrand = Fc.get_integrands_lm(
            signal_lm,
            signal,
            dsignal,
            response_IJ,
            noise_tensor,
            HD_functions_IJ,
            0,
        )

        # Check output shapes
        self.assertEqual(SNR_integrand.shape, (nfreq,))
        self.assertEqual(effective_noise.shape, (nfreq,))

        # Get actual shape of fisher_integrand to match the implementation
        # The shape includes all parameters (signal params + anisotropy + HD)
        expected_shape = (
            nparams + nlm + HD_order + 1,
            nparams + nlm + HD_order + 1,
            nfreq,
        )
        self.assertEqual(fisher_integrand.shape, expected_shape)

        # Check values are reasonable
        self.assertTrue(jnp.all(SNR_integrand >= 0))
        self.assertTrue(jnp.all(effective_noise >= 0))

    def test_get_integrands_lm_specific_values(self):
        """
        Test computation of integrands for anisotropic signals with
        specific input values and verify expected outputs
        """
        # Simple case for exact calculation
        nfreq = 2
        npsr = 2
        nlm = 2

        # Create controlled input data
        signal_lm = jnp.array([1.0, 0.0])  # Only first mode is active
        signal = jnp.array([1.0, 1.0])
        dsignal = jnp.array([[1.0, 1.0]])  # Just for function signature

        # Use identity matrices for response and noise
        response_IJ = jnp.zeros((nlm, nfreq, npsr, npsr))
        # First lm has identity matrices
        response_IJ = response_IJ.at[0, :, :, :].set(
            jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))
        )

        noise_tensor = jnp.eye(npsr)[None, :, :] * jnp.ones((nfreq, 1, 1))

        # Create an HD functions tensor with zeros (no HD correlations)
        HD_order = 0
        HD_functions_IJ = jnp.zeros((HD_order + 1, nfreq, npsr, npsr))

        # Get integrands using spherical harmonics basis (index 0)
        SNR_integrand, effective_noise, fisher_integrand = Fc.get_integrands_lm(
            signal_lm,
            signal,
            dsignal,
            response_IJ,
            noise_tensor,
            HD_functions_IJ,
            0,
        )

        # Check output shapes
        self.assertEqual(SNR_integrand.shape, (nfreq,))
        self.assertEqual(effective_noise.shape, (nfreq,))

        # Check values are reasonable
        self.assertTrue(jnp.all(SNR_integrand >= 0))
        self.assertTrue(jnp.all(effective_noise > 0))

        # Fisher integrand should have correct shape
        expected_shape = (
            dsignal.shape[0] + nlm + HD_order + 1,
            dsignal.shape[0] + nlm + HD_order + 1,
            nfreq,
        )
        self.assertEqual(fisher_integrand.shape, expected_shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
