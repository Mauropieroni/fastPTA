# Global
import unittest

import numpy as np

import jax
import jax.numpy as jnp

# Local
import utils as tu

import fastPTA.utils as ut
from fastPTA.inference_tools import iterative_estimation as ie
from fastPTA.inference_tools import signal_covariance as sc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])

n_frequencies = 35
iterative_data = np.load(tu.iterative_estimation_data_path)


class TestIterativeEstimation(unittest.TestCase):
    def test_get_update_estimate_diagonal(self):
        """
        Test the computation of parameter updates using diagonal estimation
        method with specific inputs and outputs
        """
        # Use small test dimensions

        # Create controlled input data
        parameters = jnp.array([1.0, 0.5])

        # Create specific tensors with well-defined properties
        gamma_IJ_lm = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],  # First lm component
                [[0.5, 0.2], [0.2, 0.5]],  # Second lm component
            ]
        )

        # Define frequencies and spectrum
        ff = jnp.array([1e-8, 1e-7, 1e-6])
        S_f = jnp.array([2.0, 1.0, 0.5])

        # Create a controlled data tensor that's different from the covariance
        # First compute what the covariance would be
        C = sc.get_covariance_diagonal(parameters, gamma_IJ_lm, ff, S_f)
        # Then create data with a controlled offset
        data_tensor = C + 0.1

        # Get parameter update and Fisher matrix
        update, fisher_inv = ie.get_update_estimate_diagonal(
            parameters, data_tensor, gamma_IJ_lm, ff, S_f
        )

        # Check output shapes
        self.assertEqual(update.shape, parameters.shape)
        self.assertEqual(fisher_inv.shape, (len(parameters), len(parameters)))

        # Verify outputs match expected values (captured from a successful run)
        # With the specific test input, we'll get these expected values
        expected_update = jnp.array([1.190345e-11, -1.983933e-11])

        # Use higher tolerance for floating point comparisons
        np.testing.assert_allclose(
            update, expected_update, rtol=1e-5, atol=1e-13
        )

    def test_get_update_estimate_full(self):
        """
        Test the computation of parameter updates using full estimation method
        with specific inputs and outputs
        """

        # Create controlled input data
        parameters = jnp.array([1.0, 0.5])

        # Create specific tensors with well-defined properties
        gamma_IJ_lm = jnp.array(
            [
                [[1.0, 0.5], [0.5, 1.0]],  # First lm component
                [[0.5, 0.2], [0.2, 0.5]],  # Second lm component
            ]
        )

        # Define frequency correlation matrix (non-singular)
        C_ff = jnp.array([[1.0, 0.3], [0.3, 1.0]])

        # Create a controlled data tensor that's different from the covariance
        # First compute what the covariance would be
        C = sc.get_covariance_full(parameters, gamma_IJ_lm, C_ff)
        # Then create data with a controlled offset
        data_tensor = C + 0.1

        # Get parameter update and Fisher matrix
        update, fisher_inv = ie.get_update_estimate_full(
            parameters, data_tensor, gamma_IJ_lm, C_ff
        )

        # Check output shapes
        self.assertEqual(update.shape, parameters.shape)
        self.assertEqual(fisher_inv.shape, (len(parameters), len(parameters)))

        # For these inputs this should be the output
        expected_update = jnp.array([0.46153846, -0.76923077])

        # Check that our update matches the expected values with higher
        # tolerance for numerical stability
        np.testing.assert_allclose(
            update, expected_update, rtol=1e-5, atol=1e-13
        )

    def test_iterative_estimation(self):
        """
        Test the iterative_estimation function with specific inputs and outputs.
        Verifies that for known inputs, the function converges to expected
        values.
        """

        # Run iterative estimation
        result_theta, uncertainties, converged = ie.iterative_estimation(
            ie.get_update_estimate_diagonal,
            iterative_data["initial_theta"],
            iterative_data["data"],
            iterative_data["gamma_IJ_lm"],
            iterative_data["frequencies"],
            iterative_data["S_f"],
            i_max=100,
        )

        assert converged, "Iterative estimation did not converge"

        np.testing.assert_allclose(
            result_theta, iterative_data["theta"], rtol=1e-6
        )
        np.testing.assert_allclose(
            uncertainties, iterative_data["uncertainties"], rtol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
