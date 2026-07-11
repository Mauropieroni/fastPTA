# Global
import unittest

import jax
import jax.numpy as jnp
import numpy as np

# Local
import utils as tu
from fastPTA import transmission_functions as tf
from fastPTA import utils as ut

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])


class TestTransmissionFunctions(unittest.TestCase):
    """Test class for transmission functions."""

    def setUp(self):
        """Set up test fixtures."""

        # Load saved data
        self.data = np.load(tu.transmission_data_path)

        # Test frequencies
        self.frequencies_single = self.data["frequencies"][
            0
        ]  # Single frequency in Hz

        # Test observation times
        self.T_obs_single = self.data["T_obs"][
            0
        ]  # Single observation time (~3 years)

    def test_transmission_function_approx_shape(self):
        """Test the shape of transmission_function_approx output."""
        # Test with single frequency and single T_obs
        result = tf.transmission_function_approx(
            self.frequencies_single, self.T_obs_single
        )
        self.assertEqual(result.ndim, 0)  # Scalar output

        # Test with array of frequencies and single T_obs
        result = tf.transmission_function_approx(
            self.data["frequencies"], self.T_obs_single
        )
        self.assertEqual(result.shape, self.data["frequencies"].shape)

        # Test with single frequency and array of T_obs
        result = tf.transmission_function_approx(
            self.frequencies_single, self.data["T_obs"]
        )
        self.assertEqual(result.shape, self.data["T_obs"].shape)

    def test_transmission_function_approx_values(self):
        """Test the values of transmission_function_approx."""
        # Test with single frequency and T_obs
        result = tf.transmission_function_approx(
            self.frequencies_single, self.T_obs_single
        )
        self.assertTrue(jnp.isfinite(result))
        # Transmission should be between 0 and 1
        self.assertTrue(0 <= result <= 1)

        # Vector of frequencies and single T_obs
        result = tf.transmission_function_approx(
            self.data["frequencies"], self.T_obs_single
        )
        expected = self.data["transmission_approx_single"]

        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )

        # Single frequency and vector of T_obs
        result = tf.transmission_function_approx(
            self.data["frequencies"][:, None], self.data["T_obs"][None, :]
        )
        expected = self.data["transmission_approx_tensor"]

        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )

    def test_transmission_function_quadratic_shape(self):
        """Test the shape of transmission_function_quadratic output."""
        # Test with single frequency and single T_obs
        result = tf.transmission_function_quadratic(
            self.frequencies_single, self.T_obs_single
        )
        self.assertEqual(result.ndim, 0)  # Scalar output

        # Test with array of frequencies and single T_obs
        result = tf.transmission_function_quadratic(
            self.data["frequencies"], self.T_obs_single
        )
        self.assertEqual(result.shape, self.data["frequencies"].shape)

    def test_transmission_function_quadratic_values(self):
        """Test the values of transmission_function_quadratic."""
        # Test with single frequency and T_obs
        result = tf.transmission_function_quadratic(
            self.frequencies_single, self.T_obs_single
        )
        self.assertTrue(jnp.isfinite(result))
        # Quadratic transmission can be negative due to subtraction of terms
        self.assertTrue(result <= 1)

        # Vector of frequencies and single T_obs
        result = tf.transmission_function_quadratic(
            self.data["frequencies"], self.T_obs_single
        )
        expected = self.data["transmission_quadratic_single"]

        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )
        # Vector of frequencies and vector of T_obs
        result = tf.transmission_function_quadratic(
            self.data["frequencies"][:, None], self.data["T_obs"][None, :]
        )
        expected = self.data["transmission_quadratic_tensor"]

        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )

    def test_transmission_function_quadratic_1yr_peak_shape(self):
        """Test the shape of transmission_function_quadratic_1yr_peak output."""
        # Test with single frequency and single T_obs
        result = tf.transmission_function_quadratic_1yr_peak(
            self.frequencies_single, self.T_obs_single
        )
        self.assertEqual(result.ndim, 0)  # Scalar output

        # Test with array of frequencies and single T_obs
        result = tf.transmission_function_quadratic_1yr_peak(
            self.data["frequencies"], self.T_obs_single
        )
        self.assertEqual(result.shape, self.data["frequencies"].shape)

    def test_transmission_function_quadratic_1yr_peak_values(self):
        """Test the values of transmission_function_quadratic_1yr_peak."""
        # Test with single frequency and T_obs
        result = tf.transmission_function_quadratic_1yr_peak(
            self.frequencies_single, self.T_obs_single
        )
        self.assertTrue(jnp.isfinite(result))
        # Quadratic transmission can be negative due to subtraction of terms
        self.assertTrue(result <= 1)

        # Test array values
        # Vector of frequencies and single T_obs
        result = tf.transmission_function_quadratic_1yr_peak(
            self.data["frequencies"], self.T_obs_single
        )
        expected = self.data["transmission_quadratic_1yr_peak_single"]

        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )
        # Vector of frequencies and vector of T_obs
        result = tf.transmission_function_quadratic_1yr_peak(
            self.data["frequencies"][:, None], self.data["T_obs"][None, :]
        )
        expected = self.data["transmission_quadratic_1yr_peak_tensor"]
        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )

    def test_transmission_function_matrix_values(self):
        """Test the values of transmission_function_matrix."""

        # Test array values
        # Vector of frequencies and single T_obs
        result = tf.transmission_function_matrix(
            self.data["frequencies"], self.data["t"], self.data["Mmat"]
        )
        expected = self.data["transmission_matrix_single"]

        self.assertAlmostEqual(
            float(jnp.sum(result - expected)), 0.0, places=10
        )

    def test_transmission_function_matrix_shape(self):
        """Test the shape of transmission_function_matrix output."""

        result = tf.transmission_function_matrix(
            self.data["frequencies"], self.data["t"], self.data["Mmat"]
        )

        self.assertEqual(result.shape, self.data["frequencies"].shape)
        self.assertEqual(len(result), len(self.data["frequencies"]))

    def test_transmission_function_matrix_finite_and_real(self):
        """Test the values of transmission_function_matrix."""
        result = tf.transmission_function_matrix(
            self.data["frequencies"], self.data["t"], self.data["Mmat"]
        )

        # Check that all values are finite
        self.assertTrue(jnp.all(jnp.isfinite(result)))

        # Check that result is real (should be due to the jnp.real call)
        self.assertTrue(jnp.all(jnp.isreal(result)))

    def test_transmission_function_matrix_multi_pulsar(self):
        """Stacked multi-pulsar input matches the single-pulsar result."""
        # Build list inputs (one entry per pulsar), as stated in the docstring
        toas = [self.data["t"], self.data["t"]]
        Mmats = [self.data["Mmat"], self.data["Mmat"]]

        # Stack the same way get_tensors now does, then call the function
        result = tf.transmission_function_matrix(
            self.data["frequencies"],
            jnp.asarray(toas),
            jnp.asarray(Mmats),
        )

        # Two pulsars, n frequencies: expected shape (2, n_frequencies)
        n_freq = len(self.data["frequencies"])
        self.assertEqual(result.shape, (2, n_freq))

        # Each pulsar must reproduce the known single-pulsar result
        expected = self.data["transmission_matrix_single"]
        for row in result:
            self.assertAlmostEqual(
                float(jnp.sum(row - expected)), 0.0, places=10
            )


if __name__ == "__main__":
    unittest.main()
