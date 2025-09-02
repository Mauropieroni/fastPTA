import unittest
import numpy as np
import jax
import jax.numpy as jnp

from fastPTA.inference_tools import likelihoods
from fastPTA.signal_templates.power_law_template import power_law_model
from fastPTA.inference_tools.priors import Priors


class TestLikelihoods(unittest.TestCase):
    """Test cases for the likelihoods module."""

    def setUp(self):
        """Set up common test data."""
        # Set random seed for reproducibility
        np.random.seed(42)
        jax.config.update("jax_enable_x64", True)

        # Create test data
        self.n_frequencies = 5
        self.n_pulsars = 3
        self.frequency = jnp.logspace(-9, -7, self.n_frequencies)

        # Sample response_IJ tensor
        self.response_IJ = jnp.array(
            np.random.normal(
                0, 1, (self.n_frequencies, self.n_pulsars, self.n_pulsars)
            )
        )
        # Make it symmetric for each frequency
        for i in range(self.n_frequencies):
            self.response_IJ = self.response_IJ.at[i].set(
                (self.response_IJ[i] + self.response_IJ[i].T) / 2
            )

        # Sample strain_omega tensor (positive values for diagonal)
        self.strain_omega = jnp.array(
            np.random.gamma(
                2, 1, (self.n_frequencies, self.n_pulsars, self.n_pulsars)
            )
        )
        # Make it diagonal for each frequency
        for i in range(self.n_frequencies):
            self.strain_omega = self.strain_omega.at[i].set(
                jnp.diag(jnp.diag(self.strain_omega[i]))
            )

        # Generate mock data
        self.signal_value = jnp.array(
            np.random.gamma(2, 0.1, self.n_frequencies)
        )
        # Create covariance (not used directly but for reference)
        self.test_cov = (
            jnp.einsum("ijk,i->ijk", self.response_IJ, self.signal_value)
            + self.strain_omega
        )

        # Generate random data based on the covariance (real part)
        self.data = jnp.array(
            np.random.normal(
                0, 1, (self.n_frequencies, self.n_pulsars, self.n_pulsars)
            )
        )
        # Make data symmetric for each frequency
        for i in range(self.n_frequencies):
            self.data = self.data.at[i].set((self.data[i] + self.data[i].T) / 2)

        # Create parameters for power law model
        self.parameters = jnp.array([-15.0, 4.33])  # log_amplitude, tilt

        # Create prior object
        self.prior_dict = {
            "log_amplitude": {"uniform": {"loc": -18.0, "scale": 4.0}},
            "tilt": {"uniform": {"loc": 0.0, "scale": 7.0}},
        }
        self.priors = Priors(self.prior_dict)

        # Setup for log_likelihood_full
        self.l_max = 2
        self.num_coeffs = (self.l_max + 1) ** 2
        self.parameters_full = jnp.ones(self.num_coeffs)
        self.gamma_IJ_lm = jnp.ones(
            (self.num_coeffs, self.n_pulsars, self.n_pulsars)
        )
        self.C_ff = jnp.eye(self.n_frequencies)

        # Mock data for the full likelihood
        self.data_full = jnp.ones(
            (
                self.n_frequencies,
                self.n_frequencies,
                self.n_pulsars,
                self.n_pulsars,
            )
        )

        # For spherical harmonics testing
        self.Nside = 8

    def test_log_likelihood(self):
        """Test the log_likelihood function."""
        # Make sure strain_omega is positive definite
        for i in range(self.n_frequencies):
            self.strain_omega = self.strain_omega.at[i].set(
                jnp.diag(jnp.abs(jnp.diag(self.strain_omega[i])) + 1.0)
            )

        # Use a smaller signal value to avoid numerical issues
        self.signal_value = jnp.ones_like(self.signal_value) * 0.1

        # Compute log likelihood
        log_lik = likelihoods.log_likelihood(
            self.data, self.signal_value, self.response_IJ, self.strain_omega
        )

        # Check type and finite value
        self.assertTrue(jnp.isscalar(log_lik))
        self.assertTrue(jnp.isfinite(log_lik))

        # Test with different signal values
        signal_value_zero = jnp.zeros_like(self.signal_value)
        log_lik_zero = likelihoods.log_likelihood(
            self.data, signal_value_zero, self.response_IJ, self.strain_omega
        )
        self.assertTrue(jnp.isfinite(log_lik_zero))

        # Test that changing signal values affects the likelihood
        signal_value_diff = self.signal_value * 2
        log_lik_diff = likelihoods.log_likelihood(
            self.data, signal_value_diff, self.response_IJ, self.strain_omega
        )
        self.assertTrue(jnp.isfinite(log_lik_diff))
        self.assertNotEqual(log_lik, log_lik_diff)

    def test_log_posterior(self):
        """Test the log_posterior function."""
        # Compute log posterior with valid parameters
        log_post = likelihoods.log_posterior(
            self.parameters,
            self.data,
            self.frequency,
            power_law_model,
            self.response_IJ,
            self.strain_omega,
            self.priors,
        )

        # Check type and finite value
        self.assertTrue(jnp.isscalar(log_post))
        self.assertTrue(jnp.isfinite(log_post))

        # Test with parameters outside prior bounds
        parameters_outside = jnp.array([-19.0, 8.0])  # Outside prior bounds
        log_post_outside = likelihoods.log_posterior(
            parameters_outside,
            self.data,
            self.frequency,
            power_law_model,
            self.response_IJ,
            self.strain_omega,
            self.priors,
        )
        self.assertEqual(log_post_outside, -jnp.inf)

    def test_log_likelihood_full(self):
        """Test the log_likelihood_full function."""
        # Make a valid test case
        self.gamma_IJ_lm = (
            jnp.ones((self.num_coeffs, self.n_pulsars, self.n_pulsars))
            + jnp.eye(self.n_pulsars)[None, :, :]
        )  # Add identity to make pd

        # Compute full log likelihood
        log_lik_full = likelihoods.log_likelihood_full(
            self.parameters_full, self.data_full, self.gamma_IJ_lm, self.C_ff
        )

        # Check type and finite value
        self.assertTrue(jnp.isscalar(log_lik_full))
        self.assertTrue(jnp.isfinite(log_lik_full))

        # Test with different parameter values
        params_half = 0.5 * jnp.ones_like(self.parameters_full)
        log_lik_half = likelihoods.log_likelihood_full(
            params_half, self.data_full, self.gamma_IJ_lm, self.C_ff
        )
        self.assertTrue(jnp.isfinite(log_lik_half))
        self.assertNotEqual(log_lik_full, log_lik_half)

    def test_log_posterior_full(self):
        """Test the log_posterior_full function."""
        # Make a valid test case
        self.gamma_IJ_lm = (
            jnp.ones((self.num_coeffs, self.n_pulsars, self.n_pulsars))
            + jnp.eye(self.n_pulsars)[None, :, :]
        )  # Make positive definite

        # Create a valid map with all positive values
        self.parameters_full = jnp.ones_like(self.parameters_full)

        # Compute full log posterior with valid parameters
        log_post_full = likelihoods.log_posterior_full(
            self.parameters_full,
            self.Nside,
            self.l_max,
            self.data_full,
            self.gamma_IJ_lm,
            self.C_ff,
        )

        # Check type
        self.assertTrue(jnp.isscalar(log_post_full))

        # We just test that it runs, as the actual value depends on
        # the spherical harmonics implementation and the valid values
        # in the reconstructed map


if __name__ == "__main__":
    unittest.main()
