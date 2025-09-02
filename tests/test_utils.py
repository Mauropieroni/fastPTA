# Global
import os
import unittest

import jax.numpy as jnp
import numpy as np
import yaml

# Local
from fastPTA import utils as ut


class TestUtils(unittest.TestCase):

    def test_compare_versions(self):
        """
        Test the compare_versions function
        """
        # Test cases where version1 > version2
        self.assertTrue(ut.compare_versions("1.2.3", "1.2.2"))
        self.assertTrue(ut.compare_versions("2.0.0", "1.9.9"))
        self.assertTrue(ut.compare_versions("1.3.0", "1.2.9"))

        # Test cases where version1 < version2
        self.assertFalse(ut.compare_versions("1.2.2", "1.2.3"))
        self.assertFalse(ut.compare_versions("1.9.9", "2.0.0"))
        self.assertFalse(ut.compare_versions("1.2.9", "1.3.0"))

        # Test cases where version1 == version2
        self.assertTrue(ut.compare_versions("1.2.3", "1.2.3"))
        self.assertTrue(ut.compare_versions("0.0.0", "0.0.0"))

    def test_dot_product(self):
        """
        Test the dot_product function
        """
        # Test parallel vectors (along z-axis)
        theta1 = 0.0  # z-axis
        phi1 = 0.0
        theta2 = 0.0  # z-axis
        phi2 = 0.0
        result = ut.dot_product(theta1, phi1, theta2, phi2)
        self.assertAlmostEqual(float(result), 1.0)

        # Test anti-parallel vectors (along z-axis)
        theta1 = 0.0  # z-axis
        phi1 = 0.0
        theta2 = np.pi  # negative z-axis
        phi2 = 0.0
        result = ut.dot_product(theta1, phi1, theta2, phi2)
        self.assertAlmostEqual(float(result), -1.0)

        # Test perpendicular vectors (z-axis and x-axis)
        theta1 = 0.0  # z-axis
        phi1 = 0.0
        theta2 = np.pi / 2  # x-axis
        phi2 = 0.0
        result = ut.dot_product(theta1, phi1, theta2, phi2)
        self.assertAlmostEqual(float(result), 0.0)

        # Test with arrays
        thetas1 = jnp.array([0.0, np.pi / 4, np.pi / 2])
        phis1 = jnp.array([0.0, 0.0, 0.0])
        thetas2 = jnp.array([0.0, np.pi / 4, np.pi / 2])
        phis2 = jnp.array([0.0, 0.0, 0.0])
        results = ut.dot_product(thetas1, phis1, thetas2, phis2)
        expected = jnp.array([1.0, 1.0, 1.0])
        np.testing.assert_allclose(results, expected)

    def test_characteristic_strain_to_Omega(self):
        """
        Test the characteristic_strain_to_Omega function
        """
        # Test with a single frequency
        frequency = 1e-8  # Hz
        result = ut.characteristic_strain_to_Omega(frequency)
        expected = 2 * np.pi**2 * frequency**2 / 3 / ut.Hubble_over_h**2
        self.assertAlmostEqual(float(result), float(expected))

        # Test with an array of frequencies
        frequencies = jnp.array([1e-9, 1e-8, 1e-7])
        results = ut.characteristic_strain_to_Omega(frequencies)
        expected = 2 * np.pi**2 * frequencies**2 / 3 / ut.Hubble_over_h**2
        np.testing.assert_allclose(results, expected)

    def test_strain_to_Omega(self):
        """
        Test the strain_to_Omega function
        """
        # Test with a single frequency
        frequency = 1e-8  # Hz
        result = ut.strain_to_Omega(frequency)
        expected = 2 * np.pi**2 * frequency**3 / 3 / ut.Hubble_over_h**2
        self.assertAlmostEqual(float(result), float(expected))

        # Test with an array of frequencies
        frequencies = jnp.array([1e-9, 1e-8, 1e-7])
        results = ut.strain_to_Omega(frequencies)
        expected = 2 * np.pi**2 * frequencies**3 / 3 / ut.Hubble_over_h**2
        np.testing.assert_allclose(results, expected)

    def test_hc_from_CP(self):
        """
        Test the hc_from_CP function
        """
        # Test with a single frequency
        CP = 1e-30  # seconds^3
        frequency = 1e-8  # Hz
        T_obs_s = 3.0 * 365.25 * 24 * 3600  # 3 years in seconds
        result = ut.hc_from_CP(CP, frequency, T_obs_s)
        expected = (
            2 * np.sqrt(3) * CP * frequency**1.5 * np.pi * np.sqrt(T_obs_s)
        )
        self.assertAlmostEqual(float(result), float(expected))

        # Test with arrays
        CPs = jnp.array([1e-30, 2e-30, 3e-30])
        frequencies = jnp.array([1e-9, 1e-8, 1e-7])
        results = ut.hc_from_CP(CPs, frequencies, T_obs_s)
        expected = (
            2 * np.sqrt(3) * CPs * frequencies**1.5 * np.pi * np.sqrt(T_obs_s)
        )
        np.testing.assert_allclose(results, expected)

    def test_load_yaml(self):
        """
        Test the load_yaml function
        """
        # Create a temporary YAML file
        test_yaml_path = os.path.join(
            os.path.dirname(__file__), "test_data", "test_config.yaml"
        )
        os.makedirs(os.path.dirname(test_yaml_path), exist_ok=True)

        test_data = {
            "test_key": "test_value",
            "test_list": [1, 2, 3],
            "test_dict": {"nested_key": "nested_value"},
        }

        with open(test_yaml_path, "w") as f:
            yaml.dump(test_data, f)

        # Test loading the file
        loaded_data = ut.load_yaml(test_yaml_path)

        # Check that the loaded data matches the original data
        self.assertEqual(loaded_data["test_key"], test_data["test_key"])
        self.assertEqual(loaded_data["test_list"], test_data["test_list"])
        nested_key = "nested_key"
        self.assertEqual(
            loaded_data["test_dict"][nested_key],
            test_data["test_dict"][nested_key],
        )

        # Clean up
        os.remove(test_yaml_path)

    def test_compute_inverse(self):
        """
        Test the compute_inverse function
        """
        # Test with a simple 2x2 matrix
        matrix = jnp.array([[4.0, 1.0], [1.0, 3.0]])
        result = ut.compute_inverse(matrix)
        # Analytically computed inverse
        expected = jnp.array([[3 / 11, -1 / 11], [-1 / 11, 4 / 11]])
        np.testing.assert_allclose(result, expected, rtol=1e-5)

        # Test with a batch of matrices
        matrices = jnp.array(
            [[[4.0, 1.0], [1.0, 3.0]], [[2.0, 0.5], [0.5, 2.0]]]
        )
        results = ut.compute_inverse(matrices)
        expected1 = jnp.array([[3 / 11, -1 / 11], [-1 / 11, 4 / 11]])

        # For the second matrix, compute inverse directly
        matrix2 = matrices[1]
        expected2 = jnp.linalg.inv(matrix2)

        np.testing.assert_allclose(results[0], expected1, rtol=1e-5)
        np.testing.assert_allclose(results[1], expected2, rtol=1e-5)

        # Test that A * inv(A) = I
        identity = jnp.matmul(matrix, result)
        np.testing.assert_allclose(identity, jnp.eye(2), atol=1e-14)

    def test_logdet_kronecker_product(self):
        """
        Test the logdet_kronecker_product function
        """
        # Test with simple matrices
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        B = jnp.array([[4.0, 1.0], [1.0, 5.0]])

        result = ut.logdet_kronecker_product(A, B)

        # Calculate expected logdet manually
        # For Kronecker product, det(A ⊗ B) = det(A)^dim(B) * det(B)^dim(A)
        # Thus, log(det(A ⊗ B)) = dim(B) * log(det(A)) + dim(A) * log(det(B))
        logdet_A = jnp.log(jnp.linalg.det(A))
        logdet_B = jnp.log(jnp.linalg.det(B))
        expected = 2 * logdet_A + 2 * logdet_B

        self.assertAlmostEqual(float(result), float(expected))

    def test_compute_D_IJ_mean(self):
        """
        Test the compute_D_IJ_mean function
        """
        # Create test data
        x = np.linspace(0, np.pi, 100)
        y = np.cos(x)  # Simple function with known behavior
        nbins = 10

        # Compute results
        bin_means, bin_std, bin_centers = ut.compute_D_IJ_mean(x, y, nbins)

        # Check shape of results
        self.assertEqual(len(bin_means), nbins)
        self.assertEqual(len(bin_std), nbins)
        self.assertEqual(len(bin_centers), nbins)

        # Check if bin centers are evenly spaced
        bin_width = bin_centers[1] - bin_centers[0]
        for i in range(1, nbins - 1):
            self.assertAlmostEqual(
                bin_centers[i + 1] - bin_centers[i], bin_width, places=5
            )

        # Check if mean values follow the expected trend (decreasing for cosine)
        for i in range(nbins - 1):
            self.assertGreaterEqual(bin_means[i], bin_means[i + 1])

    def test_compute_pulsar_average_D_IJ(self):
        """
        Test the compute_pulsar_average_D_IJ function
        """
        # Create test data: 3 realizations, each with a 4x4 correlation matrix
        n_realizations = 3
        n_pulsars = 4

        # Create angles between pulsars (symmetric matrix)
        ang_list = np.zeros((n_realizations, n_pulsars, n_pulsars))
        for r in range(n_realizations):
            for i in range(n_pulsars):
                for j in range(i + 1, n_pulsars):
                    ang = np.random.uniform(0, np.pi)
                    ang_list[r, i, j] = ang
                    ang_list[r, j, i] = ang

        # Create correlation values based on angles (using HD curve as example)
        def hd_curve(ang):
            x = 0.5 * (1 - np.cos(ang))
            return x * np.log(x) - 0.25 * (1 - np.cos(ang))

        D_IJ_list = np.zeros_like(ang_list)
        for r in range(n_realizations):
            for i in range(n_pulsars):
                for j in range(n_pulsars):
                    if i != j:
                        D_IJ_list[r, i, j] = hd_curve(ang_list[r, i, j])

        # Compute results
        x_avg, y_avg = ut.compute_pulsar_average_D_IJ(
            ang_list, D_IJ_list, nbins=5
        )

        # Check shapes
        self.assertEqual(x_avg.shape[0], n_realizations)
        self.assertEqual(y_avg.shape[0], n_realizations)
        self.assertEqual(x_avg.shape[1], 5)  # nbins
        self.assertEqual(y_avg.shape[1], 5)  # nbins

    def test_get_R(self):
        """
        Test the get_R function for computing Gelman-Rubin statistic
        """
        # Create mock MCMC samples with known properties
        # 3 chains, 2 parameters, 1000 steps each
        n_steps = 1000
        n_chains = 3
        n_params = 2

        # Create samples with different means for different chains
        samples = np.zeros((n_steps, n_chains, n_params))
        for chain in range(n_chains):
            # Parameter 1: All chains have same mean - should converge
            samples[:, chain, 0] = np.random.normal(5.0, 1.0, n_steps)

            # Parameter 2: Different means - should not converge well
            samples[:, chain, 1] = np.random.normal(5.0 + chain, 1.0, n_steps)

        # Compute R statistic
        R = ut.get_R(samples)

        # Check shape
        self.assertEqual(len(R), n_params)

        # First parameter should have R close to 1 (good convergence)
        self.assertLess(R[0], 1.1)

        # Second parameter should have R > 1 (poor convergence)
        self.assertGreater(R[1], 1.1)


if __name__ == "__main__":
    unittest.main()
