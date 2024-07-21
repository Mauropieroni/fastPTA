# Global
import unittest

import numpy as np

import jax
import jax.numpy as jnp

# Local
import test_utils as tu

import fastPTA.utils as ut
from fastPTA import Fisher_code as Fc

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])


class TestFisher(unittest.TestCase):

    def test_compute_SNR_integrand(self):
        """
        Test the function that computes the integrand of the SNR.
        """

        # Generate 3 random matrixes 5 x 5
        data = np.random.uniform(0, 1, size=(3, 5, 5))
        # Symmetrize
        data += np.moveaxis(data, -1, -2)

        inverse = np.linalg.inv(data)

        result = Fc.get_SNR_integrand(data, inverse)

        to_assert = jnp.sum(jnp.abs(result - 5 * np.ones(3)))

        self.assertAlmostEqual(float(to_assert), 0.0, places=10)

    def test_get_fisher_integrand(self):
        """
        Test the function that computes the integrand of the Fisher matrix.
        """

        # Generate 3 random matrixes 5 x 5
        data = np.random.uniform(0, 1, size=(2, 5, 5))
        # Symmetrize
        data += np.moveaxis(data, -1, -2)

        inverse = np.linalg.inv(data)

        data = np.array([data, data, data])

        result = Fc.get_fisher_integrand(data, inverse)

        to_assert = jnp.sum(jnp.abs(result - 5 * np.ones(shape=(3, 3, 2))))

        self.assertAlmostEqual(float(to_assert), 0.0, places=10)

    def test_compute_fisher(self):
        """
        Test the function that computes the Fisher matrix.
        """

        n_frequencies = 30
        data = np.load(tu.Fisher_data_path)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "verbose": True,
        }

        res = Fc.compute_fisher(n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs)  # type: ignore

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
            float(jnp.sum(jnp.abs(res[5] - data["SNR"]))), 0.0, places=7
        )
        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(res[6] - data["fisher"]))), 0.0, places=7
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "method": "legendre",
            "order": 6,
            "verbose": True,
        }

        res_HD_legendre = Fc.compute_fisher(get_tensors_kwargs=get_tensors_kwargs)  # type: ignore

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_legendre[6] - data["fisher_HD_legendre"])),
            0.0,  # type: ignore
            places=8,
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "method": "binned",
            "order": 10,
            "verbose": True,
        }

        res_HD_binned = Fc.compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_binned[6] - data["fisher_HD_binned"])),
            0.0,  # type: ignore
            places=7,
        )

    def test_compute_fisher_future(self):
        """
        Test the function that computes the Fisher matrix using more frequencies
        and a different catalog
        """

        n_frequencies = 100
        data = np.load(tu.Fisher_data_path2)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path3,
            "verbose": True,
        }

        res = Fc.compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertEqual(jnp.sum(jnp.abs(res[0] - data["frequency"])), 0.0)
        self.assertEqual(jnp.sum(jnp.abs(res[1] - data["signal"])), 0.0)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res[4] - data["effective_noise"])),
            0.0,  # type: ignore
        )
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res[5] - data["SNR"])), 0.0  # type: ignore
        )
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res[6] - data["fisher"])), 0.0  # type: ignore
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path3,
            "method": "legendre",
            "order": 6,
            "verbose": True,
        }

        res_HD_legendre = Fc.compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_legendre[6] - data["fisher_HD_legendre"])),
            0.0,  # type: ignore
            places=7,
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path3,
            "method": "binned",
            "order": 10,
            "verbose": True,
        }

        res_HD_binned = Fc.compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_binned[6] - data["fisher_HD_binned"])),
            0.0,  # type: ignore
            places=7,
        )

    def test_compute_fisher_anisotropic(self):
        """
        TO ADD !!!
        """


if __name__ == "__main__":
    unittest.main(verbosity=2)
