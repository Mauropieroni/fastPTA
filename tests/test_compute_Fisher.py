# Global
import unittest

import numpy as np

import jax
import jax.numpy as jnp

# Local
import test_utils as tu
from fastPTA.Fisher_code import compute_fisher

jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])


class TestFisher(unittest.TestCase):

    def test_compute_fisher(self):

        data = np.load(tu.Fisher_data_path)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "verbose": True,
        }

        res = compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertEqual(jnp.sum(jnp.abs(res[0] - data["frequency"])), 0.0)
        self.assertEqual(jnp.sum(jnp.abs(res[1] - data["signal"])), 0.0)
        self.assertEqual(
            jnp.sum(jnp.abs(res[4] - data["effective_noise"])), 0.0
        )
        self.assertEqual(jnp.sum(jnp.abs(res[5] - data["SNR"])), 0.0)
        self.assertEqual(jnp.sum(jnp.abs(res[6] - data["fisher"])), 0.0)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "method": "legendre",
            "order": 6,
            "verbose": True,
        }

        res_HD_legendre = compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_legendre[6] - data["fisher_HD_legendre"])),
            0.0,  # type: ignore
            places=10,
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path,
            "method": "binned",
            "order": 10,
            "verbose": True,
        }

        res_HD_binned = compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_binned[6] - data["fisher_HD_binned"])),
            0.0,  # type: ignore
            places=10,
        )

    def test_compute_fisher_future(self):

        n_frequencies = 100
        data = np.load(tu.Fisher_data_path2)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path3,
            "verbose": True,
        }

        res = compute_fisher(
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

        res_HD_legendre = compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_legendre[6] - data["fisher_HD_legendre"])),
            0.0,
            places=10,
        )  # type: ignore

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": tu.test_catalog_path3,
            "method": "binned",
            "order": 10,
            "verbose": True,
        }

        res_HD_binned = compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res_HD_binned[6] - data["fisher_HD_binned"])),
            0.0,
            places=10,
        )  # type: ignore


if __name__ == "__main__":
    unittest.main(verbosity=2)
