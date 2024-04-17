# Global
import unittest

# Local
from test_utils import *
from fastPTA.Fisher_code import *


class TestFisher(unittest.TestCase):

    def test_compute_fisher(self):

        data = np.load(Fisher_data_path)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": test_catalog_path,
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
            "path_to_pulsar_catalog": test_catalog_path,
            "method": "legendre",
            "order": 6,
            "verbose": True,
        }

        res_HD_legendre = compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertEqual(
            jnp.sum(jnp.abs(res_HD_legendre[6] - data["fisher_HD_legendre"])),
            0.0,
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": test_catalog_path,
            "method": "binned",
            "order": 10,
            "verbose": True,
        }

        res_HD_binned = compute_fisher(get_tensors_kwargs=get_tensors_kwargs)

        self.assertEqual(
            jnp.sum(jnp.abs(res_HD_binned[6] - data["fisher_HD_binned"])), 0.0
        )

    def test_compute_fisher_future(self):

        n_frequencies = 100
        data = np.load(Fisher_data_path2)

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": test_catalog_path3,
            "verbose": True,
        }

        res = compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertEqual(jnp.sum(jnp.abs(res[0] - data["frequency"])), 0.0)
        self.assertEqual(jnp.sum(jnp.abs(res[1] - data["signal"])), 0.0)
        self.assertAlmostEqual(
            jnp.sum(jnp.abs(res[4] - data["effective_noise"])), 0.0  # type: ignore
        )  # type: ignore
        self.assertAlmostEqual(jnp.sum(jnp.abs(res[5] - data["SNR"])), 0.0)  # type: ignore # type: ignore
        self.assertAlmostEqual(jnp.sum(jnp.abs(res[6] - data["fisher"])), 0.0)  # type: ignore

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": test_catalog_path3,
            "method": "legendre",
            "order": 6,
            "verbose": True,
        }

        res_HD_legendre = compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertEqual(
            jnp.sum(jnp.abs(res_HD_legendre[6] - data["fisher_HD_legendre"])),
            0.0,
        )

        get_tensors_kwargs = {
            "path_to_pulsar_catalog": test_catalog_path3,
            "method": "binned",
            "order": 10,
            "verbose": True,
        }

        res_HD_binned = compute_fisher(
            n_frequencies=n_frequencies, get_tensors_kwargs=get_tensors_kwargs
        )

        self.assertEqual(
            jnp.sum(jnp.abs(res_HD_binned[6] - data["fisher_HD_binned"])), 0.0
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
