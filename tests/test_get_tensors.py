# Global
import unittest

import numpy as np

import jax
import jax.numpy as jnp

# Local
import test_utils as tu
from fastPTA.get_tensors import get_tensors


jax.config.update("jax_enable_x64", True)

# If you want to use your GPU change here
jax.config.update("jax_default_device", jax.devices("cpu")[0])


@tu.not_a_test
def save_tensors(path_to_pulsar_catalog, n_pulsars, method, order):
    res = get_tensors(
        tu.test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        save_catalog=True,
        n_pulsars=n_pulsars,
        regenerate_catalog=True,
        method=method,
        order=order,
        **tu.EPTAlike_test
    )

    HD_shape = order + 1 if order else order
    test_shapes = [
        (len(tu.test_frequency), n_pulsars, n_pulsars),
        (len(tu.test_frequency), n_pulsars, n_pulsars),
        (HD_shape, len(tu.test_frequency), n_pulsars, n_pulsars),
        (HD_shape,),
    ]

    return res, test_shapes


@tu.not_a_test
def get_tensors_data(path_to_pulsar_catalog, method, order, data_path):

    get_tensor_results = get_tensors(
        tu.test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        method=method,
        order=order,
    )

    return get_tensor_results, np.load(data_path)


class TestGetTensors(unittest.TestCase):

    def test_get_tensors_generation(self):
        result, test_shapes = save_tensors(
            tu.test_catalog_path2, 30, "legendre", 0
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_generation_Legendre(self):
        result, test_shapes = save_tensors(
            tu.test_catalog_path2, 50, "legendre", 6
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_generation_Binned(self):
        result, test_shapes = save_tensors(
            tu.test_catalog_path2, 30, "binned", 10
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_results(self):

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "legendre",
            0,
            tu.get_tensors_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            self.assertAlmostEqual(
                jnp.sum(
                    jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
                ),
                0.0,  # type: ignore
            )

    def test_get_tensors_Binned_results(self):

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "binned",
            6,
            tu.get_tensors_Binned_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            self.assertAlmostEqual(
                jnp.sum(
                    jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
                ),
                0.0,  # type: ignore
            )

    def test_get_tensors_Legendre_results(self):

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "legendre",
            6,
            tu.get_tensors_Legendre_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            self.assertAlmostEqual(
                jnp.sum(
                    jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
                ),
                0.0,  # type: ignore
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
