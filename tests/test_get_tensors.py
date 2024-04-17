# Global
import unittest

# Local
from test_utils import *
from fastPTA.get_tensors import *


@not_a_test
def save_tensors(path_to_pulsar_catalog, n_pulsars, method, order):
    res = get_tensors(
        test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        save_catalog=True,
        n_pulsars=n_pulsars,
        regenerate_catalog=True,
        method=method,
        order=order,
        **EPTAlike_test
    )

    HD_shape = order + 1 if order else order
    test_shapes = [
        (len(test_frequency), n_pulsars, n_pulsars),
        (len(test_frequency), n_pulsars, n_pulsars),
        (HD_shape, len(test_frequency), n_pulsars, n_pulsars),
        (HD_shape,),
    ]

    return res, test_shapes


@not_a_test
def get_tensors_data(path_to_pulsar_catalog, method, order, data_path):

    get_tensor_results = get_tensors(
        test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        method=method,
        order=order,
    )

    return get_tensor_results, np.load(data_path)


class TestGetTensors(unittest.TestCase):

    def test_get_tensors_generation(self):
        res, test_shapes = save_tensors(test_catalog_path2, 30, "legendre", 0)
        for i in range(len(get_tensor_labels)):
            self.assertTupleEqual(res[i].shape, test_shapes[i])

    def test_get_tensors_generation_Legendre(self):
        res, test_shapes = save_tensors(test_catalog_path2, 50, "legendre", 6)
        for i in range(len(get_tensor_labels)):
            self.assertTupleEqual(res[i].shape, test_shapes[i])

    def test_get_tensors_generation_Binned(self):
        res, test_shapes = save_tensors(test_catalog_path2, 30, "binned", 10)
        for i in range(len(get_tensor_labels)):
            self.assertTupleEqual(res[i].shape, test_shapes[i])

    def test_get_tensors_results(self):

        results, loaded_data = get_tensors_data(
            test_catalog_path,
            "legendre",
            0,
            get_tensors_data_path,
        )

        for i in range(len(get_tensor_labels)):
            self.assertAlmostEqual(
                jnp.sum(
                    jnp.abs(results[i] - loaded_data[get_tensor_labels[i]])
                ),
                0.0,
            )

    def test_get_tensors_Binned_results(self):

        results, loaded_data = get_tensors_data(
            test_catalog_path,
            "binned",
            6,
            get_tensors_Binned_data_path,
        )

        for i in range(len(get_tensor_labels)):
            self.assertAlmostEqual(
                jnp.sum(
                    jnp.abs(results[i] - loaded_data[get_tensor_labels[i]])
                ),
                0.0,
            )

    def test_get_tensors_Legendre_results(self):

        results, loaded_data = get_tensors_data(
            test_catalog_path,
            "legendre",
            6,
            get_tensors_Legendre_data_path,
        )

        for i in range(len(get_tensor_labels)):
            self.assertAlmostEqual(
                jnp.sum(
                    jnp.abs(results[i] - loaded_data[get_tensor_labels[i]])
                ),
                0.0,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
