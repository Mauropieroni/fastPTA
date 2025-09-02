# Global
import unittest

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

# Local
import utils as tu

from fastPTA import get_tensors as gt
from fastPTA import utils as ut
from fastPTA.data import datastream as gds

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])


nside = 64
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, jnp.arange(npix))
theta = jnp.array(theta)
phi = jnp.array(phi)


@tu.not_a_test
def get_tensors_and_shapes(
    path_to_pulsar_catalog, n_pulsars, HD_basis, HD_order
):
    """
    A utility function that returns the results of the get_tensors function and
    the expected shapes of the results.

    """

    res = gt.get_tensors(
        tu.test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        save_catalog=True,
        n_pulsars=n_pulsars,
        regenerate_catalog=True,
        HD_basis=HD_basis,
        HD_order=HD_order,
        **tu.EPTAlike_test,
    )

    HD_shape = HD_order + 1 if HD_order else HD_order
    test_shapes = [
        (len(tu.test_frequency), n_pulsars, n_pulsars),
        (len(tu.test_frequency), n_pulsars, n_pulsars),
        (HD_shape, len(tu.test_frequency), n_pulsars, n_pulsars),
        (HD_shape,),
    ]

    return res, test_shapes


@tu.not_a_test
def get_tensors_data(
    path_to_pulsar_catalog, HD_basis, HD_order, data_path, anisotropies=False
):
    """
    A utility function that returns the results of the get_tensors function and
    the expected results.

    """

    get_tensor_results = gt.get_tensors(
        tu.test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        HD_basis=HD_basis,
        HD_order=HD_order,
        anisotropies=anisotropies,
    )

    return get_tensor_results, np.load(data_path)


class TestGetTensors(unittest.TestCase):

    def test_gamma(self):
        """
        Test the gamma function

        """

        npoints = 10

        theta, phi = hp.pix2ang(
            nside, np.linspace(0, npix - 1, npoints, dtype=int)
        )

        data = np.load(tu.get_tensors_data_path)

        p_I = data["p_vec"]
        theta_p = data["theta_p"]
        phi_p = data["phi_p"]

        analytical = gt.gamma_analytical(
            theta_p,
            phi_p,
            theta,
            phi,
        )

        hat_k = gds.unit_vector(theta, phi)
        gamma = gt.gamma(p_I, hat_k)

        self.assertAlmostEqual(
            float(jnp.sum(np.abs(gamma - analytical))),
            0.0,
            delta=1e-7,
        )

    def test_get_correlations_lm_IJ(self):
        """
        Test the get_correlations_lm_IJ function

        """

        l_max = 6
        nside = 32
        npixels = 35

        npix = hp.nside2npix(nside)
        pixels = np.linspace(0, npix - 1, npixels, dtype=int)
        p_I = jnp.array(hp.pix2vec(nside, pixels))
        GammalmIJ = gt.get_correlations_lm_IJ(p_I.T, l_max, nside)

        data = np.load(tu.get_correlations_lm_IJ_data_path)["data"]

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(data - GammalmIJ))), 0.0, delta=1e-5
        )

    def test_get_tensors_generation(self):
        """
        Test the get_tensors function

        """

        npulsars = 30
        HD_order = 0

        result, test_shapes = get_tensors_and_shapes(
            tu.test_catalog_path2, npulsars, "legendre", HD_order
        )

        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_generation_Legendre(self):
        """
        Test the get_tensors function assuming you want the Legendre projection

        """

        npulsars = 50
        HD_basis = "legendre"
        HD_order = 6

        result, test_shapes = get_tensors_and_shapes(
            tu.test_catalog_path2, npulsars, HD_basis, HD_order
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_generation_Binned(self):
        """
        Test the get_tensors function assuming you want the Binned projection

        """

        npulsars = 30
        HD_basis = "legendre"
        HD_order = 10

        result, test_shapes = get_tensors_and_shapes(
            tu.test_catalog_path2, npulsars, HD_basis, HD_order
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_results(self):
        """
        Test the get_tensors function results

        """

        HD_basis = "legendre"
        HD_order = 0

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            HD_basis,
            HD_order,
            tu.get_tensors_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            to_assert = jnp.sum(
                jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
            )
            self.assertAlmostEqual(float(to_assert), 0.0, delta=1e-9)

    def test_Legendre_projection(self):
        """
        Test the results of the Legendre projection of the get_tensors function
        (going to large HD_order the projection gives chi_IJ)

        """

        HD_order = 75
        n_pulsars = 70

        vec = np.random.uniform(-1.0, 1.0, n_pulsars)

        matrix = vec[:, None] * vec[None, :]

        for i in range(n_pulsars):
            matrix[i, i] = 1.0

        chi_IJ = gt.get_chi_tensor_IJ(matrix) - 0.5 * np.eye(n_pulsars)

        HD_coefficients = gt.get_HD_Legendre_coefficients(HD_order)
        polynomials = gt.get_polynomials_IJ(matrix, HD_order)
        HD_val = HD_coefficients[:, None, None] * polynomials

        diff = chi_IJ - np.sum(HD_val, axis=0)
        self.assertAlmostEqual(float(jnp.mean(jnp.abs(diff))), 0.0, delta=1e-4)

    def test_get_tensors_Binned_results(self):
        """
        Test the results of the Binned projection of the get_tensors function

        """

        HD_basis = "binned"
        HD_order = 6

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            HD_basis,
            HD_order,
            tu.get_tensors_Binned_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            to_assert = jnp.sum(
                jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
            )
            self.assertAlmostEqual(float(to_assert), 0.0, delta=1e-9)

    def test_get_tensors_Legendre_results(self):
        """
        Another test of the Legendre projection of the get_tensors function

        """

        HD_basis = "legendre"
        HD_order = 6

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            HD_basis,
            HD_order,
            tu.get_tensors_Legendre_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            to_assert = jnp.sum(
                jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
            )

            self.assertAlmostEqual(float(to_assert), 0.0, delta=1e-7)

    def test_get_tensors_anisotropies(self):
        """
        Test the results of the get_tensors function with anisotropies, check
        that the monopole is correct

        """

        HD_basis = "legendre"
        HD_order = 6

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            HD_basis,
            HD_order,
            tu.get_tensors_Legendre_data_path,
            anisotropies=True,
        )

        to_assert = jnp.max(
            jnp.abs(
                results[1][0] / np.sqrt(4 * np.pi) - loaded_data["response_IJ"]
            )
        )

        self.assertAlmostEqual(to_assert, 0.0, delta=1e-3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
