# Global
import sys
import unittest

import numpy as np
import healpy as hp

if sys.version_info.minor > 10:
    from scipy.special import sph_harm_y
else:
    from scipy.special import sph_harm

    sph_harm_y = lambda l, m, theta, phi: sph_harm(m, l, phi, theta)

import jax
import jax.numpy as jnp

# Local
import utils as tu
from fastPTA import utils as ut
from fastPTA import get_tensors as gt

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
    A utility function that returns the results of the get_tensors function and the expected shapes of the results.

    """

    res = gt.get_tensors(
        tu.test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        save_catalog=True,
        n_pulsars=n_pulsars,
        regenerate_catalog=True,
        HD_basis=HD_basis,
        HD_order=HD_order,
        **tu.EPTAlike_test
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
    A utility function that returns the results of the get_tensors function and the expected results.

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
        pulsars = 15

        theta, phi = hp.pix2ang(
            nside, np.linspace(0, npix - 1, npoints, dtype=int)
        )

        theta_p, phi_p = hp.pix2ang(
            nside, np.array(np.random.uniform(0, npix - 1, pulsars), dtype=int)
        )

        analytical = gt.gamma_analytical(
            theta_p,
            phi_p,
            theta,
            phi,
        )

        p_I = gt.unit_vector(theta_p, phi_p)
        hat_k = gt.unit_vector(theta, phi)
        gamma = gt.gamma(p_I, hat_k)

        self.assertAlmostEqual(
            float(jnp.sum(np.abs(gamma - analytical))),
            0.0,
            places=7,
        )

    def test_spherical_harmonics_multipoles_2(self):
        """
        Test the spherical harmonics projection function for pulsars pulsar matrix

        """

        Y00 = np.array([[sph_harm_y(0, 0, theta, phi).real]])
        Y1m1 = np.array([[np.sqrt(2.0) * sph_harm_y(1, -1, theta, phi).imag]])
        Y10 = np.array([[sph_harm_y(1, 0, theta, phi).real]])
        Y1p1 = np.array([[-np.sqrt(2.0) * sph_harm_y(1, 1, theta, phi).real]])

        Y2m2 = np.array([[np.sqrt(2.0) * sph_harm_y(2, -2, theta, phi).imag]])
        Y2m1 = np.array([[np.sqrt(2.0) * sph_harm_y(2, -1, theta, phi).imag]])
        Y20 = np.array([[sph_harm_y(2, 0, theta, phi).real]])
        Y2p1 = np.array([[-np.sqrt(2.0) * sph_harm_y(2, 1, theta, phi).real]])
        Y2p2 = np.array([[np.sqrt(2.0) * sph_harm_y(2, 2, theta, phi).real]])

        Y3m3 = np.array([[np.sqrt(2.0) * sph_harm_y(3, -3, theta, phi).imag]])
        Y3p3 = np.array([[-np.sqrt(2.0) * sph_harm_y(3, 3, theta, phi).real]])

        res_Y00 = gt.projection_spherial_harmonics_basis(Y00, 1)
        res_Y1m1 = gt.projection_spherial_harmonics_basis(Y1m1, 1)
        res_Y10 = gt.projection_spherial_harmonics_basis(Y10, 1)
        res_Y1p1 = gt.projection_spherial_harmonics_basis(Y1p1, 1)

        res_Y2m2 = gt.projection_spherial_harmonics_basis(Y2m2, 2)
        res_Y2m1 = gt.projection_spherial_harmonics_basis(Y2m1, 2)
        res_Y20 = gt.projection_spherial_harmonics_basis(Y20, 2)
        res_Y2p1 = gt.projection_spherial_harmonics_basis(Y2p1, 2)
        res_Y2p2 = gt.projection_spherial_harmonics_basis(Y2p2, 2)

        res_Y3m3 = gt.projection_spherial_harmonics_basis(Y3m3, 3)
        res_Y3p3 = gt.projection_spherial_harmonics_basis(Y3p3, 3)

        self.assertAlmostEqual(jnp.abs(res_Y00[0] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y1m1[1] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y10[2] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y1p1[3] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y2m2[4] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y2m1[5] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y20[6] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y2p1[7] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y2p2[8] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y3m3[9] - 1.0), 0.0, places=4)
        self.assertAlmostEqual(jnp.abs(res_Y3p3[15] - 1.0), 0.0, places=4)

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
            float(jnp.sum(jnp.abs(data - GammalmIJ))), 0.0, places=5
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
            self.assertAlmostEqual(float(to_assert), 0.0)

    def test_Legendre_projection(self):
        """
        Test the results of the Legendre projection of the get_tensors function (going to large HD_order the projection gives chi_IJ)

        """

        HD_order = 30
        n_pulsars = 70

        vec = np.random.uniform(-1.0, 1.0, n_pulsars)

        matrix = vec[:, None] * vec[None, :]

        for i in range(n_pulsars):
            matrix[i, i] = 1.0

        chi_IJ = gt.get_chi_tensor_IJ(matrix) - 0.5 * np.eye(n_pulsars)

        HD_coefficients = gt.get_HD_Legendre_coefficients(HD_order)
        polynomials = gt.get_polynomials_IJ(matrix, HD_order)
        HD_val = HD_coefficients[:, None, None] * polynomials

        diff = 1.0 - np.sum(HD_val, axis=0) / chi_IJ
        self.assertAlmostEqual(float(jnp.mean(jnp.abs(diff))), 0.0, places=3)

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
            self.assertAlmostEqual(float(to_assert), 0.0)

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

            self.assertAlmostEqual(float(to_assert), 0.0)

    def test_get_tensors_anisotropies(self):
        """
        Test the results of the get_tensors function with anisotropies, check that the monopole is correct

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

        to_assert = jnp.mean(
            jnp.abs(
                results[1][0] / np.sqrt(4 * np.pi) - loaded_data["response_IJ"]
            )
        )
        self.assertAlmostEqual(float(to_assert), 0.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
