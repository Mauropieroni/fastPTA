# Global
from cgi import test
import unittest

import numpy as np
import healpy as hp
from scipy.special import sph_harm

import jax
import jax.numpy as jnp

# Local
import test_utils as tu
from fastPTA import utils as ut
from fastPTA import get_tensors as gt

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])


@tu.not_a_test
def get_tensors_and_shapes(path_to_pulsar_catalog, n_pulsars, method, order):
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
def get_tensors_data(
    path_to_pulsar_catalog, method, order, data_path, anisotropies=False
):
    """
    A utility function that returns the results of the get_tensors function and
    the expected results.
    """

    get_tensor_results = gt.get_tensors(
        tu.test_frequency,
        path_to_pulsar_catalog=path_to_pulsar_catalog,
        method=method,
        order=order,
        anisotropies=anisotropies,
    )

    return get_tensor_results, np.load(data_path)


class TestGetTensors(unittest.TestCase):

    def test_gamma(self):
        """
        Test the gamma function.
        """

        nside = 16
        npoints = 10
        pulsars = 15

        npix = hp.nside2npix(nside)

        theta, phi = hp.pix2ang(
            nside, np.array(np.random.uniform(0, npix - 1, pulsars), dtype=int)
        )

        theta_k, phi_k = hp.pix2ang(
            nside, np.linspace(0, npix - 1, npoints, dtype=int)
        )

        analytical = gt.gamma_analytical(
            theta,
            phi,
            theta_k,
            phi_k,
        )

        p_I = gt.unit_vector(theta, phi)
        hat_k = gt.unit_vector(theta_k, phi_k)
        gamma = gt.gamma(p_I, hat_k)

        self.assertAlmostEqual(
            float(jnp.sum(np.abs(gamma - analytical))),
            0.0,
            places=7,
        )

    def test_spherical_harmonics_multipoles(self):
        """
        Test the spherical harmonics projection function.
        """

        nside = 64
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, jnp.arange(npix))
        theta = jnp.array(theta)
        phi = jnp.array(phi)

        Y00 = np.array(sph_harm(0.0, 0.0, phi, theta).real)
        Y1m1 = np.array(np.sqrt(2.0) * sph_harm(-1.0, 1.0, phi, theta).imag)
        Y10 = np.array(sph_harm(0.0, 1.0, phi, theta).real)
        Y1p1 = np.array(-np.sqrt(2.0) * sph_harm(1.0, 1.0, phi, theta).real)

        Y2m2 = np.array(np.sqrt(2.0) * sph_harm(-2.0, 2.0, phi, theta).imag)
        Y2m1 = np.array(np.sqrt(2.0) * sph_harm(-1.0, 2.0, phi, theta).imag)
        Y20 = np.array(sph_harm(0.0, 2.0, phi, theta).real)
        Y2p1 = np.array(-np.sqrt(2.0) * sph_harm(1.0, 2.0, phi, theta).real)
        Y2p2 = np.array(np.sqrt(2.0) * sph_harm(2.0, 2.0, phi, theta).real)

        Y3m3 = np.array(np.sqrt(2.0) * sph_harm(-3.0, 3.0, phi, theta).imag)
        Y3p3 = np.array(-np.sqrt(2.0) * sph_harm(3.0, 3.0, phi, theta).real)

        res_Y00 = gt.spherial_harmonics_projection(Y00, 1)
        res_Y1m1 = gt.spherial_harmonics_projection(Y1m1, 1)
        res_Y10 = gt.spherial_harmonics_projection(Y10, 1)
        res_Y1p1 = gt.spherial_harmonics_projection(Y1p1, 1)

        res_Y2m2 = gt.spherial_harmonics_projection(Y2m2, 2)
        res_Y2m1 = gt.spherial_harmonics_projection(Y2m1, 2)
        res_Y20 = gt.spherial_harmonics_projection(Y20, 2)
        res_Y2p1 = gt.spherial_harmonics_projection(Y2p1, 2)
        res_Y2p2 = gt.spherial_harmonics_projection(Y2p2, 2)

        res_Y3m3 = gt.spherial_harmonics_projection(Y3m3, 3)
        res_Y3p3 = gt.spherial_harmonics_projection(Y3p3, 3)

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

    def test_spherical_harmonics_multipoles_2(self):
        """
        Test the spherical harmonics projection function for pulsars pulsar
        matrix
        """

        nside = 64
        npix = hp.nside2npix(nside)
        theta, phi = hp.pix2ang(nside, jnp.arange(npix))
        theta = jnp.array(theta)
        phi = jnp.array(phi)

        Y00 = np.array([[sph_harm(0.0, 0.0, phi, theta).real]])
        Y1m1 = np.array([[np.sqrt(2.0) * sph_harm(-1.0, 1.0, phi, theta).imag]])
        Y10 = np.array([[sph_harm(0.0, 1.0, phi, theta).real]])
        Y1p1 = np.array([[-np.sqrt(2.0) * sph_harm(1.0, 1.0, phi, theta).real]])

        Y2m2 = np.array([[np.sqrt(2.0) * sph_harm(-2.0, 2.0, phi, theta).imag]])
        Y2m1 = np.array([[np.sqrt(2.0) * sph_harm(-1.0, 2.0, phi, theta).imag]])
        Y20 = np.array([[sph_harm(0.0, 2.0, phi, theta).real]])
        Y2p1 = np.array([[-np.sqrt(2.0) * sph_harm(1.0, 2.0, phi, theta).real]])
        Y2p2 = np.array([[np.sqrt(2.0) * sph_harm(2.0, 2.0, phi, theta).real]])

        Y3m3 = np.array([[np.sqrt(2.0) * sph_harm(-3.0, 3.0, phi, theta).imag]])
        Y3p3 = np.array([[-np.sqrt(2.0) * sph_harm(3.0, 3.0, phi, theta).real]])

        res_Y00 = gt.spherial_harmonics_projection_pulsars(Y00, 1)
        res_Y1m1 = gt.spherial_harmonics_projection_pulsars(Y1m1, 1)
        res_Y10 = gt.spherial_harmonics_projection_pulsars(Y10, 1)
        res_Y1p1 = gt.spherial_harmonics_projection_pulsars(Y1p1, 1)

        res_Y2m2 = gt.spherial_harmonics_projection_pulsars(Y2m2, 2)
        res_Y2m1 = gt.spherial_harmonics_projection_pulsars(Y2m1, 2)
        res_Y20 = gt.spherial_harmonics_projection_pulsars(Y20, 2)
        res_Y2p1 = gt.spherial_harmonics_projection_pulsars(Y2p1, 2)
        res_Y2p2 = gt.spherial_harmonics_projection_pulsars(Y2p2, 2)

        res_Y3m3 = gt.spherial_harmonics_projection_pulsars(Y3m3, 3)
        res_Y3p3 = gt.spherial_harmonics_projection_pulsars(Y3p3, 3)

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
        Test the get_correlations_lm_IJ function.
        """

        l_max = 6
        nside = 32
        npix = hp.nside2npix(nside)
        pixels = np.linspace(0, npix - 1, 35, dtype=int)
        p_I = jnp.array(hp.pix2vec(nside, pixels))
        GammalmIJ = gt.get_correlations_lm_IJ(p_I.T, l_max, nside)

        data = np.load(tu.get_correlations_lm_IJ_data_path)["data"]

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(data - GammalmIJ))), 0.0, places=5
        )

    def test_get_tensors_generation(self):
        """
        Test the get_tensors function.
        """
        result, test_shapes = get_tensors_and_shapes(
            tu.test_catalog_path2, 30, "legendre", 0
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_generation_Legendre(self):
        """
        Test the get_tensors function assuming you want the Legendre projection.
        """
        result, test_shapes = get_tensors_and_shapes(
            tu.test_catalog_path2, 50, "legendre", 6
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_generation_Binned(self):
        """
        Test the get_tensors function assuming you want the Binned projection.
        """
        result, test_shapes = get_tensors_and_shapes(
            tu.test_catalog_path2, 30, "binned", 10
        )
        for i in range(len(tu.get_tensor_labels)):
            self.assertTupleEqual(result[i].shape, test_shapes[i])

    def test_get_tensors_results(self):
        """
        Test the get_tensors function results
        """

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "legendre",
            0,
            tu.get_tensors_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            to_assert = jnp.sum(
                jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
            )
            self.assertAlmostEqual(float(to_assert), 0.0)

    def test_Legendre_projection(self):
        """
        Test the results of the Legendre projection of the get_tensors function
        """

        order = 70
        n_pulsars = 10

        vec = np.random.uniform(-1, 1, n_pulsars)

        matrix = vec[:, None] * vec[None, :]

        for i in range(n_pulsars):
            matrix[i, i] = 1

        chi_IJ = gt.get_chi_tensor_IJ(matrix) - 0.5 * np.eye(n_pulsars)

        HD_coefficients = gt.get_HD_Legendre_coefficients(order)
        polynomials = gt.get_polynomials_IJ(matrix, order)
        HD_val = HD_coefficients[:, None, None] * polynomials

        diff = 1 - np.sum(HD_val, axis=0) / chi_IJ
        self.assertAlmostEqual(float(jnp.mean(jnp.abs(diff))), 0.0, places=3)

    def test_get_tensors_Binned_results(self):
        """
        Test the results of the Binned projection of the get_tensors function
        """

        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "binned",
            6,
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
        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "legendre",
            6,
            tu.get_tensors_Legendre_data_path,
        )

        for i in range(len(tu.get_tensor_labels)):
            to_assert = jnp.sum(
                jnp.abs(results[i] - loaded_data[tu.get_tensor_labels[i]])
            )

            self.assertAlmostEqual(float(to_assert), 0.0)

    def test_get_tensors_anisotropies(self):
        """
        Test the results of the get_tensors function with anisotropies, check
        that the monopole is correct
        """
        results, loaded_data = get_tensors_data(
            tu.test_catalog_path,
            "legendre",
            0,
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
