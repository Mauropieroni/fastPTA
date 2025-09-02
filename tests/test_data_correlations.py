# Global
import unittest

import healpy as hp
import jax
import jax.numpy as jnp
import numpy as np

# Local
import utils as tu

from fastPTA.data import data_correlations as dc
from fastPTA import utils as ut

# Enable 64-bit precision for JAX
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices(ut.which_device)[0])


nside = 64
npix = hp.nside2npix(nside)
theta, phi = hp.pix2ang(nside, jnp.arange(npix))
theta = jnp.array(theta)
phi = jnp.array(phi)


data_correlations = np.load(tu.get_correlations_data_path)


class TestGetTensors(unittest.TestCase):

    def test_get_correlations_IJ(self):
        """
        Test the get_D_IJ function
        """
        # Get test data from data_correlations
        s_I = data_correlations["s_I"]
        expected_D_IJ = data_correlations["D_IJ"]

        # Compute D_IJ
        D_IJ = dc.get_correlation_IJ(s_I)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_D_IJ - D_IJ))), 0.0, delta=1e-5
        )

    def test_get_DI_IJ(self):
        """
        Test the get_D_IJ function
        """
        # Get test data from data_correlations
        fi = data_correlations["fvec"]
        h_tilde = data_correlations["h_tildei"]
        distances = data_correlations["dist"]
        p_vec = data_correlations["p_vec"]
        theta = data_correlations["theta"]
        phi = data_correlations["phi"]
        expected_D_IJ = data_correlations["D_IJ"]

        # Compute D_IJ
        D_IJ = dc.get_D_IJ(fi, h_tilde, distances, p_vec, theta, phi)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_D_IJ - D_IJ))),
            0.0,
            delta=1e-5,
        )

    def test_get_D_IJ_fi(self):
        """
        Test the get_D_IJ_fi function
        """
        # Get test data from data_correlations
        Tspan = data_correlations["Tspan"]
        fi = data_correlations["fvec"]
        ff = data_correlations["ff"]
        h_tilde = data_correlations["h_tilde"]
        distances = data_correlations["dist"]
        p_vec = data_correlations["p_vec"]
        theta = data_correlations["theta"]
        phi = data_correlations["phi"]
        expected_D_IJ_fi = data_correlations["D_IJ_fi"]

        # Compute D_IJ_fi
        D_IJ_fi = dc.get_D_IJ_fi(
            Tspan, fi, ff, h_tilde, distances, p_vec, theta, phi
        )

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_D_IJ_fi - D_IJ_fi))),
            0.0,
            delta=1e-5,
        )

    def test_get_D_IJ_fi_normalization(self):
        """
        Test the get_D_IJ_fi_normalization function
        """
        # Get test data from data_correlations
        Tspan = data_correlations["Tspan"]
        fi = data_correlations["fvec"]
        ff = data_correlations["ff"]
        H_p_ff = data_correlations["H_p_ff"]
        expected_norm = data_correlations["D_IJ_fi_norm"]

        # Compute normalization
        D_IJ_fi_norm = dc.get_D_IJ_fi_normalization(Tspan, fi, ff, H_p_ff)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_norm - D_IJ_fi_norm))),
            0.0,
            delta=1e-5,
        )

    def test_get_D_IJ_fifj(self):
        """
        Test the get_D_IJ_fifj function
        """
        # Get test data from data_correlations
        Tspan = data_correlations["Tspan"]
        fi = data_correlations["fvec"]
        ff = data_correlations["ff"]
        h_tilde = data_correlations["h_tilde"]
        distances = data_correlations["dist"]
        p_vec = data_correlations["p_vec"]
        theta = data_correlations["theta"]
        phi = data_correlations["phi"]
        expected_D_IJ_fifj = data_correlations["D_IJ_fifj"]

        # Compute D_IJ_fifj
        D_IJ_fifj = dc.get_D_IJ_fifj(
            Tspan, fi, ff, h_tilde, distances, p_vec, theta, phi
        )

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_D_IJ_fifj - D_IJ_fifj))),
            0.0,
            delta=1e-5,
        )

    def test_get_D_IJ_fifj_normalization(self):
        """
        Test the get_D_IJ_fifj_normalization function
        """
        # Get test data from data_correlations
        Tspan = data_correlations["Tspan"]
        fi = data_correlations["fvec"]
        ff = data_correlations["ff"]
        H_p_ff = data_correlations["H_p_ff"]
        expected_norm = data_correlations["D_IJ_fifj_norm"]

        # Compute normalization
        D_IJ_fifj_norm = dc.get_D_IJ_fifj_normalization(Tspan, fi, ff, H_p_ff)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_norm - D_IJ_fifj_norm))),
            0.0,
            delta=1e-5,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
