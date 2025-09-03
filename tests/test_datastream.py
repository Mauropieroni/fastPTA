# Global
import unittest
import jax.numpy as jnp
import numpy as np

# Local
import utils as tu
import fastPTA.data.datastream as gds


datastream = np.load(tu.get_datastream_data_path)


class TestGetTensors(unittest.TestCase):

    def test_unit_vector(self):
        """
        Test the unit vector function
        """

        theta = datastream["theta"]
        phi = datastream["phi"]
        k_vec = gds.unit_vector(theta, phi)

        self.assertAlmostEqual(
            np.sum(k_vec - datastream["k_vec"]), 0.0, delta=1e-13
        )

    def test_get_u_vec(self):
        """
        Test the unit vector function
        """

        theta = datastream["theta"]
        phi = datastream["phi"]
        u_vec = gds.get_u(theta, phi)

        self.assertAlmostEqual(
            np.sum(u_vec - datastream["u_vec"]), 0.0, delta=1e-13
        )

    def test_get_v_vec(self):
        """
        Test the unit vector function
        """

        theta = datastream["theta"]
        phi = datastream["phi"]
        v_vec = gds.get_v(theta, phi)

        self.assertAlmostEqual(
            np.sum(v_vec - datastream["v_vec"]), 0.0, delta=1e-13
        )

    def test_get_plus_cross(self):
        """
        Test the unit vector function
        """

        theta = datastream["theta"]
        phi = datastream["phi"]
        e_p, e_c = gds.get_plus_cross(theta, phi)

        self.assertAlmostEqual(
            np.sum(e_p - datastream["e_p"]), 0.0, delta=1e-13
        )

        self.assertAlmostEqual(
            np.sum(e_c - datastream["e_c"]), 0.0, delta=1e-13
        )

    def test_get_F_pc(self):
        """
        Test the unit vector function
        """

        p_vec = datastream["p_vec"]
        theta = datastream["theta"]
        phi = datastream["phi"]

        k_vec = gds.unit_vector(theta, phi)
        e_p, e_c = gds.get_plus_cross(theta, phi)

        F_p, F_c = gds.get_F_pc(p_vec, k_vec, e_p, e_c)

        self.assertAlmostEqual(
            np.sum(F_p - datastream["F_p"]), 0.0, delta=1e-13
        )

        self.assertAlmostEqual(
            np.sum(F_c - datastream["F_c"]), 0.0, delta=1e-13
        )

    def test_get_R_pc(self):
        """
        Test the unit vector function
        """

        ff = datastream["ff"]
        dist = datastream["dist"]
        p_vec = datastream["p_vec"]
        theta = datastream["theta"]
        phi = datastream["phi"]

        R_p, R_c = gds.get_R_pc(ff, dist, p_vec, theta, phi)

        self.assertAlmostEqual(
            np.sum(R_p - datastream["R_p"]), 0.0, delta=1e-13
        )

        self.assertAlmostEqual(
            np.sum(R_c - datastream["R_c"]), 0.0, delta=1e-13
        )

    def test_get_s_I(self):
        """
        Test the get_s_I function
        """
        # Get test data from datastream
        h_p = datastream["h_pi"]
        h_c = datastream["h_ci"]
        R_p = datastream["R_pi"]
        R_c = datastream["R_ci"]
        expected_s_I = datastream["s_I"]

        # Compute s_I
        s_I = gds.get_s_I(h_p, h_c, R_p, R_c)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_s_I - s_I))), 0.0, delta=1e-5
        )

    def test_get_s_I_fi(self):
        """
        Test the get_s_I_fi function
        """
        # Get test data from datastream
        Tspan = datastream["Tspan"]
        fi = datastream["fvec"]
        ff = datastream["ff"]
        h_p = datastream["h_p"]
        h_c = datastream["h_c"]
        R_p = datastream["R_p"]
        R_c = datastream["R_c"]
        expected_s_I_fi = datastream["s_I_fi"]

        # Compute s_I_fi
        s_I_fi = gds.get_s_I_fi(Tspan, fi, ff, h_p, h_c, R_p, R_c)

        self.assertAlmostEqual(
            float(jnp.sum(jnp.abs(expected_s_I_fi - s_I_fi))), 0.0, delta=1e-5
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
