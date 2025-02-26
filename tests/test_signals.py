# Global
import unittest

import jax.numpy as jnp


# Local
import utils as tu
from fastPTA.signals import SMBBH_parameters, get_signal_model


@tu.not_a_test
def test_dmodel(model_name="power_law", parameters=SMBBH_parameters, **kwargs):
    """
    Given some model name and parameters, checks the analytical derivatives
    against the numerical derivatives computed with the forward diff.

    """

    fvec = jnp.geomspace(1e-9, 1e-7, 100)
    model = get_signal_model(model_name)

    d1 = model.d1(fvec, parameters, **kwargs)
    d1j = model.dtemplate_forward(fvec, parameters, **kwargs)

    return d1, d1j


@tu.not_a_test
def test_d2model(model_name="power_law", parameters=SMBBH_parameters, **kwargs):
    """
    Given some model name and parameters, checks the analytical derivatives
    against the numerical derivatives computed with the forward diff.

    """

    fvec = jnp.geomspace(1e-9, 1e-7, 100)
    model = get_signal_model(model_name)

    d2 = model.d2(fvec, parameters, **kwargs)
    d2j = model.d2template_forward(fvec, parameters, **kwargs)

    return d2, d2j


class TestSignals(unittest.TestCase):

    def test_dflat(self):
        """
        Test function for the first derivative of a flat signal model.

        """

        d1, d1j = test_dmodel(model_name="flat", parameters=jnp.array([-7.0]))

        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)

    def test_dpower_law(self):
        """
        Test function for the first derivative of a power law signal model.

        """

        d1, d1j = test_dmodel(
            model_name="power_law", parameters=SMBBH_parameters, pivot=1e-8
        )

        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)

    def test_dlognormal(self):
        """
        Test function for the first derivative of a lognormal signal model.

        """

        d1, d1j = test_dmodel(
            model_name="lognormal",
            parameters=jnp.array([-3, -1.5, -8.0]),
        )
        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)

    def test_d2flat(self):
        """
        Test function for the first derivative of a flat signal model.

        """

        d1, d1j = test_d2model(model_name="flat", parameters=jnp.array([-7.0]))

        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)

    def test_d2power_law(self):
        """
        Test function for the first derivative of a power law signal model.

        """

        d1, d1j = test_d2model(
            model_name="power_law", parameters=SMBBH_parameters, pivot=1e-8
        )

        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
