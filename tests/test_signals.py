# Global
import unittest

import jax.numpy as jnp


# Local
import utils as tu
from fastPTA.signals import SMBBH_parameters, get_signal_model


@tu.not_a_test
def test_model(model_name="power_law", parameters=SMBBH_parameters, **kwargs):
    """
    Given some model name and parameters, checks the analytical derivatives
    against the numerical derivatives computed with the forward diff.

    """

    fvec = jnp.geomspace(1e-9, 1e-7, 100)
    model = get_signal_model(model_name)

    d1 = model.d1(fvec, parameters, **kwargs)
    d1j = model.dtemplate_forward(fvec, parameters, **kwargs)

    return d1, d1j


class TestSignals(unittest.TestCase):

    def test_flat(self):
        """
        Test function for the flat signal model.

        """

        d1, d1j = test_model(model_name="flat", parameters=jnp.array([-7.0]))

        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)

    def test_power_law(self):
        """
        Test function for the power law signal model.

        """

        d1, d1j = test_model(
            model_name="power_law", parameters=SMBBH_parameters, pivot=1e-8
        )
        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)

    def test_lognormal(self):
        """
        Test function for the lognormal signal model.

        """

        d1, d1j = test_model(
            model_name="lognormal",
            parameters=jnp.array([-3, -1.5, -8.0]),
        )
        self.assertAlmostEqual(jnp.sum(d1 - d1j), 0.0, places=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
