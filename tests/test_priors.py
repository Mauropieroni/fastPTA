# Global
import unittest

import jax.numpy as jnp
from scipy import stats

# Local
from fastPTA.inference_tools.priors import Priors


class TestSignals(unittest.TestCase):

    def test_prior1(self):
        """
        Test function for the flat signal model.

        """

        my_dict = {
            "p1": {"norm": {"loc": 5.0, "scale": 3.0}},
            "p2": {"beta": {"a": 2.0, "b": 4.0}},
        }

        my_priors = Priors(my_dict)
        x = jnp.array([1.0, 0.3])

        log_prior = my_priors.evaluate_log_priors({"p1": x[0], "p2": x[1]})
        log_prior1 = jnp.log(stats.norm.pdf(x[0], loc=5.0, scale=3.0))
        log_prior2 = jnp.log(stats.beta.pdf(x[1], a=2.0, b=4.0))
        self.assertEqual(log_prior, log_prior1 + log_prior2)

    def test_prior2(self):
        """
        Test function for the flat signal model.

        """

        def func(x):
            return x**2

        my_dict = {
            "p1": {"uniform": {"loc": 5.0, "scale": 3.0}},
            "p2": func,
        }

        my_priors = Priors(my_dict)
        x = jnp.array([6.0, 0.3])

        log_prior = my_priors.evaluate_log_priors({"p1": x[0], "p2": x[1]})
        log_prior1 = jnp.log(stats.uniform.pdf(x[0], loc=5.0, scale=3.0))
        log_prior2 = jnp.log(func(x[1]))
        self.assertEqual(log_prior, log_prior1 + log_prior2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
