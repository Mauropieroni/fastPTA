# Global
import unittest

import numpy as np
import jax.numpy as jnp
from scipy import stats

# Local
from fastPTA.inference_tools.priors import Priors
from fastPTA.run_inference import get_nested_samples


class TestNestedSampling(unittest.TestCase):

    def test_2d_gaussian(self):
        """
        Test nested sampling against the analytic evidence and posterior
        mean of a Gaussian likelihood with a uniform prior.

        """
        ndims = 2
        n_live = 200
        mean = np.random.uniform(0.1, 5, ndims)
        sigma = np.random.uniform(0.3, 1.5, ndims)
        loc, scale = -5.0, 15.0

        priors = Priors(
            {
                f"p{i}": {"uniform": {"loc": loc, "scale": scale}}
                for i in range(ndims)
            }
        )

        def logprior_fn(parameters):
            return priors.evaluate_log_priors(
                dict(zip(priors.parameter_names, parameters))
            )

        def loglikelihood_fn(parameters):
            return jnp.sum(
                -0.5 * ((parameters - mean) / sigma) ** 2
                - jnp.log(sigma * jnp.sqrt(2 * jnp.pi))
            )

        initial = priors.sample(n_live)

        samples, logZ_mean, _ = get_nested_samples(
            logprior_fn, loglikelihood_fn, initial, print_progress=False
        )

        # Analytic evidence for a Gaussian likelihood with a uniform prior
        logZ_analytic = 0.0
        for i in range(ndims):
            Z_i = (
                stats.norm.cdf(loc + scale, mean[i], sigma[i])
                - stats.norm.cdf(loc, mean[i], sigma[i])
            ) / scale
            logZ_analytic += np.log(Z_i)

        self.assertTrue(np.abs(logZ_mean - logZ_analytic) < 1.0)
        self.assertTrue(np.allclose(np.mean(samples, axis=0), mean, atol=0.5))


if __name__ == "__main__":
    unittest.main(verbosity=2)
