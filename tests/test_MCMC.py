# Global
import unittest

import numpy as np
from scipy.stats import kstest


# Local
import utils as tu
from fastPTA.MCMC_code import get_MCMC_samples


i_max_def = 10
R_convergence_def = 5e-2
burnin_steps_def = 1000
MCMC_iteration_steps_def = 1000


@tu.not_a_test
def log_posterior_gaussian(parameters, mean, covariance_matrix):
    """ """

    return (
        -0.5
        * (parameters - mean).T
        @ np.linalg.inv(covariance_matrix)
        @ (parameters - mean)
    )


class TestMCMC(unittest.TestCase):

    def test_2d(self):
        """
        Test the MCMC sampler on an N-dimensional Gaussian distribution.

        """
        ndims = 2
        nwalkers = 10

        walkers = 2 * ndims if nwalkers < 2 * ndims else nwalkers
        mean = np.random.uniform(0.1, 10, ndims)
        covariance = np.diag(np.random.uniform(0.1, 10, ndims))
        initial = np.random.multivariate_normal(mean, covariance, size=walkers)

        samples, _ = get_MCMC_samples(
            log_posterior_gaussian,
            initial,
            [mean, covariance],
            i_max=i_max_def,
            R_convergence=R_convergence_def,
            burnin_steps=burnin_steps_def,
            MCMC_iteration_steps=MCMC_iteration_steps_def,
            print_progress=False,
        )

        samples -= np.mean(samples, axis=0)
        samples_gaussian = np.random.multivariate_normal(
            0 * mean, covariance, size=1000
        )
        for i in range(ndims):
            _, p = kstest(
                samples[:, i],
                samples_gaussian[:, i],
            )

            self.assertTrue(p > 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
