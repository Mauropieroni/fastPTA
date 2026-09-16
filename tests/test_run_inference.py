# Global
import os
import shutil
import tempfile
import unittest

import numpy as np

# Local
import utils as tu

from fastPTA.inference_tools.priors import Priors
from fastPTA.run_inference import run_inference


class TestRunInference(unittest.TestCase):
    """
    End-to-end smoke tests for run_inference: exercises the full pipeline
    (data generation, likelihood prep, sampling, saving) that get_MCMC_data
    /run_MCMC/run_nested_sampling had no dedicated test for before being
    consolidated here. Uses the small test pulsar catalog and tiny sampler
    budgets to keep this fast; get_MCMC_samples/get_nested_samples already
    have their own accuracy tests, so this only checks the pipeline runs
    and produces the expected shapes.

    """

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.priors = Priors(
            {
                "log_amplitude": {"uniform": {"loc": -8.5, "scale": 3.0}},
                "tilt": {"uniform": {"loc": 1.0, "scale": 2.0}},
            }
        )
        self.common_kwargs = dict(
            priors=self.priors,
            n_frequencies=5,
            regenerate_inference_data=True,
            save_inference_data=False,
            print_progress=False,
            get_tensors_kwargs={"path_to_pulsar_catalog": tu.test_catalog_path},
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_invalid_method(self):
        """
        Unknown methods should fail fast instead of silently doing nothing.

        """

        with self.assertRaises(ValueError):
            run_inference(self.priors, method="not_a_method")

    def test_mcmc(self):
        """
        method="mcmc" should return samples/pdfs with matching lengths and
        write the chains file.

        """

        path_to_MCMC_chains = os.path.join(self.tmp_dir, "MCMC_chains.npz")

        samples, pdfs = run_inference(
            method="mcmc",
            i_max=1,
            burnin_steps=10,
            MCMC_iteration_steps=10,
            path_to_MCMC_chains=path_to_MCMC_chains,
            **self.common_kwargs,
        )

        self.assertEqual(samples.shape[1], len(self.priors.parameter_names))
        self.assertEqual(samples.shape[0], pdfs.shape[0])
        self.assertTrue(os.path.exists(path_to_MCMC_chains))

    def test_nested_sampling(self):
        """
        method="nested_sampling" should return equally-weighted samples and
        a finite evidence estimate, and write the chains file.

        """

        path_to_NS_chains = os.path.join(self.tmp_dir, "NS_chains.npz")

        samples, logZ_mean, logZ_std = run_inference(
            method="nested_sampling",
            n_live=25,
            n_posterior_samples=50,
            path_to_NS_chains=path_to_NS_chains,
            **self.common_kwargs,
        )

        self.assertEqual(samples.shape[1], len(self.priors.parameter_names))
        self.assertTrue(np.isfinite(logZ_mean))
        self.assertTrue(np.isfinite(logZ_std))
        self.assertTrue(os.path.exists(path_to_NS_chains))


if __name__ == "__main__":
    unittest.main(verbosity=2)
