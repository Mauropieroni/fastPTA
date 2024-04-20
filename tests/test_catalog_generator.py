# Global
import unittest

import numpy as np

from scipy.stats import kstest
from scipy import stats as scipy_stats

# Local
import test_utils as tu
from fastPTA.generate_new_pulsar_configuration import generate_pulsars_catalog


@tu.not_a_test
def test_generation(parameter, n_pulsars, pulsar_dictionary):

    test_dictionary = {**tu.test_distributions}
    for k, v in pulsar_dictionary.items():
        if k != "noise_probabilities_dict":
            test_dictionary[k] = v

    test_catalog = generate_pulsars_catalog(
        n_pulsars=n_pulsars, save_catalog=False, **pulsar_dictionary
    )

    x = test_catalog[tu.parameters_to_test[parameter]]
    x = x[x > -40]

    if parameter in ["dt", "T_span", "wn"]:
        x = np.log10(x)
    elif parameter in ["theta"]:
        x = np.cos(x)

    dist = test_dictionary[parameter + "_dict"]
    # This is a Kolmogorov–Smirnov test to check whether the pulsars in the
    # catalog are distributed as expected

    if dist["which_distribution"] == "uniform" and len(x) > 3:
        _stats, p = kstest(
            x,
            scipy_stats.uniform(
                dist["min"], scale=dist["max"] - dist["min"]
            ).cdf,
        )

    elif dist["which_distribution"] == "gaussian" and len(x) > 3:
        _stats, p = kstest(
            x, scipy_stats.norm(dist["mean"], scale=dist["std"]).cdf
        )

    elif len(x) <= 3:
        p = 1.0

    else:
        raise ValueError("Cannot use distribution", dist["which_distribution"])

    return p


class TestCatalogGenerator(unittest.TestCase):
    def test_generation_EPTA(self):
        for p in tu.parameters_to_test.keys():
            self.assertTrue(test_generation(p, 30, tu.EPTAlike_test) > 1e-4)

    def test_generation_EPTA_noiseless(self):
        for p in tu.parameters_to_test.keys():
            self.assertTrue(
                test_generation(p, 50, tu.EPTAlike_noiseless_test) > 1e-4
            )

    def test_generation_SKAlike(self):
        for p in tu.parameters_to_test.keys():
            self.assertTrue(test_generation(p, 500, tu.mockSKA10_test) > 1e-4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
