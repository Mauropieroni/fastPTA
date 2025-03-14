# Global
import unittest

import numpy as np


# Local
import utils as tu
from fastPTA import compute_PBH_Abundance as cpa


f_PBH_lognormal_data = np.load(tu.f_PBH_lognormal_data_path)
find_A_PBH_lognormal_data = np.load(tu.find_A_PBH_lognormal_data_path)


class Test_Abundance(unittest.TestCase):

    def test_f_PBH(self):
        """
        Test function for f_PBH_NL_QCD_lognormal

        """

        for v in f_PBH_lognormal_data["data"]:
            self.assertAlmostEqual(
                cpa.f_PBH_NL_QCD_lognormal(*v[:3]),
                v[-1],
                places=5,
            )

    def test_find_A(self):
        """
        Test function for find_A_NL_QCD

        """

        for v in find_A_PBH_lognormal_data["data"]:
            self.assertAlmostEqual(cpa.find_A_NL_QCD(*v[:3]), v[-1], places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
