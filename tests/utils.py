# Global imports
import os
import numpy as np


# Local imports
from fastPTA.utils import load_yaml


test_data_path = os.path.join(os.path.dirname(__file__), "test_data/")

# Default parameters for the pulsars
EPTAlike_test = load_yaml(test_data_path + "/EPTAlike_pulsar_parameters.yaml")
EPTAlike_noiseless_test = load_yaml(
    test_data_path + "/EPTAlike_pulsar_parameters_noiseless.yaml"
)
mockSKA10_test = load_yaml(test_data_path + "/mockSKA10_pulsar_parameters.yaml")
NANOGrav_positions = test_data_path + "/NANOGrav_positions.txt"

# Paths to the test data
lm_indexes = test_data_path + "lm_indexes.txt"
test_catalog_path = test_data_path + "test_catalog.txt"
test_catalog_path2 = test_data_path + "test_catalog2.txt"
test_catalog_path3 = test_data_path + "test_catalog3.txt"
get_tensors_data_path = test_data_path + "get_tensors.npz"
get_datastream_data_path = test_data_path + "datastream.npz"
get_correlations_data_path = test_data_path + "data_correlations.npz"
get_correlations_lm_IJ_data_path = test_data_path + "get_correlations_lm_IJ.npz"
get_tensors_Binned_data_path = test_data_path + "get_tensors_Binned.npz"
get_tensors_Legendre_data_path = test_data_path + "get_tensors_Legendre.npz"
iterative_estimation_data_path = test_data_path + "iterative_estimation.npz"
Fisher_data_path = test_data_path + "Fisher_data.npz"
Fisher_data_path2 = test_data_path + "Fisher_data2.npz"
f_PBH_lognormal_data_path = test_data_path + "f_PBH_lognormal.npz"
find_A_PBH_lognormal_data_path = test_data_path + "find_A_PBH_lognormal.npz"
mock_pulsar_noises = test_data_path + "mock_pulsar_noises.npz"

parameters_to_test = {
    "phi": "phi",
    "theta": "theta",
    "dt": "dt",
    "T_span": "Tspan",
    "wn": "wn",
    "dm_noise_log_10_A": "log10_A_dm",
    "red_noise_log_10_A": "log10_A_red",
    "sv_noise_log_10_A": "log10_A_sv",
    "dm_noise_g": "g_dm",
    "red_noise_g": "g_red",
    "sv_noise_g": "g_sv",
}


test_distributions = {
    "phi_dict": {"which_distribution": "uniform", "min": 0, "max": 2 * np.pi},
    "theta_dict": {"which_distribution": "uniform", "min": -1, "max": 1},
}


get_tensor_labels = [
    "strain_omega",
    "response_IJ",
    "HD_functions_IJ",
    "HD_coefficients",
]


def not_a_test(object):
    object.__test__ = False
    return object


test_frequency = np.arange(1e-10, 1e-6, 1e-8)
