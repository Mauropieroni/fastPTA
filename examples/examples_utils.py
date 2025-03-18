# Global
import os
import sys

# Local
import fastPTA.utils as ut


sys.path.append(os.path.split(os.path.abspath(__file__))[0])


# Creates some folders you need to store data/plots
for k in [
    "pulsar_configurations",
    "generated_data",
    "generated_chains",
    "plots",
]:
    if not os.path.exists(k):
        os.makedirs(k)


path_to_file = os.path.dirname(__file__)

path_to_pulsar_parameters = os.path.join(path_to_file, "pulsar_configurations/")

# Default parameters for the pulsars
EPTAlike = ut.load_yaml(
    path_to_pulsar_parameters + "EPTAlike_pulsar_parameters.yaml"
)

# Default parameters for the pulsars
mockSKA10 = ut.load_yaml(
    path_to_pulsar_parameters + "mockSKA10_pulsar_parameters.yaml"
)

# Default parameters for the pulsars
EPTAlike_noiseless = ut.load_yaml(
    path_to_pulsar_parameters + "EPTAlike_pulsar_parameters_noiseless.yaml"
)
