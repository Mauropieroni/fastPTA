# Global
import os

# Local
import fastPTA.utils as ut


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
