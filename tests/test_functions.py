# Global
import os, sys
import numpy as np
import pandas as pd

# Local
from fastPTA.utils import *
from fastPTA.signals import SMBBH_parameters, CGW_SIGW_parameters
from fastPTA.get_tensors import get_tensors
from fastPTA.Fisher_code import compute_fisher
from fastPTA.MCMC_code import run_MCMC
from fastPTA.plotting_functions import plot_corner
from fastPTA.generate_new_pulsar_configuration import generate_pulsars_catalog

# Setting the path to this file
file_path = os.path.dirname(__file__)
if file_path:
    file_path += "/"

sys.path.append(os.path.join(file_path, "../examples/"))
from get_forecasts import get_constraints


fmin = 1e-10
### Maximal frequency to use in the analyses
fmax = 1e-6
### Frequency vector to use in the analyses
frequency = np.arange(fmin, fmax, 1e-8)

n_pulsars1 = 100
n_pulsars2 = 120
method = "Legendre"
order = 6

### Default parameters for the pulsars
EPTAlike = load_yaml(
    file_path + "../pulsar_configurations/EPTAlike_pulsar_parameters.yaml"
)

### Default parameters for the pulsars
mockSKA10 = load_yaml(
    file_path + "../pulsar_configurations/mockSKA10_pulsar_parameters.yaml"
)


def test_generation(
    n_pulsars=n_pulsars1,
    pulsar_configuration=EPTAlike,
    outname=file_path + "pulsar_configurations/test_catalog1.txt",
    save_it=True,
):
    print("-- Test generation --")
    test_catalog = generate_pulsars_catalog(
        n_pulsars=n_pulsars,
        outname=outname,
        save_it=save_it,
        **pulsar_configuration
    )

    for k, v in test_catalog.items():
        if k != "names":
            if len(v) != n_pulsars:
                raise ValueError("Something wrong in the generation")
    print("Generated %d pulsars" % len(v))
    print("-- Test completed --\n")


def test_get_tensors_generation(n_pulsars=n_pulsars2):
    print("-- Test get tensors --")
    strain_omega, response_IJ, HD_functions_IJ, HD_coefficients = get_tensors(
        frequency,
        path_to_pulsars=file_path + "pulsar_configurations/test_catalog2.txt",
        save_it=True,
        n_pulsars=n_pulsars,
        regenerate_catalog=True,
        method=method,
        order=order,
    )

    ### Just checking shapes, some other tests must be added here!!
    if strain_omega.shape != (len(frequency), n_pulsars, n_pulsars):
        raise ValueError("Something wrong in the generation")

    if response_IJ.shape != (len(frequency), n_pulsars, n_pulsars):
        raise ValueError("Something wrong in the generation")

    if HD_functions_IJ.shape != (
        order + 1,
        len(frequency),
        n_pulsars,
        n_pulsars,
    ):
        raise ValueError("Something wrong in the generation")

    if HD_coefficients.shape != (order + 1,):
        raise ValueError("Something wrong in the generation")

    print("Checked that get_tensors returns objects with the right shape")
    print("-- Test completed --\n")


def test_get_tensors():
    print("-- Test get tensors --")

    ### Checking that all outputs are correct
    strain_omega, response_IJ, HD_functions_IJ, HD_coefficients = get_tensors(
        frequency,
        path_to_pulsars=file_path + "test_data/test_catalog.txt",
        method=method,
        order=order,
    )

    test_data = np.load("test_data/test_catalog_output.npz")

    if not np.allclose(strain_omega, test_data["strain_omega"]):
        raise ValueError("Something wrong in the generation")

    if not np.allclose(response_IJ, test_data["response_IJ"]):
        raise ValueError("Something wrong in the generation")

    if not np.allclose(HD_functions_IJ, test_data["HD_functions_IJ"]):
        raise ValueError("Something wrong in the generation")

    if not np.allclose(HD_coefficients, test_data["HD_coefficients"]):
        raise ValueError("Something wrong in the generation")

    print("Checked that get_tensors outputs are correct")
    print("-- Test completed --\n")


def test_current_EPTA(
    path_to_pulsars="test_data/EPTAlike_ml.txt",
    add_curn=True,
    len_fisher_data=10000,
    chains_path="test_data/EPTA_power_law.txt",
    parameter_labels=[r"$\alpha_{\rm PL}$", r"$n_{\rm T}$"],
    rerun_MCMC=True,
    regenerate_MCMC_data=True,
    realization=True,
    path_to_MCMC_data="generated_data/MCMC_data_current.npz",
    i_max=10,
    R_convergence=1e-1,
    burnin_steps=500,
    MCMC_iteration_steps=1000,
    path_to_MCMC_chains="generated_chains/MCMC_chains_current.npz",
):
    print("-- Test current EPTA --")
    get_tensors_kwargs = {"path_to_pulsars": path_to_pulsars, "add_curn": True}

    with open(chains_path) as f:
        line1 = f.readline()

    free_spectrum_results = pd.read_csv(
        chains_path,
        skipinitialspace=True,
        sep=" ",
        names=line1.replace("#", " ").split(),
        comment="#",
    )

    free_spectrum_log_10amplitude = free_spectrum_results["log10_Omega"]
    free_spectrum_tilt = free_spectrum_results["tilt"]
    free_spectrum_weights = free_spectrum_results["weight"]
    free_spectrum_data = np.vstack(
        (free_spectrum_log_10amplitude, free_spectrum_tilt)
    ).T

    signal_parameters = np.array(
        [
            np.average(
                free_spectrum_log_10amplitude, weights=free_spectrum_weights
            ),
            np.average(free_spectrum_tilt, weights=free_spectrum_weights),
        ]
    )

    (
        frequency,
        signal,
        HD_functions_IJ,
        HD_coeffs,
        effective_noise,
        SNR,
        fisher,
    ) = compute_fisher(
        signal_parameters=signal_parameters,
        get_tensors_kwargs=get_tensors_kwargs,
    )

    fisher_covariance = compute_inverse(fisher)
    print("Fisher errors", np.sqrt(np.diag(fisher_covariance)))

    fisher_data = np.random.multivariate_normal(
        signal_parameters, fisher_covariance, size=len_fisher_data
    )

    errors = np.sqrt(np.diag(fisher_covariance))
    ranges = [
        (
            signal_parameters[i] - 5 * errors[i],
            signal_parameters[i] + 5 * errors[i],
        )
        for i in range(len(errors))
    ]

    try:
        if rerun_MCMC:
            print("Flag forces MCMC chains regeneration")
            raise FileNotFoundError
        MCMC_results = np.load(path_to_MCMC_chains)
        MCMC_data = MCMC_results["samples"]

    except FileNotFoundError:
        MCMC_data, pdfs = run_MCMC(
            np.array([[-12, -4], [-5, 8]]).T,
            signal_parameters=signal_parameters,
            get_tensors_kwargs=get_tensors_kwargs,
            regenerate_MCMC_data=regenerate_MCMC_data,
            realization=realization,
            path_to_MCMC_data=path_to_MCMC_data,
            i_max=i_max,
            R_convergence=R_convergence,
            burnin_steps=burnin_steps,
            MCMC_iteration_steps=MCMC_iteration_steps,
            path_to_MCMC_chains=path_to_MCMC_chains,
        )

    datasets = [free_spectrum_data, fisher_data, MCMC_data]
    weights = [
        free_spectrum_weights,
        np.ones(len_fisher_data),
        np.ones(MCMC_data.shape[0]),
    ]
    smooth = [0.5, 0.5, 0.5]

    plot_corner(
        datasets,
        colors=[my_colormap["red"], my_colormap["green"], my_colormap["blue"]],
        truths=signal_parameters,
        chain_labels=["Free Spectrum EPTA", "Fisher EPTA", "MCMC EPTA"],
        weights=weights,
        smooth=smooth,
        labels=parameter_labels,
        range=ranges,
        truth_color="black",
    )
    print("-- Test completed --\n")


def test_future(
    add_curn=True,
    regenerate_catalog=True,
    pulsar_configuration=mockSKA10,
    realization=False,
    rerun_MCMC=True,
    regenerate_MCMC_data=True,
    path_to_MCMC_data="generated_data/MCMC_data_future.npz",
    path_to_MCMC_chains="generated_chains/MCMC_chains_future.npz",
):
    MCMC_kwargs = {
        "regenerate_MCMC_data": regenerate_MCMC_data,
        "realization": realization,
        "path_to_MCMC_data": path_to_MCMC_data,
        "i_max": 10,
        "R_convergence": 1e-1,
        "R_criterion": "max",
        "burnin_steps": 300,
        "MCMC_iteration_steps": 500,
        "save_MCMC_data": True,
        "path_to_MCMC_chains": path_to_MCMC_chains,
    }

    get_tensors_kwargs = {
        "path_to_pulsars": "pulsar_configurations/future.txt",
        "add_curn": add_curn,
        "regenerate_catalog": regenerate_catalog,
    }

    generate_catalog_kwargs = {
        "n_pulsars": n_pulsars1,
        "save_it": True,
        **pulsar_configuration,
    }


    get_constraints(
        "power_law_SIGW",
        np.concatenate([SMBBH_parameters, CGW_SIGW_parameters]),
        np.array([[-14, -5], [-3, 3], [-10, 2], [-1, 0.1], [-10, -5]]).T, 
        #np.array([[-11, -3], [-2, 6], [-3.7, 1.3], [-0.5, 0.1], [-8, -7.5]]).T,
        T_obs_yrs=10, 
        n_frequencies=100,  
        rerun_MCMC=rerun_MCMC,
        path_to_MCMC_data=path_to_MCMC_data,
        path_to_MCMC_chains=path_to_MCMC_chains,
        MCMC_kwargs=MCMC_kwargs,
        get_tensors_kwargs=get_tensors_kwargs,
        generate_catalog_kwargs=generate_catalog_kwargs,
        parameter_labels=[
            "$A$",
            "$n_T$",
            "$Log_{10}A$",
            "$Log_{10} \\Delta$",
            "$Log_{10} f_*$",
        ],
    )


if __name__ == "__main__":
    test_generation()
    test_generation(
        pulsar_configuration=mockSKA10,
        outname=file_path + "pulsar_configurations/future_test.txt",
        save_it=True,
    )

    test_get_tensors_generation()
    test_get_tensors()

    test_current_EPTA(
        add_curn=False,
        rerun_MCMC=False,
        realization=False,
    )

    test_future(
        add_curn=False,
        regenerate_catalog=False,
        pulsar_configuration=mockSKA10,
        realization=False,
        rerun_MCMC=True,
    )

    plt.show(block=True)
