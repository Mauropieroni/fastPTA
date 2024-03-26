import os, sys

file_path = os.path.dirname(__file__)
if file_path:
    file_path += "/"


# Setting the path to this file
import os, sys

file_path = os.path.dirname(__file__)
if file_path:
    file_path += "/"

sys.path.append(os.path.join(file_path, "../fastPTA/"))


from utils import *
from signals import SMBBH_parameters, CGW_LN_parameters
from Fisher_code import compute_fisher
from MCMC_code import run_MCMC
from plotting_functions import (
    plot_corner,
    plot_HD_Legendre,
    plot_HD_binned,
)


### Default parameters for the pulsars
EPTAlike = load_yaml(
    file_path + "../pulsar_configurations/EPTAlike_pulsar_parameters.yaml"
)

### Default parameters for the pulsars
mockSKA10 = load_yaml(
    file_path + "../pulsar_configurations/mockSKA10_pulsar_parameters.yaml"
)


def scan_parameter(
    signal_label,
    signal_parameters,
    parameter_index,
    parameter_range,
    T_obs_yrs=10.33,
    n_frequencies=30,
    get_tensors_kwargs={
        "path_to_pulsars": "pulsar_configurations/Mock_SKA.txt"
    },
    generate_catalog_kwargs={"n_pulsars": 30, "save_it": True, **mockSKA10},
    parameter_labels=[
        r"$\alpha_{\rm PL}$",
        r"$n_{\rm T}$",
    ],
    plot_kwargs={},
):

    results = np.zeros(shape=(len(signal_parameters), len(parameter_range)))
    s_params = np.array(signal_parameters)

    for i in range(len(parameter_range)):
        s_params[parameter_index] = parameter_range[i]

        fisher_kwargs = {
            "T_obs_yrs": T_obs_yrs,
            "n_frequencies": n_frequencies,
            "signal_label": signal_label,
            "signal_parameters": s_params,
        }

        (
            frequency,
            signal,
            HD_functions_IJ,
            HD_coeffs,
            effective_noise,
            SNR,
            fisher,
        ) = compute_fisher(
            **fisher_kwargs,
            get_tensors_kwargs=get_tensors_kwargs,
            generate_catalog_kwargs=generate_catalog_kwargs,
        )

        covariance = compute_inverse(fisher)
        results[:, i] = np.sqrt(np.diag(covariance))

    cols = list(my_colormap.keys())
    for i in range(len(signal_parameters)):
        plt.semilogy(
            parameter_range,
            results[i, :] / np.abs(signal_parameters[i]),
            color=my_colormap[cols[i]],
            label=parameter_labels[i],
        )

    plt.ylabel(r"$\rm Relative \ error \ (1 \sigma)$")
    plt.xlabel(parameter_labels[parameter_index])
    plt.axhline(0.3, linestyle="dashed", color="black")
    plt.legend(loc=1, ncols=2)
    plt.xlim(parameter_range[0], parameter_range[-1])
    plt.tight_layout()


def HD_constraints(
    signal_label,
    signal_parameters,
    T_obs_yrs=10.33,
    n_frequencies=30,
    get_tensors_kwargs={
        "order": 2,
        "method": "legendre",
        "path_to_pulsars": "pulsar_configurations/Mock_SKA.txt",
    },
    generate_catalog_kwargs={"n_pulsars": 30, "save_it": True, **mockSKA10},
    add_HD_prior=True,
    len_fisher_data=10000,
    parameter_labels=[
        r"$\alpha_{\rm PL}$",
        r"$n_{\rm T}$",
    ],
    plot_kwargs={},
):

    fisher_kwargs = {
        "T_obs_yrs": T_obs_yrs,
        "n_frequencies": n_frequencies,
        "signal_label": signal_label,
        "signal_parameters": signal_parameters,
    }

    (
        frequency,
        signal,
        HD_functions_IJ,
        HD_coeffs,
        effective_noise,
        SNR,
        fisher,
    ) = compute_fisher(
        **fisher_kwargs,
        get_tensors_kwargs=get_tensors_kwargs,
        generate_catalog_kwargs=generate_catalog_kwargs,
    )

    len_signal = len(signal_parameters)
    if get_tensors_kwargs["order"] and add_HD_prior:
        fisher += np.diag(
            np.append(np.zeros(len_signal), np.ones(len(fisher) - len_signal))
        )

    covariance = compute_inverse(fisher)
    fisher_data = np.random.multivariate_normal(
        np.append(signal_parameters, HD_coeffs),
        covariance,
        size=len_fisher_data,
    )

    if get_tensors_kwargs["method"].lower() == "legendre":
        plot_HD_Legendre(
            1000,
            fisher_data[:, len_signal:],
            r"$\rm HD \ reconstruction \ Legendre$",
        )
    elif get_tensors_kwargs["method"].lower() == "binned":
        plot_HD_binned(
            fisher_data[:, len_signal:],
            HD_coeffs,
            r"$\rm HD \ reconstruction \ binned$",
        )
    else:
        raise ValueError("Cannot use that")


def get_constraints(
    signal_label,
    signal_parameters,
    priors=[],
    T_obs_yrs=10.33,
    n_frequencies=30,
    rerun_MCMC=True,
    path_to_MCMC_data="generated_data/MCMC_data.npz",
    path_to_MCMC_chains="generated_chains/MCMC_chains.npz",
    MCMC_kwargs={},
    get_tensors_kwargs={},
    generate_catalog_kwargs={"n_pulsars": 30, "save_it": True, **mockSKA10},
    len_fisher_data=10000,
    parameter_labels=[
        r"$\alpha_{\rm PL}$",
        r"$n_{\rm T}$",
    ],
):

    print("-- Getting constraints --")

    fisher_kwargs = {
        "T_obs_yrs": T_obs_yrs,
        "n_frequencies": n_frequencies,
        "signal_label": signal_label,
        "signal_parameters": signal_parameters,
    }

    MCMC_kwargs["path_to_MCMC_data"] = path_to_MCMC_data
    MCMC_kwargs["path_to_MCMC_chains"] = path_to_MCMC_chains

    if "regenerate_catalog" in get_tensors_kwargs.keys():
        if get_tensors_kwargs["regenerate_catalog"]:
            rerun_MCMC = True
            regenerate_MCMC_data = True

    (
        frequency,
        signal,
        HD_functions_IJ,
        HD_coeffs,
        effective_noise,
        SNR,
        fisher,
    ) = compute_fisher(
        **fisher_kwargs,
        get_tensors_kwargs=get_tensors_kwargs,
        generate_catalog_kwargs=generate_catalog_kwargs,
    )

    covariance = compute_inverse(fisher)
    fisher_data = np.random.multivariate_normal(
        signal_parameters, covariance, size=len_fisher_data
    )
    errors = np.sqrt(np.diag(covariance))
    print("Fisher errors", errors)

    get_tensors_kwargs["regenerate_catalog"] = False

    try:
        if rerun_MCMC:
            raise FileNotFoundError("Flag forces MCMC chains regeneration")
        MCMC_results = np.load(path_to_MCMC_chains)
        MCMC_data = MCMC_results["samples"]
        pdfs = MCMC_results["pdfs"]

    except FileNotFoundError:
        if not priors:
            priors = signal_parameters[None, :] + 5 * np.array(
                [-errors, errors]
            )

        MCMC_data, pdfs = run_MCMC(
            priors,
            **fisher_kwargs,
            **MCMC_kwargs,
            get_tensors_kwargs=get_tensors_kwargs,
        )

    ### This part should be moved out of this function
    datasets = [fisher_data, MCMC_data]
    weights = [
        np.ones(len_fisher_data),
        np.ones(MCMC_data.shape[0]),
    ]
    smooth = [1.0, 1.0]

    ranges = [
        (
            signal_parameters[i] - 5 * errors[i],
            signal_parameters[i] + 5 * errors[i],
        )
        for i in range(len(errors))
    ]

    plot_corner(
        datasets,
        colors=[my_colormap["red"], my_colormap["green"]],
        truths=signal_parameters,
        chain_labels=["Fisher future", "MCMC future"],
        weights=weights,
        smooth=smooth,
        labels=parameter_labels,
        range=ranges,
        truth_color="black",
    )

    datasets = [d - np.mean(d, axis=0) for d in datasets]
    ranges = [(-5 * errors[i], +5 * errors[i]) for i in range(len(errors))]

    parameter_labels = [r"$\Delta " + p[1:] for p in parameter_labels]

    plot_corner(
        datasets,
        colors=[my_colormap["red"], my_colormap["green"]],
        truths=np.zeros(len(signal_parameters)),
        chain_labels=["Fisher future", "MCMC future"],
        weights=weights,
        smooth=smooth,
        labels=parameter_labels,
        range=ranges,
        truth_color="black",
    )
    print("-- Done --\n")


if __name__ == "__main__":
    scan_parameter(
        "power_law_lognormal",
        np.concatenate([SMBBH_parameters, CGW_LN_parameters]),
        0,
        np.linspace(-13, -7, 15),
        n_frequencies=100,
        generate_catalog_kwargs={"n_pulsars": 30, "save_it": True, **mockSKA10},
        get_tensors_kwargs={
            "path_to_pulsars": "pulsar_configurations/future.txt",
            "add_curn": True,
            "regenerate_catalog": False,
        },
        parameter_labels=[
            r"$\alpha_{\rm PL}$",
            r"$n_{\rm T}$",
            r"$\alpha_{\rm LN}$",
            r"$\beta_{\rm LN}$",
            r"$\gamma{\rm LN}$",
        ],
    )

    HD_constraints(
        "power_law",
        SMBBH_parameters,
        get_tensors_kwargs={
            "order": 2,
            "method": "legendre",
            "path_to_pulsars": "pulsar_configurations/Mock_SKA.txt",
            "add_curn": True,
            "regenerate_catalog": True,
        },
        n_frequencies=200,
        generate_catalog_kwargs={
            "n_pulsars": 30,
            "save_it": True,
            "outname": "pulsar_configurations/Mock_SKA.txt",
            **mockSKA10,
        },
    )

    get_constraints(
        "power_law_lognormal",
        np.concatenate([SMBBH_parameters, CGW_LN_parameters]),
        n_frequencies=100,
        rerun_MCMC=False,
        path_to_MCMC_data="generated_data/MCMC_data_ln_future.npz",
        path_to_MCMC_chains="generated_chains/MCMC_chains_ln_future.npz",
        MCMC_kwargs={},
        generate_catalog_kwargs={"n_pulsars": 30, "save_it": True, **mockSKA10},
        get_tensors_kwargs={
            "path_to_pulsars": "pulsar_configurations/future.txt",
            "add_curn": True,
            "regenerate_catalog": False,
        },
        parameter_labels=[
            r"$\alpha_{\rm PL}$",
            r"$n_{\rm T}$",
            r"$\alpha_{\rm LN}$",
            r"$\beta_{\rm LN}$",
            r"$\gamma{\rm LN}$",
        ],
    )
    plt.show(block=True)
