# Global
import tqdm
import numpy as np
import matplotlib.pyplot as plt


# Local
import examples_utils as eu
import fastPTA.utils as ut
import fastPTA.plotting_functions as pf
from fastPTA.Compute_PBH_Abundance import f_PBH_NL_QCD
from fastPTA.signals import (
    SMBBH_parameters,
    CGW_SIGW_parameters,
)
from fastPTA.Fisher_code import compute_fisher
from fastPTA.MCMC_code import run_MCMC
from fastPTA.plotting_functions import (
    plot_corner,
    plot_HD_Legendre,
    plot_HD_binned,
)


def scan_parameter(
    signal_label,
    signal_parameters,
    parameter_index,
    parameter_range,
    T_obs_yrs=10.33,
    n_frequencies=30,
    get_tensors_kwargs={
        "path_to_pulsar_catalog": "pulsar_configurations/Mock_SKA.txt"
    },
    generate_catalog_kwargs={
        "n_pulsars": 30,
        "save_catalog": True,
        **eu.mockSKA10,
    },
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

        covariance = ut.compute_inverse(fisher)
        results[:, i] = np.sqrt(np.diag(covariance))

    cols = list(pf.my_colormap.keys())
    for i in range(len(signal_parameters)):
        plt.semilogy(
            parameter_range,
            results[i, :] / np.abs(signal_parameters[i]),
            color=pf.my_colormap[cols[i]],
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
        "path_to_pulsar_catalog": "pulsar_configurations/Mock_SKA.txt",
    },
    generate_catalog_kwargs={
        "n_pulsars": 30,
        "save_catalog": True,
        **eu.mockSKA10,
    },
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

    covariance = ut.compute_inverse(fisher)
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
    T_obs_yrs=16.03,
    n_frequencies=30,
    rerun_MCMC=True,
    path_to_MCMC_data="generated_data/MCMC_data.npz",
    path_to_MCMC_chains="generated_chains/MCMC_chains.npz",
    MCMC_kwargs={},
    get_tensors_kwargs={},
    generate_catalog_kwargs={
        "n_pulsars": 30,
        "save_catalog": True,
        **eu.mockSKA10,
    },
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

    covariance = ut.compute_inverse(fisher)
    fisher_data = np.random.multivariate_normal(
        signal_parameters, covariance, size=len_fisher_data
    )
    errors = np.sqrt(np.diag(covariance))
    print("\nFisher errors", errors)

    get_tensors_kwargs["regenerate_catalog"] = False

    # MCMC_results = np.load(path_to_MCMC_chains)
    # MCMC_data = MCMC_results["samples"]
    # pdfs = MCMC_results["pdfs"]

    if not np.any(priors):
        # priors = signal_parameters[None, :] + 5 * np.array(
        #     [-errors, errors]
        # )
        priors = np.array(
            [
                [
                    signal_parameters[0] - 5 * errors[0],
                    signal_parameters[0] + 5 * errors[0],
                ],
                [
                    signal_parameters[1] - 5 * errors[1],
                    signal_parameters[1] + 5 * errors[1],
                ],
                [
                    max([-3.5, signal_parameters[2] - 5 * errors[2]]),
                    min([0, signal_parameters[2] + 5 * errors[2]]),
                ],
                [
                    max([-1.5, signal_parameters[3] - 5 * errors[3]]),
                    min([0.9, signal_parameters[3] + 5 * errors[3]]),
                ],
                [
                    max([-10, signal_parameters[4] - 5 * errors[4]]),
                    min([-6, signal_parameters[4] + 5 * errors[4]]),
                ],
            ]
        ).T
        print(priors)

        # i = 0
        # while i < nwalkers:
        #     PBH_abundance = f_PBH_NL_QCD(
        #         10 ** initial[i, 2],
        #         10 ** initial[i, 3],
        #         10 ** initial[i, 4] * 2 * np.pi / (9.7156e-15),
        #     )
        #     if PBH_abundance > 1:
        #         initial[i] = np.random.uniform(
        #             priors[0, :], priors[1, :], size=(1, len(priors.T))
        #         )
        #     else:
        #         i = i + 1

    try:
        if rerun_MCMC:
            raise FileNotFoundError("Flag forces MCMC chains regeneration")
        MCMC_results = np.load(path_to_MCMC_chains)
        MCMC_data = MCMC_results["samples"]
        pdfs = MCMC_results["pdfs"]

    except FileNotFoundError:

        print("Entering MCMC")
        MCMC_data, pdfs = run_MCMC(
            priors,
            **fisher_kwargs,
            **MCMC_kwargs,
            get_tensors_kwargs=get_tensors_kwargs,
        )

    # Generate a FIM without the points with fpbh>1
    fisher_data_prior = fisher_data.copy()

    for i in range(len(priors.T)):
        fisher_data_prior = fisher_data_prior[
            (fisher_data_prior[:, i] > priors[0, i])
            & (fisher_data_prior[:, i] < priors[1, i])
        ]

    for i in tqdm.tqdm(range(len(fisher_data_prior))):
        PBH_abundance = f_PBH_NL_QCD(
            10 ** fisher_data_prior[i][2],
            10 ** fisher_data_prior[i][3],
            10 ** fisher_data_prior[i][4] * 2 * np.pi / (9.7156e-15),
        )
        if PBH_abundance > 1:
            fisher_data_prior[i] = np.nan

    fisher_data_prior = fisher_data_prior[
        ~np.isnan(fisher_data_prior).any(axis=1)
    ]

    # This part should be moved out of this function
    datasets = [fisher_data, MCMC_data, fisher_data_prior]
    weights = [
        np.ones(len_fisher_data),
        np.ones(MCMC_data.shape[0]),
        np.ones(len(fisher_data_prior)),  # type: ignore
    ]
    smooth = [1.0, 1.0, 1.0]

    ranges = [
        (
            signal_parameters[i] - 5 * errors[i],
            signal_parameters[i] + 5 * errors[i],
        )
        for i in range(len(errors))
    ]

    plot_corner(
        datasets,
        colors=[
            pf.my_colormap["red"],
            pf.my_colormap["green"],
            pf.my_colormap["blue"],
        ],
        truths=signal_parameters,
        chain_labels=["Fisher", "MCMC", "Fisher with priors"],
        weights=weights,
        smooth=smooth,
        labels=parameter_labels,
        range=ranges,
        truth_color="black",
        bbox_to_anchor=(0.0, 4.25, 1.0, 1.0),
    )
    plt.suptitle("Chains", fontsize=20)

    datasets = [d - np.mean(d, axis=0) for d in datasets]
    ranges = [(-5 * errors[i], +5 * errors[i]) for i in range(len(errors))]

    parameter_labels = [r"$\Delta " + p[1:] for p in parameter_labels]

    plot_corner(
        datasets,
        colors=[
            pf.my_colormap["red"],
            pf.my_colormap["green"],
            pf.my_colormap["blue"],
        ],
        truths=np.zeros(len(signal_parameters)),
        chain_labels=["Fisher", "MCMC", "Fisher with priors"],
        weights=weights,
        smooth=smooth,
        labels=parameter_labels,
        range=ranges,
        truth_color="black",
        bbox_to_anchor=(0.0, 4.25, 1.0, 1.0),
    )
    plt.suptitle("Shifted chains", fontsize=20)

    print("-- Done --\n")


if __name__ == "__main__":
    # scan_parameter(
    #     "power_law_SIGW",
    #     np.concatenate([SMBBH_parameters, CGW_SIGW_parameters]),
    #     0,
    #     np.linspace(-7, 4, 15),
    #     n_frequencies=100,
    #     generate_catalog_kwargs={
    #         "n_pulsars": 60,
    #         "save_catalog": True,
    #         **eu.mockSKA10,
    #     },
    #     get_tensors_kwargs={
    #         "path_to_pulsar_catalog": "pulsar_configurations/future.txt",
    #         "add_curn": True,
    #         "regenerate_catalog": False,
    #     },
    #     parameter_labels=[
    #         r"$A$",
    #         r"$n_T$",
    #         r"$Log_{10}A$",
    #         r"$Log_{10} \\Delta$",
    #         r"$Log_{10} f_*$",
    #     ],
    # )

    # HD_constraints(
    #     "power_law_SIGW",
    #     np.concatenate([SMBBH_parameters, CGW_SIGW_parameters]),
    #     get_tensors_kwargs={
    #         "order": 2,
    #         "method": "legendre",
    #         "path_to_pulsar_catalog": "pulsar_configurations/Mock_SKA.txt",
    #         "add_curn": True,
    #         "regenerate_catalog": True,
    #     },
    #     n_frequencies=200,
    #     generate_catalog_kwargs={
    #         "n_pulsars": 30,
    #         "save_catalog": True,
    #         "outname": "pulsar_configurations/Mock_SKA.txt",
    #         **eu.mockSKA10,
    #     },
    # )

    get_constraints(
        "power_law_SIGW",
        np.concatenate([SMBBH_parameters, CGW_SIGW_parameters]),
        T_obs_yrs=10.33,
        n_frequencies=30,
        rerun_MCMC=False,  # True,
        path_to_MCMC_data="generated_data/MCMC_data_Pl+SIGW_200p.npz",
        path_to_MCMC_chains="generated_chains/MCMC_chains_Pl+SIGW_200p.npz",
        MCMC_kwargs={
            "realization": False,
            "regenerate_MCMC_data": True,
            "i_max": 20,
            "R_convergence": 1e-1,
            "R_criterion": "max",
            "burnin_steps": 1000,
            "MCMC_iteration_steps": 500,
        },
        generate_catalog_kwargs={
            "n_pulsars": 200,
            "save_catalog": True,
            **eu.mockSKA10,
        },
        get_tensors_kwargs={
            "path_to_pulsar_catalog": "pulsar_configurations/SKA200p.txt",
            "add_curn": False,
            "regenerate_catalog": True,
        },
    )

    plt.show(block=True)
