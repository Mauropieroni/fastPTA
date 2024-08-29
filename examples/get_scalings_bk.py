# Global
import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Local
import examples_utils as eu
import fastPTA.utils as ut
import fastPTA.plotting_functions as pf
from fastPTA.signals import SMBBH_parameters
from fastPTA.Fisher_code import compute_fisher


def T_scaling(
    rerun=False,
    T_min_yrs=0.1,
    T_max_yrs=100,
    N_times=10,
    N_runs=20,
    n_pulsars=30,
    signal_label="power_law",
    signal_parameters=SMBBH_parameters,
    order=0,
    method="Legendre",
    add_curn=True,
    default_pulsars=eu.EPTAlike,
    add_HD_prior=True,
    signal_labels=[r"$\alpha_{*}$", r"$n_{\rm T}$"],
    save_path="Default",
):
    # test evolution of SNR with T_obs
    parameter_len = (
        len(signal_parameters) + order + 1 if order else len(signal_parameters)
    )

    get_tensors_kwargs = {
        "add_curn": add_curn,
        "order": order,
        "method": method,
        "regenerate_catalog": True,
    }

    if save_path != "Default":
        pass
    elif order == 0:
        save_path = "generated_data/T_scaling.npz"
    else:
        save_path = "generated_data/T_scaling_HD.npz"

    try:
        if rerun:
            raise FileNotFoundError("Forcing regeneration")

        data = np.load(save_path)

        T_obs_values = data["T_obs_values"]
        SNR_mean = data["SNR_mean"]
        SNR_std = data["SNR_std"]

        parameters_mean = data["parameters_mean"]
        parameters_std = data["parameters_std"]

    except FileNotFoundError:
        T_obs_values = np.geomspace(T_min_yrs, T_max_yrs, N_times)
        SNR_mean = np.zeros(shape=(N_times,))
        SNR_std = np.zeros(shape=(N_times,))
        parameters_mean = np.zeros(shape=(N_times, parameter_len))
        parameters_std = np.zeros(shape=(N_times, parameter_len))

        for i in range(N_times):
            SNR_iteration = np.zeros(shape=(N_runs,))
            parameters_iteration = np.zeros(shape=(N_runs, parameter_len))

            generate_catalog_kwargs = default_pulsars.copy()
            generate_catalog_kwargs["n_pulsars"] = n_pulsars
            generate_catalog_kwargs["T_span_dict"] = {
                "which_distribution": "gaussian",
                "mean": np.log10(T_obs_values[i]),
                "std": 0.0125,
            }

            print(
                "Here starts T = %.2f yrs (iteration %d of %d)"
                % (T_obs_values[i], i + 1, N_times)
            )

            for j in tqdm.tqdm(range(N_runs)):
                (
                    frequency,
                    signal,
                    HD_functions_IJ,
                    HD_coeffs,
                    effective_noise,
                    SNR,
                    fisher,
                ) = compute_fisher(
                    T_obs_yrs=T_obs_values[i],
                    n_frequencies=30 + int(T_obs_values[i]),
                    signal_label=signal_label,
                    signal_parameters=signal_parameters,
                    get_tensors_kwargs=get_tensors_kwargs,
                    generate_catalog_kwargs=generate_catalog_kwargs,
                )

                if order and add_HD_prior:
                    fisher += np.diag(
                        np.append(
                            np.zeros(len(signal_parameters)), np.ones(order + 1)
                        )
                    )
                c_inverse = ut.compute_inverse(fisher)
                errors = np.sqrt(np.diag(c_inverse))

                SNR_iteration[j] = SNR
                parameters_iteration[j] = errors

            SNR_mean[i] = np.mean(SNR_iteration, axis=0)
            SNR_std[i] = np.std(SNR_iteration, axis=0)

            parameters_mean[i] = np.mean(parameters_iteration, axis=0)
            parameters_std[i] = np.std(parameters_iteration, axis=0)

            print("SNR=%.2e +-%.2e \n" % (SNR_mean[i], SNR_std[i]))

        to_save = {
            "T_obs_values": T_obs_values,
            "SNR_mean": SNR_mean,
            "SNR_std": SNR_std,
            "parameters_mean": parameters_mean,
            "parameters_std": parameters_std,
        }

        np.savez(save_path, **to_save)

    # here plot scaling of SNR with T_obs
    plt.figure(figsize=(6, 4))
    plt.errorbar(
        T_obs_values,
        SNR_mean,
        yerr=SNR_std,
        color=pf.my_colormap["cyan"],
        fmt="o",
        markersize=4,
        linestyle="dashed",
        capsize=7,
    )

    # plot x**3
    plt.loglog(
        T_obs_values, 0.1 * T_obs_values**3, linestyle="--", color="black"
    )
    plt.text(1, 5e-3, s=r"$\propto T^3$", fontsize=15)

    # plot x**(1/2)
    plt.loglog(
        T_obs_values, 2 * T_obs_values**0.5, linestyle="--", color="black"
    )
    plt.text(30, 1.5, s=r"$\propto \sqrt{T}$", fontsize=15)

    plt.xlabel(r"$T_{\rm obs} \rm [yr]$")
    plt.ylabel(r"$\rm SNR$")
    plt.ylim(5e-6, 50)
    plt.tight_layout()
    plt.savefig("plots/SNR_T_scaling.pdf")

    plt.figure(figsize=(6, 4))
    for i in range(len(signal_parameters)):
        colors = list(pf.my_colormap.keys())
        col = colors[np.mod(i, len(colors))]
        plt.errorbar(
            T_obs_values,
            parameters_mean[:, i],
            yerr=parameters_std[:, i],
            color=col,
            fmt="o",
            markersize=4,
            linestyle="dashed",
            capsize=7,
            label=signal_labels[i],
        )

    plt.loglog(
        T_obs_values,
        100 / T_obs_values**3,
        linestyle="--",
        color="black",
    )

    plt.text(0.5, 1e4, s=r"$\propto 1/T^3$", fontsize=15)

    plt.loglog(
        T_obs_values,
        0.6 / T_obs_values ** (1 / 2),
        linestyle="--",
        color="black",
    )

    plt.text(30, 1, s=r"$\propto 1/\sqrt{T}$", fontsize=15)

    plt.xlabel(r"$T_{\rm obs} \rm [yr]$")
    plt.ylabel(r"$\rm Uncertainties$")
    plt.ylim(1e-2, 1e6)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("plots/Error_T_scaling.pdf")

    if order:
        if method == "Legendre":
            label = r"$\rm Polynomial \ \ell = \ $"
        else:
            label = r"$\rm Bin \ = \ $"
        # here plot scaling of SNR with T_obs
        plt.figure(figsize=(6, 4))
        for i in range(order + 1):
            plt.errorbar(
                T_obs_values,
                parameters_mean[:, len(signal_parameters) + i],
                yerr=parameters_std[:, len(signal_parameters) + i],
                label=label + str(i),
                color=pf.cmap_HD(0.1 + i / 1.1 / (order + 1)),
            )

        plt.xlabel(r"$T_{\rm obs} \rm [yr]$")
        plt.ylabel(r"$\rm Uncertainties$")
        plt.ylim(1e-2, 1e5)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=12, ncols=2)
        plt.tight_layout()
        plt.savefig("plots/SNR_T_scaling_HD.pdf")


def N_scaling(
    rerun=False,
    N_min=5,
    N_max=200,
    N_times=20,
    N_runs=20,
    signal_label="power_law",
    signal_parameters=SMBBH_parameters,
    order=0,
    method="Legendre",
    add_curn=True,
    default_pulsars=eu.EPTAlike,
    add_HD_prior=True,
    signal_labels=[r"$\alpha_{*}$", r"$n_{\rm T}$"],
    save_path="Default",
):
    # test evolution of SNR with T_obs
    parameter_len = (
        len(signal_parameters) + order + 1 if order else len(signal_parameters)
    )

    get_tensors_kwargs = {
        "add_curn": add_curn,
        "order": order,
        "method": method,
        "regenerate_catalog": True,
    }

    if save_path != "Default":
        pass
    elif order == 0:
        save_path = "generated_data/N_scaling.npz"
    else:
        save_path = "generated_data/N_scaling_HD.npz"

    try:
        if rerun:
            raise FileNotFoundError("Forcing regeneration")

        data = np.load(save_path)

        N_pulsars = data["N_pulsars"]
        SNR_mean = data["SNR_mean"]
        SNR_std = data["SNR_std"]

        parameters_mean = data["parameters_mean"]
        parameters_std = data["parameters_std"]

    except FileNotFoundError:
        N_pulsars = np.unique(np.geomspace(N_min, N_max, N_times, dtype=int))
        N_times = len(N_pulsars)
        SNR_mean = np.zeros(shape=(N_times,))
        SNR_std = np.zeros(shape=(N_times,))
        parameters_mean = np.zeros(shape=(N_times, parameter_len))
        parameters_std = np.zeros(shape=(N_times, parameter_len))

        for i in range(N_times):
            SNR_iteration = np.zeros(shape=(N_runs,))
            parameters_iteration = np.zeros(shape=(N_runs, parameter_len))

            generate_catalog_kwargs = default_pulsars.copy()
            generate_catalog_kwargs["n_pulsars"] = N_pulsars[i]

            print("Here starts N = %d" % (N_pulsars[i]))
            for j in tqdm.tqdm(range(N_runs)):
                (
                    frequency,
                    signal,
                    HD_functions_IJ,
                    HD_coeffs,
                    effective_noise,
                    SNR,
                    fisher,
                ) = compute_fisher(
                    n_frequencies=30,
                    signal_label=signal_label,
                    signal_parameters=signal_parameters,
                    get_tensors_kwargs=get_tensors_kwargs,
                    generate_catalog_kwargs=generate_catalog_kwargs,
                )

                if order and add_HD_prior:
                    fisher += np.diag(
                        np.append(
                            np.zeros(len(signal_parameters)), np.ones(order + 1)
                        )
                    )

                c_inverse = ut.compute_inverse(fisher)
                errors = np.sqrt(np.diag(c_inverse))

                SNR_iteration[j] = SNR
                parameters_iteration[j] = errors

            SNR_mean[i] = np.mean(SNR_iteration, axis=0)
            SNR_std[i] = np.std(SNR_iteration, axis=0)

            parameters_mean[i] = np.mean(parameters_iteration, axis=0)
            parameters_std[i] = np.std(parameters_iteration, axis=0)

            print("SNR=%.2e +-%.2e \n" % (SNR_mean[i], SNR_std[i]))

        to_save = {
            "N_pulsars": N_pulsars,
            "SNR_mean": SNR_mean,
            "SNR_std": SNR_std,
            "parameters_mean": parameters_mean,
            "parameters_std": parameters_std,
        }

        np.savez(save_path, **to_save)

    # here plot scaling of SNR with N_pulsars
    plt.figure(figsize=(6, 4))

    plt.errorbar(
        N_pulsars,
        SNR_mean,
        yerr=SNR_std,
        color=pf.my_colormap["cyan"],
        fmt="o",
        markersize=4,
        linestyle="dashed",
        capsize=7,
    )

    plt.loglog(
        N_pulsars,
        np.sqrt(N_pulsars) / 2,
        linestyle="--",
        color="black",
    )

    plt.text(2e1, 7, s=r"$\propto \sqrt{N_{\rm pulsars}}$", fontsize=15)
    plt.ylim(0.5, 2e1)
    plt.xlabel(r"$N_{\rm pulsars}$", fontsize=20)
    plt.ylabel(r"$\rm SNR$", fontsize=20)
    plt.tight_layout()
    plt.savefig("plots/SNR_N_scaling.pdf")

    plt.figure(figsize=(6, 4))
    for i in range(len(signal_parameters)):
        colors = list(pf.my_colormap.keys())
        col = colors[np.mod(i, len(colors))]
        plt.errorbar(
            N_pulsars,
            parameters_mean[:, i],
            yerr=parameters_std[:, i],
            color=col,
            fmt="o",
            markersize=4,
            linestyle="dashed",
            capsize=7,
            label=signal_labels[i],
        )

    plt.loglog(
        N_pulsars,
        2 / N_pulsars**0.5,
        linestyle="--",
        color="black",
    )

    plt.text(2e2, 0.3, s=r"$\propto 1/\sqrt{N_{\rm pulsars}}$", fontsize=15)

    plt.xlabel(r"$N_{\rm pulsars}$", fontsize=20)
    plt.ylabel(r"$\rm Uncertainties$", fontsize=20)
    plt.legend(fontsize=15)
    plt.tight_layout()
    plt.savefig("plots/Error_N_scaling.pdf")

    if order:
        if method == "Legendre":
            label = r"$\rm Polynomial \ \ell = \ $"
        else:
            label = r"$\rm Bin \ = \ $"
        # here plot scaling of SNR with T_obs
        plt.figure(figsize=(6, 4))
        for i in range(order + 1):
            plt.errorbar(
                N_pulsars,
                parameters_mean[:, len(signal_parameters) + i],
                yerr=parameters_std[:, len(signal_parameters) + i],
                label=label + str(i),
                color=pf.cmap_HD(0.1 + i / 1.1 / (order + 1)),
            )

            plt.loglog(
                N_pulsars,
                2 / N_pulsars,
                linestyle="--",
                color="black",
            )

        plt.xlabel(r"$N_{\rm pulsars}$")
        plt.ylabel(r"$\rm Uncertainties$")
        plt.ylim(1e-3, 1e5)
        plt.xscale("log")
        plt.yscale("log")
        plt.legend(fontsize=12, ncols=2)
        plt.tight_layout()
        plt.savefig("plots/SNR_N_scaling_HD.pdf")


if __name__ == "__main__":
    N_scaling()
    N_scaling(order=6)
    T_scaling(T_max_yrs=1e3)  # type: ignore
    plt.show(block=True)
