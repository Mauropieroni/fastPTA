# Global imports
import corner
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt

import numpy as np
from scipy.special import legendre

# Local imports
from fastPTA import get_tensors as gt


# Setting plotting parameters
matplotlib.rcParams["text.usetex"] = True
plt.rc("xtick", labelsize=20)
plt.rc("ytick", labelsize=20)
plt.rcParams.update(
    {"axes.labelsize": 20, "legend.fontsize": 20, "axes.titlesize": 22}
)

# Some other useful things
cmap_HD = matplotlib.colormaps["coolwarm"]
cmap1_grid = matplotlib.colormaps["hot"]
cmap2_grid = matplotlib.colormaps["PiYG"]

my_colormap = {
    "red": "#EE6677",
    "green": "#228833",
    "blue": "#4477AA",
    "yellow": "#CCBB44",
    "purple": "#AA3377",
    "cyan": "#66CCEE",
    "gray": "#BBBBBB",
}


def plot_HD_Legendre(x_points, data_HD_coeffs, plot_suplabel):
    x = np.linspace(-1, 1, x_points)
    p_sum = np.zeros(shape=(x_points, len(data_HD_coeffs)))
    for i in range(data_HD_coeffs.shape[-1]):
        pol_i = data_HD_coeffs[None, :, i] * (legendre(i)(x))[:, None]
        p_sum += pol_i

    res = np.quantile(p_sum, [0.025, 0.16, 0.5, 0.84, 0.975], axis=-1)

    plt.figure(figsize=(6, 4))

    plt.plot(np.arccos(x), res[2, :], color="dimgrey", label=r"$\rm Injection$")

    plt.fill_between(
        np.arccos(x),
        y1=res[1, :],
        y2=res[-2, :],
        color="lightskyblue",
        label=r"$1 \sigma$",
        zorder=-1,
    )

    plt.fill_between(
        np.arccos(x),
        y1=res[0, :],
        y2=res[-1, :],
        color="dodgerblue",
        label=r"$2 \sigma$",
        zorder=-2,
    )

    plt.xlabel(r"$\zeta_{IJ} \equiv \arccos(\hat{p}_I \cdot \hat{p}_J)$")
    plt.xticks(
        np.pi / 4 * np.arange(5),
        labels=[r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3 \pi / 4$", r"$\pi$"],
    )

    plt.xlim(0, np.pi)
    plt.ylim(-0.4, 0.75)
    plt.legend(fontsize=20, loc=9)
    plt.suptitle(plot_suplabel, fontsize=20)
    plt.tight_layout()


def plot_HD_binned(data_HD_coeffs, HD_coeffs, plot_suplabel):
    plt.figure(figsize=(6, 4))

    bin_edges = np.linspace(0, np.pi, len(HD_coeffs) + 1)
    bin_val = 0.5 * (bin_edges[1:] + bin_edges[:-1])

    plt.violinplot(
        data_HD_coeffs,
        widths=0.2,
        positions=bin_val,
        showextrema=True,
        quantiles=[[0.16, 0.5, 0.84] for xx in range(len(HD_coeffs))],
    )

    plt.scatter(bin_val, HD_coeffs, color="dimgrey", label=r"$\rm Injection$")
    plt.plot(
        np.arccos(gt.x),
        gt.HD_value,
        color="red",
        linestyle="--",
        label="HD curve",
    )

    plt.xlabel(r"$\zeta_{IJ} \equiv \arccos(\hat{p}_I \cdot \hat{p}_J)$")
    plt.xticks(
        np.pi / 4 * np.arange(5),
        labels=[r"$0$", r"$\pi/4$", r"$\pi/2$", r"$3 \pi / 4$", r"$\pi$"],
    )

    plt.xlim(0, np.pi)
    plt.legend(fontsize=20, loc=9)
    plt.suptitle(plot_suplabel, fontsize=20)
    plt.tight_layout()


def plot_corner(
    samples,
    colors=False,
    smooth=False,
    weights=False,
    fig=False,
    chain_labels=False,
    show_titles=False,
    plot_density=True,
    plot_datapoints=False,
    fill_contours=True,
    bins=25,
    title_kwargs={"fontsize": 20, "pad": 12},
    hist_kwargs={"density": True, "linewidth": 2},
    bbox_to_anchor=(0.0, 1.0, 1.0, 1.0),
    **kwargs,
):

    if not colors:
        vals = list(my_colormap.values())
        colors = []
        for i in range(len(samples)):
            colors = [
                vals[np.mod(i, len(samples))] for i in range(len(samples))
            ]

    if not smooth:
        smooth = np.zeros(len(samples))

    if not weights:
        weights = [None for i in range(len(samples))]

    if not fig:
        fig = plt.figure(figsize=(10, 8))

    if not chain_labels:
        chain_labels = [None for i in range(len(samples))]

    for i in range(len(samples)):
        hist_kwargs["color"] = colors[i]  # type: ignore

        if samples[i].shape[-1] > 1:
            fig = corner.corner(
                samples[i],
                color=colors[i],  # type: ignore
                smooth=smooth[i],  # type: ignore
                weights=weights[i],  # type: ignore
                fig=fig,
                show_titles=show_titles,
                plot_density=plot_density,
                plot_datapoints=plot_datapoints,
                fill_contours=fill_contours,
                bins=bins,
                title_kwargs=title_kwargs,
                hist_kwargs=hist_kwargs,
                **kwargs,
            )

        else:
            plt.hist(
                samples[i],
                color=colors[i],  # type: ignore
                weights=weights[i],  # type: ignore
                bins=bins,
                histtype="step",
                label=chain_labels[i],  # type: ignore
                density=True,
            )

            if "truths" in kwargs.keys():
                truths = kwargs["truths"]
            if "truth_color" in kwargs.keys():
                truth_color = kwargs["truth_color"]
            else:
                truth_color = "black"

            plt.axvline(truths, color=truth_color)

    custom_lines = []

    if samples[0].shape[-1] > 1:
        for i in range(len(chain_labels)):  # type: ignore
            custom_lines.append(
                matplotlib.patches.Patch(
                    facecolor=colors[i], label=chain_labels[i]  # type: ignore
                )
            )

        plt.legend(handles=custom_lines, bbox_to_anchor=bbox_to_anchor, loc=0)
    else:
        plt.legend(loc=0)


def plot_chain_results(
    flat_samples, posterior_mean, posterior_std, injected_means
):
    n_params = flat_samples.shape[1]

    param_groups = []
    param_idx, ell = 0, 0
    while param_idx < n_params:
        n_params_this_l = 2 * ell + 1
        if param_idx + n_params_this_l <= n_params:
            param_groups.append(
                list(range(param_idx, param_idx + n_params_this_l))
            )
            param_idx += n_params_this_l
            ell += 1
        else:
            param_groups.append(list(range(param_idx, n_params)))
            break

    labels = [""] * n_params
    for ell, group in enumerate(param_groups):
        for i, param_idx in enumerate(group):
            m = i - ell
            labels[param_idx] = f"$c_{{{ell}{m if m != 0 else '0'}}}$"

    n_rows, max_cols = len(param_groups), max(
        len(group) for group in param_groups
    )
    fig, axes = plt.subplots(
        n_rows, max_cols, figsize=(3 * max_cols, 3 * n_rows)
    )

    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif max_cols == 1:
        axes = axes.reshape(-1, 1)
    if n_rows == 1 and max_cols == 1:
        axes = np.array([[axes]])

    legend_created = False

    for row_idx, group in enumerate(param_groups):
        for col_idx, param_idx in enumerate(group):
            ax = axes[row_idx, col_idx]

            ax.hist(
                flat_samples[:, param_idx],
                bins=25,
                density=True,
                alpha=0.3,
                color="steelblue",
                edgecolor="black",
                linewidth=0.5,
            )

            q_values = np.percentile(
                flat_samples[:, param_idx],
                [5, 50, 95],
            )

            ax.axvline(
                q_values[0],
                color="red",
                linestyle="--",
                alpha=1,
                linewidth=3,
                label="5th/95th percentile" if not legend_created else "",
            )
            ax.axvline(
                q_values[2],
                color="red",
                linestyle="--",
                alpha=1,
                linewidth=3,
            )

            ax.axvline(
                q_values[1],
                color="red",
                linestyle="-",
                alpha=1,
                linewidth=3,
                label="Median (50th percentile)" if not legend_created else "",
            )

            mean = posterior_mean[param_idx]
            std = posterior_std[param_idx]

            x = np.linspace(mean - 4 * std, mean + 4 * std, 100)
            y = (
                1
                / (std * np.sqrt(2 * np.pi))
                * np.exp(-0.5 * ((x - mean) / std) ** 2)
            )

            ax.plot(x, y, color="blue", linewidth=2, label="Posterior PDF")

            if param_idx < len(injected_means):
                ax.axvline(
                    injected_means[param_idx],
                    color="Black",
                    linestyle="-",
                    linewidth=3,
                    alpha=1,
                    label="True value" if not legend_created else "",
                )

            ax.set_xlabel(labels[param_idx], fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            median, upper, lower = (
                q_values[1],
                q_values[2] - q_values[1],
                q_values[1] - q_values[0],
            )
            ax.set_title(
                f"{median:.3f}$^{{+{upper:.3f}}}_{{-{lower:.3f}}}$", fontsize=8
            )
            ax.tick_params(axis="both", labelsize=6)
            ax.locator_params(nbins=3)
            ax.grid(True, alpha=0.3)

            legend_created = True

    for row_idx, group in enumerate(param_groups):
        for col_idx in range(len(group), max_cols):
            axes[row_idx, col_idx].set_visible(False)

    plt.tight_layout()

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=len(labels),
        fontsize=10,
        frameon=True,
    )

    plt.subplots_adjust(bottom=0.1)

    plt.show()
