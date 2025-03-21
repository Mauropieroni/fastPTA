# Global
import corner
import matplotlib
import matplotlib.patches
import matplotlib.pyplot as plt

import numpy as np
from scipy.special import legendre


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
