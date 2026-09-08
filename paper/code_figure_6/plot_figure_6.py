"""Render the three-panel Figure 6 from prepared data only."""

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from paper.utils.figures import (
    COLORS,
    IQR_ALPHA,
    SEQUENTIAL_HEATMAP_COLORS,
    add_panel_label,
    apply_publication_style,
    full_width_single_row_figure_size,
    load_prepared_data,
    save_publication_figure,
    style_axis,
    use_fixed_exponent,
)

CURVE_STYLES = ((COLORS["blue"], "--"), (COLORS["red"], (0, (5.0, 1.2, 1.2, 1.2))))


def _panel_a(axis, data, metadata):
    display_max = float(metadata["experiment"]["kernel_display_t_max"])
    first_L = int(metadata["experiment"]["displayed_pairs"][0][1])
    truth_t = data[f"kernel_t__L{first_L}"]
    truth_mask = truth_t <= display_max
    axis.plot(
        truth_t[truth_mask],
        data[f"kernel_truth__L{first_L}"][truth_mask],
        color=COLORS["truth"],
        linewidth=1.55,
        label="Truth",
        zorder=1,
    )
    displayed_pairs = metadata["experiment"]["displayed_pairs"]
    for curve_index, pair in enumerate(displayed_pairs):
        M, L = map(int, pair)
        style = CURVE_STYLES[curve_index]
        t = data[f"kernel_t__L{L}"]
        mask = t <= display_max
        key = f"kernel_estimate__M{M}__L{L}"
        is_top_curve = curve_index == len(displayed_pairs) - 1
        axis.plot(
            t[mask],
            data[key][mask],
            color=style[0],
            linestyle=style[1],
            linewidth=1.10,
            label=rf"$M={M},\,L={L}$",
            zorder=6 if is_top_curve else 3,
        )
    axis.set_xlim(0.0, display_max)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\gamma(t)$")
    axis.legend(loc="best", fontsize=7.0)


def _grid(data, metadata):
    M_values = np.asarray(metadata["experiment"]["M_values"], dtype=int)
    L_values = np.asarray(metadata["experiment"]["L_values"], dtype=int)
    seeds = list(map(int, metadata["plot_seeds"]))
    error = np.full((len(seeds), len(L_values), len(M_values)), np.nan)
    for row in range(len(data["M"])):
        i = int(np.flatnonzero(L_values == data["L"][row])[0])
        j = int(np.flatnonzero(M_values == data["M"][row])[0])
        seed = int(data["seed"][row])
        error[seeds.index(seed), i, j] = float(data["kernel_error"][row])
    if not np.all(np.isfinite(error)):
        grid_points = len(M_values) * len(L_values)
        raise RuntimeError(
            f"Figure 6(b,c) requires {len(seeds)} complete seeds over all " f"{grid_points} M--L grid points."
        )
    return M_values, L_values, error


def _panel_b(axis, colorbar_axis, figure, data, metadata):
    M_values, L_values, error_by_seed = _grid(data, metadata)
    error = np.nanmedian(error_by_seed, axis=0)
    x = np.arange(len(M_values) + 1) - 0.5
    y = np.arange(len(L_values) + 1) - 0.5
    log_error = np.log10(error)
    color_map = mcolors.LinearSegmentedColormap.from_list("figure6_kernel_error", SEQUENTIAL_HEATMAP_COLORS)
    image = axis.pcolormesh(x, y, log_error, shading="flat", cmap=color_map, vmin=-0.75, vmax=-0.65)
    for i in range(len(L_values)):
        for j in range(len(M_values)):
            axis.text(j, i, rf"${error[i,j]:.3f}$", ha="center", va="center", fontsize=6.8, color="white")
    axis.set_xticks(np.arange(len(M_values)))
    axis.set_xticklabels([rf"${value}$" for value in M_values])
    axis.set_yticks(np.arange(len(L_values)))
    axis.set_yticklabels([rf"${value}$" for value in L_values])
    axis.set_xlabel(r"$M$")
    axis.set_ylabel(r"$L$")
    axis.tick_params(axis="both", which="both", length=0)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks([-0.750, -0.725, -0.700, -0.675, -0.650])
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    colorbar.ax.set_title(r"$\log_{10}(E_\gamma)$", pad=5.0, fontsize=6.7)
    colorbar.ax.tick_params(direction="in", length=2.4, width=0.65, labelsize=6.6)


def _panel_c(axis, data, metadata):
    M_values, L_values, error = _grid(data, metadata)
    M_factors = M_values.astype(float) / float(M_values[0])
    L_factors = L_values.astype(float) / float(L_values[0])
    M_slice = error[:, -1, :]
    L_slice = error[:, :, -1]

    def draw_summary(x, values, color, fill, marker, label):
        q25 = np.nanquantile(values, 0.25, axis=0)
        median = np.nanmedian(values, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        axis.fill_between(
            x, q25, q75, color=fill, alpha=IQR_ALPHA, linewidth=0.0, zorder=2
        )
        axis.plot(
            x,
            median,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.75,
            linewidth=1.25,
            label=label,
            zorder=4,
        )

    slopes_M = np.asarray([np.polyfit(np.log(M_factors), np.log(values), 1)[0] for values in M_slice])
    slopes_L = np.asarray([np.polyfit(np.log(L_factors), np.log(values), 1)[0] for values in L_slice])
    slope_M = float(np.median(slopes_M))
    slope_L = float(np.median(slopes_L))
    draw_summary(
        M_factors,
        M_slice,
        COLORS["blue"],
        COLORS["blue_fill"],
        "o",
        rf"increase $M$: $\chi_M={slope_M:.3f}$",
    )
    draw_summary(
        L_factors,
        L_slice,
        COLORS["red"],
        COLORS["red_fill"],
        "s",
        rf"increase $L$: $\chi_L={slope_L:.3f}$",
    )
    axis.set_xscale("log")
    factor_ticks = M_factors
    axis.set_xticks(factor_ticks)
    axis.set_xticklabels([rf"${value:g}$" for value in factor_ticks])
    axis.xaxis.set_minor_formatter(mticker.NullFormatter())
    axis.set_xlabel(r"Data factor")
    use_fixed_exponent(axis, -1)
    axis.set_ylabel(r"$E_\gamma$")
    axis.legend(loc="best", fontsize=7.0)


def make_figure(
    data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str, dpi: int
) -> dict[str, Any]:
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure = plt.figure(figsize=full_width_single_row_figure_size())
    outer = figure.add_gridspec(
        1, 3, width_ratios=(1.08, 1.0, 1.08), wspace=0.50, left=0.058, right=0.988, bottom=0.205, top=0.915
    )
    heatmap_b = outer[0, 1].subgridspec(1, 2, width_ratios=(1.0, 0.060), wspace=0.10)
    axes = [figure.add_subplot(outer[0, 0]), figure.add_subplot(heatmap_b[0, 0]), figure.add_subplot(outer[0, 2])]
    colorbar_axis = figure.add_subplot(heatmap_b[0, 1])
    _panel_a(axes[0], data, metadata)
    _panel_b(axes[1], colorbar_axis, figure, data, metadata)
    _panel_c(axes[2], data, metadata)
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    outputs = save_publication_figure(
        figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 6 workflow"
    )
    plt.close(figure)
    return outputs
