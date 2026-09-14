"""Render the three-panel Figure 5 from prepared data only."""

from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle

from paper.utils.figures import (
    COLORS,
    SEQUENTIAL_HEATMAP_COLORS,
    add_panel_label,
    apply_publication_style,
    full_width_single_row_figure_size,
    load_prepared_data,
    save_publication_figure,
    style_axis,
)

FORCE_STYLE = {"zero": {"label": "Zero"}, "sin": {"label": "Sinusoidal"}, "bistable": {"label": "Bistable"}}


def _external_matrix(data, metadata, force_types, metric):
    seeds = list(map(int, metadata["external_plot_seeds"]))
    matrix = np.full((len(seeds), len(force_types)), np.nan)
    for column, force in enumerate(force_types):
        mask = (data["panel"] == "external") & (data["force_type"] == force) & np.isfinite(data[metric])
        for row in np.flatnonzero(mask):
            seed = int(data["seed"][row])
            if seed in seeds:
                matrix[seeds.index(seed), column] = float(data[metric][row])
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(
            f"Figure 5(a) lacks one or more {metric} values for its " f"declared {len(seeds)} complete seeds."
        )
    return matrix


def _panel_a(axis, data, metadata):
    force_types = list(metadata["experiment"]["external_force"]["force_types"])
    x = np.arange(len(force_types), dtype=float)
    series = (
        ("sigma_error", COLORS["blue"], "o", r"$E_\sigma$", -0.12),
        ("kernel_error", COLORS["teal"], "s", r"$E_\gamma$", 0.0),
        ("memory_error", COLORS["red"], "^", r"$E_{\mathcal M}$", 0.12),
    )
    for metric, color, marker, label, offset in series:
        matrix = _external_matrix(data, metadata, force_types, metric)
        q25 = np.quantile(matrix, 0.25, axis=0)
        median = np.median(matrix, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        axis.errorbar(
            x + offset, median, yerr=[median - q25, q75 - median],
            fmt="none", ecolor=color, elinewidth=1.1, capsize=2.5, capthick=1.0, zorder=3,
        )
        axis.plot(
            x + offset,
            median,
            color=color,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=0.75,
            linestyle="none",
            label=label,
            zorder=4,
        )
    axis.set_xticks(x)
    axis.set_xticklabels([FORCE_STYLE[name]["label"] for name in force_types], rotation=0, ha="center")
    axis.set_yscale("log")
    ticks = [0.02, 0.04, 0.06, 0.08, 0.10]
    axis.set_yticks(ticks)
    axis.set_yticklabels([f"{value:.2f}" for value in ticks])
    axis.set_ylabel("Error")
    lower, upper = axis.get_ylim()
    axis.set_ylim(lower, max(upper, 1.15e-1))
    axis.set_xlim(-0.45, len(force_types) - 0.55)
    axis.legend(loc="upper center", ncol=3, columnspacing=0.85, handletextpad=0.35)


def _cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])]))


def _phase_matrix(data, metadata, metric):
    alpha = np.asarray(metadata["experiment"]["phase_map"]["alpha_values"], dtype=float)
    kappa = np.asarray(metadata["experiment"]["phase_map"]["kappa_values"], dtype=float)
    report_seed = int(metadata["phase_report_seed"])
    success_field = "tail_success" if metric == "joint_levy_error" else "sigma_success"
    error = np.full((len(kappa), len(alpha)), np.nan)
    for row_index, kappa_value in enumerate(kappa):
        for column_index, alpha_value in enumerate(alpha):
            mask = (
                (data["panel"] == "phase")
                & np.isclose(data["kappa_true"], kappa_value)
                & np.isclose(data["alpha_true"], alpha_value)
                & (data["seed"] == report_seed)
                & data[success_field]
                & np.isfinite(data[metric])
            )
            indices = np.flatnonzero(mask)
            if len(indices) == 1:
                error[row_index, column_index] = float(data[metric][indices[0]])
            elif len(indices) > 1:
                raise RuntimeError(f"Figure 5 has duplicate {metric} rows for one phase cell.")
    if not np.any(np.isfinite(error)):
        raise RuntimeError(f"Figure 5 contains no finite {metric} cells at seed {report_seed}.")
    return alpha, kappa, error


def _phase_heatmap(
    axis,
    colorbar_axis,
    figure,
    data,
    metadata,
    *,
    metric,
    colorbar_title,
    colors,
    logarithmic=False,
    unacceptable_threshold=None,
):
    alpha, kappa, error = _phase_matrix(data, metadata, metric)
    color_map = mcolors.LinearSegmentedColormap.from_list(f"figure5_{metric}", colors).with_extremes(bad="#EEEEEE")
    finite = error[np.isfinite(error)]
    if logarithmic:
        if np.any(finite <= 0.0):
            raise RuntimeError(f"Log-scaled Figure 5 heatmap {metric} contains nonpositive values.")
        normalization = mcolors.LogNorm(vmin=float(np.min(finite)), vmax=float(np.max(finite)))
        plot_error = error
    elif unacceptable_threshold is not None:
        unacceptable_threshold = float(unacceptable_threshold)
        if not np.isfinite(unacceptable_threshold) or unacceptable_threshold <= 0.0:
            raise ValueError("The Figure 5 unacceptable-error threshold must be positive.")
        normalization = mcolors.Normalize(vmin=0.0, vmax=unacceptable_threshold, clip=True)
        plot_error = np.minimum(error, unacceptable_threshold)
    else:
        normalization = None
        plot_error = error
    image = axis.pcolormesh(
        _cell_edges(alpha),
        _cell_edges(kappa),
        np.ma.masked_invalid(plot_error),
        shading="flat",
        cmap=color_map,
        norm=normalization,
    )
    alpha_edges = _cell_edges(alpha)
    kappa_edges = _cell_edges(kappa)
    for row_index, column_index in np.argwhere(~np.isfinite(error)):
        axis.add_patch(
            Rectangle(
                (alpha_edges[column_index], kappa_edges[row_index]),
                alpha_edges[column_index + 1] - alpha_edges[column_index],
                kappa_edges[row_index + 1] - kappa_edges[row_index],
                facecolor="#EEEEEE",
                edgecolor="#777777",
                linewidth=0.28,
                hatch="////",
                zorder=2,
            )
        )
    if unacceptable_threshold is not None:
        for row_index, column_index in np.argwhere(np.isfinite(error) & (error >= unacceptable_threshold)):
            axis.add_patch(
                Rectangle(
                    (alpha_edges[column_index], kappa_edges[row_index]),
                    alpha_edges[column_index + 1] - alpha_edges[column_index],
                    kappa_edges[row_index + 1] - kappa_edges[row_index],
                    facecolor="none",
                    edgecolor="#555555",
                    linewidth=0.36,
                    hatch="xxxx",
                    zorder=3,
                )
            )
    axis.set_xlabel(r"$\alpha$")
    axis.set_ylabel(r"$\kappa$")
    axis.set_xticks(alpha)
    axis.set_yticks(kappa)
    axis.tick_params(axis="x", labelsize=6.8, pad=1.8)
    axis.tick_params(axis="y", labelsize=6.9, pad=1.8)
    axis.tick_params(axis="both", which="both", length=0)
    colorbar = figure.colorbar(
        image, cax=colorbar_axis, extend="max" if unacceptable_threshold is not None else "neither"
    )
    colorbar.ax.set_title(colorbar_title, pad=2.0, fontsize=7.0)
    colorbar.ax.tick_params(direction="in", length=2.4, width=0.65, labelsize=6.7)


def _panel_b(axis, colorbar_axis, figure, data, metadata):
    _phase_heatmap(
        axis,
        colorbar_axis,
        figure,
        data,
        metadata,
        metric="joint_levy_error",
        colorbar_title=r"$E_{\alpha,\kappa}$",
        colors=SEQUENTIAL_HEATMAP_COLORS,
        unacceptable_threshold=metadata["experiment"]["phase_map"].get("joint_error_unacceptable_threshold", 0.5),
    )


def _panel_c(axis, colorbar_axis, figure, data, metadata):
    _phase_heatmap(
        axis,
        colorbar_axis,
        figure,
        data,
        metadata,
        metric="sigma_error",
        colorbar_title=r"$E_{\sigma}$",
        colors=SEQUENTIAL_HEATMAP_COLORS,
        logarithmic=True,
        unacceptable_threshold=metadata["experiment"]["phase_map"].get("sigma_error_unacceptable_threshold", 0.5),
    )


def make_figure(
    data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str, dpi: int
) -> dict[str, Any]:
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure = plt.figure(figsize=full_width_single_row_figure_size())
    outer = figure.add_gridspec(
        1, 3, width_ratios=(1.08, 1.0, 1.0), wspace=0.50, left=0.060, right=0.985, bottom=0.285, top=0.925
    )
    heatmap_b = outer[0, 1].subgridspec(1, 2, width_ratios=(1.0, 0.060), wspace=0.10)
    heatmap_c = outer[0, 2].subgridspec(1, 2, width_ratios=(1.0, 0.060), wspace=0.10)
    axes = [figure.add_subplot(outer[0, 0]), figure.add_subplot(heatmap_b[0, 0]), figure.add_subplot(heatmap_c[0, 0])]
    colorbar_axes = [figure.add_subplot(heatmap_b[0, 1]), figure.add_subplot(heatmap_c[0, 1])]
    _panel_a(axes[0], data, metadata)
    _panel_b(axes[1], colorbar_axes[0], figure, data, metadata)
    _panel_c(axes[2], colorbar_axes[1], figure, data, metadata)
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    unacceptable_threshold = float(metadata["experiment"]["phase_map"].get("joint_error_unacceptable_threshold", 0.5))
    unacceptable = Patch(
        facecolor=SEQUENTIAL_HEATMAP_COLORS[-1],
        edgecolor="#555555",
        linewidth=0.45,
        hatch="xxxx",
        label=rf"Unacceptable ($E_{{\alpha,\kappa}}\geq {unacceptable_threshold:g}$)",
    )
    axes[1].legend(
        handles=[unacceptable],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        fontsize=6.5,
        frameon=True,
        fancybox=False,
        shadow=False,
        facecolor="white",
        edgecolor="none",
        framealpha=0.30,
        handlelength=1.0,
        handletextpad=0.4,
        handleheight=0.85,
        borderaxespad=0.0,
    )
    sigma_unacceptable_threshold = float(
        metadata["experiment"]["phase_map"].get("sigma_error_unacceptable_threshold", 0.5)
    )
    sigma_unacceptable = Patch(
        facecolor=SEQUENTIAL_HEATMAP_COLORS[-1],
        edgecolor="#555555",
        linewidth=0.45,
        hatch="xxxx",
        label=rf"Unacceptable ($E_{{\sigma}}\geq {sigma_unacceptable_threshold:g}$)",
    )
    axes[2].legend(
        handles=[sigma_unacceptable],
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        fontsize=6.5,
        frameon=True,
        fancybox=False,
        shadow=False,
        facecolor="white",
        edgecolor="none",
        framealpha=0.30,
        handlelength=1.0,
        handletextpad=0.4,
        handleheight=0.85,
        borderaxespad=0.0,
    )
    outputs = save_publication_figure(
        figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 5 workflow"
    )
    plt.close(figure)
    return outputs
