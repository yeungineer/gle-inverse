"""Plot the seed-42 additive-Brownian external comparison."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from paper.utils.figures import (
    COLORS,
    add_panel_label,
    apply_publication_style,
    full_width_single_row_figure_size,
    load_prepared_data,
    save_publication_figure,
    style_axis,
)

METHOD_STYLE = {
    "proposed": {
        "label": "Proposed",
        "color": COLORS["blue"],
        "marker": "o",
        "linestyle": "--",
    },
    "prony_sobolev": {
        "label": r"Prony--Sobolev",
        "color": COLORS["red"],
        "marker": "s",
        "linestyle": "-.",
    },
    "vroylandt_em": {
        "label": r"Likelihood--EM",
        "color": COLORS["teal"],
        "marker": "^",
        "linestyle": ":",
    },
}
METHODS = ("proposed", "prony_sobolev", "vroylandt_em")
PLOTTED_METHODS = METHODS


def _panel_a(axis, data: dict[str, np.ndarray], metadata: dict[str, Any]) -> None:
    lower, upper = map(float, metadata["kernel_time_window"])
    value_lower, value_upper = map(float, metadata["kernel_value_window"])
    truth_time = np.asarray(data["kernel_t__proposed"], dtype=float)
    truth_mask = (truth_time >= lower) & (truth_time <= upper)
    axis.plot(
        truth_time[truth_mask],
        np.asarray(data["kernel_truth__proposed"], dtype=float)[truth_mask],
        color=COLORS["truth"],
        linewidth=1.55,
        label="Truth",
        zorder=1,
    )
    for method in PLOTTED_METHODS:
        style = METHOD_STYLE[method]
        time = np.asarray(data[f"kernel_t__{method}"], dtype=float)
        mask = (time >= lower) & (time <= upper)
        error = float(np.asarray(data[f"kernel_error__{method}"], dtype=float)[0])
        estimate = np.asarray(data[f"kernel_estimate__{method}"], dtype=float)[mask]
        outside = (estimate < value_lower) | (estimate > value_upper)
        visible = estimate.copy()
        visible[outside] = np.nan
        off_scale = bool(np.any(outside))
        error_text = _compact_value(error, 3)
        label = rf"{style['label']}, $E_\gamma={error_text}$"
        axis.plot(
            time[mask],
            visible,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.20,
            label=label,
            zorder=3,
        )
        if off_scale:
            first = int(np.flatnonzero(outside)[0])
            above = bool(estimate[first] > value_upper)
            axis.plot(
                time[mask][first],
                value_upper if above else value_lower,
                marker="^" if above else "v",
                markersize=3.8,
                markerfacecolor=style["color"],
                markeredgecolor=style["color"],
                linestyle="none",
                clip_on=False,
                zorder=5,
            )
    axis.set_xlim(lower, upper)
    axis.set_ylim(value_lower, value_upper)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\gamma(t)$")
    axis.legend(loc="upper right", fontsize=7.0)


def _compact_value(value: float, decimal_places: int) -> str:
    """Return a LaTeX number with a fixed number of decimal places."""
    value = float(value)
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0.0 else 0
    if exponent <= -3 or exponent >= 3:
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.{max(int(decimal_places), 0)}f}\times10^{{{exponent}}}"
    return f"{value:.{max(int(decimal_places), 0)}f}"


def _kernel_cost_axis(axis, data, *, metric, scale, xlabel, decimals) -> None:
    positions = {method: float(len(PLOTTED_METHODS) - index - 1) for index, method in enumerate(PLOTTED_METHODS)}
    summaries = {}
    for method in PLOTTED_METHODS:
        style = METHOD_STYLE[method]
        values = scale * np.asarray(data[f"{metric}__{method}"], dtype=float)
        summaries[method] = (float(np.median(values)), float(np.min(values)), float(np.max(values)))
        median, lower, upper = summaries[method]
        y = positions[method]
        axis.plot([lower, upper], [y, y], color=style["color"], linewidth=1.2, solid_capstyle="butt", zorder=1)
        axis.plot(
            median,
            y,
            marker=style["marker"],
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=style["color"],
            markeredgewidth=1.0,
            linestyle="none",
            zorder=3,
        )

    smallest = min(lower for _, lower, _ in summaries.values())
    largest = max(upper for _, _, upper in summaries.values())
    positive = [value for values in summaries.values() for value in values if value > 0.0]
    if smallest < 0.0 or not positive:
        raise ValueError(f"Figure 2 {metric} values must be nonnegative and not all zero.")
    if smallest == 0.0:
        linear_threshold = min(positive) / 10.0
        axis.set_xscale("symlog", linthresh=linear_threshold, linscale=0.8)
        axis.set_xlim(0.0, largest * 3.2)
    else:
        axis.set_xscale("log")
        axis.set_xlim(smallest / 1.8, largest * 3.2)
    for method, (median, _, _) in summaries.items():
        annotate_left = median == max(value[0] for value in summaries.values())
        axis.annotate(
            rf"${_compact_value(median, int(decimals))}$",
            xy=(median, positions[method]),
            xytext=((-9.0 if annotate_left else 10.0), 0.0),
            textcoords="offset points",
            ha=("right" if annotate_left else "left"),
            va="center",
            fontsize=6.8,
            color="#333333",
        )
    axis.set_ylim(-0.60, len(PLOTTED_METHODS) - 0.40)
    axis.set_yticks([positions[method] for method in PLOTTED_METHODS])
    axis.set_yticklabels([METHOD_STYLE[method]["label"] for method in PLOTTED_METHODS], fontsize=7.0)
    axis.tick_params(axis="y", which="both", length=0.0, right=False)
    axis.set_xlabel(xlabel)
    axis.minorticks_off()


def make_figure(
    data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str, dpi: int
) -> dict[str, Any]:
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)

    figure = plt.figure(figsize=full_width_single_row_figure_size())
    outer = figure.add_gridspec(
        1, 2, width_ratios=(1.18, 1.0), wspace=0.32, left=0.070, right=0.990, bottom=0.205, top=0.925
    )
    panel_a = figure.add_subplot(outer[0, 0])
    cost_grid = outer[0, 1].subgridspec(2, 1, hspace=0.46)
    cost_time = figure.add_subplot(cost_grid[0, 0])
    cost_memory = figure.add_subplot(cost_grid[1, 0])

    _panel_a(panel_a, data, metadata)
    _kernel_cost_axis(cost_time, data, metric="kernel_elapsed_seconds", scale=1.0, xlabel=r"Time (s)", decimals=3)
    _kernel_cost_axis(
        cost_memory, data, metric="kernel_incremental_peak_rss_mib", scale=1.0, xlabel=r"Peak memory (MiB)", decimals=3
    )
    for axis in (panel_a, cost_time, cost_memory):
        style_axis(axis)
    add_panel_label(panel_a, "a", y_offset_points=5.5)
    add_panel_label(cost_time, "b", y_offset_points=5.5)
    outputs = save_publication_figure(
        figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 2 Brownian comparison workflow"
    )
    plt.close(figure)
    return outputs
