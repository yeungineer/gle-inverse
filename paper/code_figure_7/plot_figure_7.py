from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from paper.utils.figures import (
    COLORS,
    add_panel_label,
    apply_publication_style,
    full_width_figure_size,
    load_prepared_data,
    save_publication_figure,
    style_axis,
)

PROTOCOL_STYLES = {
    "euler_oracle": (COLORS["blue"], "--", "Matched history"),
    "fine_branch": (COLORS["red"], "-.", "Short-lag release"),
    "replay_branch": (COLORS["teal"], (0, (2.0, 1.2)), "Feedback-prepared"),
}


def _set_y_limits(axis, curves):
    values = np.concatenate([np.asarray(curve, dtype=float) for curve in curves])
    margin = 0.08 * max(float(np.ptp(values)), float(np.max(np.abs(values))) * 0.10)
    axis.set_ylim(float(np.min(values)) - margin, float(np.max(values)) + margin)
    axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def _plot_sigma(axis, data, protocols, case):
    x = data[f"{case}__sigma_x"]
    truth = data[f"{case}__sigma_truth"]
    curves = [truth]
    axis.plot(x, truth, color=COLORS["truth"], linewidth=1.15, label="Truth", zorder=1)
    for protocol in protocols:
        estimate = data[f"{case}__sigma__{protocol}"]
        color, linestyle, label = PROTOCOL_STYLES[protocol]
        axis.plot(x, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label, zorder=3)
        curves.append(estimate)
    _set_y_limits(axis, curves)
    axis.set_xlabel(r"$v$")
    axis.set_ylabel(r"$\sigma(v)$")


def _plot_kernel(axis, data, protocols, case):
    time = data[f"{case}__kernel_t"]
    truth = data[f"{case}__kernel_truth"]
    curves = [truth]
    axis.plot(time, truth, color=COLORS["truth"], linewidth=1.15, label="Truth", zorder=1)
    for protocol in protocols:
        estimate = data[f"{case}__kernel__{protocol}"]
        color, linestyle, label = PROTOCOL_STYLES[protocol]
        axis.plot(time, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label, zorder=3)
        curves.append(estimate)
    _set_y_limits(axis, curves)
    axis.axhline(0.0, color=COLORS["neutral_light"], linewidth=0.65, linestyle="--", zorder=0)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\gamma(t)$")


def _plot_memory(axis, data, protocols, case):
    time = data[f"{case}__memory_t"]
    truth = data[f"{case}__memory_truth"]
    curves = [truth]
    axis.plot(time, truth, color=COLORS["truth"], linewidth=1.15, label="Truth", zorder=1)
    for protocol in protocols:
        estimate = data[f"{case}__memory__{protocol}"]
        color, linestyle, label = PROTOCOL_STYLES[protocol]
        axis.plot(time, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label, zorder=3)
        curves.append(estimate)
    _set_y_limits(axis, curves)
    axis.axhline(0.0, color=COLORS["neutral_light"], linewidth=0.65, linestyle="--", zorder=0)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$-\mathcal{M}(t)$")


def make_figure(data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str, dpi: int) -> dict[str, Any]:
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    protocols = list(metadata["protocols"])
    figure, axes = plt.subplots(2, 3, figsize=full_width_figure_size(), squeeze=False)
    for row, case in enumerate(("brownian", "brownian_levy")):
        _plot_sigma(axes[row, 0], data, protocols, case)
        _plot_kernel(axes[row, 1], data, protocols, case)
        _plot_memory(axes[row, 2], data, protocols, case)
    for label, axis in zip("abcdef", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    axes[0, 0].yaxis.labelpad = 0.0
    axes[1, 0].yaxis.labelpad = 0.0
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.94, wspace=0.42, hspace=0.38)
    figure.text(0.0121, 0.749, r"\textbf{Brownian}", rotation=90, ha="center", va="center", fontsize=7.0)
    figure.text(0.0121, 0.273, r"\textbf{Brownian--L\'evy}", rotation=90, ha="center", va="center", fontsize=7.0)
    for row in axes:
        for column, axis in enumerate(row):
            lower, upper = axis.get_ylim()
            span = upper - lower
            # Reserve space inside the axes so legends do not cover the curves.
            if column == 0:
                axis.set_ylim(lower - 0.42 * span, upper)
            elif column == 2:
                axis.set_ylim(lower, upper + 0.48 * span)
            axis.legend(
                loc="lower center" if column == 0 else "upper right",
                fontsize=6.7, handlelength=2.0, labelspacing=0.25, borderpad=0.3,
            )
    outputs = save_publication_figure(figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 7 workflow")
    plt.close(figure)
    return outputs
