"""Render the three-panel Figure 3 from prepared data only."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from paper.utils.figures import (
    COLORS,
    IQR_ALPHA,
    add_panel_label,
    apply_publication_style,
    full_width_single_row_figure_size,
    load_prepared_data,
    save_publication_figure,
    style_axis,
    use_fixed_exponent,
)

PALETTE = {
    "blue": (COLORS["blue"], COLORS["blue_fill"]),
    "red": (COLORS["red"], COLORS["red_fill"]),
    "teal": (COLORS["teal"], COLORS["teal_fill"]),
}


def _matching(data: dict[str, np.ndarray], experiment: str, *, L: int | None = None) -> np.ndarray:
    mask = (data["experiment"] == experiment) & data["included_in_plot"].astype(bool) & data["success"].astype(bool)
    if L is not None:
        mask &= data["L"] == int(L)
    return mask


def _series_matrix(
    data: dict[str, np.ndarray],
    metadata: dict[str, Any],
    *,
    experiment: str,
    x_field: str,
    x_values: list[float] | list[int],
    metric: str,
    L: int | None = None,
) -> np.ndarray:
    seeds = list(map(int, metadata["plot_seeds"]))
    x_numeric = np.asarray(x_values, dtype=float)
    matrix = np.full((len(seeds), len(x_numeric)), np.nan, dtype=float)
    for row_index in np.flatnonzero(_matching(data, experiment, L=L)):
        seed = int(data["seed"][row_index])
        if seed not in seeds:
            continue
        x = float(data[x_field][row_index])
        candidates = np.flatnonzero(np.isclose(x_numeric, x, rtol=0.0, atol=1.0e-12))
        if len(candidates) != 1:
            raise RuntimeError(f"Could not uniquely place {experiment} {x_field}={x}.")
        i = seeds.index(seed)
        j = int(candidates[0])
        if np.isfinite(matrix[i, j]):
            raise RuntimeError(f"Duplicate prepared value for {experiment}, seed={seed}.")
        matrix[i, j] = float(data[metric][row_index])
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Figure 3 has an incomplete {experiment}/{metric} seed matrix.")
    return matrix


def _plot_seed_summary(
    axis,
    x: np.ndarray,
    matrix: np.ndarray,
    *,
    color: str,
    fill: str,
    label: str,
    marker: str,
    iqr_alpha: float,
    linestyle: str = "-",
) -> np.ndarray:
    q25 = np.quantile(matrix, 0.25, axis=0)
    median = np.median(matrix, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    axis.fill_between(x, q25, q75, color=fill, alpha=iqr_alpha, linewidth=0.0, zorder=2)
    axis.plot(
        x,
        median,
        color=color,
        linewidth=1.35,
        linestyle=linestyle,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=color,
        markeredgewidth=0.75,
        label=label,
        zorder=3,
    )
    return median


def _linear_x(axis, ticks: np.ndarray) -> None:
    axis.set_xticks(ticks)
    margin = 0.04 * float(ticks[-1] - ticks[0])
    axis.set_xlim(float(ticks[0]) - margin, float(ticks[-1]) + margin)


def _panel_sampling(axis, data, metadata, settings) -> None:
    x = np.asarray(settings["sampling_M"]["values"], dtype=float)
    d1 = _series_matrix(
        data, metadata, experiment="sampling_M_theory", x_field="M", x_values=x.tolist(), metric="d1_normalized_rmse"
    )
    d2 = _series_matrix(
        data, metadata, experiment="sampling_M_theory", x_field="M", x_values=x.tolist(), metric="d2_normalized_rmse"
    )
    median = _plot_seed_summary(
        axis,
        x,
        d1,
        color=PALETTE["blue"][0],
        fill=PALETTE["blue"][1],
        label=r"$\widehat D^{(1)}$",
        marker="o",
        iqr_alpha=IQR_ALPHA,
    )
    _plot_seed_summary(
        axis,
        x,
        d2,
        color=PALETTE["red"][0],
        fill=PALETTE["red"][1],
        label=r"$\widehat D^{(2)}$",
        marker="s",
        linestyle=(0, (4.0, 1.4, 1.0, 1.4)),
        iqr_alpha=IQR_ALPHA,
    )
    guide = median[2] * (x / x[2]) ** (-0.5)
    axis.plot(x, guide, color=COLORS["truth"], linewidth=0.90, linestyle=(0, (3.0, 2.0)), label=r"$M^{-1/2}$", zorder=5)
    _linear_x(axis, x)
    axis.set_xticklabels([rf"${int(value)}$" for value in x])
    axis.set_yticks([1.0e-2, 2.0e-2, 3.0e-2])
    axis.set_ylim(9.5e-3, 3.8e-2)
    use_fixed_exponent(axis, -2)
    axis.set_xlabel(r"$M$")
    axis.set_ylabel(r"NRMSE")
    axis.legend(loc="upper center", ncol=3, fontsize=7.0, columnspacing=0.85, handletextpad=0.35)


def _panel_kernel(axis, data, metadata, settings) -> None:
    """Show how the conditional sampling law reaches the recovered kernel."""
    x = np.asarray(settings["sampling_M"]["values"], dtype=float)
    matrix = _series_matrix(
        data, metadata, experiment="kernel_M", x_field="M", x_values=x.tolist(), metric="kernel_error"
    )
    median = _plot_seed_summary(
        axis,
        x,
        matrix,
        color=PALETTE["blue"][0],
        fill=PALETTE["blue"][1],
        label=r"$E_\gamma$",
        marker="o",
        iqr_alpha=IQR_ALPHA,
    )
    guide = median[2] * (x / x[2]) ** (-0.5)
    axis.plot(x, guide, color=COLORS["truth"], linewidth=0.90, linestyle=(0, (3.0, 2.0)), label=r"$M^{-1/2}$", zorder=5)
    _linear_x(axis, x)
    axis.set_xticklabels([rf"${int(value)}$" for value in x])
    axis.set_yticks([0.0, 4.0e-2, 8.0e-2, 1.2e-1])
    axis.set_ylim(0.0, 1.4e-1)
    use_fixed_exponent(axis, -2)
    axis.set_xlabel(r"$M$")
    axis.set_ylabel(r"$E_\gamma$")
    axis.legend(loc="upper center", ncol=2, fontsize=7.0, columnspacing=0.85, handletextpad=0.35)


def _panel_fixed_L(axis, data, metadata, settings) -> None:
    fixed_L_settings = settings["fixed_L"]
    curves = list(fixed_L_settings["curves"])
    display_L_values = set(map(int, metadata["panels"]["c"].get("display_L_values", [])))
    if display_L_values:
        curves = [curve for curve in curves if int(curve["L"]) in display_L_values]
    styles = [
        (PALETTE["blue"][0], PALETTE["blue"][1], "o"),
        (PALETTE["red"][0], PALETTE["red"][1], "D"),
        (PALETTE["teal"][0], PALETTE["teal"][1], "^"),
    ]
    all_M: list[float] = []
    for curve, (color, fill, marker) in zip(curves, styles):
        L = int(curve["L"])
        x = np.asarray(curve["M_values"], dtype=float)
        all_M.extend(x.tolist())
        matrix = _series_matrix(
            data, metadata, experiment="fixed_L_brownian", x_field="M", x_values=x.tolist(), metric="kernel_error", L=L
        )
        _plot_seed_summary(
            axis, x, matrix, color=color, fill=fill, label=rf"$L={L}$", marker=marker, iqr_alpha=IQR_ALPHA
        )
    labelled_M = np.asarray(curves[0]["M_values"], dtype=float)
    _linear_x(axis, labelled_M)
    axis.set_xticklabels([rf"${int(M)}$" for M in labelled_M], fontsize=7.0)
    margin = 0.04 * (max(all_M) - min(all_M))
    axis.set_xlim(min(all_M) - margin, max(all_M) + margin)
    axis.set_yscale("log")
    axis.set_xlabel(r"$M$")
    axis.set_ylabel(r"$E_\gamma$")
    axis.legend(loc="upper center", ncol=len(curves), fontsize=6.3, columnspacing=0.55, handlelength=1.0, handletextpad=0.3)


def make_figure(
    data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str = "Figure_3", dpi: int = 600
) -> dict[str, Any]:
    """Create Figure 3 without running data generation or inversion."""
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    settings = metadata["scientific_configuration"]
    figure, axes = plt.subplots(1, 3, figsize=full_width_single_row_figure_size(), squeeze=False)
    panels = axes[0]
    _panel_sampling(panels[0], data, metadata, settings)
    _panel_kernel(panels[1], data, metadata, settings)
    _panel_fixed_L(panels[2], data, metadata, settings)
    for label, axis in zip("abc", panels):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    figure.subplots_adjust(left=0.073, right=0.992, bottom=0.22, top=0.92, wspace=0.35)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=dpi, creator="GLE inverse Figure 3 workflow")
    plt.close(figure)
    return outputs
