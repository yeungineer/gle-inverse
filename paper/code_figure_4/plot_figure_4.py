"""Render the compact 1x3 Figure 4 from prepared data only."""

import warnings
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

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

ROUTE_STYLE = {
    "endpoint": {"line": COLORS["blue"], "fill": COLORS["blue_fill"], "label": "Final point"},
    "start_endpoint": {
        "line": COLORS["red"],
        "fill": COLORS["red_fill"],
        "label": "Initial and final points",
    },
    "full_path": {"line": COLORS["teal"], "fill": COLORS["teal_fill"], "label": "Full trajectory"},
}

CASE_STYLE = {
    "brownian": {"linestyle": "-", "marker": "o", "label": "Brownian"},
    "brownian_levy": {"linestyle": (0, (3.0, 1.8)), "marker": "s", "label": r"Brownian--L\'evy"},
}

METRIC_STYLE = {
    "alpha_error": {"linestyle": "-", "marker": "o", "label": r"$|\widehat\alpha-\alpha|$"},
    "kappa_log_error": {"linestyle": (0, (3.0, 1.8)), "marker": "s", "label": r"$|\log(\widehat\kappa/\kappa)|$"},
}


def _matrix(
    data: dict[str, np.ndarray], seeds: list[int], levels: np.ndarray, *, case: str, route: str, metric: str
) -> np.ndarray:
    matrix = np.full((len(seeds), len(levels)), np.nan, dtype=float)
    mask = (data["case"] == case) & (data["route"] == route) & np.isfinite(data[metric])
    for row in np.flatnonzero(mask):
        seed = int(data["seed"][row])
        if seed not in seeds:
            continue
        eta = float(data["eta_ratio"][row])
        matches = np.flatnonzero(np.isclose(levels, eta, rtol=0.0, atol=1.0e-14))
        if len(matches) != 1:
            raise RuntimeError(f"Could not place {case}/{route}/{metric}/eta={eta}.")
        i = seeds.index(seed)
        j = int(matches[0])
        if np.isfinite(matrix[i, j]):
            raise RuntimeError("Duplicate Figure 4 prepared value.")
        matrix[i, j] = float(data[metric][row])
    return matrix


def _plot_summary(
    axis, x: np.ndarray, matrix: np.ndarray, *, line: str, fill: str, linestyle, marker: str
) -> bool:
    if not np.any(np.isfinite(matrix)):
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q25 = np.nanquantile(matrix, 0.25, axis=0)
        median = np.nanmedian(matrix, axis=0)
        q75 = np.nanquantile(matrix, 0.75, axis=0)
    axis.fill_between(x, q25, q75, color=fill, alpha=IQR_ALPHA, linewidth=0.0, zorder=2)
    axis.plot(
        x,
        median,
        color=line,
        linestyle=linestyle,
        linewidth=1.25,
        marker=marker,
        markerfacecolor="white",
        markeredgecolor=line,
        markeredgewidth=0.72,
        zorder=4,
    )
    counts = np.sum(np.isfinite(matrix), axis=0)
    incomplete = counts < matrix.shape[0]
    mark = incomplete & np.isfinite(median)
    if np.any(mark):
        axis.scatter(
            x[mark], median[mark], marker="x", s=13.0, linewidths=0.75, color=COLORS["neutral"], zorder=6
        )
    return bool(np.any(incomplete))


def _format_axis(axis, x: np.ndarray, ylabel: str) -> None:
    positive = np.sort(np.unique(x[x > 0.0]))
    if positive.size == 0:
        raise ValueError("Figure 4 requires at least one positive noise level.")
    linear_threshold = float(positive[0])
    axis.set_xscale("symlog", base=10, linthresh=linear_threshold, linscale=0.8)
    axis.set_xticks(x)
    use_fixed_exponent(axis, -2, axis_name="x")
    axis.tick_params(axis="x", which="major", labelsize=6.9, pad=2.0)
    xmax = float(np.max(x))
    axis.set_xlim(-0.22 * linear_threshold, 1.18 * xmax)
    axis.set_yscale("log")
    axis.set_yticks([1.0e-2, 1.0e-1, 1.0e0])
    axis.yaxis.set_minor_formatter(mticker.NullFormatter())
    axis.set_xlabel(r"$\sigma_{\rm obs}$")
    axis.set_ylabel(ylabel)


def _route_handles(routes):
    return [
        Line2D([0], [0], color=ROUTE_STYLE[route]["line"], linewidth=1.4, label=ROUTE_STYLE[route]["label"])
        for route in routes
    ]


def _incomplete_handle(required_successes: int):
    return Line2D(
        [0],
        [0],
        color=COLORS["neutral"],
        marker="x",
        linestyle="none",
        markersize=4.0,
        label=rf"$n_{{\rm success}}<{int(required_successes)}$",
    )


def _panel_case_metric(axis, data, metadata, *, metric: str, ylabel: str) -> None:
    seeds = list(map(int, metadata["seeds"]))
    levels = np.asarray(metadata["scientific_configuration"]["noise_levels"], dtype=float)
    routes = tuple(metadata["scientific_configuration"]["routes"])
    x = levels
    incomplete = False
    for route in routes:
        for case in ("brownian", "brownian_levy"):
            matrix = _matrix(data, seeds, levels, case=case, route=route, metric=metric)
            incomplete |= _plot_summary(
                axis,
                x,
                matrix,
                line=ROUTE_STYLE[route]["line"],
                fill=ROUTE_STYLE[route]["fill"],
                linestyle=CASE_STYLE[case]["linestyle"],
                marker=CASE_STYLE[case]["marker"],
            )
    _format_axis(axis, x, ylabel)
    case_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["neutral"],
            linestyle=CASE_STYLE[case]["linestyle"],
            marker=CASE_STYLE[case]["marker"],
            markerfacecolor="white",
            markersize=3.5,
            label=CASE_STYLE[case]["label"],
        )
        for case in ("brownian", "brownian_levy")
    ]
    if incomplete:
        case_handles.append(_incomplete_handle(metadata["required_successes"]))
    axis.legend(handles=case_handles, loc="upper left", fontsize=7.0)


def _panel_levy(axis, data, metadata) -> None:
    seeds = list(map(int, metadata["seeds"]))
    levels = np.asarray(metadata["scientific_configuration"]["noise_levels"], dtype=float)
    routes = tuple(metadata["scientific_configuration"]["routes"])
    x = levels
    incomplete = False
    for route in routes:
        for metric in ("alpha_error", "kappa_log_error"):
            matrix = _matrix(data, seeds, levels, case="brownian_levy", route=route, metric=metric)
            incomplete |= _plot_summary(
                axis,
                x,
                matrix,
                line=ROUTE_STYLE[route]["line"],
                fill=ROUTE_STYLE[route]["fill"],
                linestyle=METRIC_STYLE[metric]["linestyle"],
                marker=METRIC_STYLE[metric]["marker"],
            )
    _format_axis(axis, x, r"L\'evy parameter error")
    metric_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["neutral"],
            linestyle=METRIC_STYLE[metric]["linestyle"],
            marker=METRIC_STYLE[metric]["marker"],
            markerfacecolor="white",
            markersize=3.5,
            label=METRIC_STYLE[metric]["label"],
        )
        for metric in ("alpha_error", "kappa_log_error")
    ]
    if incomplete:
        metric_handles.append(_incomplete_handle(metadata["required_successes"]))
    axis.legend(handles=metric_handles, loc="upper left", fontsize=7.0)


def make_figure(
    data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str = "Figure_4", dpi: int = 600
) -> dict[str, Any]:
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure, axes = plt.subplots(1, 3, figsize=full_width_single_row_figure_size(), squeeze=False)
    _panel_case_metric(axes[0, 0], data, metadata, metric="sigma_error", ylabel=r"$E_\sigma$")
    _panel_case_metric(axes[0, 1], data, metadata, metric="kernel_error", ylabel=r"$E_\gamma$")
    _panel_levy(axes[0, 2], data, metadata)

    for label, axis in zip("abc", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    figure.subplots_adjust(left=0.070, right=0.985, bottom=0.205, top=0.92, wspace=0.34)
    routes = metadata["scientific_configuration"]["routes"]
    for axis in axes.ravel():
        lower, upper = axis.get_ylim()
        axis.set_ylim(lower / 5.0, upper)
        axis.add_artist(axis.get_legend())
        axis.legend(handles=_route_handles(routes), loc="lower right", fontsize=6.5)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=dpi, creator="GLE inverse Figure 4 workflow")
    plt.close(figure)
    return outputs
