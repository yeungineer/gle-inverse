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
    style_inset,
)


def _percent(value: float) -> str:
    return rf"{100.0 * float(value):.3f}\%"


def _plot_band(axis, x, lower, upper, *, color=COLORS["estimate_band"]):
    if np.allclose(lower, upper, rtol=0.0, atol=1.0e-12):
        return
    axis.fill_between(x, lower, upper, color=color, alpha=0.20, linewidth=0.0, zorder=1, label="IQR")


def _panel_brownian_trajectory(axis, data, metadata):
    axis.plot(
        data["brownian__trajectory_time"], data["brownian__trajectory_velocity"], color=COLORS["truth"], linewidth=0.75
    )
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$v(t)$")


def _panel_sigma(axis, data, metadata, prefix, *, levy=False):
    case = metadata["cases"][prefix]["representative"]
    grid = data[f"{prefix}__sigma_grid"]
    truth = data[f"{prefix}__sigma_truth"]
    _plot_band(axis, grid, data[f"{prefix}__sigma_q25"], data[f"{prefix}__sigma_q75"])
    axis.plot(grid, truth, color=COLORS["truth"], label="Truth", zorder=4)
    label = rf"Estimate ($E_\sigma={_percent(case['sigma_mu_relative_error'])}$)"
    axis.plot(
        grid, data[f"{prefix}__sigma_representative"], color=COLORS["estimate"], linestyle="--", label=label, zorder=5
    )
    if levy:
        axis.plot(
            grid,
            data[f"{prefix}__sigma_without_correction_representative"],
            color=COLORS["uncorrected"],
            linestyle=(0, (2.0, 1.25)),
            linewidth=0.9,
            label=(r"Uncorrected " rf"($E_\sigma={_percent(case['sigma_without_correction_relative_error'])}$)"),
            zorder=3,
        )
    values = np.concatenate(
        [np.asarray(truth, dtype=float), np.asarray(data[f"{prefix}__sigma_representative"], dtype=float)]
    )
    if levy:
        values = np.concatenate(
            [values, np.asarray(data[f"{prefix}__sigma_without_correction_representative"], dtype=float)]
        )
    value_range = float(np.ptp(values))
    minimum_margin = 0.06 * max(float(np.max(np.abs(truth))), 1.0e-8)
    margin = max(0.10 * value_range, minimum_margin)
    axis.set_ylim(float(np.min(values)) - margin, float(np.max(values)) + margin)
    axis.set_xlabel(r"$v$")
    axis.set_ylabel(r"$\sigma(v)$")
    axis.legend(loc="lower left")


def _panel_kernel(axis, data, metadata, prefix):
    case = metadata["cases"][prefix]["representative"]
    time = data[f"{prefix}__kernel_time"]
    _plot_band(axis, time, data[f"{prefix}__kernel_q25"], data[f"{prefix}__kernel_q75"])
    axis.plot(time, data[f"{prefix}__kernel_truth"], color=COLORS["truth"], label="Truth", zorder=4)
    axis.plot(
        time,
        data[f"{prefix}__kernel_representative"],
        color=COLORS["estimate"],
        linestyle="--",
        label=rf"Estimate ($E_\gamma={_percent(case['kernel_relative_error'])}$)",
        zorder=5,
    )
    axis.axhline(0.0, color=COLORS["neutral_light"], linewidth=0.65, linestyle="--", zorder=0)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\gamma(t)$")
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.legend(loc="upper right")

    inset = axis.inset_axes([0.48, 0.35, 0.47, 0.35])
    inset.plot(
        data[f"{prefix}__memory_time"],
        data[f"{prefix}__memory_estimate"],
        color=COLORS["estimate"],
        linestyle="--",
        linewidth=0.85,
    )
    inset.plot(
        data[f"{prefix}__memory_time"],
        data[f"{prefix}__memory_truth"],
        color=COLORS["truth"],
        linestyle="-",
        linewidth=0.60,
    )
    inset.set_xlabel(r"$t$")
    inset.set_ylabel(r"$-\mathcal{M}$")
    inset.set_title(rf"$E_\mathcal{{M}}={_percent(case['memory_force_relative_error'])}$", fontsize=6.8, pad=2.5)
    style_inset(inset)


def _panel_levy_trajectory(axis, data, metadata):
    case = metadata["cases"]["brownian_levy"]["representative"]
    axis.plot(
        data["brownian_levy__trajectory_time"],
        data["brownian_levy__trajectory_velocity"],
        color=COLORS["truth"],
        linewidth=0.72,
        zorder=2,
    )
    jump_time = data["brownian_levy__trajectory_jump_time"]
    jump_velocity = data["brownian_levy__trajectory_jump_velocity"]
    axis.scatter(
        jump_time,
        jump_velocity,
        s=13.0,
        facecolors="none",
        edgecolors=COLORS["levy"],
        linewidths=0.8,
        label=rf"Exceedances ({len(jump_time)})",
        zorder=4,
    )
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$v(t)$")
    axis.set_xlim(0.0, 200.0)
    axis.legend(loc="lower left")
    axis.text(
        0.03,
        0.96,
        (rf"$\widehat{{\alpha}}={case['alpha_est']:.3f}$, " rf"$\widehat{{\kappa}}={case['kappa_est']:.3f}$"),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
    )


def make_figure(
    data_path: str | Path, metadata_path: str | Path, output_dir: str | Path, *, stem: str = "Figure_1", dpi: int = 600
) -> dict[str, Any]:
    """Create Figure 1 without running any numerical inversion."""
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure, axes = plt.subplots(2, 3, figsize=full_width_figure_size(), squeeze=False)

    _panel_brownian_trajectory(axes[0, 0], data, metadata)
    _panel_sigma(axes[0, 1], data, metadata, "brownian", levy=False)
    axes[0, 1].set_ylim(0.40, 0.60)
    axes[0, 1].set_yticks([0.40, 0.45, 0.50, 0.55, 0.60])
    axes[0, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _panel_kernel(axes[0, 2], data, metadata, "brownian")
    _panel_levy_trajectory(axes[1, 0], data, metadata)
    _panel_sigma(axes[1, 1], data, metadata, "brownian_levy", levy=True)
    axes[1, 1].set_ylim(0.0, 1.15)
    axes[1, 1].set_yticks([0.30, 0.50, 0.70, 0.90, 1.10])
    axes[1, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    _panel_kernel(axes[1, 2], data, metadata, "brownian_levy")

    for label, axis in zip("abcdef", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)

    figure.subplots_adjust(left=0.072, right=0.985, bottom=0.105, top=0.940, wspace=0.42, hspace=0.30)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=dpi)
    plt.close(figure)
    return outputs
