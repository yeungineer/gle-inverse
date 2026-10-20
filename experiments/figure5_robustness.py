"""Figure 5: robustness."""
from copy import deepcopy

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch, Rectangle
from scipy.signal import fftconvolve

from model.fit import (
    configured_case, evaluate_sigma, fit_sigma, float_column, predict_sigma, prepare, run_point, save_figure_data, string_column,
)
from model.inverse import effective_rank
from model.simulation import generate_trajectory
from model.style import (
    COLORS, ROW_FIGURE_SIZE, SEQUENTIAL_HEATMAP_COLORS, add_panel_label, apply_publication_style, load_prepared_data,
    save_publication_figure, style_axis,
)

PARAMS = {
    "figure": {"stem": "Figure_5", "dpi": 600, "seeds": [10, 11, 12, 13, 14, 15]},
    "experiment": {
        "km_samples": "euler_oracle",
        "diffusion_type": "cosine",
        "dt": 0.01,
        "N": 12,
        "test_seed_offset": 70000000,
        "external_force": {"force_types": ["zero", "sin", "bistable"], "M": 2000, "L": 3000, "test_L": 6000},
        "phase_map": {
            "force_type": "zero",
            "joint_error_unacceptable_threshold": 0.5,
            "sigma_error_unacceptable_threshold": 0.5,
            "alpha_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            "kappa_values": [0.04, 0.08, 0.12, 0.18, 0.24],
            "M": 2000,
            "L": 3000,
            "seed": 42,
        },
    },
}


def _parameters(settings, seed, M, L, force_type, *, alpha=None, kappa=None):
    values = {
        "generation.simulation.seed": int(seed),
        "generation.simulation.dt": float(settings["dt"]),
        "generation.simulation.num_steps": int(M),
        "generation.simulation.L": int(L),
        "generation.km_samples": str(settings["km_samples"]),
        "noise.diffusion": str(settings["diffusion_type"]),
        "force.type": str(force_type),
        "phase2.estimation.num_basis": int(settings["N"]),
        "phase2.estimation.t_max": 30.0,
    }
    if alpha is not None:
        values["noise.alpha"] = float(alpha)
        values["noise.kappa"] = float(kappa)
    return values


def _levy_metrics(problem):
    estimate = problem["km_samples"]["levy"]
    alpha_true, kappa_true = (problem["driver_config"]["noise"]["alpha"], problem["driver_config"]["noise"]["kappa"])
    alpha_est = float(estimate["alpha"])
    kappa_est = float(estimate["kappa"])
    alpha_error = abs(alpha_est - alpha_true)
    kappa_error = abs(float(np.log(kappa_est / kappa_true)))
    return {
        "alpha_true": float(alpha_true),
        "kappa_true": float(kappa_true),
        "alpha_est": alpha_est,
        "kappa_est": kappa_est,
        "joint_levy_error": float(np.hypot(alpha_error, kappa_error)),
    }


def _sigma_error(problem):
    config = deepcopy(problem["driver_config"])
    config["estimation"] = deepcopy(problem["config"]["phase1"]["estimation"])
    samples = problem["phase1_km_samples"]
    net, _ = fit_sigma(samples["x"], samples["d2"], config)
    sigma_est = predict_sigma(net, samples["x"])
    _, relative_error, _ = evaluate_sigma(config, samples["x"], sigma_est)
    return float(relative_error)


def _independent_test(problem, seed, L):
    driver = deepcopy(problem["driver_config"])
    driver["simulation"]["seed"] = int(seed)
    driver["simulation"]["L"] = int(L)
    t, v, _, _, memory, _, _ = generate_trajectory(driver, 0)
    return (np.asarray(t, dtype=float), np.asarray(v, dtype=float), np.asarray(memory, dtype=float))


def _external_rows(config):
    settings = config["experiment"]
    experiment = settings["external_force"]
    rows = []
    seeds = list(map(int, config["figure"]["seeds"]))
    forces = list(experiment["force_types"])
    for seed in seeds:
        for force_type in forces:
            problem, arrays, metrics = run_point("brownian_levy", _parameters(settings, seed, experiment["M"], experiment["L"], force_type))
            t, velocity, memory = _independent_test(problem, seed + int(settings["test_seed_offset"]), int(experiment["test_L"]))
            kernel = np.interp(
                t - t[0], np.asarray(arrays["t"], dtype=float), np.asarray(arrays["kernel_est"], dtype=float),
                left=float(arrays["kernel_est"][0]), right=0.0,
            )
            convolution = fftconvolve(velocity, kernel, mode="full")[: len(velocity)]
            # Held-out evaluation uses trapezoids; simulation uses left rectangles.
            convolution -= 0.5 * (velocity[0] * kernel + velocity * kernel[0])
            memory_est = -float(t[1] - t[0]) * convolution
            memory_true = -memory
            mask = t >= 30.0
            row = {
                "panel": "external",
                "seed": seed,
                "force_type": str(force_type),
                **_levy_metrics(problem),
                "sigma_error": float(metrics["sigma"]["relative_l2_error"]),
                "kernel_error": float(metrics["kernel"]["relative_l2_error"]),
                "memory_error": float(np.linalg.norm(memory_est[mask] - memory_true[mask]) / np.linalg.norm(memory_true[mask])),
                "selected_lambda": float(metrics["regularization"]["selected_lambda"]),
                "effective_rank": effective_rank(arrays["regularization_singular_values"]),
            }
            rows.append(row)
    return rows


def _phase_rows(config):
    settings = config["experiment"]
    phase = settings["phase_map"]
    seed = int(phase["seed"])
    points = [(float(alpha), float(kappa)) for alpha in phase["alpha_values"] for kappa in phase["kappa_values"]]
    rows = []
    for alpha, kappa in points:
        problem = prepare(
            configured_case(
                "brownian_levy", _parameters(settings, seed, phase["M"], phase["L"], phase["force_type"], alpha=alpha, kappa=kappa)
            )
        )
        rows.append(
            {
                "panel": "phase",
                "seed": seed,
                "force_type": str(phase["force_type"]),
                **_levy_metrics(problem),
                "sigma_error": _sigma_error(problem),
                "kernel_error": None,
                "memory_error": None,
                "selected_lambda": None,
                "effective_rank": None,
            }
        )
    return rows


def run(config, out):
    rows = _external_rows(config) + _phase_rows(config)
    arrays, metadata = pack_rows(config, rows)
    save_figure_data(out, 5, arrays, metadata)
    return rows


def pack_rows(config, rows):
    rows.sort(key=lambda row: (row["panel"], row["force_type"], row["alpha_true"], row["kappa_true"], row["seed"]))
    arrays = {
        "panel": string_column(rows, "panel"),
        "force_type": string_column(rows, "force_type"),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
    }
    for field in (
        "alpha_true",
        "kappa_true",
        "alpha_est",
        "kappa_est",
        "joint_levy_error",
        "sigma_error",
        "kernel_error",
        "memory_error",
        "selected_lambda",
        "effective_rank",
    ):
        arrays[field] = float_column(rows, field)
    arrays["tail_success"] = np.ones(len(rows), dtype=bool)
    arrays["sigma_success"] = np.ones(len(rows), dtype=bool)
    metadata = {
        "external_plot_seeds": list(map(int, config["figure"]["seeds"])),
        "phase_report_seed": int(config["experiment"]["phase_map"]["seed"]),
        "experiment": config["experiment"],
    }
    return (arrays, metadata)


FORCE_STYLE = {"zero": {"label": "Zero"}, "sin": {"label": "Sinusoidal"}, "bistable": {"label": "Bistable"}}


def f5_external_matrix(data, metadata, force_types, metric):
    seeds = list(map(int, metadata["external_plot_seeds"]))
    matrix = np.full((len(seeds), len(force_types)), np.nan)
    for column, force in enumerate(force_types):
        mask = (data["panel"] == "external") & (data["force_type"] == force) & np.isfinite(data[metric])
        for row in np.flatnonzero(mask):
            seed = int(data["seed"][row])
            if seed in seeds:
                matrix[seeds.index(seed), column] = float(data[metric][row])
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Figure 5(a) lacks one or more {metric} values for its declared {len(seeds)} complete seeds.")
    return matrix


def f5_cell_edges(values):
    values = np.asarray(values, dtype=float)
    middle = 0.5 * (values[:-1] + values[1:])
    return np.concatenate(([values[0] - (middle[0] - values[0])], middle, [values[-1] + (values[-1] - middle[-1])]))


def f5_phase_matrix(data, metadata, metric):
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
    return (alpha, kappa, error)


def f5_phase_heatmap(
    axis, colorbar_axis, figure, data, metadata, *, metric, colorbar_title, colors, logarithmic=False, unacceptable_threshold=None
):
    alpha, kappa, error = f5_phase_matrix(data, metadata, metric)
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
        f5_cell_edges(alpha), f5_cell_edges(kappa), np.ma.masked_invalid(plot_error), shading="flat", cmap=color_map, norm=normalization
    )
    alpha_edges = f5_cell_edges(alpha)
    kappa_edges = f5_cell_edges(kappa)
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
    axis.set_xlabel("$\\alpha$")
    axis.set_ylabel("$\\kappa$")
    axis.set_xticks(alpha)
    axis.set_yticks(kappa)
    axis.tick_params(axis="x", labelsize=6.8, pad=1.8)
    axis.tick_params(axis="y", labelsize=6.9, pad=1.8)
    axis.tick_params(axis="both", which="both", length=0)
    colorbar = figure.colorbar(image, cax=colorbar_axis, extend="max" if unacceptable_threshold is not None else "neither")
    colorbar.ax.set_title(colorbar_title, pad=2.0, fontsize=7.0)
    colorbar.ax.tick_params(direction="in", length=2.4, width=0.65, labelsize=6.7)


def plot(data_path, metadata_path, output_dir, *, stem, dpi):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure = plt.figure(figsize=ROW_FIGURE_SIZE)
    outer = figure.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.0), wspace=0.5, left=0.06, right=0.985, bottom=0.285, top=0.925)
    heatmap_b = outer[0, 1].subgridspec(1, 2, width_ratios=(1.0, 0.06), wspace=0.1)
    heatmap_c = outer[0, 2].subgridspec(1, 2, width_ratios=(1.0, 0.06), wspace=0.1)
    axes = [figure.add_subplot(outer[0, 0]), figure.add_subplot(heatmap_b[0, 0]), figure.add_subplot(heatmap_c[0, 0])]
    colorbar_axes = [figure.add_subplot(heatmap_b[0, 1]), figure.add_subplot(heatmap_c[0, 1])]
    force_types = list(metadata["experiment"]["external_force"]["force_types"])
    x = np.arange(len(force_types), dtype=float)
    series = (
        ("sigma_error", COLORS["blue"], "o", "$E_\\sigma$", -0.12),
        ("kernel_error", COLORS["teal"], "s", "$E_\\gamma$", 0.0),
        ("memory_error", COLORS["red"], "^", "$E_{\\mathcal M}$", 0.12),
    )
    for metric, color, marker, label, offset in series:
        matrix = f5_external_matrix(data, metadata, force_types, metric)
        q25 = np.quantile(matrix, 0.25, axis=0)
        median = np.median(matrix, axis=0)
        q75 = np.quantile(matrix, 0.75, axis=0)
        axes[0].errorbar(
            x + offset, median, yerr=[median - q25, q75 - median], fmt="none", ecolor=color, elinewidth=1.1, capsize=2.5, capthick=1.0,
            zorder=3,
        )
        axes[0].plot(
            x + offset, median, color=color, marker=marker, markerfacecolor="white", markeredgewidth=0.75, linestyle="none", label=label,
            zorder=4,
        )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([FORCE_STYLE[name]["label"] for name in force_types], rotation=0, ha="center")
    axes[0].set_yscale("log")
    ticks = [0.02, 0.04, 0.06, 0.08, 0.1]
    axes[0].set_yticks(ticks)
    axes[0].set_yticklabels([f"{value:.2f}" for value in ticks])
    axes[0].set_ylabel("Error")
    lower, upper = axes[0].get_ylim()
    axes[0].set_ylim(lower, max(upper, 0.115))
    axes[0].set_xlim(-0.45, len(force_types) - 0.55)
    axes[0].legend(loc="upper center", ncol=3, columnspacing=0.85, handletextpad=0.35)
    f5_phase_heatmap(
        axes[1], colorbar_axes[0], figure, data, metadata, metric="joint_levy_error", colorbar_title="$E_{\\alpha,\\kappa}$",
        colors=SEQUENTIAL_HEATMAP_COLORS,
        unacceptable_threshold=metadata["experiment"]["phase_map"].get("joint_error_unacceptable_threshold", 0.5),
    )
    f5_phase_heatmap(
        axes[2], colorbar_axes[1], figure, data, metadata, metric="sigma_error", colorbar_title="$E_{\\sigma}$",
        colors=SEQUENTIAL_HEATMAP_COLORS, logarithmic=True,
        unacceptable_threshold=metadata["experiment"]["phase_map"].get("sigma_error_unacceptable_threshold", 0.5),
    )
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    unacceptable_threshold = float(metadata["experiment"]["phase_map"].get("joint_error_unacceptable_threshold", 0.5))
    unacceptable = Patch(
        facecolor=SEQUENTIAL_HEATMAP_COLORS[-1],
        edgecolor="#555555",
        linewidth=0.45,
        hatch="xxxx",
        label=f"Unacceptable ($E_{{\\alpha,\\kappa}}\\geq {unacceptable_threshold:g}$)",
    )
    axes[1].legend(
        handles=[unacceptable], loc="lower right", bbox_to_anchor=(1.0, 1.005), fontsize=6.5, frameon=True, fancybox=False, shadow=False,
        facecolor="white", edgecolor="none", framealpha=0.3, handlelength=1.0, handletextpad=0.4, handleheight=0.85, borderaxespad=0.0,
    )
    sigma_unacceptable_threshold = float(metadata["experiment"]["phase_map"].get("sigma_error_unacceptable_threshold", 0.5))
    sigma_unacceptable = Patch(
        facecolor=SEQUENTIAL_HEATMAP_COLORS[-1],
        edgecolor="#555555",
        linewidth=0.45,
        hatch="xxxx",
        label=f"Unacceptable ($E_{{\\sigma}}\\geq {sigma_unacceptable_threshold:g}$)",
    )
    axes[2].legend(
        handles=[sigma_unacceptable], loc="lower right", bbox_to_anchor=(1.0, 1.005), fontsize=6.5, frameon=True, fancybox=False,
        shadow=False, facecolor="white", edgecolor="none", framealpha=0.3, handlelength=1.0, handletextpad=0.4, handleheight=0.85,
        borderaxespad=0.0,
    )
    outputs = save_publication_figure(figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 5 workflow")
    plt.close(figure)
    return outputs
