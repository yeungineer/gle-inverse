"""Figure 7: preparation."""

from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from experiments.figure1_reconstruction import _interpolate, _memory_force, _relative_l2
from model.fit import configured_case, prepare, run_point, save_figure_data, solve
from model.simulation import conditional_moments
from model.style import (
    COLORS, FULL_FIGURE_SIZE, add_panel_label, apply_publication_style, load_prepared_data, save_publication_figure, style_axis,
)

PARAMS = {
    "figure": {"stem": "Figure_7", "dpi": 600, "seed": 42, "state_quantiles": [0.01, 0.99], "memory_time_window": [2.0, 16.0]},
    "experiment": {
        "dt": 0.01,
        "L": 3000,
        "preparation_time": 5.0,
        "M": 2000,
        "N": 12,
        "cases": {"brownian": {"diffusion": "cosine"}, "brownian_levy": {"diffusion": "cosine"}},
        "protocols": ["euler_oracle", "fine_branch", "replay_branch"],
    },
}


def _evaluation_bank(problem, preparation_steps, evaluation_L):
    raw = problem["km_samples"]
    indices = np.asarray(raw["idx"], dtype=int)
    keep = (indices > preparation_steps) & (indices < preparation_steps + evaluation_L)
    noise = problem["driver_config"]["noise"]
    context = {
        "dt": float(raw["dt"]),
        "reference_states": np.asarray(raw["x"])[keep],
        "force_values": np.asarray(raw["f"])[keep],
        "reference_indices": indices[keep],
        "noise_model": noise,
        "stable_parameters": (noise["alpha"], noise["kappa"]) if noise["type"] == "brownian_levy" else None,
    }
    generation = {
        "method": str(raw["generation_method"]),
        "substeps": int(raw["generation_substeps"]),
        "internal_dt": float(raw["generation_internal_dt"]),
    }
    return conditional_moments(
        context, np.asarray(raw["delta_v"])[keep], np.asarray(raw["mem"])[keep], generation,
        levy_estimation=problem["config"]["phase1"]["estimation"], levy_estimate=None, branch_start=np.asarray(raw["branch_start"])[keep],
    )


def run_prepared_window_point(case, parameters, evaluation_L, preparation_time):
    parameters = deepcopy(dict(parameters))
    dt = float(parameters["generation.simulation.dt"])
    preparation_steps = int(round(float(preparation_time) / dt))
    parameters["generation.simulation.L"] = preparation_steps + int(evaluation_L)
    config = configured_case(case, parameters)
    problem = prepare(config, conditional_sample_seed=None)
    bank = _evaluation_bank(problem, preparation_steps, int(evaluation_L))
    problem["km_samples"] = bank
    problem["phase1_km_samples"] = bank
    arrays, metrics = solve(problem)
    return (problem, arrays, metrics)


def _parameters(settings, case_settings, protocol, seed):
    return {
        "generation.simulation.seed": seed,
        "generation.simulation.dt": float(settings["dt"]),
        "generation.simulation.L": int(settings["L"]),
        "generation.simulation.num_steps": int(settings["M"]),
        "generation.km_samples": protocol,
        "noise.diffusion": str(case_settings["diffusion"]),
        "phase2.estimation.num_basis": int(settings["N"]),
        "phase2.estimation.t_max": min(30.0, settings["L"] * settings["dt"]),
    }


def run(config, out):
    settings = config["experiment"]
    seed = int(config["figure"]["seed"])
    memory_window = tuple(map(float, config["figure"]["memory_time_window"]))
    protocols = list(settings["protocols"])
    runs = {}
    errors = {}
    for case, case_settings in settings["cases"].items():
        errors[case] = {}
        for protocol in protocols:
            parameters = _parameters(settings, case_settings, protocol, seed)
            if protocol == "replay_branch":
                problem, arrays, metrics = run_prepared_window_point(
                    case, parameters, int(settings["L"]), float(settings["preparation_time"])
                )
            else:
                problem, arrays, metrics = run_point(case, parameters)
            runs[case, protocol] = (problem, arrays)
            errors[case][protocol] = {
                "sigma": float(metrics["sigma"]["relative_l2_error"]),
                "gamma": float(metrics["kernel"]["relative_l2_error"]),
            }
    q0, q1 = map(float, config["figure"]["state_quantiles"])
    curves = {}
    for case in settings["cases"]:
        problem, baseline = runs[case, protocols[0]]
        baseline_state = np.asarray(baseline["x_ref"], dtype=float)
        state_grid = np.linspace(np.quantile(baseline_state, q0), np.quantile(baseline_state, q1), 401)
        curves[f"{case}__sigma_x"] = state_grid
        curves[f"{case}__sigma_truth"] = _interpolate(baseline["x_ref"], baseline["sigma_true"], state_grid)
        kernel_time = np.asarray(baseline["t"], dtype=float)
        kernel_mask = kernel_time <= float(problem["config"]["phase2"]["estimation"]["t_max"])
        curves[f"{case}__kernel_t"] = kernel_time[kernel_mask]
        curves[f"{case}__kernel_truth"] = np.asarray(baseline["kernel_true"], dtype=float)[kernel_mask]
        inference_t = np.asarray(problem["t_inference"], dtype=float)
        memory_truth = -np.asarray(problem["memory_inference"], dtype=float)
        memory_mask = (inference_t >= memory_window[0]) & (inference_t <= memory_window[1])
        curves[f"{case}__memory_t"] = inference_t[memory_mask]
        curves[f"{case}__memory_truth"] = memory_truth[memory_mask]
        for protocol in protocols:
            _, arrays = runs[case, protocol]
            curves[f"{case}__sigma__{protocol}"] = _interpolate(arrays["x_ref"], arrays["sigma_est"], state_grid)
            estimate_time = np.asarray(arrays["t"], dtype=float)
            estimate_mask = estimate_time <= float(problem["config"]["phase2"]["estimation"]["t_max"])
            kernel_estimate = np.interp(
                kernel_time[kernel_mask], estimate_time[estimate_mask], np.asarray(arrays["kernel_est"], dtype=float)[estimate_mask]
            )
            curves[f"{case}__kernel__{protocol}"] = kernel_estimate
            memory_estimate = _memory_force(problem["v_inference"], kernel_estimate, float(inference_t[1] - inference_t[0]))
            curves[f"{case}__memory__{protocol}"] = memory_estimate[memory_mask]
            errors[case][protocol]["memory"] = _relative_l2(memory_estimate, memory_truth, inference_t)
    save_figure_data(out, 7, curves, {"seed": seed, "protocols": protocols, "errors": errors})
    return curves


PROTOCOL_STYLES = {
    "euler_oracle": (COLORS["blue"], "--", "Matched history"),
    "fine_branch": (COLORS["red"], "-.", "Short-lag release"),
    "replay_branch": (COLORS["teal"], (0, (2.0, 1.2)), "Feedback-prepared"),
}


def f7_set_y_limits(axis, curves):
    values = np.concatenate([np.asarray(curve, dtype=float) for curve in curves])
    margin = 0.08 * max(float(np.ptp(values)), float(np.max(np.abs(values))) * 0.1)
    axis.set_ylim(float(np.min(values)) - margin, float(np.max(values)) + margin)
    axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))


def f7_plot_sigma(axis, data, protocols, case):
    x = data[f"{case}__sigma_x"]
    truth = data[f"{case}__sigma_truth"]
    curves = [truth]
    axis.plot(x, truth, color=COLORS["truth"], linewidth=1.15, label="Truth", zorder=1)
    for protocol in protocols:
        estimate = data[f"{case}__sigma__{protocol}"]
        color, linestyle, label = PROTOCOL_STYLES[protocol]
        axis.plot(x, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label, zorder=3)
        curves.append(estimate)
    f7_set_y_limits(axis, curves)
    axis.set_xlabel("$v$")
    axis.set_ylabel("$\\sigma(v)$")


def f7_plot_kernel(axis, data, protocols, case):
    time = data[f"{case}__kernel_t"]
    truth = data[f"{case}__kernel_truth"]
    curves = [truth]
    axis.plot(time, truth, color=COLORS["truth"], linewidth=1.15, label="Truth", zorder=1)
    for protocol in protocols:
        estimate = data[f"{case}__kernel__{protocol}"]
        color, linestyle, label = PROTOCOL_STYLES[protocol]
        axis.plot(time, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label, zorder=3)
        curves.append(estimate)
    f7_set_y_limits(axis, curves)
    axis.axhline(0.0, color=COLORS["neutral_light"], linewidth=0.65, linestyle="--", zorder=0)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_xlabel("$t$")
    axis.set_ylabel("$\\gamma(t)$")


def f7_plot_memory(axis, data, protocols, case):
    time = data[f"{case}__memory_t"]
    truth = data[f"{case}__memory_truth"]
    curves = [truth]
    axis.plot(time, truth, color=COLORS["truth"], linewidth=1.15, label="Truth", zorder=1)
    for protocol in protocols:
        estimate = data[f"{case}__memory__{protocol}"]
        color, linestyle, label = PROTOCOL_STYLES[protocol]
        axis.plot(time, estimate, color=color, linestyle=linestyle, linewidth=1.0, label=label, zorder=3)
        curves.append(estimate)
    f7_set_y_limits(axis, curves)
    axis.axhline(0.0, color=COLORS["neutral_light"], linewidth=0.65, linestyle="--", zorder=0)
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.set_xlabel("$t$")
    axis.set_ylabel("$-\\mathcal{M}(t)$")


def plot(data_path, metadata_path, output_dir, *, stem, dpi):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    protocols = list(metadata["protocols"])
    figure, axes = plt.subplots(2, 3, figsize=FULL_FIGURE_SIZE, squeeze=False)
    for row, case in enumerate(("brownian", "brownian_levy")):
        f7_plot_sigma(axes[row, 0], data, protocols, case)
        f7_plot_kernel(axes[row, 1], data, protocols, case)
        f7_plot_memory(axes[row, 2], data, protocols, case)
    for label, axis in zip("abcdef", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    axes[0, 0].yaxis.labelpad = 0.0
    axes[1, 0].yaxis.labelpad = 0.0
    figure.subplots_adjust(left=0.095, right=0.985, bottom=0.105, top=0.94, wspace=0.42, hspace=0.38)
    figure.text(0.0121, 0.749, "\\textbf{Brownian}", rotation=90, ha="center", va="center", fontsize=7.0)
    figure.text(0.0121, 0.273, "\\textbf{Brownian--L\\'evy}", rotation=90, ha="center", va="center", fontsize=7.0)
    for row in axes:
        for column, axis in enumerate(row):
            lower, upper = axis.get_ylim()
            span = upper - lower
            if column == 0:
                axis.set_ylim(lower - 0.42 * span, upper)
            elif column == 2:
                axis.set_ylim(lower, upper + 0.48 * span)
            axis.legend(
                loc="lower center" if column == 0 else "upper right", fontsize=6.7, handlelength=2.0, labelspacing=0.25, borderpad=0.3
            )
    for column in (0, 2):
        pair = axes[:, column]
        lower = min((axis.get_ylim()[0] for axis in pair))
        upper = max((axis.get_ylim()[1] for axis in pair))
        for axis in pair:
            axis.set_ylim(lower, upper)
            axis.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    outputs = save_publication_figure(figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 7 workflow")
    plt.close(figure)
    return outputs
