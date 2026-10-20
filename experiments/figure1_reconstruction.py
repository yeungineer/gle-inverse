"""Figure 1: reconstruction."""

from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from scipy.signal import fftconvolve

from model.fit import ROOT, fit_sigma, load_config, predict_sigma, prepare, replace, save_figure_data, solve
from model.simulation import generate_trajectory
from model.style import (
    COLORS, FULL_FIGURE_SIZE, add_panel_label, apply_publication_style, load_prepared_data, save_publication_figure, style_axis,
    style_inset,
)

PARAMS = {
    "figure": {
        "stem": "Figure_1",
        "dpi": 600,
        "seeds": [42],
        "brownian_trajectory_time_window": [0.0, 20.0],
        "levy_trajectory_time_window": [0.0, 200.0],
        "memory_time_window": [2.0, 16.0],
        "state_quantiles": [0.01, 0.99],
    },
    "cases": {
        "brownian": {
            "source_config": "brownian.yaml",
            "overrides": {"generation.km_samples": "euler_oracle", "generation.simulation.num_steps": 2000, "noise.diffusion": "constant"},
        },
        "brownian_levy": {
            "source_config": "brownian_levy.yaml",
            "overrides": {"generation.km_samples": "euler_oracle", "generation.simulation.num_steps": 2000, "noise.diffusion": "cosine"},
        },
    },
}


def _interpolate(x, y, grid):
    order = np.argsort(x)
    unique, first = np.unique(np.asarray(x)[order], return_index=True)
    return np.interp(grid, unique, np.asarray(y)[order][first])


def _relative_l2(estimate, truth, grid):
    return float(np.sqrt(np.trapezoid((estimate - truth) ** 2, x=grid)) / np.sqrt(np.trapezoid(truth**2, x=grid)))


def _memory_force(velocity, kernel, dt):
    kernel = np.asarray(kernel)[: len(velocity)]
    convolution = fftconvolve(velocity, kernel, mode="full")[: len(velocity)]
    convolution -= velocity * kernel[0]
    return -dt * convolution


def _uncorrected_sigma(problem):
    samples = problem["phase1_km_samples"]
    driver = deepcopy(problem["driver_config"])
    driver["estimation"] = deepcopy(problem["config"]["phase1"]["estimation"])
    net, _ = fit_sigma(samples["x"], samples["d2_truncated"], driver)
    estimate = predict_sigma(net, samples["x"])
    return np.asarray(estimate, dtype=float)


def _run_case(name, case, seed, display_horizon=None):
    config = replace(load_config(ROOT / case["source_config"]), case.get("overrides", {}))
    config["generation"]["simulation"]["seed"] = seed
    problem = prepare(config)
    arrays, metrics = solve(problem)
    uncorrected = _uncorrected_sigma(problem) if name == "brownian_levy" else None
    display = None
    if display_horizon is not None:
        driver = deepcopy(problem["driver_config"])
        driver["simulation"]["L"] = int(np.ceil(display_horizon / driver["simulation"]["dt"]))
        display = generate_trajectory(driver, 0)
    return (problem, arrays, metrics, uncorrected, display)


def _plot_data(name, problem, arrays, metrics, uncorrected, state_quantiles, memory_window):
    x = np.asarray(arrays["x_ref"], dtype=float)
    lower, upper = np.quantile(x, state_quantiles)
    grid = np.linspace(lower, upper, 401)
    sigma_truth = _interpolate(x, arrays["sigma_true"], grid)
    sigma_estimate = _interpolate(x, arrays["sigma_est"], grid)
    t = np.asarray(arrays["t"], dtype=float)
    kernel_mask = t <= problem["t_inference"][-1]
    inference_t = np.asarray(problem["t_inference"], dtype=float)
    memory_truth = -np.asarray(problem["memory_inference"], dtype=float)
    memory_estimate = _memory_force(problem["v_inference"], arrays["kernel_est"], float(inference_t[1] - inference_t[0]))
    memory_mask = (inference_t >= memory_window[0]) & (inference_t <= memory_window[1])
    data = {
        f"{name}__sigma_grid": grid,
        f"{name}__sigma_truth": sigma_truth,
        f"{name}__sigma_representative": sigma_estimate,
        f"{name}__sigma_q25": sigma_estimate,
        f"{name}__sigma_q75": sigma_estimate,
        f"{name}__kernel_time": t[kernel_mask],
        f"{name}__kernel_truth": np.asarray(arrays["kernel_true"])[kernel_mask],
        f"{name}__kernel_representative": np.asarray(arrays["kernel_est"])[kernel_mask],
        f"{name}__kernel_q25": np.asarray(arrays["kernel_est"])[kernel_mask],
        f"{name}__kernel_q75": np.asarray(arrays["kernel_est"])[kernel_mask],
        f"{name}__memory_time": inference_t[memory_mask],
        f"{name}__memory_truth": memory_truth[memory_mask],
        f"{name}__memory_estimate": memory_estimate[memory_mask],
    }
    metadata = {
        "sigma_mu_relative_error": float(metrics["sigma"]["relative_l2_error"]),
        "kernel_relative_error": float(metrics["kernel"]["relative_l2_error"]),
        "memory_force_relative_error": _relative_l2(memory_estimate, memory_truth, inference_t),
    }
    if uncorrected is not None:
        uncorrected = _interpolate(x, uncorrected, grid)
        data[f"{name}__sigma_without_correction_representative"] = uncorrected
        metadata["sigma_without_correction_relative_error"] = _relative_l2(uncorrected, sigma_truth, grid)
    return (data, metadata)


def run(config, out):
    figure = config["figure"]
    seed = int(figure["seeds"][0])
    state_quantiles = tuple(map(float, figure["state_quantiles"]))
    memory_window = tuple(map(float, figure["memory_time_window"]))
    results = {}
    metadata = {"cases": {}}
    runs = {}
    for name, case in config["cases"].items():
        display_horizon = float(figure["levy_trajectory_time_window"][1]) if name == "brownian_levy" else None
        runs[name] = _run_case(name, case, seed, display_horizon)
        problem, arrays, metrics, uncorrected, _ = runs[name]
        data, case_metadata = _plot_data(name, problem, arrays, metrics, uncorrected, state_quantiles, memory_window)
        results.update(data)
        metadata["cases"][name] = {"representative": case_metadata}
    brownian, _, _, _, _ = runs["brownian"]
    window = tuple(map(float, figure["brownian_trajectory_time_window"]))
    keep = (brownian["t_observed"] >= window[0]) & (brownian["t_observed"] <= window[1])
    results["brownian__trajectory_time"] = brownian["t_observed"][keep]
    results["brownian__trajectory_velocity"] = brownian["v_observed"][keep]
    _, _, levy_metrics, _, display = runs["brownian_levy"]
    t, v = map(np.asarray, display[:2])
    window = tuple(map(float, figure["levy_trajectory_time_window"]))
    keep = (t >= window[0]) & (t <= window[1])
    t, v = (t[keep], v[keep])
    levy = levy_metrics["levy"]
    jumps = np.abs(np.diff(v)) >= float(levy["epsilon"])
    results.update(
        {
            "brownian_levy__trajectory_time": t,
            "brownian_levy__trajectory_velocity": v,
            "brownian_levy__trajectory_jump_time": t[1:][jumps],
            "brownian_levy__trajectory_jump_velocity": v[1:][jumps],
        }
    )
    metadata["cases"]["brownian_levy"]["representative"].update(
        {"alpha_est": float(levy["alpha_est"]), "kappa_est": float(levy["kappa_est"])}
    )
    save_figure_data(out, 1, results, metadata)
    return results


def f1_percent(value):
    return f"{100.0 * float(value):.3f}\\%"


def f1_plot_band(axis, x, lower, upper, *, color=COLORS["estimate_band"]):
    if np.allclose(lower, upper, rtol=0.0, atol=1e-12):
        return
    axis.fill_between(x, lower, upper, color=color, alpha=0.2, linewidth=0.0, zorder=1, label="IQR")


def f1_panel_sigma(axis, data, metadata, prefix, *, levy=False):
    case = metadata["cases"][prefix]["representative"]
    grid = data[f"{prefix}__sigma_grid"]
    truth = data[f"{prefix}__sigma_truth"]
    f1_plot_band(axis, grid, data[f"{prefix}__sigma_q25"], data[f"{prefix}__sigma_q75"])
    axis.plot(grid, truth, color=COLORS["truth"], label="Truth", zorder=4)
    label = f"Estimate ($E_\\sigma={f1_percent(case['sigma_mu_relative_error'])}$)"
    axis.plot(grid, data[f"{prefix}__sigma_representative"], color=COLORS["estimate"], linestyle="--", label=label, zorder=5)
    if levy:
        axis.plot(
            grid, data[f"{prefix}__sigma_without_correction_representative"], color=COLORS["uncorrected"], linestyle=(0, (2.0, 1.25)),
            linewidth=0.9, label=f"Uncorrected ($E_\\sigma={f1_percent(case['sigma_without_correction_relative_error'])}$)", zorder=3,
        )
    values = np.concatenate([np.asarray(truth, dtype=float), np.asarray(data[f"{prefix}__sigma_representative"], dtype=float)])
    if levy:
        values = np.concatenate([values, np.asarray(data[f"{prefix}__sigma_without_correction_representative"], dtype=float)])
    value_range = float(np.ptp(values))
    minimum_margin = 0.06 * max(float(np.max(np.abs(truth))), 1e-08)
    margin = max(0.1 * value_range, minimum_margin)
    axis.set_ylim(float(np.min(values)) - margin, float(np.max(values)) + margin)
    axis.set_xlabel("$v$")
    axis.set_ylabel("$\\sigma(v)$")
    axis.legend(loc="lower left", fontsize=7.0, handlelength=1.8)


def f1_panel_kernel(axis, data, metadata, prefix):
    case = metadata["cases"][prefix]["representative"]
    time = data[f"{prefix}__kernel_time"]
    f1_plot_band(axis, time, data[f"{prefix}__kernel_q25"], data[f"{prefix}__kernel_q75"])
    axis.plot(time, data[f"{prefix}__kernel_truth"], color=COLORS["truth"], label="Truth", zorder=4)
    axis.plot(
        time, data[f"{prefix}__kernel_representative"], color=COLORS["estimate"], linestyle="--",
        label=f"Estimate ($E_\\gamma={f1_percent(case['kernel_relative_error'])}$)", zorder=5,
    )
    axis.axhline(0.0, color=COLORS["neutral_light"], linewidth=0.65, linestyle="--", zorder=0)
    axis.set_xlabel("$t$")
    axis.set_ylabel("$\\gamma(t)$")
    axis.set_xlim(float(time[0]), float(time[-1]))
    axis.legend(loc="upper right")
    inset = axis.inset_axes([0.48, 0.35, 0.47, 0.35])
    inset.plot(data[f"{prefix}__memory_time"], data[f"{prefix}__memory_estimate"], color=COLORS["estimate"], linestyle="--", linewidth=0.85)
    inset.plot(data[f"{prefix}__memory_time"], data[f"{prefix}__memory_truth"], color=COLORS["truth"], linestyle="-", linewidth=0.6)
    inset.set_xlabel("$t$")
    inset.set_ylabel("$-\\mathcal{M}$")
    inset.set_title(f"$E_\\mathcal{{M}}={f1_percent(case['memory_force_relative_error'])}$", fontsize=6.8, pad=2.5)
    style_inset(inset)


def plot(data_path, metadata_path, output_dir, *, stem="Figure_1", dpi=600):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure, axes = plt.subplots(2, 3, figsize=FULL_FIGURE_SIZE, squeeze=False)
    axes[0, 0].plot(data["brownian__trajectory_time"], data["brownian__trajectory_velocity"], color=COLORS["truth"], linewidth=0.75)
    axes[0, 0].set_xlabel("$t$")
    axes[0, 0].set_ylabel("$v(t)$")
    f1_panel_sigma(axes[0, 1], data, metadata, "brownian", levy=False)
    axes[0, 1].set_ylim(0.4, 0.6)
    axes[0, 1].set_yticks([0.4, 0.45, 0.5, 0.55, 0.6])
    axes[0, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    f1_panel_kernel(axes[0, 2], data, metadata, "brownian")
    case = metadata["cases"]["brownian_levy"]["representative"]
    axes[1, 0].plot(
        data["brownian_levy__trajectory_time"], data["brownian_levy__trajectory_velocity"], color=COLORS["truth"], linewidth=0.72, zorder=2
    )
    jump_time = data["brownian_levy__trajectory_jump_time"]
    jump_velocity = data["brownian_levy__trajectory_jump_velocity"]
    axes[1, 0].scatter(
        jump_time, jump_velocity, s=13.0, facecolors="none", edgecolors=COLORS["levy"], linewidths=0.8,
        label="$|\\Delta v|\\geq\\varepsilon$", zorder=4,
    )
    axes[1, 0].set_xlabel("$t$")
    axes[1, 0].set_ylabel("$v(t)$")
    axes[1, 0].set_xlim(0.0, 200.0)
    lower, upper = axes[1, 0].get_ylim()
    axes[1, 0].set_ylim(lower, upper + 0.15 * (upper - lower))
    axes[1, 0].legend(loc="lower left", fontsize=7.0, handlelength=1.8)
    axes[1, 0].text(
        0.03, 0.96, f"$\\widehat{{\\alpha}}={case['alpha_est']:.3f}$, $\\widehat{{\\kappa}}={case['kappa_est']:.3f}$",
        transform=axes[1, 0].transAxes, ha="left", va="top", fontsize=7.0,
    )
    f1_panel_sigma(axes[1, 1], data, metadata, "brownian_levy", levy=True)
    axes[1, 1].set_ylim(0.0, 1.15)
    axes[1, 1].set_yticks([0.3, 0.5, 0.7, 0.9, 1.1])
    axes[1, 1].yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    f1_panel_kernel(axes[1, 2], data, metadata, "brownian_levy")
    for label, axis in zip("abcdef", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    figure.subplots_adjust(left=0.072, right=0.985, bottom=0.105, top=0.94, wspace=0.42, hspace=0.3)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=dpi)
    plt.close(figure)
    return outputs
