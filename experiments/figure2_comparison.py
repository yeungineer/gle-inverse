"""Figure 2: comparison."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import os
import subprocess
import threading
import time
from tempfile import TemporaryDirectory

import numpy as np
import psutil
import yaml

from model.baselines import continuous_memory_kernel, fit_em, reconstruct_prony_sobolev, run_prony_sobolev, run_vroylandt_likelihood_em
from model.fit import ROOT, load_config, prepare, replace, save_figure_data, save_yaml, solve
from model.inverse import SplineBasis, fit_kernel

WORKER = Path(__file__)


METHODS = ("proposed", "prony_sobolev", "vroylandt_em")


PARAMS = {
    "figure": {"stem": "Figure_2", "dpi": 600, "kernel_time_window": [0.0, 30.0], "kernel_value_window": [-0.4, 2.1]},
    "source": "brownian.yaml",
    "common_task": {
        "seed": 42,
        "dt": 0.01,
        "L": 3000,
        "M": 1000,
        "kernel_type": "complex_exp",
        "force_type": "zero",
        "noise_type": "brownian",
        "diffusion": "constant",
        "observation_noise": 0.0,
    },
    "proposed": {"km_samples": "euler_oracle", "basis_count": 12},
    "prony_sobolev": {
        "initial_velocity_std": 1.0,
        "prony": {"sample_count": 24, "modes": 10, "lag_stride": 70, "drop_zero": True},
        "sobolev": {"omega": 0.05, "t_max": 30.0, "knot_count": 30, "grid_count": 2001},
    },
    "vroylandt_em": {
        "hidden_dimension": 5,
        "initialization_seed": 42,
        "evaluation_t_max": 30.0,
        "evaluation_grid_count": 2001,
        "em": {"likelihood_tolerance": 1e-08, "max_iterations": 2000, "minimum_iterations": 2, "covariance_floor": 1e-10},
    },
    "kernel_cost": {"repetitions": 6, "poll_interval_seconds": 0.0005},
}


def _method_configs(config):
    task = config["common_task"]
    source = Path(config["source"])
    if not source.is_absolute():
        source = ROOT / source
    proposed = replace(
        load_config(source),
        {
            "generation.simulation.seed": int(task["seed"]),
            "generation.simulation.dt": float(task["dt"]),
            "generation.simulation.L": int(task["L"]),
            "generation.simulation.num_steps": int(task["M"]),
            "generation.km_samples": str(config["proposed"]["km_samples"]),
            "kernel.type": str(task["kernel_type"]),
            "noise.type": str(task["noise_type"]),
            "noise.diffusion": str(task["diffusion"]),
            "force.type": str(task["force_type"]),
            "phase1.observation.sigma_obs": float(task["observation_noise"]),
            "phase2.observation.sigma_obs": float(task["observation_noise"]),
            "phase2.estimation.num_basis": int(config["proposed"]["basis_count"]),
        },
    )
    common = {
        "seed": int(task["seed"]),
        "dt": float(task["dt"]),
        "M": int(task["M"]),
        "L": int(task["L"]),
        "kernel_type": str(task["kernel_type"]),
        "diffusion": str(task["diffusion"]),
    }
    prony_settings = config["prony_sobolev"]
    prony = {
        **common,
        "initial_velocity_std": float(prony_settings["initial_velocity_std"]),
        "prony": dict(prony_settings["prony"]),
        "sobolev": dict(prony_settings["sobolev"]),
    }
    em_settings = config["vroylandt_em"]
    em = {
        **common,
        "hidden_dimension": int(em_settings["hidden_dimension"]),
        "initialization_seed": int(em_settings["initialization_seed"]),
        "evaluation_t_max": float(em_settings["evaluation_t_max"]),
        "evaluation_grid_count": int(em_settings["evaluation_grid_count"]),
        "em": dict(em_settings["em"]),
    }
    return {"proposed": proposed, "prony_sobolev": prony, "vroylandt_em": em}


def _accuracy(method, config_path, data_root):
    curve_path = data_root / f"{method}.npz"
    report_path = data_root / f"{method}_accuracy.yaml"
    command = [sys.executable, str(WORKER), "accuracy", method, str(config_path), str(curve_path), str(report_path), "0"]
    subprocess.run(command, cwd=ROOT, env=_environment(), check=True)
    report = load_config(report_path)
    return {"method": method, "curve_path": str(curve_path), "kernel_error": report["kernel_error"]}


def _environment():
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["MPLBACKEND"] = "Agg"
    return environment


def _benchmark(method, config_path, input_path, report_path, poll_interval):
    command = [sys.executable, str(WORKER), "cost", method, str(config_path), str(input_path), str(report_path), str(poll_interval)]
    subprocess.run(command, cwd=ROOT, env=_environment(), check=True)
    return load_config(report_path)


def run(config, out):
    with TemporaryDirectory(prefix="gle_comparison_") as tmp:
        data_root = Path(tmp)
        configs = _method_configs(config)
        config_paths = {}
        for method, task in configs.items():
            config_paths[method] = data_root / f"{method}.yaml"
            save_yaml(config_paths[method], task)
        accuracy = [_accuracy(method, config_paths[method], data_root) for method in METHODS]
        costs = []
        for repeat in range(int(config["kernel_cost"]["repetitions"])):
            for row in accuracy:
                method = row["method"]
                result = _benchmark(
                    method, config_paths[method], row["curve_path"], data_root / f"{method}_{repeat}.yaml",
                    float(config["kernel_cost"]["poll_interval_seconds"]),
                )
                costs.append(dict(method=method, repeat=repeat, **result))
        arrays = {}
        for row in accuracy:
            method = row["method"]
            with np.load(row["curve_path"], allow_pickle=False) as a:
                arrays[f"kernel_t__{method}"] = np.asarray(a["t" if method == "proposed" else "kernel_t"], dtype=float)
                arrays[f"kernel_truth__{method}"] = np.asarray(a["kernel_truth"], dtype=float)
                arrays[f"kernel_estimate__{method}"] = np.asarray(a["kernel_estimate"], dtype=float)
            arrays[f"kernel_error__{method}"] = np.asarray([row["kernel_error"]], dtype=float)
            times = sorted((r for r in costs if r["method"] == method), key=lambda r: r["repeat"])
            arrays[f"kernel_elapsed_seconds__{method}"] = np.asarray([r["elapsed_seconds"] for r in times], dtype=float)
            arrays[f"kernel_incremental_peak_rss_mib__{method}"] = (
                np.asarray([r["incremental_peak_rss_bytes"] for r in times], dtype=float) / 1024.0**2
            )
        metadata = {k: config["figure"][k] for k in ("kernel_time_window", "kernel_value_window")}
        save_figure_data(out, 2, arrays, metadata)
    return arrays


def _proposed(config, source):
    settings = config["phase2"]["estimation"]
    t = np.asarray(source["t_inference"], dtype=float)
    basis = SplineBasis(int(settings["num_basis"]), float(settings.get("t_max", t[-1] - t[0])))
    return fit_kernel(
        source["v_inference"], t, source["kernel_reference_index"], source["d1_empirical"], source["force_ref"], source["t"], basis
    )[0]


def _prony(config, source):
    return reconstruct_prony_sobolev(
        np.asarray(source["correlation_t"], dtype=float), np.asarray(source["correlation_samples"], dtype=float), config
    )[1]


def _em(config, source):
    settings = dict(config["em"], hidden_dimension=int(config["hidden_dimension"]), seed=int(config["initialization_seed"]))
    fit = fit_em(np.asarray(source["velocity_ensemble"], dtype=float), float(config["dt"]), settings)
    return continuous_memory_kernel(fit["transition"], float(config["dt"]), np.asarray(source["kernel_t"], dtype=float))[0]


def _fit_baseline(method, config_path, curve_path, report_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if method == "proposed":
        arrays, metrics = solve(prepare(config))
        saved = {key: np.asarray(value) for (key, value) in arrays.items() if value is not None}
        saved["kernel_estimate"] = np.asarray(arrays["kernel_est"])
        saved["kernel_truth"] = np.asarray(arrays["kernel_true"])
        error = float(metrics["kernel"]["relative_l2_error"])
    elif method == "prony_sobolev":
        saved, metrics = run_prony_sobolev(config)
        error = float(metrics["kernel_relative_l2_error"])
    else:
        saved, metrics = run_vroylandt_likelihood_em(config)
        error = float(metrics["kernel_relative_l2_error"])
    np.savez_compressed(curve_path, **saved)
    save_yaml(report_path, {"kernel_error": error})


def benchmark():
    action, method, config_path, input_path, report_path, interval = sys.argv[1:]
    if action == "accuracy":
        _fit_baseline(method, config_path, input_path, report_path)
        return
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    with np.load(input_path, allow_pickle=False) as stored:
        source = {key: np.asarray(stored[key]) for key in stored.files}
    solve = {"proposed": _proposed, "prony_sobolev": _prony, "vroylandt_em": _em}[method]
    np.linalg.svd(np.asarray([[2.0, 0.5], [0.5, 1.0]]), full_matrices=False)
    np.fft.rfft(np.arange(16, dtype=float))
    process = psutil.Process(os.getpid())
    baseline = int(process.memory_info().rss)
    peak = baseline
    stop = threading.Event()

    def monitor():
        nonlocal peak
        while not stop.is_set():
            peak = max(peak, int(process.memory_info().rss))
            time.sleep(float(interval))

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    started = time.perf_counter()
    try:
        solve(config, source)
    finally:
        elapsed = time.perf_counter() - started
        stop.set()
        thread.join()
    peak = max(peak, int(process.memory_info().rss))
    save_yaml(report_path, {"elapsed_seconds": elapsed, "incremental_peak_rss_bytes": max(peak - baseline, 0)})


# Worker measurements exclude plotting imports.
if __name__ == "__main__":
    benchmark()
    sys.exit()

import matplotlib.pyplot as plt

from model.style import (
    COLORS, ROW_FIGURE_SIZE, add_panel_label, apply_publication_style, load_prepared_data, save_publication_figure, style_axis,
)

METHOD_STYLE = {
    "proposed": {"label": "Proposed", "color": COLORS["blue"], "marker": "o", "linestyle": "--"},
    "prony_sobolev": {"label": "Prony--Sobolev", "color": COLORS["red"], "marker": "s", "linestyle": "-."},
    "vroylandt_em": {"label": "Likelihood--EM", "color": COLORS["teal"], "marker": "^", "linestyle": ":"},
}


PLOTTED_METHODS = METHODS


def f2_compact_value(value, decimal_places):
    value = float(value)
    exponent = int(np.floor(np.log10(abs(value)))) if value != 0.0 else 0
    if exponent <= -3 or exponent >= 3:
        mantissa = value / 10.0**exponent
        return f"{mantissa:.{max(int(decimal_places), 0)}f}\\times10^{{{exponent}}}"
    return f"{value:.{max(int(decimal_places), 0)}f}"


def f2_kernel_cost_axis(axis, data, *, metric, scale, xlabel, decimals):
    positions = {method: float(len(PLOTTED_METHODS) - index - 1) for (index, method) in enumerate(PLOTTED_METHODS)}
    summaries = {}
    for method in PLOTTED_METHODS:
        style = METHOD_STYLE[method]
        values = scale * np.asarray(data[f"{metric}__{method}"], dtype=float)
        summaries[method] = (float(np.median(values)), float(np.min(values)), float(np.max(values)))
        median, lower, upper = summaries[method]
        y = positions[method]
        axis.plot([lower, upper], [y, y], color=style["color"], linewidth=1.2, solid_capstyle="butt", zorder=1)
        axis.plot(
            median, y, marker=style["marker"], markersize=5.0, markerfacecolor="white", markeredgecolor=style["color"], markeredgewidth=1.0,
            linestyle="none", zorder=3,
        )
    smallest = min((lower for (_, lower, _) in summaries.values()))
    largest = max((upper for (_, _, upper) in summaries.values()))
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
        annotate_left = median == max((value[0] for value in summaries.values()))
        axis.annotate(
            f"${f2_compact_value(median, int(decimals))}$",
            xy=(median, positions[method]),
            xytext=(-9.0 if annotate_left else 10.0, 0.0),
            textcoords="offset points",
            ha="right" if annotate_left else "left",
            va="center",
            fontsize=6.8,
            color="#333333",
        )
    axis.set_ylim(-0.6, len(PLOTTED_METHODS) - 0.4)
    axis.set_yticks([positions[method] for method in PLOTTED_METHODS])
    axis.set_yticklabels([METHOD_STYLE[method]["label"] for method in PLOTTED_METHODS], fontsize=7.0)
    axis.tick_params(axis="y", which="both", length=0.0, right=False)
    axis.set_xlabel(xlabel)
    axis.minorticks_off()


def plot(data_path, metadata_path, output_dir, *, stem, dpi):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure = plt.figure(figsize=ROW_FIGURE_SIZE)
    outer = figure.add_gridspec(1, 2, width_ratios=(1.18, 1.0), wspace=0.32, left=0.07, right=0.99, bottom=0.205, top=0.925)
    panel_a = figure.add_subplot(outer[0, 0])
    cost_grid = outer[0, 1].subgridspec(2, 1, hspace=0.46)
    cost_time = figure.add_subplot(cost_grid[0, 0])
    cost_memory = figure.add_subplot(cost_grid[1, 0])
    lower, upper = map(float, metadata["kernel_time_window"])
    value_lower, value_upper = map(float, metadata["kernel_value_window"])
    truth_time = np.asarray(data["kernel_t__proposed"], dtype=float)
    truth_mask = (truth_time >= lower) & (truth_time <= upper)
    panel_a.plot(
        truth_time[truth_mask], np.asarray(data["kernel_truth__proposed"], dtype=float)[truth_mask], color=COLORS["truth"], linewidth=1.55,
        label="Truth", zorder=1,
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
        error_text = f2_compact_value(error, 3)
        label = f"{style['label']}, $E_\\gamma={error_text}$"
        panel_a.plot(time[mask], visible, color=style["color"], linestyle=style["linestyle"], linewidth=1.2, label=label, zorder=3)
        if off_scale:
            first = int(np.flatnonzero(outside)[0])
            above = bool(estimate[first] > value_upper)
            panel_a.plot(
                time[mask][first], value_upper if above else value_lower, marker="^" if above else "v", markersize=3.8,
                markerfacecolor=style["color"], markeredgecolor=style["color"], linestyle="none", clip_on=False, zorder=5,
            )
    panel_a.set_xlim(lower, upper)
    panel_a.set_ylim(value_lower, value_upper)
    panel_a.set_xlabel("$t$")
    panel_a.set_ylabel("$\\gamma(t)$")
    panel_a.legend(loc="upper right", fontsize=7.0)
    f2_kernel_cost_axis(cost_time, data, metric="kernel_elapsed_seconds", scale=1.0, xlabel="Time (s)", decimals=3)
    f2_kernel_cost_axis(cost_memory, data, metric="kernel_incremental_peak_rss_mib", scale=1.0, xlabel="Peak memory (MiB)", decimals=3)
    for axis in (panel_a, cost_time, cost_memory):
        style_axis(axis)
    add_panel_label(panel_a, "a", y_offset_points=5.5)
    add_panel_label(cost_time, "b", y_offset_points=5.5)
    outputs = save_publication_figure(
        figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 2 Brownian comparison workflow"
    )
    plt.close(figure)
    return outputs
