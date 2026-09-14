from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from gle_inverse.simulation.generation import generate_and_save_trajectory
from gle_inverse.solvers.kramers_moyal_sigma import SigmaSolver
from gle_inverse.utils.artifacts import ArtifactStore
from gle_inverse.utils.runtime import load_config, replace
from gle_inverse.workflows.kramers_moyal import prepare_problem, solve_problem
from paper.utils.experiments import PROJECT_ROOT, temporary_work_directory
from paper.utils.figures import resolve_code_path


def _interpolate(x, y, grid):
    order = np.argsort(x)
    unique, first = np.unique(np.asarray(x)[order], return_index=True)
    return np.interp(grid, unique, np.asarray(y)[order][first])


def _relative_l2(estimate, truth, grid):
    return float(
        np.sqrt(np.trapezoid((estimate - truth) ** 2, x=grid))
        / np.sqrt(np.trapezoid(truth**2, x=grid))
    )


def _memory_force(velocity, kernel, dt):
    kernel = np.asarray(kernel)[: len(velocity)]
    convolution = fftconvolve(velocity, kernel, mode="full")[: len(velocity)]
    convolution -= velocity * kernel[0]
    return -dt * convolution


def _uncorrected_sigma(problem):
    samples = problem.km_samples if problem.phase1_km_samples is None else problem.phase1_km_samples
    driver = deepcopy(problem.driver_config)
    driver["estimation"] = deepcopy(problem.config["phase1"]["estimation"])
    estimate, _ = SigmaSolver(driver).solve(samples["x"], samples["d2_truncated"])
    return np.asarray(estimate, dtype=float)


def _run_case(name, case, seed, data_root, display_horizon=None):
    source = Path(case["source_config"])
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    config = replace(load_config(source), case.get("overrides", {}))
    config["generation"]["simulation"]["seed"] = seed
    with temporary_work_directory(data_root) as work:
        problem = prepare_problem(
            config,
            experiment_name=source.stem,
            data_dir=work / "generated_data",
            output_dir=work,
        )
        arrays, metrics = solve_problem(problem, make_plots=False)
        uncorrected = _uncorrected_sigma(problem) if name == "brownian_levy" else None
        display = None
        if display_horizon is not None:
            driver = {
                "simulation": deepcopy(config["generation"]["simulation"]),
                "kernel": deepcopy(config["kernel"]),
                "force": deepcopy(config["force"]),
                "noise": deepcopy(config["noise"]),
            }
            driver["simulation"]["L"] = int(np.ceil(display_horizon / driver["simulation"]["dt"]))
            display = generate_and_save_trajectory(driver, work / "display", f"{source.stem}_display", trajectory_index=0)
    return problem, arrays, metrics, uncorrected, display


def _plot_data(name, problem, arrays, metrics, uncorrected, state_quantiles, memory_window):
    x = np.asarray(arrays["x_ref"], dtype=float)
    lower, upper = np.quantile(x, state_quantiles)
    grid = np.linspace(lower, upper, 401)
    sigma_truth = _interpolate(x, arrays["sigma_true"], grid)
    sigma_estimate = _interpolate(x, arrays["sigma_est"], grid)
    t = np.asarray(arrays["t"], dtype=float)
    kernel_mask = t <= problem.t_inference[-1]
    inference_t = np.asarray(problem.t_inference, dtype=float)
    memory_truth = -np.asarray(problem.memory_inference, dtype=float)
    memory_estimate = _memory_force(problem.v_inference, arrays["kernel_est"], float(inference_t[1] - inference_t[0]))
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
    return data, metadata


def generate_all(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    data_root.mkdir(parents=True, exist_ok=True)
    figure = config["figure"]
    seed = int(figure["seeds"][0])
    state_quantiles = tuple(map(float, figure["state_quantiles"]))
    memory_window = tuple(map(float, figure["memory_time_window"]))
    results = {}
    metadata = {"cases": {}}
    runs = {}
    for name, case in config["cases"].items():
        display_horizon = float(figure["levy_trajectory_time_window"][1]) if name == "brownian_levy" else None
        runs[name] = _run_case(name, case, seed, data_root, display_horizon)
        problem, arrays, metrics, uncorrected, _ = runs[name]
        data, case_metadata = _plot_data(name, problem, arrays, metrics, uncorrected, state_quantiles, memory_window)
        results.update(data)
        metadata["cases"][name] = {"representative": case_metadata}

    brownian, _, _, _, _ = runs["brownian"]
    window = tuple(map(float, figure["brownian_trajectory_time_window"]))
    keep = (brownian.t_observed >= window[0]) & (brownian.t_observed <= window[1])
    results["brownian__trajectory_time"] = brownian.t_observed[keep]
    results["brownian__trajectory_velocity"] = brownian.v_observed[keep]

    _, _, levy_metrics, _, display = runs["brownian_levy"]
    t, v = map(np.asarray, display[:2])
    window = tuple(map(float, figure["levy_trajectory_time_window"]))
    keep = (t >= window[0]) & (t <= window[1])
    t, v = t[keep], v[keep]
    levy = levy_metrics["levy"]
    jumps = np.abs(np.diff(v)) >= float(levy["epsilon"])
    results.update({
        "brownian_levy__trajectory_time": t,
        "brownian_levy__trajectory_velocity": v,
        "brownian_levy__trajectory_jump_time": t[1:][jumps],
        "brownian_levy__trajectory_jump_velocity": v[1:][jumps],
    })
    metadata["cases"]["brownian_levy"]["representative"].update({
        "alpha_est": float(levy["alpha_est"]),
        "kappa_est": float(levy["kappa_est"]),
    })
    prepared = ArtifactStore(data_root / "prepared").prepare()
    prepared.save_npz("figure_1_data.npz", **results)
    prepared.save_yaml("figure_1_metadata.yaml", metadata)
    return results
