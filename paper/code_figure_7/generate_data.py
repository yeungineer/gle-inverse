from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from gle_inverse.utils.artifacts import ArtifactStore
from paper.code_figure_7.preparation_window import run_prepared_window_point
from paper.utils.experiments import log_figure_progress, run_point
from paper.utils.figures import resolve_code_path


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


def generate_all(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    settings = config["experiment"]
    seed = int(config["figure"]["seed"])
    memory_window = tuple(map(float, config["figure"]["memory_time_window"]))
    protocols = list(settings["protocols"])
    runs = {}
    errors = {}
    total = len(settings["cases"]) * len(protocols)
    point = 0
    for case, case_settings in settings["cases"].items():
        errors[case] = {}
        for protocol in protocols:
            point += 1
            log_figure_progress(7, seed=seed, stage="point", current=point, total=total, detail=f"{case}, {protocol}")
            parameters = _parameters(settings, case_settings, protocol, seed)
            name = f"figure_7__{case}__{protocol}__seed_{seed:04d}"
            if protocol == "replay_branch":
                problem, arrays, metrics = run_prepared_window_point(
                    case, parameters, int(settings["L"]), float(settings["preparation_time"]), data_root, name
                )
            else:
                problem, arrays, metrics = run_point(case, parameters, data_root, name)
            runs[(case, protocol)] = (problem, arrays)
            errors[case][protocol] = {
                "sigma": float(metrics["sigma"]["relative_l2_error"]),
                "gamma": float(metrics["kernel"]["relative_l2_error"]),
            }

    q0, q1 = map(float, config["figure"]["state_quantiles"])
    curves = {}
    for case in settings["cases"]:
        problem, baseline = runs[(case, protocols[0])]
        baseline_state = np.asarray(baseline["x_ref"], dtype=float)
        state_grid = np.linspace(np.quantile(baseline_state, q0), np.quantile(baseline_state, q1), 401)
        curves[f"{case}__sigma_x"] = state_grid
        curves[f"{case}__sigma_truth"] = _interpolate(baseline["x_ref"], baseline["sigma_true"], state_grid)
        kernel_time = np.asarray(baseline["t"], dtype=float)
        kernel_mask = kernel_time <= float(problem.config["phase2"]["estimation"]["t_max"])
        curves[f"{case}__kernel_t"] = kernel_time[kernel_mask]
        curves[f"{case}__kernel_truth"] = np.asarray(baseline["kernel_true"], dtype=float)[kernel_mask]
        inference_t = np.asarray(problem.t_inference, dtype=float)
        memory_truth = -np.asarray(problem.memory_inference, dtype=float)
        memory_mask = (inference_t >= memory_window[0]) & (inference_t <= memory_window[1])
        curves[f"{case}__memory_t"] = inference_t[memory_mask]
        curves[f"{case}__memory_truth"] = memory_truth[memory_mask]
        for protocol in protocols:
            _, arrays = runs[(case, protocol)]
            curves[f"{case}__sigma__{protocol}"] = _interpolate(arrays["x_ref"], arrays["sigma_est"], state_grid)
            estimate_time = np.asarray(arrays["t"], dtype=float)
            estimate_mask = estimate_time <= float(problem.config["phase2"]["estimation"]["t_max"])
            kernel_estimate = np.interp(
                kernel_time[kernel_mask], estimate_time[estimate_mask], np.asarray(arrays["kernel_est"], dtype=float)[estimate_mask]
            )
            curves[f"{case}__kernel__{protocol}"] = kernel_estimate
            memory_estimate = _memory_force(problem.v_inference, kernel_estimate, float(inference_t[1] - inference_t[0]))
            curves[f"{case}__memory__{protocol}"] = memory_estimate[memory_mask]
            errors[case][protocol]["memory"] = _relative_l2(memory_estimate, memory_truth, inference_t)

    prepared = ArtifactStore(data_root / "prepared").prepare()
    prepared.save_npz("figure_7_data.npz", **curves)
    prepared.save_yaml("figure_7_metadata.yaml", {"seed": seed, "protocols": protocols, "errors": errors})
    return curves
