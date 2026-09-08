from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from gle_inverse.simulation.generation import generate_and_save_trajectory
from gle_inverse.simulation.noises import parse_noise_config
from gle_inverse.solvers.kramers_moyal_sigma import SigmaSolver
from gle_inverse.utils.artifacts import ArtifactStore
from gle_inverse.utils.metrics import Evaluator
from gle_inverse.workflows.kramers_moyal import KramersMoyalProblem
from paper.utils.experiments import log_figure_progress, prepare_point, run_point
from paper.utils.figures import resolve_code_path


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
    estimate = problem.km_samples["levy"]
    alpha_true, kappa_true = parse_noise_config(problem.driver_config["noise"]).stable_parameters()
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
    config = deepcopy(problem.driver_config)
    config["estimation"] = deepcopy(problem.config["phase1"]["estimation"])
    samples = problem.km_samples if problem.phase1_km_samples is None else problem.phase1_km_samples
    sigma_est, _ = SigmaSolver(config).solve(samples["x"], samples["d2"])
    _, relative_error, _ = Evaluator(config).evaluate_sigma(samples["x"], sigma_est)
    return float(relative_error)


def _effective_rank(singular_values):
    energy = np.asarray(singular_values, dtype=float) ** 2
    probability = energy / np.sum(energy)
    probability = probability[probability > 0.0]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def _independent_test(problem: KramersMoyalProblem, data_root, name, seed, L):
    driver = deepcopy(problem.driver_config)
    driver["simulation"]["seed"] = int(seed)
    driver["simulation"]["L"] = int(L)
    scratch = ArtifactStore(Path(data_root) / "work").prepare().root
    with TemporaryDirectory(prefix="paper_test_", dir=scratch) as directory:
        t, v, _, _, memory, _, _ = generate_and_save_trajectory(driver, Path(directory), name, trajectory_index=0)
    return np.asarray(t, dtype=float), np.asarray(v, dtype=float), np.asarray(memory, dtype=float)


def _external_rows(config, data_root):
    settings = config["experiment"]
    experiment = settings["external_force"]
    rows = []
    seeds = list(map(int, config["figure"]["seeds"]))
    forces = list(experiment["force_types"])
    for seed in seeds:
        for index, force_type in enumerate(forces, 1):
            log_figure_progress(
                5, seed=seed, stage="external force", current=index, total=len(forces), detail=force_type
            )
            name = f"figure_5_external_{force_type}__seed_{seed:04d}"
            problem, arrays, metrics = run_point(
                "brownian_levy",
                _parameters(settings, seed, experiment["M"], experiment["L"], force_type),
                data_root,
                name,
            )
            t, velocity, memory = _independent_test(
                problem,
                data_root,
                f"figure_5_test_{force_type}__seed_{seed + int(settings['test_seed_offset']):04d}",
                seed + int(settings["test_seed_offset"]),
                int(experiment["test_L"]),
            )
            kernel = np.interp(
                t - t[0],
                np.asarray(arrays["t"], dtype=float),
                np.asarray(arrays["kernel_est"], dtype=float),
                left=float(arrays["kernel_est"][0]),
                right=0.0,
            )
            convolution = fftconvolve(velocity, kernel, mode="full")[: len(velocity)]
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
                "memory_error": float(
                    np.linalg.norm(memory_est[mask] - memory_true[mask]) / np.linalg.norm(memory_true[mask])
                ),
                "selected_lambda": float(metrics["regularization"]["selected_lambda"]),
                "effective_rank": _effective_rank(arrays["regularization_singular_values"]),
            }
            rows.append(row)
    return rows


def _phase_rows(config, data_root):
    settings = config["experiment"]
    phase = settings["phase_map"]
    seed = int(phase["seed"])
    points = [(float(alpha), float(kappa)) for alpha in phase["alpha_values"] for kappa in phase["kappa_values"]]
    rows = []
    for index, (alpha, kappa) in enumerate(points, 1):
        log_figure_progress(
            5,
            seed=seed,
            stage="phase point",
            current=index,
            total=len(points),
            detail=f"alpha={alpha:g}, kappa={kappa:g}",
        )
        tag = f"a{alpha:.3f}_k{kappa:.3f}".replace(".", "p")
        problem = prepare_point(
            "brownian_levy",
            _parameters(settings, seed, phase["M"], phase["L"], phase["force_type"], alpha=alpha, kappa=kappa),
            data_root,
            f"figure_5_phase_{tag}__seed_{seed:04d}",
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


def generate_all(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    data_root.mkdir(parents=True, exist_ok=True)
    rows = _external_rows(config, data_root) + _phase_rows(config, data_root)
    ArtifactStore(data_root).save_yaml("rows.yaml", {"rows": rows})
    return rows
