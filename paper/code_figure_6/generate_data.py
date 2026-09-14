from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.utils.artifacts import ArtifactStore
from paper.code_figure_6.kernel_inverse import solve_kernel
from paper.utils.experiments import log_figure_progress, run_point
from paper.utils.figures import resolve_code_path


def expected_points(config):
    return [
        (int(M), int(L))
        for M in config["experiment"]["M_values"]
        for L in config["experiment"]["L_values"]
    ]


def _run(config, data_root, seed, M, L):
    settings = config["experiment"]
    name = f"figure_6__seed_{seed:04d}__M{M}__L{L}"
    parameters = {
        "generation.simulation.seed": seed,
        "generation.simulation.dt": float(settings["dt"]),
        "generation.simulation.num_steps": M,
        "generation.simulation.L": L,
        "generation.km_samples": str(settings["km_samples"]),
        "force.type": str(settings["force_type"]),
        "kernel.type": str(settings["kernel_type"]),
        "noise.diffusion": str(settings["diffusion_type"]),
        "phase2.estimation.num_basis": int(settings["N"]),
        "phase2.estimation.t_max": 30.0,
    }
    problem, arrays, _ = run_point("brownian", parameters, data_root, name)
    result = solve_kernel(
        problem,
        n_basis=int(settings["N"]),
        t_max=30.0,
    )
    return arrays, result


def generate_all(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    data_root.mkdir(parents=True, exist_ok=True)
    points = expected_points(config)
    rows = []
    for seed in map(int, config["figure"]["seeds"]):
        for index, (M, L) in enumerate(points, 1):
            log_figure_progress(6, seed=seed, stage="point", current=index, total=len(points), detail=f"M={M}, L={L}")
            _, result = _run(config, data_root, seed, M, L)
            rows.append({
                "seed": seed,
                "M": M,
                "L": L,
                "kernel_error": result.kernel_relative_error,
            })

    representative = int(config["figure"]["representative_seed"])
    curves = {}
    for M, L in config["experiment"]["displayed_pairs"]:
        M, L = int(M), int(L)
        arrays, result = _run(config, data_root, representative, M, L)
        curves[f"kernel_t__L{L}"] = np.asarray(arrays["t"], dtype=float)
        curves[f"kernel_truth__L{L}"] = np.asarray(arrays["kernel_true"], dtype=float)
        curves[f"kernel_estimate__M{M}__L{L}"] = np.asarray(result.kernel_estimate, dtype=float)

    store = ArtifactStore(data_root)
    store.save_yaml("rows.yaml", {"rows": rows})
    store.save_npz("curves.npz", **curves)
    return rows
