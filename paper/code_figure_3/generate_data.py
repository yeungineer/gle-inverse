from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.numerics.kramers_moyal import brownian_euler_error_scales
from gle_inverse.simulation.noises import diffusion_profile, parse_noise_config
from gle_inverse.utils.artifacts import ArtifactStore
from paper.utils.experiments import log_figure_progress, run_point
from paper.utils.figures import resolve_code_path

METRIC_FIELDS = (
    "d1_normalized_rmse",
    "d2_normalized_rmse",
    "d1_rmse",
    "d2_rmse",
    "kernel_error",
    "sigma_error",
    "effective_rank",
    "smallest_relative_singular",
    "selected_lambda",
    "regularization_selected_at_boundary",
)
STRING_DIAGNOSTIC_FIELDS = ("regularization_method",)


def expected_points(config):
    experiments = config["experiments"]
    dt = float(experiments["core"]["dt"])
    reference_rows = int(experiments["core"]["reference_rows"])
    points = []
    for M in map(int, experiments["sampling_M"]["values"]):
        for experiment in ("sampling_M_theory", "kernel_M"):
            points.append({
                "experiment": experiment, "case": "brownian", "M": M,
                "L": reference_rows, "dt": dt, "budget": M * reference_rows,
            })
    for curve in experiments["fixed_L"]["curves"]:
        L = int(curve["L"])
        for M in map(int, curve["M_values"]):
            points.append({
                "experiment": "fixed_L_brownian", "case": "brownian",
                "M": M, "L": L, "dt": dt, "budget": M * L,
            })
    for point in points:
        point["point_id"] = "__".join((
            point["experiment"], f"M{point['M']}", f"L{point['L']}",
            f"dt{point['dt']:.12g}", f"B{point['budget']}",
        )).replace(".", "p")
    return points


def _effective_rank(singular_values: np.ndarray) -> float:
    energy = np.asarray(singular_values, dtype=float) ** 2
    probability = energy / np.sum(energy)
    probability = probability[probability > 0.0]
    return float(np.exp(-np.sum(probability * np.log(probability))))


def _metrics(problem, arrays, metrics) -> dict[str, Any]:
    d1_error = arrays["d1_empirical"] - arrays["d1_reference"]
    d2_error = arrays["d2_diffusion_target"] - arrays["d2_finite_step_reference"]
    d1_rmse = float(np.sqrt(np.mean(d1_error**2)))
    d2_rmse = float(np.sqrt(np.mean(d2_error**2)))

    km = problem.km_samples
    drift = np.asarray(km["f"]) - np.asarray(km["mem"])
    noise = parse_noise_config(problem.driver_config["noise"])
    diffusion = diffusion_profile(km["x"], noise.diffusion_name())
    scales = brownian_euler_error_scales(drift, diffusion, float(km["dt"]), sample_size=1)
    d1_scale = np.sqrt(np.mean(np.asarray(scales["d1_standard_deviation"]) ** 2))
    d2_scale = np.sqrt(np.mean(np.asarray(scales["d2_standard_deviation"]) ** 2))
    singular = np.asarray(arrays["regularization_singular_values"])
    regularization = metrics["regularization"]
    return {
        "d1_rmse": d1_rmse,
        "d2_rmse": d2_rmse,
        "d1_normalized_rmse": float(d1_rmse / d1_scale),
        "d2_normalized_rmse": float(d2_rmse / d2_scale),
        "kernel_error": float(metrics["kernel"]["relative_l2_error"]),
        "sigma_error": float(metrics["sigma"]["relative_l2_error"]),
        "effective_rank": _effective_rank(singular),
        "smallest_relative_singular": float(singular[-1] / singular[0]),
        "selected_lambda": float(regularization["selected_lambda"]),
        "regularization_selected_at_boundary": bool(
            regularization["selected_at_search_boundary"]
        ),
        "regularization_method": str(regularization["mode"]),
    }


def _run_point(config, spec, seed: int, data_root: Path) -> dict[str, Any]:
    reconstruction = config["experiments"]["kernel_reconstruction"]
    overrides = {
        "generation.simulation.seed": seed,
        "generation.simulation.dt": float(spec["dt"]),
        "generation.simulation.num_steps": int(spec["M"]),
        "generation.simulation.L": int(spec["L"]),
        "generation.km_samples": "euler_oracle",
        "noise.diffusion": str(config["experiments"]["core"]["diffusion_type"]),
        "phase2.estimation.num_basis": int(reconstruction["num_basis"]),
        "phase2.estimation.t_max": float(reconstruction["t_max"]),
    }
    problem, arrays, metrics = run_point(
        "brownian",
        parameters=overrides,
        data_root=data_root,
        experiment_name=f"figure_3__{spec['point_id']}__seed_{seed:04d}",
    )
    values = _metrics(problem, arrays, metrics)
    row = {**spec, "seed": seed, "success": True}
    fields = (
        ("d1_rmse", "d2_rmse", "d1_normalized_rmse", "d2_normalized_rmse")
        if spec["experiment"] == "sampling_M_theory"
        else (
            "kernel_error",
            "sigma_error",
            "effective_rank",
            "smallest_relative_singular",
            "selected_lambda",
            "regularization_selected_at_boundary",
            "regularization_method",
        )
    )
    row.update({field: values[field] for field in fields})
    return row


def generate_all(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    ArtifactStore(data_root).prepare()
    points = expected_points(config)
    rows = []
    for seed in map(int, config["figure"]["seeds"]):
        for index, spec in enumerate(points, start=1):
            log_figure_progress(3, seed=seed, stage="point", current=index, total=len(points))
            rows.append(_run_point(config, spec, seed, data_root))
    ArtifactStore(data_root).save_yaml("rows.yaml", {"rows": rows})
    return {"rows": len(rows)}
