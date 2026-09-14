from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.numerics.kramers_moyal import (
    empirical_conditional_moments,
    remove_symmetric_jump_second_moment,
    truncated_conditional_moments,
)
from gle_inverse.numerics.levy import estimate_symmetric_stable, resolve_levy_threshold
from gle_inverse.utils.artifacts import ArtifactStore
from gle_inverse.workflows.kramers_moyal import prepare_problem, solve_problem
from paper.utils.experiments import (
    configured_case,
    log_figure_progress,
    temporary_work_directory,
)
from paper.utils.figures import resolve_code_path

ROUTE_SEED = {"endpoint": 401, "start_endpoint": 402, "full_path": 403}
METRIC_FIELDS = (
    "sigma_error",
    "kernel_error",
    "alpha_error",
    "kappa_log_error",
    "alpha_est",
    "kappa_est",
    "epsilon",
    "tail_count",
    "selected_lambda",
)
BOOLEAN_DIAGNOSTIC_FIELDS = ("regularization_selected_at_boundary",)
STRING_DIAGNOSTIC_FIELDS = ("regularization_method",)


def expected_points(config):
    return [
        {
            "experiment": "measurement_noise",
            "case": case,
            "route": route,
            "eta_ratio": float(eta),
            "point_id": f"{case}__{route}__eta{float(eta):.12g}".replace(".", "p"),
        }
        for case in ("brownian", "brownian_levy")
        for route in config["experiment"]["routes"]
        for eta in config["experiment"]["noise_levels"]
    ]


def _apply_endpoint_noise(problem, route: str, sigma: float, seed: int) -> None:
    samples = problem.km_samples
    if route == "full_path":
        problem.phase1_km_samples = samples
        return
    delta = np.asarray(samples["delta_v"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, ROUTE_SEED[route]]))
    start_noise = sigma * rng.standard_normal(delta.shape) if route == "start_endpoint" else 0.0
    end_noise = sigma * rng.standard_normal(delta.shape)
    observed = delta + end_noise - start_noise
    branch_start = np.asarray(samples["branch_start"]) + start_noise
    d1, d2 = empirical_conditional_moments(observed, float(samples["dt"]))
    samples.update(
        {
            "delta_v": observed,
            "branch_start": branch_start,
            "v_next": branch_start + observed,
            "d1": d1,
            "d2": d2,
            "d1_raw": d1,
            "d2_raw": d2,
        }
    )
    if samples["model"] == "brownian_levy":
        epsilon = resolve_levy_threshold(problem.config["phase1"]["estimation"])
        levy = estimate_symmetric_stable(observed, float(samples["dt"]), epsilon=epsilon)
        d1, d2, retained = truncated_conditional_moments(
            observed, float(samples["dt"]), levy["epsilon"]
        )
        samples.update(
            {
                "d1": d1,
                "d2": remove_symmetric_jump_second_moment(
                    d2, levy["small_jump_second_moment"]
                ),
                "d1_truncated": d1,
                "d2_truncated": d2,
                "retained_fraction": np.mean(retained, axis=1),
                "levy": levy,
            }
        )
    problem.phase1_km_samples = samples


def _run_point(config, spec, seed: int, data_root: Path) -> dict[str, Any]:
    settings = config["experiment"]
    case = str(spec["case"])
    route = str(spec["route"])
    sigma = float(spec["eta_ratio"])
    overrides = {
        "generation.simulation.seed": seed,
        "generation.simulation.dt": float(settings["dt"]),
        "generation.simulation.num_steps": int(settings["M"]),
        "generation.simulation.L": int(settings["L"]),
        "generation.km_samples": str(settings["km_samples"]),
        "noise.diffusion": str(settings["diffusion_types"][case]),
        "phase1.observation.sigma_obs": sigma if route == "full_path" else 0.0,
        "phase2.observation.sigma_obs": sigma if route == "full_path" else 0.0,
        "phase2.estimation.num_basis": int(settings["N"]),
        "phase2.estimation.t_max": min(30.0, settings["L"] * settings["dt"]),
    }
    task = configured_case(case, overrides)
    stem = f"figure_4__{spec['point_id']}__seed_{seed:04d}"
    with temporary_work_directory(data_root) as root:
        problem = prepare_problem(
            task,
            experiment_name=stem,
            data_dir=root / "generated_data",
            output_dir=root,
            conditional_sample_seed=None,
        )
        _apply_endpoint_noise(problem, route, sigma, seed)
        _, metrics = solve_problem(problem, make_plots=False)

    row = {
        **spec,
        "seed": seed,
        "eta_absolute": sigma,
        "clean_state_scale": 1.0,
        "clean_empirical_std": float(np.std(problem.v_true)),
        "success": True,
        "sigma_error": float(metrics["sigma"]["relative_l2_error"]),
        "kernel_error": float(metrics["kernel"]["relative_l2_error"]),
        "selected_lambda": float(metrics["regularization"]["selected_lambda"]),
        "regularization_method": str(metrics["regularization"]["mode"]),
        "regularization_selected_at_boundary": bool(
            metrics["regularization"]["selected_at_search_boundary"]
        ),
    }
    if case == "brownian_levy":
        levy = metrics["levy"]
        row.update(
            {
                "alpha_error": abs(levy["alpha_est"] - levy["alpha_true"]),
                "kappa_log_error": abs(np.log(levy["kappa_est"] / levy["kappa_true"])),
                "alpha_est": float(levy["alpha_est"]),
                "kappa_est": float(levy["kappa_est"]),
                "epsilon": float(problem.km_samples["levy"]["epsilon"]),
                "tail_count": int(problem.km_samples["levy"]["tail_count"]),
            }
        )
    return row


def generate_all(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    ArtifactStore(data_root).prepare()
    points = expected_points(config)
    rows = []
    for seed in map(int, config["figure"]["seeds"]):
        for index, spec in enumerate(points, start=1):
            log_figure_progress(4, seed=seed, stage="point", current=index, total=len(points))
            rows.append(_run_point(config, spec, seed, data_root))
    ArtifactStore(data_root).save_yaml("rows.yaml", {"rows": rows})
    return {"rows": len(rows)}
