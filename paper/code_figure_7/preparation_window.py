from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from gle_inverse.simulation.km_samples.common import _postprocess_conditional_increments
from gle_inverse.simulation.noises import parse_noise_config
from gle_inverse.workflows.kramers_moyal import prepare_problem, solve_problem
from paper.utils.experiments import configured_case, temporary_work_directory


def _evaluation_bank(problem, preparation_steps: int, evaluation_L: int):
    raw = problem.km_samples
    indices = np.asarray(raw["idx"], dtype=int)
    keep = (indices > preparation_steps) & (indices < preparation_steps + evaluation_L)
    noise = parse_noise_config(problem.driver_config["noise"])
    context = {
        "dt": float(raw["dt"]),
        "reference_states": np.asarray(raw["x"])[keep],
        "force_values": np.asarray(raw["f"])[keep],
        "reference_indices": indices[keep],
        "noise_model": noise,
        "stable_parameters": noise.stable_parameters() if noise.kind == "brownian_levy" else None,
    }
    generation = {
        "method": str(raw["generation_method"]),
        "substeps": int(raw["generation_substeps"]),
        "internal_dt": float(raw["generation_internal_dt"]),
    }
    return _postprocess_conditional_increments(
        context,
        np.asarray(raw["delta_v"])[keep],
        np.asarray(raw["mem"])[keep],
        generation,
        levy_estimation=problem.config["phase1"]["estimation"],
        levy_estimate=None,
        branch_start=np.asarray(raw["branch_start"])[keep],
    )


def run_prepared_window_point(
    case: str,
    parameters: Mapping[str, Any],
    evaluation_L: int,
    preparation_time: float,
    data_root: str | Path,
    experiment_name: str,
):
    parameters = deepcopy(dict(parameters))
    dt = float(parameters["generation.simulation.dt"])
    preparation_steps = int(round(float(preparation_time) / dt))
    parameters["generation.simulation.L"] = preparation_steps + int(evaluation_L)
    config = configured_case(case, parameters)
    with temporary_work_directory(data_root) as root:
        problem = prepare_problem(
            config,
            experiment_name=experiment_name,
            data_dir=root / "generated_data",
            output_dir=root,
            conditional_sample_seed=None,
        )
        bank = _evaluation_bank(problem, preparation_steps, int(evaluation_L))
        problem.km_samples = bank
        problem.phase1_km_samples = bank
        arrays, metrics = solve_problem(problem, make_plots=False)
    return problem, arrays, metrics
