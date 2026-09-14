import logging
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

from gle_inverse.utils.runtime import load_config
from gle_inverse.workflows.kramers_moyal import prepare_problem, solve_problem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CASE_CONFIGS = {
    "brownian": PROJECT_ROOT / "run/configs/kramers_moyal_brownian.yaml",
    "brownian_levy": PROJECT_ROOT / "run/configs/kramers_moyal_brownian_levy.yaml",
}


def log_figure_progress(
    figure_number: int,
    *,
    seed: int,
    stage: str,
    current: int | None = None,
    total: int | None = None,
    detail: str | None = None,
) -> None:
    parts = [f"Figure {figure_number}", f"seed {seed}"]
    parts.append(f"{stage} {current}/{total}" if current is not None else stage)
    if detail:
        parts.append(detail)
    logging.info(" | ".join(parts))


def load_case_config(case: str) -> dict[str, Any]:
    return load_config(CASE_CONFIGS[case])


def configured_case(case: str, parameters: Mapping[str, Any]) -> dict[str, Any]:
    config = deepcopy(load_case_config(case))
    for path, value in parameters.items():
        keys = path.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = deepcopy(value)
    return config


@contextmanager
def temporary_work_directory(data_root: str | Path):
    parent = Path(data_root) / "work"
    parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="point_", dir=parent) as directory:
        yield Path(directory)


def prepare_point(
    case: str,
    parameters: Mapping[str, Any],
    data_root: str | Path,
    experiment_name: str,
):
    config = configured_case(case, parameters)
    with temporary_work_directory(data_root) as root:
        problem = prepare_problem(
            config,
            experiment_name=experiment_name,
            data_dir=root / "generated_data",
            output_dir=root,
            conditional_sample_seed=None,
        )
    return problem


def run_point(
    case: str,
    parameters: Mapping[str, Any],
    data_root: str | Path,
    experiment_name: str,
):
    problem = prepare_point(case, parameters, data_root, experiment_name)
    arrays, metrics = solve_problem(problem, make_plots=False)
    return problem, arrays, metrics
