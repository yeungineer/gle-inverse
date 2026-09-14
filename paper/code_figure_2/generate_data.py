import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from gle_inverse.utils.artifacts import ArtifactStore
from gle_inverse.utils.runtime import load_config, replace
from paper.utils.experiments import PROJECT_ROOT
from paper.utils.figures import load_yaml, resolve_code_path

WORKER = Path(__file__).with_name("benchmark_worker.py")
METHODS = ("proposed", "prony_sobolev", "vroylandt_em")


def _method_configs(config):
    task = config["common_task"]
    source = Path(config["source"])
    if not source.is_absolute():
        source = PROJECT_ROOT / source
    proposed = replace(load_config(source), {
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
    })
    common = {
        "seed": int(task["seed"]), "dt": float(task["dt"]),
        "M": int(task["M"]), "L": int(task["L"]),
        "kernel_type": str(task["kernel_type"]), "diffusion": str(task["diffusion"]),
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
    command = [
        sys.executable, str(WORKER), "accuracy", method, str(config_path),
        str(curve_path), str(report_path), "0",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(), check=True)
    report = load_yaml(report_path)
    return {"method": method, "curve_path": str(curve_path), "kernel_error": report["kernel_error"]}


def _environment():
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    environment["MPLBACKEND"] = "Agg"
    return environment


def _benchmark(method, config_path, input_path, report_path, poll_interval):
    command = [
        sys.executable, str(WORKER), "cost", method, str(config_path), str(input_path),
        str(report_path), str(poll_interval),
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, env=_environment(), check=True)
    return load_yaml(report_path)


def generate_all(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    data_root.mkdir(parents=True, exist_ok=True)
    configs = _method_configs(config)
    config_paths = {}
    for method, method_config in configs.items():
        config_paths[method] = ArtifactStore(data_root).save_yaml(f"{method}.yaml", method_config)
    accuracy = [
        _accuracy(method, config_paths[method], data_root)
        for method in METHODS
    ]
    repetitions = int(config["kernel_cost"]["repetitions"])
    interval = float(config["kernel_cost"]["poll_interval_seconds"])
    costs = []
    for repeat in range(repetitions):
        for row in accuracy:
            method = row["method"]
            report = data_root / f"{method}_cost_{repeat}.yaml"
            result = _benchmark(method, config_paths[method], row["curve_path"], report, interval)
            costs.append({"method": method, "repeat": repeat, **result})
    ArtifactStore(data_root).save_yaml("results.yaml", {"accuracy": accuracy, "costs": costs})
    return accuracy
