import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gle_inverse.numerics.basis import get_basis
from gle_inverse.solvers.kramers_moyal_kernel import KernelSolver
from gle_inverse.utils.artifacts import ArtifactStore
from paper.code_figure_2.prony_sobolev import (
    reconstruct_prony_sobolev,
    run_prony_sobolev,
)
from paper.code_figure_2.vroylandt_likelihood_em import (
    VroylandtLikelihoodEM,
    continuous_memory_kernel,
    run_vroylandt_likelihood_em,
)
from run.run_kramers_moyal import main as run_kramers_moyal


def _proposed(config, source):
    estimation = dict(config["phase2"]["estimation"])
    t_reference = np.asarray(source["t_inference"], dtype=float)
    estimation.setdefault("t_max", float(t_reference[-1] - t_reference[0]))
    solver = KernelSolver(get_basis(estimation), t_reference)
    return solver.solve(
        np.asarray(source["v_inference"], dtype=float),
        np.asarray(source["kernel_reference_index"], dtype=int),
        np.asarray(source["d1_empirical"], dtype=float),
        np.asarray(source["force_ref"], dtype=float),
        np.asarray(source["t"], dtype=float),
    )[0]


def _prony(config, source):
    return reconstruct_prony_sobolev(
        np.asarray(source["correlation_t"], dtype=float),
        np.asarray(source["correlation_samples"], dtype=float),
        config,
    )[1]


def _em(config, source):
    settings = config["em"]
    estimator = VroylandtLikelihoodEM(
        hidden_dimension=int(config["hidden_dimension"]),
        seed=int(config["initialization_seed"]),
        likelihood_tolerance=float(settings["likelihood_tolerance"]),
        max_iterations=int(settings["max_iterations"]),
        covariance_floor=float(settings["covariance_floor"]),
        minimum_iterations=int(settings["minimum_iterations"]),
    )
    fit = estimator.fit(np.asarray(source["velocity_ensemble"], dtype=float), float(config["dt"]))
    return continuous_memory_kernel(fit.transition, float(config["dt"]), np.asarray(source["kernel_t"], dtype=float))[0]


def _accuracy(method, config_path, curve_path, report_path):
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if method == "proposed":
        _, arrays, metrics = run_kramers_moyal(
            config_path, make_plots=False, output_dir=Path(curve_path).parent / "production"
        )
        saved = {key: np.asarray(value) for key, value in arrays.items() if value is not None}
        saved["kernel_estimate"] = np.asarray(arrays["kernel_est"])
        saved["kernel_truth"] = np.asarray(arrays["kernel_true"])
        error = float(metrics["kernel"]["relative_l2_error"])
    elif method == "prony_sobolev":
        saved, metrics = run_prony_sobolev(config)
        error = float(metrics["kernel_relative_l2_error"])
    else:
        saved, metrics = run_vroylandt_likelihood_em(config)
        error = float(metrics["kernel_relative_l2_error"])
    ArtifactStore(Path(curve_path).parent).save_npz(Path(curve_path).name, **saved)
    ArtifactStore(Path(report_path).parent).save_yaml(Path(report_path).name, {"kernel_error": error})


def main():
    action, method, config_path, input_path, report_path, interval = sys.argv[1:]
    if action == "accuracy":
        _accuracy(method, config_path, input_path, report_path)
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
    ArtifactStore(Path(report_path).parent).save_yaml(
        Path(report_path).name,
        {"elapsed_seconds": elapsed, "incremental_peak_rss_bytes": max(peak - baseline, 0)},
    )


if __name__ == "__main__":
    main()
