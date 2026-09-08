from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.utils.artifacts import ArtifactStore
from paper.code_figure_2.generate_data import METHODS
from paper.utils.figures import load_yaml, resolve_code_path


def prepare_figure_data(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    results = load_yaml(data_root / "results.yaml")
    accuracy = {row["method"]: row for row in results["accuracy"]}
    arrays = {}
    for method in METHODS:
        with np.load(accuracy[method]["curve_path"], allow_pickle=False) as stored:
            time_key = "t" if method == "proposed" else "kernel_t"
            arrays[f"kernel_t__{method}"] = np.asarray(stored[time_key], dtype=float)
            arrays[f"kernel_truth__{method}"] = np.asarray(stored["kernel_truth"], dtype=float)
            arrays[f"kernel_estimate__{method}"] = np.asarray(stored["kernel_estimate"], dtype=float)
        arrays[f"kernel_error__{method}"] = np.asarray([accuracy[method]["kernel_error"]], dtype=float)
        rows = sorted((row for row in results["costs"] if row["method"] == method), key=lambda row: row["repeat"])
        arrays[f"kernel_elapsed_seconds__{method}"] = np.asarray([row["elapsed_seconds"] for row in rows], dtype=float)
        arrays[f"kernel_incremental_peak_rss_mib__{method}"] = np.asarray([row["incremental_peak_rss_bytes"] for row in rows], dtype=float) / 1024.0**2
    prepared = ArtifactStore(data_root / "prepared").prepare()
    data_path = prepared.save_npz("figure_2_data.npz", **arrays)
    metadata_path = prepared.save_yaml("figure_2_metadata.yaml", {
        "kernel_time_window": config["figure"]["kernel_time_window"],
        "kernel_value_window": config["figure"]["kernel_value_window"],
    })
    return data_path, metadata_path
