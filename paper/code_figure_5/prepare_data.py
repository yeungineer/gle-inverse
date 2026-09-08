from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.utils.artifacts import ArtifactStore
from paper.utils.figures import (
    float_column,
    load_yaml,
    resolve_code_path,
    string_column,
)


def prepare_figure_data(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    rows = load_yaml(data_root / "rows.yaml")["rows"]
    rows.sort(key=lambda row: (row["panel"], row["force_type"], row["alpha_true"], row["kappa_true"], row["seed"]))
    arrays = {
        "panel": string_column(rows, "panel"),
        "force_type": string_column(rows, "force_type"),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
    }
    for field in (
        "alpha_true", "kappa_true", "alpha_est", "kappa_est",
        "joint_levy_error", "sigma_error", "kernel_error", "memory_error",
        "selected_lambda", "effective_rank",
    ):
        arrays[field] = float_column(rows, field)
    arrays["tail_success"] = np.ones(len(rows), dtype=bool)
    arrays["sigma_success"] = np.ones(len(rows), dtype=bool)

    prepared = ArtifactStore(data_root / "prepared").prepare()
    data_path = prepared.save_npz("figure_5_data.npz", **arrays)
    metadata = {
        "external_plot_seeds": list(map(int, config["figure"]["seeds"])),
        "phase_report_seed": int(config["experiment"]["phase_map"]["seed"]),
        "experiment": config["experiment"],
    }
    metadata_path = prepared.save_yaml("figure_5_metadata.yaml", metadata)
    return data_path, metadata_path
