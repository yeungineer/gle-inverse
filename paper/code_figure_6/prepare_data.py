from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.utils.artifacts import ArtifactStore
from paper.utils.figures import load_yaml, resolve_code_path


def prepare_figure_data(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    rows = load_yaml(data_root / "rows.yaml")["rows"]
    rows.sort(key=lambda row: (row["L"], row["M"], row["seed"]))
    with np.load(data_root / "curves.npz", allow_pickle=False) as stored:
        curves = {key: np.asarray(stored[key]) for key in stored.files}
    arrays = {
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "M": np.asarray([row["M"] for row in rows], dtype=np.int64),
        "L": np.asarray([row["L"] for row in rows], dtype=np.int64),
        "kernel_error": np.asarray([row["kernel_error"] for row in rows], dtype=float),
        **curves,
    }
    prepared = ArtifactStore(data_root / "prepared").prepare()
    data_path = prepared.save_npz("figure_6_data.npz", **arrays)
    metadata = {
        "plot_seeds": list(map(int, config["figure"]["seeds"])),
        "figure_style": {"iqr_alpha": float(config["figure"]["iqr_alpha"])},
        "experiment": config["experiment"],
    }
    metadata_path = prepared.save_yaml("figure_6_metadata.yaml", metadata)
    return data_path, metadata_path
