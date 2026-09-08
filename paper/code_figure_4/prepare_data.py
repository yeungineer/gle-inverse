from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.utils.artifacts import ArtifactStore
from paper.code_figure_4.generate_data import (
    BOOLEAN_DIAGNOSTIC_FIELDS,
    METRIC_FIELDS,
    STRING_DIAGNOSTIC_FIELDS,
)
from paper.utils.figures import (
    float_column,
    load_yaml,
    resolve_code_path,
    string_column,
)


def prepare_figure_data(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    rows = load_yaml(data_root / "rows.yaml")["rows"]
    rows.sort(key=lambda row: (row["case"], row["route"], row["eta_ratio"], row["seed"]))
    arrays = {
        "point_id": string_column(rows, "point_id"),
        "experiment": string_column(rows, "experiment"),
        "case": string_column(rows, "case"),
        "route": string_column(rows, "route"),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "eta_ratio": float_column(rows, "eta_ratio"),
        "eta_absolute": float_column(rows, "eta_absolute"),
        "clean_state_scale": float_column(rows, "clean_state_scale"),
        "clean_empirical_std": float_column(rows, "clean_empirical_std"),
        "success": np.ones(len(rows), dtype=bool),
    }
    for field in METRIC_FIELDS:
        arrays[field] = float_column(rows, field)
    for field in BOOLEAN_DIAGNOSTIC_FIELDS:
        arrays[field] = np.asarray([bool(row.get(field, False)) for row in rows])
    for field in STRING_DIAGNOSTIC_FIELDS:
        arrays[field] = string_column(rows, field)

    prepared = ArtifactStore(data_root / "prepared").prepare()
    data_path = prepared.save_npz("figure_4_data.npz", **arrays)
    seeds = list(map(int, config["figure"]["seeds"]))
    metadata = {
        "seeds": seeds,
        "required_successes": len(seeds),
        "figure_style": {
            "iqr_alpha": float(config["figure"]["iqr_alpha"]),
        },
        "scientific_configuration": config["experiment"],
    }
    metadata_path = prepared.save_yaml("figure_4_metadata.yaml", metadata)
    return data_path, metadata_path
