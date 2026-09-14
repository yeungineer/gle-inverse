from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.utils.artifacts import ArtifactStore
from paper.code_figure_3.generate_data import METRIC_FIELDS, STRING_DIAGNOSTIC_FIELDS
from paper.utils.figures import (
    float_column,
    load_yaml,
    resolve_code_path,
    string_column,
)


def _integers(rows, key):
    return np.asarray([int(row[key]) for row in rows], dtype=np.int64)


def prepare_figure_data(config: dict[str, Any], config_path: Path):
    data_root = resolve_code_path(config_path, config["paths"]["data"])
    rows = load_yaml(data_root / "rows.yaml")["rows"]
    rows.sort(
        key=lambda row: (
            row["experiment"], row["budget"], row["dt"], row["M"], row["L"], row["seed"]
        )
    )
    arrays = {
        "point_id": string_column(rows, "point_id"),
        "experiment": string_column(rows, "experiment"),
        "case": string_column(rows, "case"),
        "seed": _integers(rows, "seed"),
        "M": _integers(rows, "M"),
        "L": _integers(rows, "L"),
        "dt": float_column(rows, "dt"),
        "budget": _integers(rows, "budget"),
        "success": np.ones(len(rows), dtype=bool),
        "included_in_plot": np.ones(len(rows), dtype=bool),
    }
    for field in METRIC_FIELDS:
        arrays[field] = float_column(rows, field)
    for field in STRING_DIAGNOSTIC_FIELDS:
        arrays[field] = string_column(rows, field)

    prepared = ArtifactStore(data_root / "prepared").prepare()
    data_path = prepared.save_npz("figure_3_data.npz", **arrays)
    metadata = {
        "plot_seeds": list(map(int, config["figure"]["seeds"])),
        "figure_style": {
            "iqr_alpha": float(config["figure"]["iqr_alpha"]),
        },
        "panels": {
            "c": {
                "display_L_values": [
                    int(curve["L"]) for curve in config["experiments"]["fixed_L"]["curves"]
                ]
            }
        },
        "scientific_configuration": config["experiments"],
    }
    metadata_path = prepared.save_yaml("figure_3_metadata.yaml", metadata)
    return data_path, metadata_path
