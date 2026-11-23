"""实验配置与结果文件的读写。"""

from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def replace(config, overrides):
    config = deepcopy(config)
    for path, value in overrides.items():
        keys = path.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = deepcopy(value)
    return config


def configured_case(case, parameters):
    return replace(load_config(ROOT / f"{case}.yaml"), parameters)


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(serializable(value), sort_keys=False, allow_unicode=True))


def float_column(rows, key):
    return np.asarray(
        [np.nan if row.get(key) is None or row.get(key) == "" else float(row[key]) for row in rows],
        dtype=float,
    )


def string_column(rows, key):
    return np.asarray(["" if row.get(key) is None else str(row[key]) for row in rows], dtype=str)


def save_figure_data(out, number, arrays, metadata):
    np.savez_compressed(out / f"figure_{number}_data.npz", **arrays)
    save_yaml(out / f"figure_{number}_metadata.yaml", metadata)
