"""Shared data persistence for numerical experiments."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _serializable(value: Any) -> Any:
    """Convert NumPy-rich experiment values to JSON/YAML-safe objects."""
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    return value


@dataclass(frozen=True)
class ArtifactStore:
    """Filesystem-backed output collection rooted at one directory."""

    root: Path

    def __init__(self, root: str | Path):
        object.__setattr__(self, "root", Path(root))

    def prepare(self, *relative_dirs: str | Path) -> "ArtifactStore":
        self.root.mkdir(parents=True, exist_ok=True)
        for relative in relative_dirs:
            (self.root / relative).mkdir(parents=True, exist_ok=True)
        return self

    def path(self, relative: str | Path) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_yaml(self, relative: str | Path, value: Any) -> Path:
        path = self.path(relative)
        with path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(_serializable(value), handle, allow_unicode=True, sort_keys=False)
        return path

    def save_npz(self, relative: str | Path, **arrays: Any) -> Path:
        path = self.path(relative)
        np.savez_compressed(path, **{key: np.asarray(value) for key, value in arrays.items() if value is not None})
        return path


def save_run_artifacts(
    output_dir: str | Path,
    config: Mapping[str, Any],
    arrays: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    name: str,
) -> tuple[Path, Path, Path]:
    """Save one run's configuration, arrays, and metrics."""
    store = ArtifactStore(output_dir).prepare()
    config_path = store.save_yaml("config_snapshot.yaml", config)
    array_path = store.save_npz(f"{name}.npz", **arrays)
    metric_path = store.save_yaml(f"{name}_metrics.yaml", metrics)
    return config_path, array_path, metric_path
