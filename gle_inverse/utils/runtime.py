"""Configuration, logging, and run-directory helpers."""

import logging
import sys
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from gle_inverse.utils.artifacts import ArtifactStore


def load_config(path: str | Path, relative_path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path)
    if relative_path is not None:
        config_path /= relative_path
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def replace(config: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(dict(config))
    for dotted_path, value in overrides.items():
        keys = dotted_path.split(".")
        if not keys or any(not key for key in keys):
            raise ValueError(f"Invalid configuration path: {dotted_path!r}")
        target = result
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise KeyError(f"Configuration path {dotted_path!r} does not exist (failed at {key!r}).")
            target = target[key]
        leaf = keys[-1]
        if leaf not in target:
            raise KeyError(f"Configuration path {dotted_path!r} does not exist.")
        target[leaf] = deepcopy(value)
    return result


def setup_logging(log_dir, log_name="experiment.log"):
    log_path = Path(log_dir) / log_name
    log_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(fmt="[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    for handler in logger.handlers:
        handler.close()
    logger.handlers.clear()
    file_handler = logging.FileHandler(log_path, mode="x", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.info("Logging initialized. Log file: %s", log_path)


def create_timestamped_run_directory(
    base_dir: str | Path, config_name: str, *, timestamp: datetime | None = None
) -> Path:
    """Create ``<config>__<timestamp>`` and refuse any overwrite."""
    created_at = datetime.now() if timestamp is None else timestamp
    path = Path(base_dir) / f"{Path(config_name).stem}__{created_at:%Y%m%d_%H%M%S_%f}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def setup_env(experiment_name, config_path):
    """Prepare one isolated run with no cross-run data directory."""
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = project_root / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config = load_config(config_path)
    output_dir = create_timestamped_run_directory(project_root / "run" / "outputs" / experiment_name, config_path.stem)
    setup_logging(output_dir, log_name="run.log")
    logging.info("[RUN] Experiment: %s", experiment_name)
    logging.info("[RUN] Run ID: %s", output_dir.name)
    logging.info("[RUN] Output directory: %s", output_dir)
    ArtifactStore(output_dir).save_yaml("config_snapshot.yaml", config)
    return config, output_dir
