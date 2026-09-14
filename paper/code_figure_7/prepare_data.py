from pathlib import Path
from typing import Any

from paper.utils.figures import resolve_code_path


def prepare_figure_data(config: dict[str, Any], config_path: Path):
    prepared = resolve_code_path(config_path, config["paths"]["data"]) / "prepared"
    return prepared / "figure_7_data.npz", prepared / "figure_7_metadata.yaml"
