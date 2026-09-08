import logging
from pathlib import Path
from typing import Any

import numpy as np

from gle_inverse.simulation.trajectory import (
    FORWARD_MEMORY_QUADRATURE,
    generate_trajectory,
)
from gle_inverse.utils.artifacts import ArtifactStore


def generate_and_save_trajectory(
    config: dict[str, Any],
    output_dir: str | Path,
    experiment_name: str,
    trajectory_index: int,
    *,
    brownian_base_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:

    store = ArtifactStore(output_dir).prepare()
    path = store.path(f"trajectory_{int(trajectory_index):04d}.npz")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite generated trajectory: {path}")

    logging.info("[DATA] Generating trajectory %04d for %s.", int(trajectory_index), experiment_name)
    logging.info(
        "[DATA] Integrator: explicit Euler--Maruyama; forward memory quadrature: %s.", FORWARD_MEMORY_QUADRATURE
    )
    t, v, noise, kernel, memory, force, components = generate_trajectory(
        config, int(trajectory_index), brownian_base_dt=brownian_base_dt
    )
    payload = {
        "t": t,
        "v": v,
        "noise": noise,
        "kernel": kernel,
        "memory": memory,
        "force": force,
        "gaussian_increment": components["gaussian_increment"],
        "levy_increment": components["levy_increment"],
        "trajectory_index": int(trajectory_index),
    }
    np.savez_compressed(path, **payload)
    logging.info("[DATA] Saved generated trajectory: %s", path)
    return t, v, noise, kernel, memory, force, components
