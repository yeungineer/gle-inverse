import numpy as np


def get_force(force_config, state):
    force_type = str(force_config["type"]).lower()

    if force_type == "zero":
        return np.zeros_like(state)
    if force_type == "bistable":
        return state - state**3
    if force_type == "sin":
        return np.sin(2.0 * state)

    raise ValueError(f"Unknown force type: {force_type}")
