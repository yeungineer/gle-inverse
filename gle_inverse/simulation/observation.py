import numpy as np

from gle_inverse.numerics.convolution import left_rectangle_convolution


def observation_indices(raw_sample_count, observation_config):
    """Return the exact raw-grid indices selected by an observation config."""
    raw_sample_count = int(raw_sample_count)
    if raw_sample_count < 2:
        raise ValueError("The raw path must contain at least two samples.")
    stride = max(int(observation_config.get("r_obs", 1)), 1)
    length_fraction = float(observation_config.get("length_fraction", 1.0))
    if not 0.0 < length_fraction <= 1.0:
        raise ValueError("observation.length_fraction must lie in (0, 1].")
    observation_count = int(np.floor(raw_sample_count * length_fraction / stride))
    if observation_count < 2:
        raise ValueError("The observation window must contain at least two samples.")
    return np.arange(observation_count, dtype=int) * stride


def observation_horizon(t_observed):
    """Return the closed-interval horizon of a uniform observation grid."""
    t_observed = np.asarray(t_observed, dtype=float)
    if t_observed.ndim != 1 or len(t_observed) < 2:
        raise ValueError("At least two one-dimensional observation times are required.")
    increments = np.diff(t_observed)
    if increments[0] <= 0.0 or not np.allclose(increments, increments[0], rtol=0.0, atol=1.0e-12):
        raise ValueError("Observation times must form a strictly increasing uniform grid.")
    return float(t_observed[-1] - t_observed[0])


def observe_trajectory(config, observation_config, trajectory_index, *, trajectory):
    t_raw, v_raw, _, kernel_raw, mem_raw, _, _ = trajectory

    indices = observation_indices(len(t_raw), observation_config)
    sigma_obs = float(observation_config.get("sigma_obs", 0.0))
    rng = np.random.default_rng(int(config["simulation"]["seed"]) + 1000000 + int(trajectory_index))

    t_observed = t_raw[indices]
    v_observed = v_raw[indices] + sigma_obs * rng.standard_normal(len(indices))
    if sigma_obs == 0.0:
        return t_observed, v_observed, mem_raw[indices]

    dt_observed = t_observed[1] - t_observed[0]
    kernel_observed = np.asarray(kernel_raw[indices], dtype=float)
    memory_observed = left_rectangle_convolution(v_observed, kernel_observed, dt_observed)
    return t_observed, v_observed, memory_observed
