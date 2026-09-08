"""One-step Euler-oracle conditional data for Kramers--Moyal inference."""

import logging

import numpy as np

from gle_inverse.simulation.force import get_force
from gle_inverse.simulation.noises import brownian_increments, diffusion_profile

from .common import (
    KM_GENERATION_EULER_ORACLE,
    _conditional_context,
    _measurement_rng,
    _postprocess_conditional_increments,
    _row_rng,
    _stable_noise,
    parse_km_generation_config,
)


def compute_euler_oracle_km_samples(
    config,
    t_observed,
    v_observed,
    memory_observed,
    observation_config,
    sample_size,
    trajectory_index=0,
    random_seed=None,
    levy_estimation=None,
    levy_estimate=None,
    physical_v_observed=None,
    physical_memory_observed=None,
):
    """Generate one-step Euler-oracle KM data from the physical reference history."""
    context = _conditional_context(config, t_observed, v_observed, memory_observed, observation_config, sample_size)
    sigma_obs = float(observation_config.get("sigma_obs", 0.0))
    physical_v = np.asarray(v_observed if physical_v_observed is None else physical_v_observed, dtype=float)
    physical_memory = np.asarray(
        memory_observed if physical_memory_observed is None else physical_memory_observed, dtype=float
    )
    dt = context["dt"]
    delta_v = np.empty((len(context["reference_indices"]), context["sample_size"]), dtype=float)
    memory_values = np.asarray(physical_memory[context["reference_indices"]], dtype=float)
    for row, reference_index in enumerate(context["reference_indices"]):
        rng = _row_rng(config, reference_index, trajectory_index, random_seed)
        if sigma_obs == 0.0:
            drift = (context["force_values"][row] - memory_values[row]) * dt
            noise = context["diffusion_values"][row] * brownian_increments(context["sample_size"], dt, rng)
            noise = noise + _stable_noise(context["sample_size"], dt, context["stable_parameters"], rng)
            delta_v[row] = drift + noise
        else:
            physical_reference = float(physical_v[reference_index])
            force = float(np.asarray(get_force(config["force"], physical_reference), dtype=float))
            diffusion = float(
                np.asarray(diffusion_profile(physical_reference, context["noise_model"].diffusion_name()), dtype=float)
            )
            drift = (force - memory_values[row]) * dt
            noise = diffusion * brownian_increments(context["sample_size"], dt, rng)
            noise = noise + _stable_noise(context["sample_size"], dt, context["stable_parameters"], rng)
            clean_endpoint = physical_reference + drift + noise
            measurement_rng = _measurement_rng(config, reference_index, trajectory_index, random_seed)
            observed_endpoint = clean_endpoint + sigma_obs * measurement_rng.standard_normal(context["sample_size"])
            delta_v[row] = observed_endpoint - context["reference_states"][row]

    generation = parse_km_generation_config(KM_GENERATION_EULER_ORACLE, dt)
    logging.info(
        "[DATA] KM generator=%s, observed_dt=%.8g, internal_dt=%.8g, substeps=%d.",
        generation["method"],
        dt,
        generation["internal_dt"],
        generation["substeps"],
    )
    return _postprocess_conditional_increments(
        context, delta_v, memory_values, generation, levy_estimation=levy_estimation, levy_estimate=levy_estimate
    )
