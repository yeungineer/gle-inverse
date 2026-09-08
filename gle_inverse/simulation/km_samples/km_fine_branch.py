"""Fine-step continuation from a reconstructed reference history."""

import logging

import numpy as np

from gle_inverse.simulation.force import get_force
from gle_inverse.simulation.kernels import get_kernel
from gle_inverse.simulation.noises import brownian_increments, diffusion_profile

from .common import (
    KM_GENERATION_FINE_BRANCH,
    _conditional_context,
    _measurement_rng,
    _postprocess_conditional_increments,
    _row_rng,
    _stable_noise,
    parse_km_generation_config,
)

FINE_BRANCH_SUBSTEPS = 10


def fine_generation_config(observed_dt):
    observed_dt = float(observed_dt)
    substeps = int(FINE_BRANCH_SUBSTEPS)
    return {"method": KM_GENERATION_FINE_BRANCH, "substeps": substeps, "internal_dt": observed_dt / substeps}


def _left_rectangle_branch_memory(kernel_model, t_history, v_history, branch_history, internal_dt):
    """Evaluate branch memory using only causal left endpoints."""
    branch_step = len(branch_history) - 1
    current_time = float(t_history[-1]) + branch_step * internal_dt
    history_dt = float(t_history[1] - t_history[0])
    past_lags = current_time - t_history[:-1]
    past_kernel = np.asarray(kernel_model.value(past_lags), dtype=float)
    past_values = past_kernel * np.asarray(v_history[:-1], dtype=float)
    past_memory = float(history_dt * np.sum(past_values))
    if branch_step == 0:
        return np.full_like(branch_history[0], past_memory, dtype=float)

    branch_states = np.stack(branch_history[:branch_step], axis=0)
    branch_lags = np.arange(branch_step, 0, -1, dtype=float) * internal_dt
    branch_kernel = np.asarray(kernel_model.value(branch_lags), dtype=float)
    branch_values = branch_kernel[:, None] * branch_states
    future_memory = internal_dt * np.sum(branch_values, axis=0)
    return past_memory + future_memory


def _fine_step_euler_maruyama_branches(context, physical_v, generation, trajectory_index, random_seed):
    internal_dt = generation["internal_dt"]
    substeps = generation["substeps"]
    sample_size = context["sample_size"]
    kernel_model = get_kernel(context["config"]["kernel"], internal_dt)
    delta_v = np.empty((len(context["reference_indices"]), sample_size), dtype=float)
    memory_values = np.empty(len(context["reference_indices"]), dtype=float)

    logging.info("[DATA] fine_branch memory uses causal left-rectangle Euler--Maruyama quadrature.")
    for row, reference_index in enumerate(context["reference_indices"]):
        rng = _row_rng(context["config"], reference_index, trajectory_index, random_seed)
        state = np.full(sample_size, physical_v[reference_index], dtype=float)
        branch_history = [state.copy()]
        t_history = context["t"][: reference_index + 1]
        v_history = physical_v[: reference_index + 1]

        for substep in range(substeps):
            memory = _left_rectangle_branch_memory(kernel_model, t_history, v_history, branch_history, internal_dt)
            if substep == 0:
                memory_values[row] = float(memory[0])
            force = np.asarray(get_force(context["config"]["force"], state), dtype=float)
            diffusion = np.asarray(diffusion_profile(state, context["noise_model"].diffusion_name()), dtype=float)
            noise = diffusion * brownian_increments(sample_size, internal_dt, rng)
            noise = noise + _stable_noise(sample_size, internal_dt, context["stable_parameters"], rng)
            state = state + (force - memory) * internal_dt + noise
            branch_history.append(state.copy())

        delta_v[row] = state - physical_v[reference_index]
    return delta_v, memory_values


def compute_fine_branch_km_samples(
    config,
    t_observed,
    v_observed,
    memory_observed,
    observation_config,
    sample_size,
    generation_config,
    trajectory_index=0,
    random_seed=None,
    levy_estimation=None,
    levy_estimate=None,
    physical_v_observed=None,
):
    """Generate Euler--Maruyama endpoints from the supplied physical history."""
    context = _conditional_context(config, t_observed, v_observed, memory_observed, observation_config, sample_size)
    generation = parse_km_generation_config(generation_config, context["dt"])
    sigma_obs = float(observation_config.get("sigma_obs", 0.0))
    physical_v = np.asarray(v_observed if physical_v_observed is None else physical_v_observed, dtype=float)
    clean_delta_v, memory_values = _fine_step_euler_maruyama_branches(
        context, physical_v, generation, trajectory_index, random_seed
    )

    physical_reference = physical_v[context["reference_indices"]]
    clean_endpoints = physical_reference[:, None] + clean_delta_v
    observed_reference = context["reference_states"]
    if sigma_obs == 0.0:
        observed_endpoints = clean_endpoints
    else:
        observed_endpoints = np.empty_like(clean_endpoints)
        for row, reference_index in enumerate(context["reference_indices"]):
            measurement_rng = _measurement_rng(config, reference_index, trajectory_index, random_seed)
            observed_endpoints[row] = clean_endpoints[row] + sigma_obs * measurement_rng.standard_normal(
                context["sample_size"]
            )
    delta_v = observed_endpoints - observed_reference[:, None]

    logging.info(
        "[DATA] KM generator=%s, observed_dt=%.8g, internal_dt=%.8g, substeps=%d.",
        generation["method"],
        context["dt"],
        generation["internal_dt"],
        generation["substeps"],
    )
    return _postprocess_conditional_increments(
        context, delta_v, memory_values, generation, levy_estimation=levy_estimation, levy_estimate=levy_estimate
    )
