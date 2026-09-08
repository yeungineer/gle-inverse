"""Shared configuration and postprocessing for conditional KM samples."""

import logging

import numpy as np

from gle_inverse.numerics.kramers_moyal import (
    empirical_conditional_moments,
    remove_symmetric_jump_second_moment,
    truncated_conditional_moments,
)
from gle_inverse.numerics.levy import estimate_symmetric_stable, resolve_levy_threshold
from gle_inverse.simulation.force import get_force
from gle_inverse.simulation.noises import (
    diffusion_profile,
    parse_noise_config,
    symmetric_stable_increments,
)

KM_GENERATION_EULER_ORACLE = "euler_oracle"
KM_GENERATION_FINE_BRANCH = "fine_branch"
KM_GENERATION_REPLAY_BRANCH = "replay_branch"
KM_GENERATION_METHODS = frozenset({KM_GENERATION_EULER_ORACLE, KM_GENERATION_FINE_BRANCH, KM_GENERATION_REPLAY_BRANCH})

_CONDITIONAL_PROCESS_PHASE = 101
_BRANCH_MEASUREMENT_PHASE = 102


def parse_km_generation_config(generation_config, observed_dt):
    """Validate one conditional-data generation utils."""
    observed_dt = float(observed_dt)
    if not np.isfinite(observed_dt) or observed_dt <= 0.0:
        raise ValueError("The observed KM lag must be finite and positive.")

    method = KM_GENERATION_EULER_ORACLE if generation_config is None else str(generation_config).lower()
    if method not in KM_GENERATION_METHODS:
        choices = ", ".join(sorted(KM_GENERATION_METHODS))
        raise ValueError(f"generation.km_samples must be one of: {choices}.")

    if method == KM_GENERATION_EULER_ORACLE:
        return {"method": method, "substeps": 1, "internal_dt": observed_dt}

    if method == KM_GENERATION_REPLAY_BRANCH:
        from .km_replay_branch import replay_generation_config

        return replay_generation_config(observed_dt)
    if method == KM_GENERATION_FINE_BRANCH:
        from .km_fine_branch import fine_generation_config

        return fine_generation_config(observed_dt)
    raise RuntimeError(f"Unreachable KM generation method: {method}.")


def _validate_observed_arrays(t_observed, v_observed, memory_observed):
    t_observed = np.asarray(t_observed, dtype=float)
    v_observed = np.asarray(v_observed, dtype=float)
    memory_observed = np.asarray(memory_observed, dtype=float)
    if not t_observed.shape == v_observed.shape == memory_observed.shape:
        raise ValueError("Observed time, state, and memory arrays must have the same shape.")
    if t_observed.ndim != 1 or len(t_observed) < 3:
        raise ValueError("At least three one-dimensional observations are required.")
    increments = np.diff(t_observed)
    dt = float(increments[0])
    if dt <= 0.0 or not np.allclose(increments, dt, rtol=0.0, atol=max(1.0e-12, abs(dt) * 1.0e-10)):
        raise ValueError("Observed KM times must form a strictly uniform grid.")
    return t_observed, v_observed, memory_observed, dt


def _conditional_context(config, t_observed, v_observed, memory_observed, observation_config, sample_size):
    t_observed, v_observed, memory_observed, dt = _validate_observed_arrays(t_observed, v_observed, memory_observed)
    sample_size = int(sample_size)
    if sample_size < 1:
        raise ValueError("The conditional sample size must be positive.")

    reference_stride = max(int(observation_config.get("reference_stride", 1)), 1)
    reference_indices = np.arange(1, len(v_observed) - 1, reference_stride, dtype=int)
    if len(reference_indices) == 0:
        raise ValueError("No admissible reference states remain after striding.")

    noise_model = parse_noise_config(config["noise"])
    stable_parameters = noise_model.stable_parameters() if noise_model.kind == "brownian_levy" else None
    reference_states = np.asarray(v_observed[reference_indices], dtype=float)
    return {
        "config": config,
        "t": t_observed,
        "v": v_observed,
        "memory": memory_observed,
        "dt": dt,
        "sample_size": sample_size,
        "reference_indices": reference_indices,
        "reference_states": reference_states,
        "force_values": np.asarray(get_force(config["force"], reference_states), dtype=float),
        "diffusion_values": np.asarray(diffusion_profile(reference_states, noise_model.diffusion_name()), dtype=float),
        "noise_model": noise_model,
        "stable_parameters": stable_parameters,
    }


def _restrict_context_to_full_history(context, minimum_index):
    """Drop replay targets that lack the declared observable history."""
    minimum_index = int(minimum_index)
    keep = np.asarray(context["reference_indices"]) >= minimum_index
    restricted = dict(context)
    for key in ("reference_indices", "reference_states", "force_values", "diffusion_values"):
        restricted[key] = np.asarray(context[key])[keep]
    return restricted


def _conditional_rng(config, reference_index, trajectory_index, random_seed, phase):
    seed = int(config["simulation"]["seed"] if random_seed is None else random_seed)
    identifiers = (seed, int(trajectory_index), int(reference_index), int(phase))
    return np.random.default_rng(np.random.SeedSequence(identifiers))


def _row_rng(config, reference_index, trajectory_index, random_seed):
    return _conditional_rng(config, reference_index, trajectory_index, random_seed, _CONDITIONAL_PROCESS_PHASE)


def _measurement_rng(config, reference_index, trajectory_index, random_seed):
    return _conditional_rng(config, reference_index, trajectory_index, random_seed, _BRANCH_MEASUREMENT_PHASE)


def _stable_noise(size, dt, stable_parameters, rng):
    if stable_parameters is None:
        return 0.0
    alpha, kappa = stable_parameters
    return symmetric_stable_increments(size, dt, alpha, kappa, rng)


def _postprocess_conditional_increments(
    context, delta_v, memory_values, generation, *, levy_estimation, levy_estimate, branch_start=None
):
    dt = context["dt"]
    delta_v = np.asarray(delta_v, dtype=float)
    d1_raw, d2_raw = empirical_conditional_moments(delta_v, dt)
    reference_states = context["reference_states"]
    if branch_start is None:
        branch_start = np.broadcast_to(reference_states[:, None], delta_v.shape)
    else:
        branch_start = np.asarray(branch_start, dtype=float)
    result = {
        "x": reference_states,
        "f": context["force_values"],
        "mem": np.asarray(memory_values, dtype=float),
        "delta_v": delta_v,
        "branch_start": branch_start,
        "v_next": branch_start + delta_v,
        "d1": d1_raw,
        "d2": d2_raw,
        "d1_raw": d1_raw,
        "d2_raw": d2_raw,
        "dt": dt,
        "idx": context["reference_indices"],
        "model": context["noise_model"].kind,
        "generation_method": generation["method"],
        "generation_substeps": int(generation["substeps"]),
        "generation_internal_dt": float(generation["internal_dt"]),
    }

    if context["stable_parameters"] is None:
        return result

    settings = {} if levy_estimation is None else levy_estimation
    levy_est = (
        estimate_symmetric_stable(delta_v=delta_v, dt=dt, epsilon=resolve_levy_threshold(settings))
        if levy_estimate is None
        else levy_estimate
    )
    logging.info(
        ">>> [Fixed tail MLE stable] epsilon=%.8g, q=%.3g, K=%d, counts=%s, tail_count=%s",
        levy_est["epsilon"],
        levy_est["q"],
        levy_est["K"],
        levy_est["counts"].tolist(),
        levy_est.get("tail_count", "n/a"),
    )
    d1_truncated, d2_truncated, retained = truncated_conditional_moments(delta_v, dt, levy_est["epsilon"])
    d2_corrected = remove_symmetric_jump_second_moment(d2_truncated, levy_est["small_jump_second_moment"])
    result.update(
        {
            "d1": d1_truncated,
            "d2": d2_corrected,
            "d1_truncated": d1_truncated,
            "d2_truncated": d2_truncated,
            "retained_fraction": np.mean(retained, axis=1),
            "levy": levy_est,
        }
    )
    return result


def _exponential_mode_parameters(kernel_model):
    return np.asarray(kernel_model.u, dtype=np.complex128), np.asarray(kernel_model.eta, dtype=np.complex128)
