"""Observable-history preparation and release for approximate KM samples."""

import logging
from dataclasses import dataclass

import numpy as np

from gle_inverse.simulation.force import get_force
from gle_inverse.simulation.kernels import get_kernel
from gle_inverse.simulation.noises import (
    brownian_increments,
    diffusion_profile,
    symmetric_stable_increments,
)

from .common import (
    KM_GENERATION_REPLAY_BRANCH,
    _conditional_context,
    _exponential_mode_parameters,
    _postprocess_conditional_increments,
    _restrict_context_to_full_history,
    parse_km_generation_config,
)

REPLAY_SUBSTEPS = 10
REPLAY_CONTROL_GAIN = 60.0
REPLAY_MINIMUM_PREPARATION_TIME = 5.0

REPLAY_PREPARATION_BROWNIAN_PHASE = 201
REPLAY_PREPARATION_LEVY_PHASE = 202
REPLAY_RELEASE_BROWNIAN_PHASE = 203
REPLAY_RELEASE_LEVY_PHASE = 204


@dataclass
class ReplayParticleState:
    velocity: np.ndarray
    memory_modes: np.ndarray
    alive: np.ndarray


def replay_generation_config(observed_dt):
    observed_dt = float(observed_dt)
    substeps = int(REPLAY_SUBSTEPS)
    internal_dt = observed_dt / float(substeps)
    return {
        "method": KM_GENERATION_REPLAY_BRANCH,
        "substeps": substeps,
        "internal_dt": internal_dt,
        "replay": {
            "control_gain": float(REPLAY_CONTROL_GAIN),
            "minimum_preparation_time": float(REPLAY_MINIMUM_PREPARATION_TIME),
            "minimum_preparation_steps": int(round(REPLAY_MINIMUM_PREPARATION_TIME / observed_dt)),
        },
    }


def _reference_left_rectangle_states(velocity, observed_dt, rates):
    """Propagate reference filter states with the causal left rule."""
    velocity = np.asarray(velocity, dtype=float)
    rates = np.asarray(rates, dtype=np.complex128)
    decay = np.exp(rates * float(observed_dt))
    states = np.zeros((len(velocity), len(rates)), dtype=np.complex128)
    current = np.zeros(len(rates), dtype=np.complex128)
    for observed_index in range(1, len(velocity)):
        current = decay * (current + float(observed_dt) * float(velocity[observed_index - 1]))
        states[observed_index] = current
    return states


def _structured_rng(context, trajectory_index, random_seed, phase, *keys):
    base = int(context["config"]["simulation"]["seed"] if random_seed is None else random_seed)
    entropy = [base, int(trajectory_index), int(phase), *map(int, keys)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _build_replay_model(context, generation):
    """Build the common nonlinear particle model for every supported noise."""
    kernel_model = get_kernel(context["config"]["kernel"], generation["internal_dt"])
    parameters = _exponential_mode_parameters(kernel_model)
    weights, rates = parameters
    internal_dt = float(generation["internal_dt"])
    return {
        "weights": np.asarray(weights, dtype=np.complex128),
        "rates": np.asarray(rates, dtype=np.complex128),
        "memory_decay": np.exp(np.asarray(rates, dtype=np.complex128) * internal_dt),
        "noise_name": context["noise_model"].diffusion_name(),
        "stable_parameters": (
            context["noise_model"].stable_parameters() if context["noise_model"].kind == "brownian_levy" else None
        ),
    }


def _new_particle_state(batch_size, first_velocity, model):
    return ReplayParticleState(
        velocity=np.full(int(batch_size), float(first_velocity), dtype=float),
        memory_modes=np.zeros((int(batch_size), len(model["rates"])), dtype=np.complex128),
        alive=np.ones(int(batch_size), dtype=bool),
    )


def _advance_particles(state, context, generation, model, brownian_rng, levy_rng, *, control_target=None):
    """Advance one shard with the production causal left-rule recurrence."""
    internal_dt = float(generation["internal_dt"])
    velocity = state.velocity
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        memory = np.real(state.memory_modes @ model["weights"])
        force = np.asarray(get_force(context["config"]["force"], velocity), dtype=float)
        diffusion = np.asarray(diffusion_profile(velocity, model["noise_name"]), dtype=float)
        drift = force - memory
        if control_target is not None:
            drift = drift + float(generation["replay"]["control_gain"]) * (float(control_target) - velocity)
        noise = diffusion * brownian_increments(len(velocity), internal_dt, brownian_rng)
        if model["stable_parameters"] is not None:
            alpha, kappa = model["stable_parameters"]
            noise = noise + symmetric_stable_increments(len(velocity), internal_dt, alpha, kappa, levy_rng)
        next_velocity = velocity + drift * internal_dt + noise
        next_memory_modes = model["memory_decay"][None, :] * (state.memory_modes + internal_dt * velocity[:, None])
    finite = state.alive & np.isfinite(next_velocity) & np.all(np.isfinite(next_memory_modes), axis=1)
    newly_invalid = state.alive & ~finite
    next_velocity = np.where(finite, next_velocity, 0.0)
    next_memory_modes = np.where(finite[:, None], next_memory_modes, 0.0)
    state.velocity = next_velocity
    state.memory_modes = next_memory_modes
    state.alive = finite
    return int(np.count_nonzero(newly_invalid))


def _release_prepared(prepared, context, generation, model, brownian_rng, levy_rng):
    """Release every prepared particle with fresh Brownian and jump streams."""
    release = ReplayParticleState(
        velocity=np.asarray(prepared.velocity, dtype=float).copy(),
        memory_modes=np.asarray(prepared.memory_modes, dtype=np.complex128).copy(),
        alive=np.ones(len(prepared.velocity), dtype=bool),
    )
    initial = release.velocity.copy()
    for _ in range(int(generation["substeps"])):
        _advance_particles(release, context, generation, model, brownian_rng, levy_rng, control_target=None)
    return release.velocity - initial, release.alive


def _generate_replay_branches(context, generation, trajectory_index, random_seed):
    """Prepare exactly ``M`` particles and release all of them at every row."""
    target_velocity = np.asarray(context["v"], dtype=float)
    model = _build_replay_model(context, generation)
    sample_size = int(context["sample_size"])
    reference_indices = np.asarray(context["reference_indices"], dtype=int)
    index_to_row = {int(reference_index): row for row, reference_index in enumerate(reference_indices)}
    row_count = len(reference_indices)

    reference_modes = _reference_left_rectangle_states(target_velocity, context["dt"], model["rates"])

    branch_start = np.empty((row_count, sample_size), dtype=float)
    delta_v = np.empty_like(branch_start)

    state = _new_particle_state(sample_size, target_velocity[0], model)
    preparation_brownian_rng = _structured_rng(
        context, trajectory_index, random_seed, REPLAY_PREPARATION_BROWNIAN_PHASE, 0
    )
    preparation_levy_rng = _structured_rng(context, trajectory_index, random_seed, REPLAY_PREPARATION_LEVY_PHASE, 0)

    last_reference_index = int(reference_indices[-1])
    for observed_index in range(1, last_reference_index + 1):
        left = float(target_velocity[observed_index - 1])
        right = float(target_velocity[observed_index])
        for substep in range(int(generation["substeps"])):
            fraction = substep / float(generation["substeps"])
            target = left + fraction * (right - left)
            newly_invalid = _advance_particles(
                state, context, generation, model, preparation_brownian_rng, preparation_levy_rng, control_target=target
            )
            if newly_invalid:
                raise RuntimeError(
                    "replay_branch preparation produced non-finite particles: "
                    f"observed_index={observed_index}, substep={substep}, "
                    f"newly_invalid={newly_invalid}, "
                    f"total_invalid={np.count_nonzero(~state.alive)}/{sample_size}."
                )

        if observed_index not in index_to_row:
            continue
        row = index_to_row[observed_index]
        branch_start[row] = state.velocity

        release_brownian_rng = _structured_rng(
            context, trajectory_index, random_seed, REPLAY_RELEASE_BROWNIAN_PHASE, observed_index
        )
        release_levy_rng = _structured_rng(
            context, trajectory_index, random_seed, REPLAY_RELEASE_LEVY_PHASE, observed_index
        )
        released, release_valid = _release_prepared(
            state, context, generation, model, release_brownian_rng, release_levy_rng
        )
        invalid_releases = int(np.count_nonzero(~release_valid))
        if invalid_releases:
            raise RuntimeError(
                "replay_branch release produced non-finite increments: "
                f"reference_index={observed_index}, "
                f"invalid={invalid_releases}/{sample_size}."
            )
        delta_v[row] = released
    reference_memory = np.real(reference_modes[reference_indices] @ model["weights"])
    return delta_v, branch_start, reference_memory


def compute_replay_branch_km_samples(
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
):
    """Generate localized preparation--release increments."""
    context = _conditional_context(config, t_observed, v_observed, memory_observed, observation_config, sample_size)
    generation = parse_km_generation_config(generation_config, context["dt"])
    context = _restrict_context_to_full_history(context, generation["replay"]["minimum_preparation_steps"])
    delta_v, branch_start, memory_values = _generate_replay_branches(
        context, generation, trajectory_index, random_seed
    )
    logging.info(
        "[DATA] KM generator=%s, model=%s, diffusion=%s, observed_dt=%.8g, internal_dt=%.8g, substeps=%d, rows=%d, M=%d.",
        generation["method"],
        context["noise_model"].kind,
        context["noise_model"].diffusion_name(),
        context["dt"],
        generation["internal_dt"],
        generation["substeps"],
        len(context["reference_indices"]),
        context["sample_size"],
    )
    result = _postprocess_conditional_increments(
        context,
        delta_v,
        memory_values,
        generation,
        levy_estimation=levy_estimation,
        levy_estimate=levy_estimate,
        branch_start=branch_start,
    )
    return result
