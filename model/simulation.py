import numpy as np

from model.inverse import (
    empirical_conditional_moments, estimate_symmetric_stable, left_rectangle_convolution, left_rectangle_memory,
    remove_symmetric_jump_second_moment, truncated_conditional_moments,
)


def get_force(force_config, state):
    force_type = str(force_config["type"]).lower()
    if force_type == "zero":
        return np.zeros_like(state)
    if force_type == "bistable":
        return state - state**3
    if force_type == "sin":
        return np.sin(2.0 * state)
    raise ValueError(f"Unknown force type: {force_type}")


KERNEL_WEIGHTS = np.array([0.4021, 0.4021, 0.2753, 0.4638, 0.3187])
KERNEL_RATES = np.array([-0.1425 - 0.2987j, -0.1425 + 0.2987j, -0.7814 + 0j, -0.9643 + 0j, -0.4012 + 0j])


def kernel(t, kind="complex_exp"):
    t = np.asarray(t, dtype=float)
    if kind == "complex_exp":
        return (np.exp(np.outer(t, KERNEL_RATES)) @ KERNEL_WEIGHTS).real
    if kind == "power_law":
        return (1.0 - 3.0 * t**2) / (1.0 + t**2) ** 3
    raise ValueError(kind)


def brownian_increments(size, dt, rng):
    size = int(size)
    dt = float(dt)
    return np.sqrt(dt) * rng.standard_normal(size)


def symmetric_stable_increments(size, dt, alpha, kappa, rng):
    size = int(size)
    dt = float(dt)
    alpha = float(alpha)
    kappa = float(kappa)
    # CMS stable sampler: exclude angular endpoints; alpha=1 uses the Cauchy limit.
    eps = np.finfo(float).eps
    angle = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size)
    angle = np.clip(angle, -0.5 * np.pi + eps, 0.5 * np.pi - eps)
    expo = rng.exponential(1.0, size)
    if abs(alpha - 1.0) > 1e-12:
        stable = np.sin(alpha * angle) / np.cos(angle) ** (1.0 / alpha) * (np.cos((1.0 - alpha) * angle) / expo) ** ((1.0 - alpha) / alpha)
        time_scale = kappa * dt ** (1.0 / alpha)
    else:
        stable = np.tan(angle)
        time_scale = kappa * dt
    return np.asarray(time_scale * stable, dtype=float)


def diffusion_profile(state, name):
    state = np.asarray(state)
    name = str(name).lower()
    if name == "constant":
        return np.full_like(state, 0.5, dtype=float)
    if name == "cosine":
        return 0.5 * (np.cos(state) + 1.0)
    raise ValueError(f"Unknown diffusion profile: {name}")


LEVY_INCREMENT_BATCH_SIZE = 3000


def _brownian_driver_increments(size, dt, rng, base_dt=None):
    if base_dt is None:
        return brownian_increments(size, dt, rng)
    base_dt = float(base_dt)
    ratio = float(dt) / base_dt
    steps_per_increment = int(round(ratio))
    fine = brownian_increments(int(size) * steps_per_increment, base_dt, rng)
    return fine.reshape(int(size), steps_per_increment).sum(axis=1)


def generate_trajectory(config, trajectory_index, *, brownian_base_dt=None):
    simulation = config["simulation"]
    kernel_config = config["kernel"]
    noise_config = config["noise"]
    force_config = config["force"]
    interval_count = int(simulation["L"])
    trajectory_length = interval_count + 1
    dt = float(simulation["dt"])
    t = np.arange(trajectory_length, dtype=float) * dt
    seed = int(simulation["seed"]) + int(trajectory_index)
    noise_model = noise_config
    K = np.asarray(kernel(t, kernel_config["type"]), dtype=float)
    v = np.zeros(trajectory_length, dtype=float)
    mem = np.zeros(trajectory_length, dtype=float)
    force = np.zeros(trajectory_length, dtype=float)
    noise = np.zeros(trajectory_length, dtype=float)
    gaussian_increment = np.zeros(trajectory_length, dtype=float)
    levy_increment = np.zeros(trajectory_length, dtype=float)
    initial_rng = np.random.default_rng(seed)
    v[0] = initial_rng.standard_normal()
    brownian_rng = np.random.default_rng(seed)
    brownian_rng.standard_normal()
    brownian_driver = _brownian_driver_increments(trajectory_length, dt, brownian_rng, brownian_base_dt)
    if noise_model["type"] == "brownian_levy":
        alpha, kappa = (noise_model["alpha"], noise_model["kappa"])
        stable_block_size = min(LEVY_INCREMENT_BATCH_SIZE, trajectory_length)
        levy_rng = np.random.default_rng(seed)
        levy_rng.standard_normal()
        brownian_increments(stable_block_size, dt, levy_rng)
        for start in range(0, trajectory_length, stable_block_size):
            stop = min(start + stable_block_size, trajectory_length)
            levy_increment[start:stop] = symmetric_stable_increments(stop - start, dt, alpha, kappa, levy_rng)
    for index in range(trajectory_length - 1):
        mem[index] = left_rectangle_memory(K, v[: index + 1], dt)
        force[index] = float(get_force(force_config, v[index]))
        drift = (force[index] - mem[index]) * dt
        diffusion = float(diffusion_profile(v[index], noise_model["diffusion"]))
        gaussian_increment[index] = diffusion * brownian_driver[index]
        noise[index] = gaussian_increment[index] + levy_increment[index]
        v[index + 1] = v[index] + drift + noise[index]
    mem[-1] = left_rectangle_memory(K, v, dt)
    force[-1] = float(get_force(force_config, v[-1]))
    diffusion = float(diffusion_profile(v[-1], noise_model["diffusion"]))
    gaussian_increment[-1] = diffusion * brownian_driver[-1]
    noise[-1] = gaussian_increment[-1] + levy_increment[-1]
    components = {"gaussian_increment": gaussian_increment, "levy_increment": levy_increment}
    return (t, v, noise, K, mem, force, components)


def observation_indices(raw_sample_count, observation_config):
    raw_sample_count = int(raw_sample_count)
    stride = max(int(observation_config.get("r_obs", 1)), 1)
    length_fraction = float(observation_config.get("length_fraction", 1.0))
    observation_count = int(np.floor(raw_sample_count * length_fraction / stride))
    return np.arange(observation_count, dtype=int) * stride


def observation_horizon(t_observed):
    t_observed = np.asarray(t_observed, dtype=float)
    return float(t_observed[-1] - t_observed[0])


def observe_trajectory(config, observation_config, trajectory_index, *, trajectory):
    t_raw, v_raw, _, kernel_raw, mem_raw, _, _ = trajectory
    indices = observation_indices(len(t_raw), observation_config)
    sigma_obs = float(observation_config.get("sigma_obs", 0.0))
    rng = np.random.default_rng(int(config["simulation"]["seed"]) + 1000000 + int(trajectory_index))
    t_observed = t_raw[indices]
    v_observed = v_raw[indices] + sigma_obs * rng.standard_normal(len(indices))
    if sigma_obs == 0.0:
        return (t_observed, v_observed, mem_raw[indices])
    dt_observed = t_observed[1] - t_observed[0]
    kernel_observed = np.asarray(kernel_raw[indices], dtype=float)
    memory_observed = left_rectangle_convolution(v_observed, kernel_observed, dt_observed)
    return (t_observed, v_observed, memory_observed)


KM_GENERATION_EULER_ORACLE = "euler_oracle"
KM_GENERATION_FINE_BRANCH = "fine_branch"
KM_GENERATION_REPLAY_BRANCH = "replay_branch"
KM_GENERATION_METHODS = frozenset({KM_GENERATION_EULER_ORACLE, KM_GENERATION_FINE_BRANCH, KM_GENERATION_REPLAY_BRANCH})
_CONDITIONAL_PROCESS_PHASE = 101
_BRANCH_MEASUREMENT_PHASE = 102


def conditional_state(config, t_observed, v_observed, memory_observed, observation_config, sample_size):
    t_observed = np.asarray(t_observed, dtype=float)
    v_observed = np.asarray(v_observed, dtype=float)
    memory_observed = np.asarray(memory_observed, dtype=float)
    dt = float(np.diff(t_observed)[0])
    sample_size = int(sample_size)
    reference_stride = max(int(observation_config.get("reference_stride", 1)), 1)
    reference_indices = np.arange(1, len(v_observed) - 1, reference_stride, dtype=int)
    noise_model = config["noise"]
    stable_parameters = (noise_model["alpha"], noise_model["kappa"]) if noise_model["type"] == "brownian_levy" else None
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
        "diffusion_values": np.asarray(diffusion_profile(reference_states, noise_model["diffusion"]), dtype=float),
        "noise_model": noise_model,
        "stable_parameters": stable_parameters,
    }


def _conditional_rng(config, reference_index, trajectory_index, random_seed, phase):
    seed = int(config["simulation"]["seed"] if random_seed is None else random_seed)
    identifiers = (seed, int(trajectory_index), int(reference_index), int(phase))
    return np.random.default_rng(np.random.SeedSequence(identifiers))


def _stable_noise(size, dt, stable_parameters, rng):
    if stable_parameters is None:
        return 0.0
    alpha, kappa = stable_parameters
    return symmetric_stable_increments(size, dt, alpha, kappa, rng)


def conditional_moments(context, delta_v, memory_values, generation, *, levy_estimation, levy_estimate, branch_start=None):
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
        "model": context["noise_model"]["type"],
        "generation_method": generation["method"],
        "generation_substeps": int(generation["substeps"]),
        "generation_internal_dt": float(generation["internal_dt"]),
    }
    if context["stable_parameters"] is None:
        return result
    settings = {} if levy_estimation is None else levy_estimation
    levy_est = (
        estimate_symmetric_stable(delta_v=delta_v, dt=dt, epsilon=float(settings["levy_epsilon"]))
        if levy_estimate is None
        else levy_estimate
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


def _oracle(context, physical_v, physical_memory, sigma_obs, trajectory_index, random_seed):
    config = context["config"]
    dt = context["dt"]
    delta_v = np.empty((len(context["reference_indices"]), context["sample_size"]), dtype=float)
    memory_values = np.asarray(physical_memory[context["reference_indices"]], dtype=float)
    for row, reference_index in enumerate(context["reference_indices"]):
        rng = _conditional_rng(config, reference_index, trajectory_index, random_seed, 101)
        if sigma_obs == 0.0:
            drift = (context["force_values"][row] - memory_values[row]) * dt
            noise = context["diffusion_values"][row] * brownian_increments(context["sample_size"], dt, rng)
            noise = noise + _stable_noise(context["sample_size"], dt, context["stable_parameters"], rng)
            delta_v[row] = drift + noise
        else:
            physical_reference = float(physical_v[reference_index])
            force = float(np.asarray(get_force(config["force"], physical_reference), dtype=float))
            diffusion = float(np.asarray(diffusion_profile(physical_reference, context["noise_model"]["diffusion"]), dtype=float))
            drift = (force - memory_values[row]) * dt
            noise = diffusion * brownian_increments(context["sample_size"], dt, rng)
            noise = noise + _stable_noise(context["sample_size"], dt, context["stable_parameters"], rng)
            clean_endpoint = physical_reference + drift + noise
            measurement_rng = _conditional_rng(config, reference_index, trajectory_index, random_seed, 102)
            observed_endpoint = clean_endpoint + sigma_obs * measurement_rng.standard_normal(context["sample_size"])
            delta_v[row] = observed_endpoint - context["reference_states"][row]
    return (delta_v, memory_values)


FINE_BRANCH_SUBSTEPS = 10


def _left_rectangle_branch_memory(kernel_type, t_history, v_history, branch_history, internal_dt):
    branch_step = len(branch_history) - 1
    current_time = float(t_history[-1]) + branch_step * internal_dt
    history_dt = float(t_history[1] - t_history[0])
    past_lags = current_time - t_history[:-1]
    past_kernel = np.asarray(kernel(past_lags, kernel_type), dtype=float)
    past_values = past_kernel * np.asarray(v_history[:-1], dtype=float)
    past_memory = float(history_dt * np.sum(past_values))
    if branch_step == 0:
        return np.full_like(branch_history[0], past_memory, dtype=float)
    branch_states = np.stack(branch_history[:branch_step], axis=0)
    branch_lags = np.arange(branch_step, 0, -1, dtype=float) * internal_dt
    branch_kernel = np.asarray(kernel(branch_lags, kernel_type), dtype=float)
    branch_values = branch_kernel[:, None] * branch_states
    future_memory = internal_dt * np.sum(branch_values, axis=0)
    return past_memory + future_memory


def _fine_step_euler_maruyama_branches(context, physical_v, generation, trajectory_index, random_seed):
    internal_dt = generation["internal_dt"]
    substeps = generation["substeps"]
    sample_size = context["sample_size"]
    kernel_type = context["config"]["kernel"]["type"]
    delta_v = np.empty((len(context["reference_indices"]), sample_size), dtype=float)
    memory_values = np.empty(len(context["reference_indices"]), dtype=float)
    for row, reference_index in enumerate(context["reference_indices"]):
        rng = _conditional_rng(context["config"], reference_index, trajectory_index, random_seed, 101)
        state = np.full(sample_size, physical_v[reference_index], dtype=float)
        branch_history = [state.copy()]
        t_history = context["t"][: reference_index + 1]
        v_history = physical_v[: reference_index + 1]
        for substep in range(substeps):
            memory = _left_rectangle_branch_memory(kernel_type, t_history, v_history, branch_history, internal_dt)
            if substep == 0:
                memory_values[row] = float(memory[0])
            force = np.asarray(get_force(context["config"]["force"], state), dtype=float)
            diffusion = np.asarray(diffusion_profile(state, context["noise_model"]["diffusion"]), dtype=float)
            noise = diffusion * brownian_increments(sample_size, internal_dt, rng)
            noise = noise + _stable_noise(sample_size, internal_dt, context["stable_parameters"], rng)
            state = state + (force - memory) * internal_dt + noise
            branch_history.append(state.copy())
        delta_v[row] = state - physical_v[reference_index]
    return (delta_v, memory_values)


REPLAY_SUBSTEPS = 10
REPLAY_CONTROL_GAIN = 60.0
REPLAY_MINIMUM_PREPARATION_TIME = 5.0
REPLAY_PREPARATION_BROWNIAN_PHASE = 201
REPLAY_PREPARATION_LEVY_PHASE = 202
REPLAY_RELEASE_BROWNIAN_PHASE = 203
REPLAY_RELEASE_LEVY_PHASE = 204


def _reference_left_rectangle_states(velocity, observed_dt, rates):
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
    assert context["config"]["kernel"]["type"] == "complex_exp", "Replay requires exponential memory modes."
    weights = np.asarray(KERNEL_WEIGHTS, dtype=np.complex128)
    rates = np.asarray(KERNEL_RATES, dtype=np.complex128)
    internal_dt = float(generation["internal_dt"])
    return {
        "weights": np.asarray(weights, dtype=np.complex128),
        "rates": np.asarray(rates, dtype=np.complex128),
        "memory_decay": np.exp(np.asarray(rates, dtype=np.complex128) * internal_dt),
        "noise_name": context["noise_model"]["diffusion"],
        "stable_parameters": (
            (context["noise_model"]["alpha"], context["noise_model"]["kappa"])
            if context["noise_model"]["type"] == "brownian_levy"
            else None
        ),
    }


def _new_particle_state(batch_size, first_velocity, model):
    return {
        "velocity": np.full(int(batch_size), float(first_velocity), dtype=float),
        "memory_modes": np.zeros((int(batch_size), len(model["rates"])), dtype=np.complex128),
        "alive": np.ones(int(batch_size), dtype=bool),
    }


def _advance_particles(state, context, generation, model, brownian_rng, levy_rng, *, control_target=None):
    internal_dt = float(generation["internal_dt"])
    velocity = state["velocity"]
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        memory = np.real(state["memory_modes"] @ model["weights"])
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
        next_memory_modes = model["memory_decay"][None, :] * (state["memory_modes"] + internal_dt * velocity[:, None])
    finite = state["alive"] & np.isfinite(next_velocity) & np.all(np.isfinite(next_memory_modes), axis=1)
    newly_invalid = state["alive"] & ~finite
    next_velocity = np.where(finite, next_velocity, 0.0)
    next_memory_modes = np.where(finite[:, None], next_memory_modes, 0.0)
    state["velocity"] = next_velocity
    state["memory_modes"] = next_memory_modes
    state["alive"] = finite
    return int(np.count_nonzero(newly_invalid))


def _release_prepared(prepared, context, generation, model, brownian_rng, levy_rng):
    """Release every prepared particle with fresh Brownian and jump streams."""
    release = {
        "velocity": np.asarray(prepared["velocity"], dtype=float).copy(),
        "memory_modes": np.asarray(prepared["memory_modes"], dtype=np.complex128).copy(),
        "alive": np.ones(len(prepared["velocity"]), dtype=bool),
    }
    initial = release["velocity"].copy()
    for _ in range(int(generation["substeps"])):
        _advance_particles(release, context, generation, model, brownian_rng, levy_rng, control_target=None)
    return (release["velocity"] - initial, release["alive"])


def _generate_replay_branches(context, generation, trajectory_index, random_seed):
    target_velocity = np.asarray(context["v"], dtype=float)
    model = _build_replay_model(context, generation)
    sample_size = int(context["sample_size"])
    reference_indices = np.asarray(context["reference_indices"], dtype=int)
    index_to_row = {int(reference_index): row for (row, reference_index) in enumerate(reference_indices)}
    row_count = len(reference_indices)
    reference_modes = _reference_left_rectangle_states(target_velocity, context["dt"], model["rates"])
    branch_start = np.empty((row_count, sample_size), dtype=float)
    delta_v = np.empty_like(branch_start)
    state = _new_particle_state(sample_size, target_velocity[0], model)
    preparation_brownian_rng = _structured_rng(context, trajectory_index, random_seed, REPLAY_PREPARATION_BROWNIAN_PHASE, 0)
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
                    f"replay_branch preparation produced non-finite particles: observed_index={observed_index}, substep={substep}, newly_invalid={newly_invalid}, total_invalid={np.count_nonzero(~state['alive'])}/{sample_size}."
                )
        if observed_index not in index_to_row:
            continue
        row = index_to_row[observed_index]
        branch_start[row] = state["velocity"]
        release_brownian_rng = _structured_rng(context, trajectory_index, random_seed, REPLAY_RELEASE_BROWNIAN_PHASE, observed_index)
        release_levy_rng = _structured_rng(context, trajectory_index, random_seed, REPLAY_RELEASE_LEVY_PHASE, observed_index)
        released, release_valid = _release_prepared(state, context, generation, model, release_brownian_rng, release_levy_rng)
        invalid_releases = int(np.count_nonzero(~release_valid))
        if invalid_releases:
            raise RuntimeError(
                f"replay_branch release produced non-finite increments: reference_index={observed_index}, invalid={invalid_releases}/{sample_size}."
            )
        delta_v[row] = released
    reference_memory = np.real(reference_modes[reference_indices] @ model["weights"])
    return (delta_v, branch_start, reference_memory)


def sample_conditional(
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
    generation_config=None,
    physical_v_observed=None,
    physical_memory_observed=None,
):
    """Three history-conditioning protocols, followed by the same moment estimator."""
    context = conditional_state(config, t_observed, v_observed, memory_observed, observation_config, sample_size)
    dt = context["dt"]
    method = "euler_oracle" if generation_config is None else str(generation_config).lower()
    substeps = 1 if method == "euler_oracle" else 10
    generation = {"method": method, "substeps": substeps, "internal_dt": dt / substeps}
    sigma_obs = float(observation_config.get("sigma_obs", 0.0))
    physical_v = np.asarray(v_observed if physical_v_observed is None else physical_v_observed, dtype=float)
    physical_memory = np.asarray(memory_observed if physical_memory_observed is None else physical_memory_observed, dtype=float)
    branch_start = None
    if method == "euler_oracle":
        delta_v, memory_values = _oracle(context, physical_v, physical_memory, sigma_obs, trajectory_index, random_seed)
    elif method == "fine_branch":
        clean_delta, memory_values = _fine_step_euler_maruyama_branches(context, physical_v, generation, trajectory_index, random_seed)
        clean_endpoints = physical_v[context["reference_indices"]][:, None] + clean_delta
        if sigma_obs == 0.0:
            endpoints = clean_endpoints
        else:
            endpoints = np.empty_like(clean_endpoints)
            for row, idx in enumerate(context["reference_indices"]):
                rng = _conditional_rng(config, idx, trajectory_index, random_seed, 102)
                endpoints[row] = clean_endpoints[row] + sigma_obs * rng.standard_normal(context["sample_size"])
        delta_v = endpoints - context["reference_states"][:, None]
    elif method == "replay_branch":
        generation["replay"] = {"control_gain": 60.0, "minimum_preparation_time": 5.0, "minimum_preparation_steps": int(round(5.0 / dt))}
        keep = context["reference_indices"] >= generation["replay"]["minimum_preparation_steps"]
        for key in ("reference_indices", "reference_states", "force_values", "diffusion_values"):
            context[key] = context[key][keep]
        delta_v, branch_start, memory_values = _generate_replay_branches(context, generation, trajectory_index, random_seed)
    else:
        raise ValueError(method)
    return conditional_moments(
        context, delta_v, memory_values, generation, levy_estimation=levy_estimation, levy_estimate=levy_estimate, branch_start=branch_start
    )
