import numpy as np
from scipy.linalg import expm, hankel, null_space, toeplitz
from scipy.signal import find_peaks
from scipy.stats import theilslopes

from model.fit import relative_l2_error
from model.inverse import SplineBasis, select_lambda
from model.simulation import KERNEL_RATES, KERNEL_WEIGHTS, diffusion_profile, kernel

# Minimum decay for stable Prony roots.
ROOT_NORMALIZATION_EPSILON = 1e-06
REAL_ROOT_TOLERANCE = 1e-12
SOBOLEV_R1GCV_ROBUSTNESS = 0.99999


def _spectral_coefficients(selection):
    singular_values = np.asarray(selection["singular_values"], dtype=float)
    modal_data = singular_values / (singular_values**2 + float(selection["lambda"])) * np.asarray(selection["projected_data"], dtype=float)
    return np.asarray(selection["coefficient_modes"], dtype=float) @ modal_data


def _real_system(response, data, penalty):
    response = np.asarray(response, dtype=np.complex128)
    data = np.asarray(data, dtype=np.complex128).reshape(-1)
    penalty = np.asarray(penalty, dtype=np.complex128)
    response = np.block([[response.real, -response.imag], [response.imag, response.real]])
    data = np.concatenate((data.real, data.imag))
    penalty = np.block([[penalty.real, -penalty.imag], [penalty.imag, penalty.real]])
    penalty = 0.5 * (penalty + penalty.T)
    eigenvalues, eigenvectors = np.linalg.eigh(penalty)
    penalty = eigenvectors @ np.diag(np.maximum(eigenvalues, 0.0)) @ eigenvectors.T
    return (response, data, penalty)


def _decay_rate(correlation, time):
    amplitude = np.abs(np.asarray(correlation, dtype=float))
    time = np.asarray(time, dtype=float)
    minimum = ROOT_NORMALIZATION_EPSILON / float(time[1] - time[0])
    tail_count = min(max(4, len(amplitude) // 6), len(amplitude))
    noise_floor = max(float(np.median(amplitude[-tail_count:])), np.finfo(float).tiny)
    peaks = np.unique(np.concatenate(([0], find_peaks(amplitude)[0])))
    peaks = peaks[amplitude[peaks] > 2.0 * noise_floor]
    if len(peaks) >= 3:
        rate = -float(theilslopes(np.log(amplitude[peaks]), time[peaks]).slope)
    else:
        indices = np.flatnonzero(amplitude > 2.0 * noise_floor)
        if len(indices) >= 2 and time[indices[-1]] > time[indices[0]]:
            rate = -float(np.log(amplitude[indices[-1]] / amplitude[indices[0]]) / (time[indices[-1]] - time[indices[0]]))
        else:
            rate = minimum
    if not np.isfinite(rate) or rate <= 0.0:
        rate = minimum
    return max(float(rate), minimum)


def _regularized_rates(correlation, time, modes, decay_rate):
    dt = float(time[1] - time[0])
    correlation = np.asarray(correlation, dtype=np.complex128)
    count = len(correlation)
    pencil = hankel(correlation[: count - modes], correlation[count - modes - 1 :])
    roots = np.linalg.eigvals(np.linalg.pinv(pencil[:, :-1]) @ pencil[:, 1:])
    maximum_magnitude = float(np.exp(-float(decay_rate) * dt))
    for index, root in enumerate(roots):
        magnitude = float(np.abs(root))
        if magnitude > maximum_magnitude:
            roots[index] = root * (maximum_magnitude / magnitude)
    rates = []
    for root in roots:
        real_negative = np.isclose(root.imag, 0.0, rtol=0.0, atol=REAL_ROOT_TOLERANCE * max(1.0, abs(root.real))) and root.real < 0.0
        if real_negative:
            log_magnitude = np.log(abs(root))
            rates.extend(((log_magnitude + 1j * np.pi) / dt, (log_magnitude - 1j * np.pi) / dt))
        else:
            rates.append(np.log(root) / dt)
    return np.asarray(rates, dtype=np.complex128)


def _prony_functions(correlation, time, modes, drop_zero):
    correlation = np.asarray(correlation, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    first = 1 if drop_zero and np.isclose(time[0], 0.0, rtol=0.0, atol=1e-14) else 0
    samples = correlation[first:]
    sample_time = time[first:]
    rates = _regularized_rates(samples, sample_time, modes, _decay_rate(correlation, time))
    design = np.exp(np.outer(sample_time, rates))
    constraint = null_space(rates[None, :])
    reduced_design = design @ constraint
    penalty = np.linalg.pinv(design.conj().T @ design, hermitian=True)
    penalty = constraint.conj().T @ penalty @ constraint
    response, data, penalty = _real_system(reduced_design, samples, penalty)
    # robustness=1 is ordinary GCV.
    selection = select_lambda(response, data, penalty, robustness=1.0)
    reduced = _spectral_coefficients(selection)
    size = reduced_design.shape[1]
    weights = constraint @ (reduced[:size] + 1j * reduced[size:])

    def evaluate(t, power):
        return np.real(np.exp(np.outer(np.asarray(t, dtype=float).reshape(-1), rates)) @ (weights * rates**power))

    h = lambda t: evaluate(t, 0)
    g = lambda t: -evaluate(t, 1)
    gp = lambda t: -evaluate(t, 2)
    h0 = float(h(np.asarray([0.0]))[0])
    integral = float(np.real(-np.sum(weights / rates)))
    derivative_weight = integral**2 / (h0**2 + integral**2)
    return (h, g, gp, derivative_weight)


def _sobolev_estimator(h, g, gp, settings, derivative_weight):
    t = np.linspace(0.0, float(settings["t_max"]), int(settings["grid_count"]))
    dt = float(t[1] - t[0])
    h_values = np.asarray(h(t), dtype=float)
    g_values = np.asarray(g(t), dtype=float)
    gp_values = np.asarray(gp(t), dtype=float)
    basis = SplineBasis(n_basis=int(settings["knot_count"]) + 3, t_max=float(settings["t_max"]), knot_power=1.0)
    phi = basis.matrix(t)
    row = np.zeros_like(h_values)
    row[0] = h_values[0]
    convolution = toeplitz(h_values, row)
    convolution[:, 0] *= 0.5
    np.fill_diagonal(convolution, convolution.diagonal() * 0.5)
    convolution[0, :] = 0.0
    psi = convolution @ phi * dt
    psi_prime = np.gradient(psi, dt, axis=0)
    alpha_derivative = float(derivative_weight)
    alpha_space = 1.0 - alpha_derivative
    quadrature = np.exp(-2.0 * float(settings["omega"]) * t) * dt
    response = np.vstack((np.sqrt(alpha_space * quadrature)[:, None] * psi, np.sqrt(alpha_derivative * quadrature)[:, None] * psi_prime))
    data = np.concatenate((np.sqrt(alpha_space * quadrature) * g_values, np.sqrt(alpha_derivative * quadrature) * gp_values))
    penalty = np.linalg.pinv(response.T @ response, hermitian=True)
    selection = select_lambda(response, data, penalty, robustness=SOBOLEV_R1GCV_ROBUSTNESS)
    coefficients = _spectral_coefficients(selection)
    return lambda evaluation_time: basis.matrix(np.asarray(evaluation_time, dtype=float)) @ coefficients


def reconstruct_prony_sobolev(correlation_time, correlation, config):
    prony = config["prony"]
    h, g, gp, derivative_weight = _prony_functions(correlation, correlation_time, int(prony["modes"]), bool(prony["drop_zero"]))
    settings = config["sobolev"]
    time = np.linspace(0.0, float(settings["t_max"]), int(settings["grid_count"]))
    estimate = _sobolev_estimator(h, g, gp, settings, derivative_weight)(time)
    return (time, np.asarray(estimate, dtype=float))


def _ensemble_correlation(config):
    M, L, dt = (int(config["M"]), int(config["L"]), float(config["dt"]))
    settings = config["prony"]
    sample_indices = np.arange(int(settings["sample_count"]), dtype=np.int64) * int(settings["lag_stride"])
    weights = np.asarray(KERNEL_WEIGHTS, dtype=np.complex128)
    memory_decay = np.exp(np.asarray(KERNEL_RATES, dtype=np.complex128) * dt)
    rng = np.random.default_rng(int(config["seed"]))
    initial_velocity = float(config["initial_velocity_std"]) * rng.standard_normal(M)
    velocity = initial_velocity.copy()
    memory_modes = np.zeros((M, len(weights)), dtype=np.complex128)
    brownian_scale = float(diffusion_profile(np.asarray([0.0]), str(config["diffusion"]))[0]) * np.sqrt(dt)
    correlation = np.empty(len(sample_indices), dtype=float)
    correlation[0] = float(np.mean(initial_velocity * velocity))
    next_sample = 1
    for step in range(L):
        memory = np.real(memory_modes @ weights)
        next_velocity = velocity - memory * dt + brownian_scale * rng.standard_normal(M)
        memory_modes = memory_decay[None, :] * (memory_modes + dt * velocity[:, None])
        velocity = next_velocity
        if next_sample < len(sample_indices) and step + 1 == sample_indices[next_sample]:
            correlation[next_sample] = float(np.mean(initial_velocity * velocity))
            next_sample += 1
    return (sample_indices.astype(float) * dt, correlation)


def run_prony_sobolev(config):
    correlation_time, correlation = _ensemble_correlation(config)
    kernel_time, kernel_estimate = reconstruct_prony_sobolev(correlation_time, correlation, config)
    kernel_truth = np.asarray(kernel(kernel_time, str(config["kernel_type"])), dtype=float)
    _, relative_error = relative_l2_error(kernel_estimate, kernel_truth, kernel_time)
    arrays = {
        "kernel_t": kernel_time,
        "kernel_truth": kernel_truth,
        "kernel_estimate": kernel_estimate,
        "correlation_t": correlation_time,
        "correlation_samples": correlation,
    }
    return (arrays, {"kernel_relative_l2_error": float(relative_error)})


"Compact reproduction of Vroylandt et al.'s likelihood--EM estimator."


def _symmetrize(matrix):
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)


def _positive_definite(matrix, relative_floor):
    """Apply only a numerical eigenvalue floor, not a statistical penalty."""
    matrix = _symmetrize(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = float(relative_floor) * scale
    eigenvalues = np.maximum(eigenvalues, floor)
    return _symmetrize(eigenvectors * eigenvalues[None, :] @ eigenvectors.T)


def _solve_spd(matrix, right_hand_side):
    factor = np.linalg.cholesky(_symmetrize(matrix))
    return np.linalg.solve(factor.T, np.linalg.solve(factor, right_hand_side))


def generate_common_velocity_ensemble(config):
    """Generate the common Brownian GLE with causal left-rule memory."""
    trajectory_count = int(config["M"])
    interval_count = int(config["L"])
    dt = float(config["dt"])
    weights = np.asarray(KERNEL_WEIGHTS, dtype=np.complex128)
    decay = np.exp(np.asarray(KERNEL_RATES, dtype=np.complex128) * dt)
    rng = np.random.default_rng(int(config["seed"]))
    velocity = np.empty((interval_count + 1, trajectory_count), dtype=float)
    velocity[0] = rng.standard_normal(trajectory_count)
    memory_modes = np.zeros((trajectory_count, len(weights)), dtype=np.complex128)
    brownian_scale = float(diffusion_profile(np.asarray([0.0]), str(config["diffusion"]))[0]) * np.sqrt(dt)
    for step in range(interval_count):
        current = velocity[step]
        memory = np.real(memory_modes @ weights)
        velocity[step + 1] = current - dt * memory + brownian_scale * rng.standard_normal(trajectory_count)
        memory_modes = decay[None, :] * (memory_modes + dt * current[:, None])
    return velocity


def _initialize(velocity, dt, settings):
    """Seeded stable initialization."""
    velocity = np.asarray(velocity, dtype=float)
    current = velocity[:-1].reshape(-1)
    following = velocity[1:].reshape(-1)
    denominator = max(float(current @ current), np.finfo(float).tiny)
    # Stable AR coefficient and positive innovation variance.
    visible_ar = float(np.clip(following @ current / denominator, -0.98, 0.9995))
    innovation = following - visible_ar * current
    visible_variance = max(float(np.mean(innovation**2)), 1e-08)
    rng = np.random.default_rng(settings["seed"])
    d = settings["hidden_dimension"]
    horizon = float(dt) * float(velocity.shape[0] - 1)
    slow_rate = max(1.0 / max(horizon, dt), 0.025)
    fast_rate = min(2.0 / dt, 4.0)
    rates = np.geomspace(slow_rate, fast_rate, d)
    rates *= np.exp(0.18 * rng.standard_normal(d))
    hidden_transition = np.diag(np.exp(-dt * rates))
    continuous_scale = max(np.sqrt(max(1.0 - visible_ar, 0.001) / dt), 0.2)
    visible_hidden = -dt * continuous_scale * rng.standard_normal((1, d)) / np.sqrt(d)
    hidden_visible = -dt * continuous_scale * rng.standard_normal((d, 1)) / np.sqrt(d)
    transition = np.block([[np.asarray([[visible_ar]]), visible_hidden], [hidden_visible, hidden_transition]])
    covariance = np.eye(d + 1, dtype=float) * visible_variance
    covariance[1:, 1:] *= 0.5
    covariance = _positive_definite(covariance, settings["covariance_floor"])
    initial_hidden_mean = np.zeros(d, dtype=float)
    return (transition, covariance, initial_hidden_mean)


def _e_step(velocity, transition, covariance, initial_hidden_mean, settings):
    """Exact Kalman conditioning and RTS smoothing for all trajectories."""
    velocity = np.asarray(velocity, dtype=float)
    interval_count = velocity.shape[0] - 1
    trajectory_count = velocity.shape[1]
    d = settings["hidden_dimension"]
    if transition.shape != (d + 1, d + 1) or covariance.shape != (d + 1, d + 1):
        raise ValueError("The extended transition matrices have inconsistent dimensions.")
    f_vv = float(transition[0, 0])
    f_vh = np.asarray(transition[0, 1:], dtype=float)
    f_hv = np.asarray(transition[1:, 0], dtype=float)
    f_hh = np.asarray(transition[1:, 1:], dtype=float)
    q_vv = float(covariance[0, 0])
    q_hv = np.asarray(covariance[1:, 0], dtype=float)
    q_hh = np.asarray(covariance[1:, 1:], dtype=float)
    filtered_mean = np.empty((interval_count + 1, trajectory_count, d), dtype=float)
    filtered_covariance = np.empty((interval_count + 1, d, d), dtype=float)
    filtered_mean[0] = np.asarray(initial_hidden_mean, dtype=float)[None, :]
    filtered_covariance[0] = np.eye(d, dtype=float)
    log_likelihood = 0.0
    for step in range(interval_count):
        mean = filtered_mean[step]
        p = filtered_covariance[step]
        p_fvh = p @ f_vh
        hidden_observation_covariance = f_hh @ p_fvh + q_hv
        innovation_variance = float(f_vh @ p_fvh + q_vv)
        if not np.isfinite(innovation_variance) or innovation_variance <= 0.0:
            raise FloatingPointError("The EM innovation covariance is not positive.")
        predicted_visible = f_vv * velocity[step] + mean @ f_vh
        innovation = velocity[step + 1] - predicted_visible
        predicted_hidden = velocity[step, :, None] * f_hv[None, :] + mean @ f_hh.T
        gain = hidden_observation_covariance / innovation_variance
        filtered_mean[step + 1] = predicted_hidden + innovation[:, None] * gain[None, :]
        filtered_covariance[step + 1] = _positive_definite(
            f_hh @ p @ f_hh.T + q_hh - np.outer(hidden_observation_covariance, hidden_observation_covariance) / innovation_variance,
            settings["covariance_floor"],
        )
        log_likelihood += -0.5 * (
            trajectory_count * np.log(2.0 * np.pi * innovation_variance) + float(innovation @ innovation) / innovation_variance
        )
    smoothed_mean_next = filtered_mean[-1].copy()
    smoothed_covariance_next = filtered_covariance[-1].copy()
    s_xx = np.zeros((d + 1, d + 1), dtype=float)
    s_yx = np.zeros((d + 1, d + 1), dtype=float)
    s_yy = np.zeros((d + 1, d + 1), dtype=float)
    for step in range(interval_count - 1, -1, -1):
        filtered_at_step = filtered_mean[step].copy()
        p = filtered_covariance[step]
        p_fvh = p @ f_vh
        innovation_variance = float(f_vh @ p_fvh + q_vv)
        hidden_observation_covariance = f_hh @ p_fvh + q_hv
        innovation = velocity[step + 1] - (f_vv * velocity[step] + filtered_at_step @ f_vh)
        updated_mean_at_step = filtered_at_step + innovation[:, None] * (p_fvh / innovation_variance)[None, :]
        updated_covariance_at_step = _symmetrize(p - np.outer(p_fvh, p_fvh) / innovation_variance)
        cross_after_observation = p @ f_hh.T - np.outer(p_fvh, hidden_observation_covariance) / innovation_variance
        smoother_gain = _solve_spd(filtered_covariance[step + 1], cross_after_observation.T).T
        next_mean = smoothed_mean_next
        current_mean = updated_mean_at_step + (next_mean - filtered_mean[step + 1]) @ smoother_gain.T
        current_covariance = _symmetrize(
            updated_covariance_at_step + smoother_gain @ (smoothed_covariance_next - filtered_covariance[step + 1]) @ smoother_gain.T
        )
        cross_current_next = smoother_gain @ smoothed_covariance_next
        visible_current = velocity[step]
        visible_next = velocity[step + 1]
        x_mean = np.column_stack((visible_current, current_mean))
        y_mean = np.column_stack((visible_next, next_mean))
        s_xx += x_mean.T @ x_mean
        s_yy += y_mean.T @ y_mean
        s_yx += y_mean.T @ x_mean
        s_xx[1:, 1:] += trajectory_count * current_covariance
        s_yy[1:, 1:] += trajectory_count * smoothed_covariance_next
        s_yx[1:, 1:] += trajectory_count * cross_current_next.T
        smoothed_mean_next = current_mean
        smoothed_covariance_next = current_covariance
    return {
        "s_xx": _symmetrize(s_xx),
        "s_yx": s_yx,
        "s_yy": _symmetrize(s_yy),
        "initial_hidden_mean_sum": np.sum(smoothed_mean_next, axis=0),
        "transition_count": int(interval_count * trajectory_count),
        "trajectory_count": int(trajectory_count),
        "log_likelihood": float(log_likelihood),
    }


def _m_step(sufficient, settings):
    """Closed-form maximizer of the paper's evidence lower bound."""
    ridge = np.finfo(float).eps * max(float(np.trace(sufficient["s_xx"])), 1.0)
    s_xx = sufficient["s_xx"] + ridge * np.eye(sufficient["s_xx"].shape[0])
    transition = np.linalg.solve(s_xx, sufficient["s_yx"].T).T
    covariance = (
        sufficient["s_yy"]
        - transition @ sufficient["s_yx"].T
        - sufficient["s_yx"] @ transition.T
        + transition @ sufficient["s_xx"] @ transition.T
    ) / float(sufficient["transition_count"])
    covariance = _positive_definite(covariance, settings["covariance_floor"])
    initial_hidden_mean = sufficient["initial_hidden_mean_sum"] / float(sufficient["trajectory_count"])
    return (transition, covariance, initial_hidden_mean)


def _observed_log_likelihood(velocity, transition, covariance, initial_hidden_mean, settings):
    interval_count = velocity.shape[0] - 1
    trajectory_count = velocity.shape[1]
    d = settings["hidden_dimension"]
    f_vv = float(transition[0, 0])
    f_vh = np.asarray(transition[0, 1:], dtype=float)
    f_hv = np.asarray(transition[1:, 0], dtype=float)
    f_hh = np.asarray(transition[1:, 1:], dtype=float)
    q_vv = float(covariance[0, 0])
    q_hv = np.asarray(covariance[1:, 0], dtype=float)
    q_hh = np.asarray(covariance[1:, 1:], dtype=float)
    mean = np.broadcast_to(initial_hidden_mean, (trajectory_count, d)).copy()
    p = np.eye(d, dtype=float)
    log_likelihood = 0.0
    for step in range(interval_count):
        p_fvh = p @ f_vh
        hidden_observation_covariance = f_hh @ p_fvh + q_hv
        innovation_variance = float(f_vh @ p_fvh + q_vv)
        innovation = velocity[step + 1] - (f_vv * velocity[step] + mean @ f_vh)
        predicted_hidden = velocity[step, :, None] * f_hv[None, :] + mean @ f_hh.T
        mean = predicted_hidden + innovation[:, None] * (hidden_observation_covariance / innovation_variance)[None, :]
        p = _positive_definite(
            f_hh @ p @ f_hh.T + q_hh - np.outer(hidden_observation_covariance, hidden_observation_covariance) / innovation_variance,
            settings["covariance_floor"],
        )
        log_likelihood += -0.5 * (
            trajectory_count * np.log(2.0 * np.pi * innovation_variance) + float(innovation @ innovation) / innovation_variance
        )
    return float(log_likelihood)


def fit_em(velocity, dt, settings):
    velocity = np.asarray(velocity, dtype=float)
    if velocity.ndim != 2 or velocity.shape[0] < 3 or velocity.shape[1] < 2:
        raise ValueError("velocity must have shape (L+1, M) with L,M >= 2.")
    if not np.all(np.isfinite(velocity)) or float(dt) <= 0.0:
        raise ValueError("The likelihood data and timestep must be finite.")
    transition, covariance, initial_hidden_mean = _initialize(velocity, float(dt), settings)
    likelihood_history = []
    converged = False
    completed_updates = 0
    for iteration in range(settings["max_iterations"]):
        sufficient = _e_step(velocity, transition, covariance, initial_hidden_mean, settings)
        likelihood_history.append(sufficient["log_likelihood"])
        if len(likelihood_history) >= 2:
            improvement = likelihood_history[-1] - likelihood_history[-2]
            numerical_allowance = 1e-09 * max(abs(likelihood_history[-2]), 1.0)
            if improvement < -numerical_allowance:
                raise FloatingPointError(
                    "The observed likelihood decreased beyond numerical tolerance; the EM sufficient statistics are inconsistent."
                )
            if iteration + 1 >= settings["minimum_iterations"] and abs(improvement) < settings["likelihood_tolerance"]:
                converged = True
                break
        transition, covariance, initial_hidden_mean = _m_step(sufficient, settings)
        completed_updates += 1
    if not converged:
        likelihood_history.append(_observed_log_likelihood(velocity, transition, covariance, initial_hidden_mean, settings))
    return {
        "transition": np.asarray(transition, dtype=float),
        "covariance": np.asarray(covariance, dtype=float),
        "initial_hidden_mean": np.asarray(initial_hidden_mean, dtype=float),
        "log_likelihood": np.asarray(likelihood_history, dtype=float),
        "converged": bool(converged),
        "iterations": int(completed_updates),
    }


def continuous_memory_kernel(transition, dt, evaluation_time):
    """K(t) = -A_vh exp(-t A_hh) A_hv, with A = (I - transition) / dt."""
    transition = np.asarray(transition, dtype=float)
    evaluation_time = np.asarray(evaluation_time, dtype=float)
    generator = (np.eye(transition.shape[0]) - transition) / float(dt)
    a_vv = float(generator[0, 0])
    a_vh = generator[0, 1:]
    a_hv = generator[1:, 0]
    a_hh = generator[1:, 1:]
    kernel = np.asarray([-float(a_vh @ expm(-a_hh * time) @ a_hv) for time in evaluation_time], dtype=float)
    return (kernel, a_vv)


def run_vroylandt_likelihood_em(config):
    velocity = generate_common_velocity_ensemble(config)
    settings = dict(config["em"], hidden_dimension=int(config["hidden_dimension"]), seed=int(config["initialization_seed"]))
    fit = fit_em(velocity, float(config["dt"]), settings)
    t = np.linspace(0.0, float(config["evaluation_t_max"]), int(config["evaluation_grid_count"]))
    truth = np.asarray(kernel(t, str(config["kernel_type"])), dtype=float)
    estimate, _ = continuous_memory_kernel(fit["transition"], float(config["dt"]), t)
    _, error = relative_l2_error(estimate, truth, t)
    arrays = {"velocity_ensemble": velocity, "kernel_t": t, "kernel_truth": truth, "kernel_estimate": estimate}
    arrays.update(fit)
    return (arrays, {"kernel_relative_l2_error": float(error)})
