import numpy as np
from scipy.linalg import hankel, null_space, toeplitz
from scipy.signal import find_peaks
from scipy.stats import theilslopes

from gle_inverse.numerics.basis import SplineBasis
from gle_inverse.numerics.errors import relative_l2_error
from gle_inverse.numerics.gcv import (
    _cross_validation,
    strong_robust_generalized_cross_validation,
)
from gle_inverse.simulation.kernels import get_kernel
from gle_inverse.simulation.noises import diffusion_profile

ROOT_NORMALIZATION_EPSILON = 1.0e-6
REAL_ROOT_TOLERANCE = 1.0e-12
SOBOLEV_R1GCV_ROBUSTNESS = 0.99999


def _spectral_coefficients(selection):
    singular_values = np.asarray(selection["singular_values"], dtype=float)
    modal_data = (
        singular_values
        / (singular_values**2 + float(selection["lambda"]))
        * np.asarray(selection["projected_data"], dtype=float)
    )
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
    return response, data, penalty


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
            rate = -float(
                np.log(amplitude[indices[-1]] / amplitude[indices[0]]) / (time[indices[-1]] - time[indices[0]])
            )
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
        real_negative = (
            np.isclose(root.imag, 0.0, rtol=0.0, atol=REAL_ROOT_TOLERANCE * max(1.0, abs(root.real)))
            and root.real < 0.0
        )
        if real_negative:
            log_magnitude = np.log(abs(root))
            rates.extend(((log_magnitude + 1j * np.pi) / dt, (log_magnitude - 1j * np.pi) / dt))
        else:
            rates.append(np.log(root) / dt)
    return np.asarray(rates, dtype=np.complex128)


def _prony_functions(correlation, time, modes, drop_zero):
    correlation = np.asarray(correlation, dtype=np.float64)
    time = np.asarray(time, dtype=np.float64)
    first = 1 if drop_zero and np.isclose(time[0], 0.0, rtol=0.0, atol=1.0e-14) else 0
    samples = correlation[first:]
    sample_time = time[first:]
    rates = _regularized_rates(samples, sample_time, modes, _decay_rate(correlation, time))
    design = np.exp(np.outer(sample_time, rates))
    constraint = null_space(rates[None, :])
    reduced_design = design @ constraint
    penalty = np.linalg.pinv(design.conj().T @ design, hermitian=True)
    penalty = constraint.conj().T @ penalty @ constraint
    response, data, penalty = _real_system(reduced_design, samples, penalty)
    selection = _cross_validation(response, data, penalty, None, method="gcv", robustness=1.0)
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
    return h, g, gp, derivative_weight


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
    psi = (convolution @ phi) * dt
    psi_prime = np.gradient(psi, dt, axis=0)
    alpha_derivative = float(derivative_weight)
    alpha_space = 1.0 - alpha_derivative
    quadrature = np.exp(-2.0 * float(settings["omega"]) * t) * dt
    response = np.vstack(
        (np.sqrt(alpha_space * quadrature)[:, None] * psi, np.sqrt(alpha_derivative * quadrature)[:, None] * psi_prime)
    )
    data = np.concatenate(
        (np.sqrt(alpha_space * quadrature) * g_values, np.sqrt(alpha_derivative * quadrature) * gp_values)
    )
    penalty = np.linalg.pinv(response.T @ response, hermitian=True)
    selection = strong_robust_generalized_cross_validation(response, data, penalty, robustness=SOBOLEV_R1GCV_ROBUSTNESS)
    coefficients = _spectral_coefficients(selection)
    return lambda evaluation_time: basis.matrix(np.asarray(evaluation_time, dtype=float)) @ coefficients


def reconstruct_prony_sobolev(correlation_time, correlation, config):
    prony = config["prony"]
    h, g, gp, derivative_weight = _prony_functions(
        correlation, correlation_time, int(prony["modes"]), bool(prony["drop_zero"])
    )
    settings = config["sobolev"]
    time = np.linspace(0.0, float(settings["t_max"]), int(settings["grid_count"]))
    estimate = _sobolev_estimator(h, g, gp, settings, derivative_weight)(time)
    return time, np.asarray(estimate, dtype=float)


def _ensemble_correlation(config):
    M, L, dt = int(config["M"]), int(config["L"]), float(config["dt"])
    settings = config["prony"]
    sample_indices = np.arange(int(settings["sample_count"]), dtype=np.int64) * int(settings["lag_stride"])
    kernel = get_kernel({"type": str(config["kernel_type"])}, dt)
    weights = np.asarray(kernel.u, dtype=np.complex128)
    memory_decay = np.exp(np.asarray(kernel.eta, dtype=np.complex128) * dt)
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
    return sample_indices.astype(float) * dt, correlation


def run_prony_sobolev(config):
    correlation_time, correlation = _ensemble_correlation(config)
    kernel_time, kernel_estimate = reconstruct_prony_sobolev(correlation_time, correlation, config)
    kernel = get_kernel({"type": str(config["kernel_type"])}, float(config["dt"]))
    kernel_truth = np.asarray(kernel.value(kernel_time), dtype=float)
    _, relative_error = relative_l2_error(kernel_estimate, kernel_truth, kernel_time)
    arrays = {
        "kernel_t": kernel_time,
        "kernel_truth": kernel_truth,
        "kernel_estimate": kernel_estimate,
        "correlation_t": correlation_time,
        "correlation_samples": correlation,
    }
    return arrays, {"kernel_relative_l2_error": float(relative_error)}
