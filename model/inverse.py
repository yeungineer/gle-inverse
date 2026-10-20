from math import gamma, pi, sqrt

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar
from scipy.signal import fftconvolve
from scipy.special import digamma


def left_rectangle_memory(kernel, history, dt):
    kernel = np.asarray(kernel, dtype=float).reshape(-1)
    history = np.asarray(history, dtype=float).reshape(-1)
    dt = float(dt)
    if history.size < 2:
        return 0.0
    values = kernel[: history.size] * history[::-1]
    return float(dt * np.sum(values[1:]))


def left_rectangle_convolution(history, kernel, dt):
    history = np.asarray(history, dtype=float).reshape(-1)
    kernel = np.asarray(kernel, dtype=float).reshape(-1)
    dt = float(dt)
    convolution = fftconvolve(history, kernel, mode="full")[: history.size]
    convolution -= history * kernel[0]
    return np.asarray(dt * convolution, dtype=float)


def left_rectangle_basis_response(history, basis_values, dt, reference_indices):
    history = np.asarray(history, dtype=float).reshape(-1)
    basis_values = np.asarray(basis_values, dtype=float)
    reference_indices = np.asarray(reference_indices, dtype=int).reshape(-1)
    dt = float(dt)
    response = np.empty((reference_indices.size, basis_values.shape[1]), dtype=float)
    for column in range(basis_values.shape[1]):
        basis_column = basis_values[:, column]
        convolution = fftconvolve(history, basis_column, mode="full")[: history.size]
        convolution -= history * basis_column[0]
        response[:, column] = -dt * convolution[reference_indices]
    return response


def empirical_conditional_moments(increments, dt):
    increments = np.asarray(increments, dtype=float)
    dt = float(dt)
    first = np.mean(increments, axis=1) / dt
    second = np.mean(increments**2, axis=1) / (2.0 * dt)
    return (first, second)


def truncated_conditional_moments(increments, dt, epsilon):
    increments = np.asarray(increments, dtype=float)
    epsilon = float(epsilon)
    retained = np.abs(increments) < epsilon
    first, second = empirical_conditional_moments(increments * retained, dt)
    return (first, second, retained)


def remove_symmetric_jump_second_moment(truncated_second, small_jump_second_moment):
    truncated_second = np.asarray(truncated_second, dtype=float)
    small_jump_second_moment = float(small_jump_second_moment)
    return truncated_second - 0.5 * small_jump_second_moment


def brownian_euler_reference(drift, diffusion, dt):
    drift = np.asarray(drift, dtype=float)
    diffusion = np.asarray(diffusion, dtype=float)
    dt = float(dt)
    first = drift
    second = 0.5 * diffusion**2 + 0.5 * dt * drift**2
    return (first, second)


def brownian_euler_error_scales(drift, diffusion, dt, sample_size):
    drift = np.asarray(drift, dtype=float)
    diffusion = np.asarray(diffusion, dtype=float)
    dt = float(dt)
    sample_size = int(sample_size)
    return {
        "d1_standard_deviation": np.abs(diffusion) / np.sqrt(sample_size * dt),
        "d2_bias": 0.5 * drift**2 * dt,
        "d2_standard_deviation": np.sqrt((0.5 * diffusion**4 + drift**2 * diffusion**2 * dt) / sample_size),
    }


def stable_constant(alpha):
    alpha = float(alpha)
    return alpha * 2.0 ** (alpha - 1.0) * gamma((1.0 + alpha) / 2.0) / (sqrt(pi) * gamma(1.0 - alpha / 2.0))


def _stable_constant_log_derivative(alpha):
    alpha = float(alpha)
    return float(1.0 / alpha + np.log(2.0) + 0.5 * digamma((1.0 + alpha) / 2.0) + 0.5 * digamma(1.0 - alpha / 2.0))


def symmetric_tail_intensity(alpha, kappa):
    alpha = float(alpha)
    kappa = float(kappa)
    return 2.0 * stable_constant(alpha) * kappa**alpha


def stable_annular_rates(alpha, kappa, epsilon, q, n_bins):
    alpha = float(alpha)
    epsilon = float(epsilon)
    q = float(q)
    index = np.arange(int(n_bins) + 1, dtype=float)
    return symmetric_tail_intensity(alpha, kappa) / alpha * epsilon ** (-alpha) * q ** (-index * alpha) * (1.0 - q ** (-alpha))


def symmetric_jump_second_moment(alpha, kappa, epsilon):
    alpha = float(alpha)
    epsilon = float(epsilon)
    return float(symmetric_tail_intensity(alpha, kappa) * epsilon ** (2.0 - alpha) / (2.0 - alpha))


def _annular_counts(delta_v, epsilon, q, n_bins):
    values = np.asarray(delta_v, dtype=float).reshape(-1)
    edges = float(epsilon) * float(q) ** np.arange(int(n_bins) + 2, dtype=float)
    positive = np.asarray([np.count_nonzero((values >= lo) & (values < hi)) for (lo, hi) in zip(edges[:-1], edges[1:])], dtype=np.int64)
    negative = np.asarray([np.count_nonzero((values <= -lo) & (values > -hi)) for (lo, hi) in zip(edges[:-1], edges[1:])], dtype=np.int64)
    return (edges, positive, negative, positive + negative)


def _tail_parameter_standard_errors(alpha, kappa, epsilon, tail_count):
    alpha = float(alpha)
    epsilon = float(epsilon)
    alpha_standard_error = alpha / sqrt(int(tail_count))
    log_total_intensity = np.log(2.0 * stable_constant(alpha)) + alpha * np.log(float(kappa)) - np.log(alpha) - alpha * np.log(epsilon)
    numerator = log_total_intensity + np.log(alpha) + alpha * np.log(epsilon) - np.log(2.0 * stable_constant(alpha))
    numerator_derivative = 1.0 / alpha + np.log(epsilon) - _stable_constant_log_derivative(alpha)
    derivative = numerator_derivative / alpha - numerator / alpha**2
    log_kappa_variance = 1.0 / (alpha**2 * int(tail_count)) + derivative**2 * alpha_standard_error**2
    return (float(alpha_standard_error), float(sqrt(max(log_kappa_variance, 0.0))))


def estimate_symmetric_stable(delta_v, dt, *, epsilon=1.0, q=2.0, n_bins=2):
    values = np.asarray(delta_v, dtype=float).reshape(-1)
    dt = float(dt)
    epsilon = float(epsilon)
    q = float(q)
    n_bins = int(n_bins)
    absolute = np.abs(values)
    tail = absolute[absolute >= epsilon]
    tail_count = int(tail.size)
    log_excess_sum = float(np.sum(np.log(tail / epsilon)))
    alpha = float(tail_count / log_excess_sum)
    exposure = values.size * dt
    total_intensity = tail_count / exposure
    kappa = float((total_intensity * alpha * epsilon**alpha / (2.0 * stable_constant(alpha))) ** (1.0 / alpha))
    edges, positive, negative, counts = _annular_counts(values, epsilon, q, n_bins)
    empirical_rates = counts.astype(float) / exposure
    fitted_rates = stable_annular_rates(alpha, kappa, epsilon, q, n_bins)
    bin_index = np.arange(n_bins + 1, dtype=float)
    factor = alpha * epsilon**alpha * q ** (bin_index * alpha) / (1.0 - q ** (-alpha)) / exposure
    total_intensity_bins = factor * counts
    kappa_bins = (total_intensity_bins / (2.0 * stable_constant(alpha))) ** (1.0 / alpha)
    alpha_se, log_kappa_se = _tail_parameter_standard_errors(alpha, kappa, epsilon, tail_count)
    return {
        "alpha_raw": alpha,
        "alpha": alpha,
        "alpha_standard_error": alpha_se,
        "kappa": kappa,
        "log_kappa_standard_error": log_kappa_se,
        "kappa_bins": kappa_bins,
        "tail_count": tail_count,
        "counts": counts,
        "positive_counts": positive,
        "negative_counts": negative,
        "bin_edges": edges,
        "empirical_rates": empirical_rates,
        "fitted_rates": fitted_rates,
        "small_jump_second_moment": symmetric_jump_second_moment(alpha, kappa, epsilon),
        "exposure": float(exposure),
        "epsilon": epsilon,
        "q": q,
        "K": n_bins,
    }


_DEFAULT_GRID_SIZE = 161
_LOWER_SPECTRAL_FACTOR = 0.0001
_UPPER_SPECTRAL_FACTOR = 100.0
_MAX_CONDITION_SPAN = 1000000000000.0
_BOUNDARY_EXPANSION_FACTOR = 10000.0
_MAX_BOUNDARY_EXPANSIONS = 3
R1GCV_ROBUSTNESS = 0.999


def _spectral_candidates(singular_values):
    singular_values = np.asarray(singular_values, dtype=float)
    squared = singular_values**2
    largest = float(np.max(squared))
    rank_tolerance = np.finfo(float).eps * max(1, singular_values.size) * float(np.max(singular_values))
    positive = squared[singular_values > rank_tolerance]
    if largest <= 0.0 or positive.size == 0:
        raise ValueError("response has no numerically identifiable direction.")
    lower = max(_LOWER_SPECTRAL_FACTOR * float(np.min(positive)), largest / _MAX_CONDITION_SPAN)
    upper = _UPPER_SPECTRAL_FACTOR * largest
    return np.geomspace(lower, upper, _DEFAULT_GRID_SIZE)


def _path_statistics(lambdas, squared_singular_values, projected_data, orthogonal_energy, n_rows):
    lambdas = np.asarray(lambdas, dtype=float)
    shrinkage = squared_singular_values[None, :] / (squared_singular_values[None, :] + lambdas[:, None])
    residual_energy = orthogonal_energy + np.sum(((1.0 - shrinkage) * projected_data[None, :]) ** 2, axis=1)
    # Projection subtraction can leave a negative squared norm at roundoff scale.
    residual_energy = np.maximum(residual_energy, 0.0)
    effective_dofs = np.sum(shrinkage, axis=1)
    denominator = np.maximum(float(n_rows) - effective_dofs, np.finfo(float).eps * max(float(n_rows), 1.0))
    scores = float(n_rows) * residual_energy / denominator**2
    return (scores, np.sqrt(residual_energy), effective_dofs)


def _strong_influence(lambdas, squared_singular_values):
    lambdas = np.asarray(lambdas, dtype=float)
    squared = np.asarray(squared_singular_values, dtype=float)
    return np.sum(squared[None, :] / (squared[None, :] + lambdas[:, None]) ** 2, axis=1)


def _selection_scores(lambdas, squared_singular_values, projected_data, orthogonal_energy, n_rows, robustness):
    gcv_scores, _, _ = _path_statistics(lambdas, squared_singular_values, projected_data, orthogonal_energy, n_rows)
    influence = _strong_influence(lambdas, squared_singular_values)
    return gcv_scores * (robustness + (1.0 - robustness) * influence)


def _locally_refined_candidates(candidates, scores, score_at_log_lambda):
    index = _finite_argmin(scores)
    if index == 0 or index == len(candidates) - 1:
        return candidates
    result = minimize_scalar(
        score_at_log_lambda, bounds=(float(np.log(candidates[index - 1])), float(np.log(candidates[index + 1]))), method="bounded",
        options={"xatol": 1e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        return candidates
    return np.unique(np.append(candidates, np.exp(result.x)))


def _finite_argmin(values):
    values = np.asarray(values, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        raise FloatingPointError("The regularization path contains no finite selection score.")
    return int(finite_indices[np.argmin(values[finite_indices])])


def _expanded_candidates(candidates, score_function):
    candidates = np.asarray(candidates, dtype=float)
    expansion_count = 0
    while expansion_count < _MAX_BOUNDARY_EXPANSIONS:
        scores = score_function(candidates)
        index = _finite_argmin(scores)
        if index not in {0, len(candidates) - 1}:
            break
        if index == 0:
            lower = max(float(candidates[0]) / _BOUNDARY_EXPANSION_FACTOR, np.finfo(float).tiny)
            if not lower < candidates[0]:
                break
            extension = np.geomspace(lower, candidates[0], 33)[:-1]
            candidates = np.unique(np.concatenate((extension, candidates)))
        else:
            upper = float(candidates[-1]) * _BOUNDARY_EXPANSION_FACTOR
            if not np.isfinite(upper) or not upper > candidates[-1]:
                break
            extension = np.geomspace(candidates[-1], upper, 33)[1:]
            candidates = np.unique(np.concatenate((candidates, extension)))
        expansion_count += 1
    return candidates


def select_lambda(response, data, gram, *, robustness=R1GCV_ROBUSTNESS):
    """Whiten the RKHS penalty, then minimize robust GCV along the SVD path."""
    response, data, gram = (np.asarray(a, dtype=float) for a in (response, data, gram))
    gram = 0.5 * (gram + gram.T)
    penalty_values, penalty_vectors = np.linalg.eigh(gram)
    penalty_scale = max(float(np.max(np.abs(penalty_values))), np.finfo(float).tiny)
    # Restrict the inverse to the numerically positive eigenspace of the penalty.
    penalty_tolerance = np.finfo(float).eps * max(gram.shape) * penalty_scale
    if np.any(penalty_values < -penalty_tolerance):
        raise ValueError("gram must be positive semidefinite.")
    positive = penalty_values > penalty_tolerance
    if not np.any(positive):
        raise ValueError("gram has no numerically penalized direction.")
    coefficient_support = penalty_vectors[:, positive]
    reduced_response = response @ coefficient_support
    reduced_gram = np.diag(penalty_values[positive])
    lower = cholesky(reduced_gram, lower=True, check_finite=False)
    whitened = solve_triangular(lower, reduced_response.T, lower=True, check_finite=False).T
    left, singular_values, right_transpose = np.linalg.svd(whitened, full_matrices=False)
    projected = left.T @ data
    orthogonal_energy = max(float(data @ data - projected @ projected), 0.0)
    squared = singular_values**2
    n_rows = response.shape[0]
    candidates = _spectral_candidates(singular_values)

    def path(candidate_values):
        return _selection_scores(candidate_values, squared, projected, orthogonal_energy, n_rows, robustness)

    candidates = _expanded_candidates(candidates, path)
    scores = path(candidates)

    def score_at_log_lambda(log_lambda):
        score = path(np.asarray([np.exp(log_lambda)]))
        return float(score[0])

    candidates = _locally_refined_candidates(candidates, scores, score_at_log_lambda)
    scores = path(candidates)
    index = _finite_argmin(scores)
    reduced_coefficient_modes = solve_triangular(lower.T, right_transpose.T, lower=False, check_finite=False)
    coefficient_modes = coefficient_support @ reduced_coefficient_modes
    return {
        "lambda": float(candidates[index]),
        "selected_at_boundary": index in {0, len(candidates) - 1},
        "singular_values": singular_values,
        "projected_data": projected,
        "coefficient_modes": coefficient_modes,
    }


def effective_rank(singular_values):
    energy = np.asarray(singular_values, dtype=float) ** 2
    probability = energy / np.sum(energy)
    probability = probability[probability > 0.0]
    return float(np.exp(-np.sum(probability * np.log(probability))))


class SplineBasis:

    def __init__(self, n_basis, t_max, knot_power=1.25):
        n_basis, t_max = int(n_basis), float(t_max)
        n_internal = n_basis - 4
        fractions = np.arange(1, n_internal + 1, dtype=float) / (n_internal + 1)
        internal = t_max * fractions ** float(knot_power)
        knots = np.concatenate((np.zeros(4), internal, np.full(4, t_max)))
        self.splines = []
        for index in range(n_basis):
            c = np.zeros(n_basis, dtype=float)
            c[index] = 1.0
            self.splines.append(BSpline(knots, c, 3, extrapolate=False))

    def matrix(self, t):
        t = np.asarray(t, dtype=float).reshape(-1)
        # Compact support: SciPy returns NaN outside the spline interval.
        return np.column_stack([np.nan_to_num(spline(t), nan=0.0) for spline in self.splines])


def fit_kernel(v, t, idx, d1, f, t_eval, basis):
    """Volterra inversion: R c = D1 - f, penalty pinv(R.T @ R)."""
    v = np.asarray(v, dtype=float)
    t = np.asarray(t, dtype=float)
    idx = np.asarray(idx, dtype=int)
    d1 = np.asarray(d1, dtype=float).reshape(-1)
    f = np.asarray(f, dtype=float).reshape(-1)
    dt = float(t[1] - t[0])
    R = left_rectangle_basis_response(v, basis.matrix(t - t[0]), dt, idx)
    P = np.linalg.pinv(R.T @ R, hermitian=True)
    selection = select_lambda(R, d1 - f, P)
    s = selection["singular_values"]
    lam = float(selection["lambda"])
    c = selection["coefficient_modes"] @ (s / (s**2 + lam) * selection["projected_data"])
    K = basis.matrix(np.asarray(t_eval, dtype=float) - t[0]) @ c
    return (K, f + R @ c, c, selection)
