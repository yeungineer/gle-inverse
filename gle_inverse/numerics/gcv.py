"""GCV parameter choices for finite-dimensional Tikhonov systems."""

import numpy as np
from scipy.linalg import cholesky, solve_triangular
from scipy.optimize import minimize_scalar

_DEFAULT_GRID_SIZE = 161
_LOWER_SPECTRAL_FACTOR = 1.0e-4
_UPPER_SPECTRAL_FACTOR = 1.0e2
_MAX_CONDITION_SPAN = 1.0e12
_BOUNDARY_EXPANSION_FACTOR = 1.0e4
_MAX_BOUNDARY_EXPANSIONS = 3

R1GCV_ROBUSTNESS = 0.999


def _validate_system(response, data, gram):
    """Return a validated finite-dimensional inverse system."""
    response = np.asarray(response, dtype=float)
    data = np.asarray(data, dtype=float)
    gram = np.asarray(gram, dtype=float)

    if response.ndim != 2 or min(response.shape) == 0:
        raise ValueError("response must be a nonempty two-dimensional array.")
    n_rows, n_columns = response.shape
    if data.shape != (n_rows,):
        raise ValueError("data must be one-dimensional with one value per row.")
    if gram.shape != (n_columns, n_columns):
        raise ValueError("gram must be square with one row per response column.")
    if not (np.all(np.isfinite(response)) and np.all(np.isfinite(data)) and np.all(np.isfinite(gram))):
        raise ValueError("response, data, and gram must contain finite values.")
    if not np.allclose(gram, gram.T, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("gram must be symmetric.")

    return response, data, 0.5 * (gram + gram.T)


def _spectral_candidates(singular_values):
    """Construct a scale-aware logarithmic regularization grid."""
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
    """Evaluate the GCV score and its diagnostics for a lambda array."""
    lambdas = np.asarray(lambdas, dtype=float)
    shrinkage = squared_singular_values[None, :] / (squared_singular_values[None, :] + lambdas[:, None])
    residual_energy = orthogonal_energy + np.sum(((1.0 - shrinkage) * projected_data[None, :]) ** 2, axis=1)
    residual_energy = np.maximum(residual_energy, 0.0)
    effective_dofs = np.sum(shrinkage, axis=1)
    denominator = np.maximum(float(n_rows) - effective_dofs, np.finfo(float).eps * max(float(n_rows), 1.0))
    scores = float(n_rows) * residual_energy / denominator**2
    return scores, np.sqrt(residual_energy), effective_dofs


def _strong_influence(lambdas, squared_singular_values):
    """Return Lukas' regularization-norm influence ``mu_12``."""
    lambdas = np.asarray(lambdas, dtype=float)
    squared = np.asarray(squared_singular_values, dtype=float)
    return np.sum(squared[None, :] / (squared[None, :] + lambdas[:, None]) ** 2, axis=1)


def _selection_scores(lambdas, squared_singular_values, projected_data, orthogonal_energy, n_rows, robustness):
    gcv_scores, _, _ = _path_statistics(
        lambdas, squared_singular_values, projected_data, orthogonal_energy, n_rows
    )
    influence = _strong_influence(lambdas, squared_singular_values)
    return gcv_scores * (robustness + (1.0 - robustness) * influence)


def _locally_refined_candidates(candidates, scores, score_at_log_lambda):
    """Add a bounded log-space minimizer around an interior grid minimum."""
    index = _finite_argmin(scores)
    if index == 0 or index == len(candidates) - 1:
        return candidates

    result = minimize_scalar(
        score_at_log_lambda,
        bounds=(float(np.log(candidates[index - 1])), float(np.log(candidates[index + 1]))),
        method="bounded",
        options={"xatol": 1.0e-10},
    )
    if not result.success or not np.isfinite(result.fun):
        return candidates
    return np.unique(np.append(candidates, np.exp(result.x)))


def _finite_argmin(values):
    """Return the minimum finite entry and reject a numerically invalid path."""
    values = np.asarray(values, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        raise FloatingPointError("The regularization path contains no finite selection score.")
    return int(finite_indices[np.argmin(values[finite_indices])])


def _expanded_candidates(candidates, score_function):
    """Expand a default path when its minimizer lies at an endpoint."""
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


def _cross_validation(response, data, gram, robustness):
    response, data, gram = _validate_system(response, data, gram)

    penalty_values, penalty_vectors = np.linalg.eigh(gram)
    penalty_scale = max(float(np.max(np.abs(penalty_values))), np.finfo(float).tiny)
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


def strong_robust_generalized_cross_validation(response, data, gram, *, robustness=R1GCV_ROBUSTNESS):
    """Select lambda by Lukas' strong robust GCV (R1GCV) criterion."""
    robustness = float(robustness)
    return _cross_validation(response, data, gram, robustness)
