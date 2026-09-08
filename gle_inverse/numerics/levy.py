from math import gamma, pi, sqrt

import numpy as np
from scipy.special import digamma


def resolve_levy_threshold(settings):
    return float(settings["levy_epsilon"])


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
    return (
        symmetric_tail_intensity(alpha, kappa)
        / alpha
        * epsilon ** (-alpha)
        * q ** (-index * alpha)
        * (1.0 - q ** (-alpha))
    )


def symmetric_jump_second_moment(alpha, kappa, epsilon):
    alpha = float(alpha)
    epsilon = float(epsilon)
    return float(symmetric_tail_intensity(alpha, kappa) * epsilon ** (2.0 - alpha) / (2.0 - alpha))


def _annular_counts(delta_v, epsilon, q, n_bins):
    values = np.asarray(delta_v, dtype=float).reshape(-1)
    edges = float(epsilon) * float(q) ** np.arange(int(n_bins) + 2, dtype=float)
    positive = np.asarray(
        [np.count_nonzero((values >= lo) & (values < hi)) for lo, hi in zip(edges[:-1], edges[1:])], dtype=np.int64
    )
    negative = np.asarray(
        [np.count_nonzero((values <= -lo) & (values > -hi)) for lo, hi in zip(edges[:-1], edges[1:])], dtype=np.int64
    )
    return edges, positive, negative, positive + negative


def _tail_parameter_standard_errors(alpha, kappa, epsilon, tail_count):
    alpha = float(alpha)
    epsilon = float(epsilon)
    alpha_standard_error = alpha / sqrt(int(tail_count))
    log_total_intensity = (
        np.log(2.0 * stable_constant(alpha)) + alpha * np.log(float(kappa)) - np.log(alpha) - alpha * np.log(epsilon)
    )
    numerator = log_total_intensity + np.log(alpha) + alpha * np.log(epsilon) - np.log(2.0 * stable_constant(alpha))
    numerator_derivative = 1.0 / alpha + np.log(epsilon) - _stable_constant_log_derivative(alpha)
    derivative = numerator_derivative / alpha - numerator / alpha**2
    log_kappa_variance = 1.0 / (alpha**2 * int(tail_count)) + derivative**2 * alpha_standard_error**2
    return float(alpha_standard_error), float(sqrt(max(log_kappa_variance, 0.0)))


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
