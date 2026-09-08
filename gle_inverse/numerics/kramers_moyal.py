"""Finite-step Kramers--Moyal identities used by simulation and analysis."""

import numpy as np


def empirical_conditional_moments(increments, dt) -> tuple[np.ndarray, np.ndarray]:
    """Return empirical first and second coefficients from branch increments."""
    increments = np.asarray(increments, dtype=float)
    dt = float(dt)
    if increments.ndim != 2 or increments.shape[1] < 1:
        raise ValueError("increments must have shape (reference states, M).")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    first = np.mean(increments, axis=1) / dt
    second = np.mean(increments**2, axis=1) / (2.0 * dt)
    return first, second


def truncated_conditional_moments(increments, dt, epsilon) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return moments truncated to increments with magnitude below epsilon."""
    increments = np.asarray(increments, dtype=float)
    epsilon = float(epsilon)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    retained = np.abs(increments) < epsilon
    first, second = empirical_conditional_moments(increments * retained, dt)
    return first, second, retained


def remove_symmetric_jump_second_moment(truncated_second, small_jump_second_moment) -> np.ndarray:
    """Remove half of the symmetric small-jump quadratic contribution."""
    truncated_second = np.asarray(truncated_second, dtype=float)
    small_jump_second_moment = float(small_jump_second_moment)
    if small_jump_second_moment < 0.0:
        raise ValueError("small_jump_second_moment must be nonnegative.")
    return truncated_second - 0.5 * small_jump_second_moment


def brownian_euler_reference(drift, diffusion, dt) -> tuple[np.ndarray, np.ndarray]:
    """Return exact first and second moments for one Brownian Euler step."""
    drift = np.asarray(drift, dtype=float)
    diffusion = np.asarray(diffusion, dtype=float)
    dt = float(dt)
    if drift.shape != diffusion.shape:
        raise ValueError("drift and diffusion must have the same shape.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")
    first = drift
    second = 0.5 * diffusion**2 + 0.5 * dt * drift**2
    return first, second


def brownian_euler_error_scales(drift, diffusion, dt, sample_size) -> dict[str, np.ndarray]:
    """Return the exact bias and standard errors for averaged Euler moments."""
    drift = np.asarray(drift, dtype=float)
    diffusion = np.asarray(diffusion, dtype=float)
    dt = float(dt)
    sample_size = int(sample_size)
    if drift.shape != diffusion.shape:
        raise ValueError("drift and diffusion must have the same shape.")
    if dt <= 0.0 or sample_size <= 0:
        raise ValueError("dt and sample_size must be positive.")
    return {
        "d1_standard_deviation": np.abs(diffusion) / np.sqrt(sample_size * dt),
        "d2_bias": 0.5 * drift**2 * dt,
        "d2_standard_deviation": np.sqrt((0.5 * diffusion**4 + drift**2 * diffusion**2 * dt) / sample_size),
    }
