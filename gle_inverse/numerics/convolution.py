"""Matrix representations of causal discrete convolutions."""

import numpy as np
from scipy.signal import fftconvolve


def left_rectangle_memory(kernel, history, dt):
    """Evaluate a causal memory integral by the explicit left rule."""
    kernel = np.asarray(kernel, dtype=float).reshape(-1)
    history = np.asarray(history, dtype=float).reshape(-1)
    dt = float(dt)
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive.")
    if kernel.size < history.size:
        raise ValueError("kernel must cover every lag in history.")
    if history.size < 2:
        return 0.0
    values = kernel[: history.size] * history[::-1]
    return float(dt * np.sum(values[1:]))


def left_rectangle_convolution(history, kernel, dt):
    """Evaluate the explicit-left memory integral along a full path."""
    history = np.asarray(history, dtype=float).reshape(-1)
    kernel = np.asarray(kernel, dtype=float).reshape(-1)
    dt = float(dt)
    if history.size != kernel.size:
        raise ValueError("history and kernel must have the same length.")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive.")
    convolution = fftconvolve(history, kernel, mode="full")[: history.size]
    convolution -= history * kernel[0]
    return np.asarray(dt * convolution, dtype=float)


def left_rectangle_basis_response(history, basis_values, dt, reference_indices):
    """Assemble explicit left-rule basis convolutions by FFT."""
    history = np.asarray(history, dtype=float).reshape(-1)
    basis_values = np.asarray(basis_values, dtype=float)
    reference_indices = np.asarray(reference_indices, dtype=int).reshape(-1)
    dt = float(dt)

    if basis_values.ndim != 2 or basis_values.shape[0] != history.size:
        raise ValueError("basis_values must have one row per history sample.")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be finite and positive.")
    if np.any(reference_indices < 0) or np.any(reference_indices >= history.size):
        raise ValueError("reference_indices lie outside history.")

    response = np.empty((reference_indices.size, basis_values.shape[1]), dtype=float)
    for column in range(basis_values.shape[1]):
        basis_column = basis_values[:, column]
        convolution = fftconvolve(history, basis_column, mode="full")[: history.size]
        convolution -= history * basis_column[0]
        response[:, column] = -dt * convolution[reference_indices]
    return response
