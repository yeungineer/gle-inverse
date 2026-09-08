"""Weighted and relative error measures for reconstructed functions."""

import numpy as np


def relative_l2_error(estimate, reference, time):
    """Return absolute and relative continuous L2 errors."""
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    time = np.asarray(time, dtype=float)
    absolute = np.sqrt(np.trapezoid((reference - estimate) ** 2, time))
    reference_norm = np.sqrt(np.trapezoid(reference**2, time))
    return float(absolute), float(absolute / max(reference_norm, 1.0e-14))
