"""Generic spline Volterra reconstruction used only for Figure 6."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from gle_inverse.numerics.basis import SplineBasis
from gle_inverse.numerics.errors import relative_l2_error
from gle_inverse.solvers.kramers_moyal_kernel import KernelSolver


@dataclass(frozen=True)
class KernelResult:
    kernel_estimate: np.ndarray
    kernel_relative_error: float


def solve_kernel(
    problem: Any,
    *,
    n_basis: int,
    t_max: float,
) -> KernelResult:
    t_reference = np.asarray(problem.t_inference, dtype=float)
    basis = SplineBasis(n_basis=int(n_basis), t_max=float(t_max))
    solver = KernelSolver(basis, t_reference)
    km = problem.km_samples
    kernel_estimate, _, _ = solver.solve(
        np.asarray(problem.v_inference, dtype=float),
        np.asarray(km["idx"], dtype=int),
        np.asarray(km["d1"], dtype=float),
        np.asarray(km["f"], dtype=float),
        np.asarray(problem.t, dtype=float),
    )

    time = np.asarray(problem.t, dtype=float)
    truth = np.asarray(problem.kernel_true, dtype=float)
    evaluation_mask = time <= float(t_max) + 1.0e-12
    _, relative_error = relative_l2_error(
        kernel_estimate[evaluation_mask], truth[evaluation_mask], time[evaluation_mask]
    )
    return KernelResult(
        kernel_estimate=np.asarray(kernel_estimate, dtype=float),
        kernel_relative_error=float(relative_error),
    )
