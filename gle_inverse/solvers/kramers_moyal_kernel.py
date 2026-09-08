"""RKHS solver for the Kramers--Moyal memory-kernel equation."""

import numpy as np

from gle_inverse.numerics.convolution import left_rectangle_basis_response
from gle_inverse.numerics.gcv import strong_robust_generalized_cross_validation


class KernelSolver:
    """Estimate the memory kernel in a finite-dimensional RKHS basis."""

    def __init__(self, basis_model, t_reference):
        self.basis = basis_model
        self.mode = "r1gcv"
        self.t_reference = np.asarray(t_reference, dtype=float)
        self.dt = float(self.t_reference[1] - self.t_reference[0])
        self.selected_lambda = None
        self.gcv = None
        self.singular_values = None

    @staticmethod
    def _coefficients_from_gcv(result):
        """Recover coefficients from a GCV-family spectral result."""
        singular_values = result["singular_values"]
        selected_lambda = float(result["lambda"])
        modal_data = singular_values / (singular_values**2 + selected_lambda) * result["projected_data"]
        return result["coefficient_modes"] @ modal_data

    def solve(self, v_reference, reference_indices, d1_empirical, force_reference, t_evaluation):
        """Solve the regularized first-moment inverse problem."""
        v_reference = np.asarray(v_reference, dtype=float)
        reference_indices = np.asarray(reference_indices, dtype=int)
        d1_empirical = np.asarray(d1_empirical, dtype=float)
        force_reference = np.asarray(force_reference, dtype=float)

        if np.any(reference_indices < 0) or np.any(reference_indices >= len(v_reference)):
            raise ValueError("reference_indices lie outside v_reference.")

        lags = self.t_reference - self.t_reference[0]
        response_matrix = left_rectangle_basis_response(
            v_reference, self.basis.matrix(lags), self.dt, reference_indices
        )

        return self._solve_from_response(response_matrix, d1_empirical, force_reference, t_evaluation)

    def _solve_from_response(self, response_matrix, d1_empirical, force_reference, t_evaluation):
        """Apply the common regularized solve to a mathematically declared response."""
        response_matrix = np.asarray(response_matrix, dtype=float)
        d1_empirical = np.asarray(d1_empirical, dtype=float).reshape(-1)
        force_reference = np.asarray(force_reference, dtype=float).reshape(-1)
        if response_matrix.ndim != 2 or response_matrix.shape[1] != self.basis.n_basis:
            raise ValueError("response_matrix must have one column per kernel-basis function.")
        if response_matrix.shape[0] != len(d1_empirical) or d1_empirical.shape != force_reference.shape:
            raise ValueError("Response rows, empirical first moments, and force values must align.")
        if (
            np.any(~np.isfinite(response_matrix))
            or np.any(~np.isfinite(d1_empirical))
            or np.any(~np.isfinite(force_reference))
        ):
            raise ValueError("The inverse response and first-moment data must be finite.")

        inverse_data = d1_empirical - force_reference
        normal_matrix = response_matrix.T @ response_matrix
        penalty = np.linalg.pinv(normal_matrix, hermitian=True)
        self.gcv = strong_robust_generalized_cross_validation(response_matrix, inverse_data, penalty)
        self.selected_lambda = float(self.gcv["lambda"])
        self.singular_values = np.asarray(self.gcv["singular_values"], dtype=float)
        coefficients = self._coefficients_from_gcv(self.gcv)

        gamma = self.basis.matrix(np.asarray(t_evaluation, dtype=float) - self.t_reference[0]) @ coefficients
        d1_fit = force_reference + response_matrix @ coefficients

        self.coefficients = coefficients
        return gamma, d1_fit, coefficients
