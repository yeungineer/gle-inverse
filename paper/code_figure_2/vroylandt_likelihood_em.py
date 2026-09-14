"""Compact reproduction of Vroylandt et al.'s likelihood--EM estimator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.linalg import expm

from gle_inverse.numerics.errors import relative_l2_error
from gle_inverse.simulation.kernels import get_kernel
from gle_inverse.simulation.noises import diffusion_profile


@dataclass(frozen=True)
class EStepResult:
    """Sufficient statistics and diagnostics from one exact E-step."""

    s_xx: np.ndarray
    s_yx: np.ndarray
    s_yy: np.ndarray
    initial_hidden_mean_sum: np.ndarray
    transition_count: int
    trajectory_count: int
    log_likelihood: float


@dataclass(frozen=True)
class EMFit:
    """Fitted discrete extended dynamics and convergence record."""

    transition: np.ndarray
    covariance: np.ndarray
    initial_hidden_mean: np.ndarray
    log_likelihood: np.ndarray
    converged: bool
    iterations: int


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)


def _positive_definite(matrix: np.ndarray, relative_floor: float) -> np.ndarray:
    """Apply only a numerical eigenvalue floor, not a statistical penalty."""
    matrix = _symmetrize(matrix)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    floor = float(relative_floor) * scale
    eigenvalues = np.maximum(eigenvalues, floor)
    return _symmetrize((eigenvectors * eigenvalues[None, :]) @ eigenvectors.T)


def _solve_spd(matrix: np.ndarray, right_hand_side: np.ndarray) -> np.ndarray:
    """Solve an SPD system using a Cholesky factorization."""
    factor = np.linalg.cholesky(_symmetrize(matrix))
    return np.linalg.solve(factor.T, np.linalg.solve(factor, right_hand_side))


def generate_common_velocity_ensemble(config: dict[str, Any]) -> np.ndarray:
    """Generate the common Brownian GLE with causal left-rule memory."""
    trajectory_count = int(config["M"])
    interval_count = int(config["L"])
    dt = float(config["dt"])

    kernel = get_kernel({"type": str(config["kernel_type"])}, dt)
    weights = np.asarray(kernel.u, dtype=np.complex128)
    decay = np.exp(np.asarray(kernel.eta, dtype=np.complex128) * dt)

    rng = np.random.default_rng(int(config["seed"]))
    velocity = np.empty((interval_count + 1, trajectory_count), dtype=float)
    velocity[0] = rng.standard_normal(trajectory_count)
    memory_modes = np.zeros((trajectory_count, len(weights)), dtype=np.complex128)
    brownian_scale = float(
        diffusion_profile(np.asarray([0.0]), str(config["diffusion"]))[0]
    ) * np.sqrt(dt)

    for step in range(interval_count):
        current = velocity[step]
        memory = np.real(memory_modes @ weights)
        velocity[step + 1] = (
            current - dt * memory + brownian_scale * rng.standard_normal(trajectory_count)
        )
        memory_modes = decay[None, :] * (
            memory_modes + dt * current[:, None]
        )

    return velocity


class VroylandtLikelihoodEM:
    """One-visible-coordinate likelihood--EM estimator."""

    def __init__(
        self,
        *,
        hidden_dimension: int,
        seed: int,
        likelihood_tolerance: float,
        max_iterations: int,
        covariance_floor: float,
        minimum_iterations: int = 2,
    ) -> None:
        self.hidden_dimension = int(hidden_dimension)
        self.seed = int(seed)
        self.likelihood_tolerance = float(likelihood_tolerance)
        self.max_iterations = int(max_iterations)
        self.covariance_floor = float(covariance_floor)
        self.minimum_iterations = int(minimum_iterations)
        if self.hidden_dimension < 1:
            raise ValueError("hidden_dimension must be positive.")
        if self.likelihood_tolerance <= 0.0 or self.max_iterations < 1:
            raise ValueError("The EM stopping parameters are invalid.")
        if self.covariance_floor <= 0.0 or self.minimum_iterations < 1:
            raise ValueError("The EM numerical safeguards are invalid.")

    def _initialize(self, velocity: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Seeded stable random initialization, as prescribed in the paper."""
        velocity = np.asarray(velocity, dtype=float)
        current = velocity[:-1].reshape(-1)
        following = velocity[1:].reshape(-1)
        denominator = max(float(current @ current), np.finfo(float).tiny)
        visible_ar = float(np.clip((following @ current) / denominator, -0.98, 0.9995))
        innovation = following - visible_ar * current
        visible_variance = max(float(np.mean(innovation**2)), 1.0e-8)

        rng = np.random.default_rng(self.seed)
        d = self.hidden_dimension
        horizon = float(dt) * float(velocity.shape[0] - 1)
        slow_rate = max(1.0 / max(horizon, dt), 2.5e-2)
        fast_rate = min(2.0 / dt, 4.0)
        rates = np.geomspace(slow_rate, fast_rate, d)
        rates *= np.exp(0.18 * rng.standard_normal(d))
        hidden_transition = np.diag(np.exp(-dt * rates))

        continuous_scale = max(np.sqrt(max(1.0 - visible_ar, 1.0e-3) / dt), 0.20)
        visible_hidden = -dt * continuous_scale * rng.standard_normal((1, d)) / np.sqrt(d)
        hidden_visible = -dt * continuous_scale * rng.standard_normal((d, 1)) / np.sqrt(d)
        transition = np.block(
            [
                [np.asarray([[visible_ar]]), visible_hidden],
                [hidden_visible, hidden_transition],
            ]
        )

        covariance = np.eye(d + 1, dtype=float) * visible_variance
        covariance[1:, 1:] *= 0.5
        covariance = _positive_definite(covariance, self.covariance_floor)
        initial_hidden_mean = np.zeros(d, dtype=float)
        return transition, covariance, initial_hidden_mean

    def _e_step(
        self,
        velocity: np.ndarray,
        transition: np.ndarray,
        covariance: np.ndarray,
        initial_hidden_mean: np.ndarray,
    ) -> EStepResult:
        """Exact Kalman conditioning and RTS smoothing for all trajectories."""
        velocity = np.asarray(velocity, dtype=float)
        interval_count = velocity.shape[0] - 1
        trajectory_count = velocity.shape[1]
        d = self.hidden_dimension
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
                f_hh @ p @ f_hh.T
                + q_hh
                - np.outer(hidden_observation_covariance, hidden_observation_covariance)
                / innovation_variance,
                self.covariance_floor,
            )
            log_likelihood += -0.5 * (
                trajectory_count * np.log(2.0 * np.pi * innovation_variance)
                + float(innovation @ innovation) / innovation_variance
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
            innovation = velocity[step + 1] - (
                f_vv * velocity[step] + filtered_at_step @ f_vh
            )

            updated_mean_at_step = (
                filtered_at_step + innovation[:, None] * (p_fvh / innovation_variance)[None, :]
            )
            updated_covariance_at_step = _symmetrize(
                p - np.outer(p_fvh, p_fvh) / innovation_variance
            )
            cross_after_observation = (
                p @ f_hh.T
                - np.outer(p_fvh, hidden_observation_covariance) / innovation_variance
            )
            smoother_gain = _solve_spd(
                filtered_covariance[step + 1], cross_after_observation.T
            ).T
            next_mean = smoothed_mean_next
            current_mean = updated_mean_at_step + (
                next_mean - filtered_mean[step + 1]
            ) @ smoother_gain.T
            current_covariance = _symmetrize(
                updated_covariance_at_step
                + smoother_gain
                @ (smoothed_covariance_next - filtered_covariance[step + 1])
                @ smoother_gain.T
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

        return EStepResult(
            s_xx=_symmetrize(s_xx),
            s_yx=s_yx,
            s_yy=_symmetrize(s_yy),
            initial_hidden_mean_sum=np.sum(smoothed_mean_next, axis=0),
            transition_count=int(interval_count * trajectory_count),
            trajectory_count=int(trajectory_count),
            log_likelihood=float(log_likelihood),
        )

    def _m_step(self, sufficient: EStepResult) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Closed-form maximizer of the paper's evidence lower bound."""
        ridge = np.finfo(float).eps * max(float(np.trace(sufficient.s_xx)), 1.0)
        s_xx = sufficient.s_xx + ridge * np.eye(sufficient.s_xx.shape[0])
        transition = np.linalg.solve(s_xx, sufficient.s_yx.T).T
        covariance = (
            sufficient.s_yy
            - transition @ sufficient.s_yx.T
            - sufficient.s_yx @ transition.T
            + transition @ sufficient.s_xx @ transition.T
        ) / float(sufficient.transition_count)
        covariance = _positive_definite(covariance, self.covariance_floor)
        initial_hidden_mean = (
            sufficient.initial_hidden_mean_sum / float(sufficient.trajectory_count)
        )
        return transition, covariance, initial_hidden_mean

    def _observed_log_likelihood(
        self,
        velocity: np.ndarray,
        transition: np.ndarray,
        covariance: np.ndarray,
        initial_hidden_mean: np.ndarray,
    ) -> float:
        """Evaluate the observed likelihood without an unnecessary smoother."""
        interval_count = velocity.shape[0] - 1
        trajectory_count = velocity.shape[1]
        d = self.hidden_dimension
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
            innovation = velocity[step + 1] - (
                f_vv * velocity[step] + mean @ f_vh
            )
            predicted_hidden = velocity[step, :, None] * f_hv[None, :] + mean @ f_hh.T
            mean = predicted_hidden + innovation[:, None] * (
                hidden_observation_covariance / innovation_variance
            )[None, :]
            p = _positive_definite(
                f_hh @ p @ f_hh.T
                + q_hh
                - np.outer(hidden_observation_covariance, hidden_observation_covariance)
                / innovation_variance,
                self.covariance_floor,
            )
            log_likelihood += -0.5 * (
                trajectory_count * np.log(2.0 * np.pi * innovation_variance)
                + float(innovation @ innovation) / innovation_variance
            )
        return float(log_likelihood)

    def fit(self, velocity: np.ndarray, dt: float) -> EMFit:
        velocity = np.asarray(velocity, dtype=float)
        if velocity.ndim != 2 or velocity.shape[0] < 3 or velocity.shape[1] < 2:
            raise ValueError("velocity must have shape (L+1, M) with L,M >= 2.")
        if not np.all(np.isfinite(velocity)) or float(dt) <= 0.0:
            raise ValueError("The likelihood data and timestep must be finite.")

        transition, covariance, initial_hidden_mean = self._initialize(velocity, float(dt))
        likelihood_history: list[float] = []
        converged = False
        completed_updates = 0

        for iteration in range(self.max_iterations):
            sufficient = self._e_step(
                velocity, transition, covariance, initial_hidden_mean
            )
            likelihood_history.append(sufficient.log_likelihood)
            if len(likelihood_history) >= 2:
                improvement = likelihood_history[-1] - likelihood_history[-2]
                numerical_allowance = 1.0e-9 * max(abs(likelihood_history[-2]), 1.0)
                if improvement < -numerical_allowance:
                    raise FloatingPointError(
                        "The observed likelihood decreased beyond numerical tolerance; "
                        "the EM sufficient statistics are inconsistent."
                    )
                if (
                    iteration + 1 >= self.minimum_iterations
                    and abs(improvement) < self.likelihood_tolerance
                ):
                    converged = True
                    break
            transition, covariance, initial_hidden_mean = self._m_step(sufficient)
            completed_updates += 1

        if not converged:
            likelihood_history.append(
                self._observed_log_likelihood(
                    velocity, transition, covariance, initial_hidden_mean
                )
            )

        return EMFit(
            transition=np.asarray(transition, dtype=float),
            covariance=np.asarray(covariance, dtype=float),
            initial_hidden_mean=np.asarray(initial_hidden_mean, dtype=float),
            log_likelihood=np.asarray(likelihood_history, dtype=float),
            converged=bool(converged),
            iterations=int(completed_updates),
        )


def continuous_memory_kernel(
    transition: np.ndarray, dt: float, evaluation_time: np.ndarray
) -> tuple[np.ndarray, float]:
    """Recover Eq. (3) from the fitted Euler drift blocks."""
    transition = np.asarray(transition, dtype=float)
    evaluation_time = np.asarray(evaluation_time, dtype=float)
    generator = (np.eye(transition.shape[0]) - transition) / float(dt)
    a_vv = float(generator[0, 0])
    a_vh = generator[0, 1:]
    a_hv = generator[1:, 0]
    a_hh = generator[1:, 1:]
    kernel = np.asarray(
        [-float(a_vh @ expm(-a_hh * time) @ a_hv) for time in evaluation_time],
        dtype=float,
    )
    return kernel, a_vv


def run_vroylandt_likelihood_em(
    config: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Generate the budget-matched trajectories and execute likelihood--EM."""
    velocity = generate_common_velocity_ensemble(config)
    em = config["em"]
    estimator = VroylandtLikelihoodEM(
        hidden_dimension=int(config["hidden_dimension"]),
        seed=int(config["initialization_seed"]),
        likelihood_tolerance=float(em["likelihood_tolerance"]),
        max_iterations=int(em["max_iterations"]),
        covariance_floor=float(em["covariance_floor"]),
        minimum_iterations=int(em["minimum_iterations"]),
    )
    fit = estimator.fit(velocity, float(config["dt"]))

    evaluation_time = np.linspace(
        0.0, float(config["evaluation_t_max"]), int(config["evaluation_grid_count"])
    )
    kernel_truth = np.asarray(
        get_kernel({"type": str(config["kernel_type"])}, float(config["dt"])).value(
            evaluation_time
        ),
        dtype=float,
    )
    kernel_estimate, _ = continuous_memory_kernel(
        fit.transition, float(config["dt"]), evaluation_time
    )
    _, relative_error = relative_l2_error(
        kernel_estimate, kernel_truth, evaluation_time
    )
    arrays = {
        "velocity_ensemble": velocity,
        "kernel_t": evaluation_time,
        "kernel_truth": kernel_truth,
        "kernel_estimate": kernel_estimate,
    }
    return arrays, {"kernel_relative_l2_error": float(relative_error)}
