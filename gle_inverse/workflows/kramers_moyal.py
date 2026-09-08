"""Kramers--Moyal diffusion and memory-kernel workflow."""

import logging
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gle_inverse.numerics.basis import get_basis
from gle_inverse.simulation.generation import generate_and_save_trajectory
from gle_inverse.simulation.km_samples import (
    KM_GENERATION_EULER_ORACLE,
    KM_GENERATION_FINE_BRANCH,
    compute_km_samples,
)
from gle_inverse.simulation.observation import observation_horizon, observe_trajectory
from gle_inverse.solvers.kramers_moyal_kernel import KernelSolver
from gle_inverse.solvers.kramers_moyal_sigma import SigmaSolver
from gle_inverse.utils.artifacts import ArtifactStore
from gle_inverse.utils.metrics import Evaluator
from gle_inverse.utils.plots import (
    PlotCounter,
    plot_annular_rates,
    plot_d1,
    plot_kernel,
    plot_kernel_est,
    plot_km_samples,
    plot_levy_parameters,
    plot_loss,
    plot_noise,
    plot_sigma_est,
    plot_traj,
)
from gle_inverse.utils.results import result_arrays, result_metrics

_DIAGNOSTIC_BRANCH_LIMIT = 3000


@dataclass
class KramersMoyalProblem:
    """Data and conditional samples shared by both inverse stages."""

    config: dict
    driver_config: dict
    output_dir: Path
    t: np.ndarray
    v_true: np.ndarray
    noise: np.ndarray
    kernel_true: np.ndarray
    noise_components: dict
    t_observed: np.ndarray
    v_observed: np.ndarray
    t_inference: np.ndarray
    v_inference: np.ndarray
    memory_inference: np.ndarray
    km_samples: dict
    phase1_t_inference: np.ndarray | None = None
    phase1_v_inference: np.ndarray | None = None
    phase1_memory_inference: np.ndarray | None = None
    phase1_km_samples: dict | None = None


def _driver_config(config):
    """Extract the canonical flat configuration used by simulation code."""
    return {
        "simulation": deepcopy(config["generation"]["simulation"]),
        "kernel": deepcopy(config["kernel"]),
        "force": deepcopy(config["force"]),
        "noise": deepcopy(config["noise"]),
    }


def prepare_problem(
    config,
    *,
    experiment_name,
    data_dir,
    output_dir,
    brownian_base_dt=None,
    conditional_sample_seed=None,
    levy_estimate=None,
):
    """Generate observations and finite-sample conditional increments."""
    experiment_config = deepcopy(config)
    driver_config = _driver_config(experiment_config)
    km_generation = deepcopy(experiment_config["generation"].get("km_samples"))
    km_generation_method = str(km_generation or "euler_oracle").lower()
    phase1_observation = deepcopy(experiment_config["phase1"]["observation"])
    phase2_observation = deepcopy(experiment_config["phase2"]["observation"])
    output_dir = Path(output_dir)
    ArtifactStore(data_dir).prepare()
    ArtifactStore(output_dir).prepare()

    trajectory = generate_and_save_trajectory(
        driver_config, data_dir, experiment_name, trajectory_index=0, brownian_base_dt=brownian_base_dt
    )
    t, v_true, noise, kernel_true, _, _, noise_components = trajectory
    phase1_t_observed, phase1_v_observed, phase1_memory_observed = observe_trajectory(
        driver_config, phase1_observation, trajectory_index=0, trajectory=trajectory
    )
    phase1_physical_v_observed = phase1_v_observed
    phase1_physical_memory_observed = phase1_memory_observed
    if (
        km_generation_method in {KM_GENERATION_EULER_ORACLE, KM_GENERATION_FINE_BRANCH}
        and float(phase1_observation.get("sigma_obs", 0.0)) > 0.0
    ):
        clean_observation = deepcopy(phase1_observation)
        clean_observation["sigma_obs"] = 0.0
        phase1_clean_t, phase1_physical_v_observed, phase1_physical_memory_observed = observe_trajectory(
            driver_config, clean_observation, trajectory_index=0, trajectory=trajectory
        )
        if not np.array_equal(phase1_clean_t, phase1_t_observed):
            raise ValueError("Clean and noisy phase-1 observation grids must align.")

    if phase2_observation == phase1_observation:
        t_observed = phase1_t_observed
        v_observed = phase1_v_observed
        memory_observed = phase1_memory_observed
        physical_v_observed = phase1_physical_v_observed
        physical_memory_observed = phase1_physical_memory_observed
    else:
        t_observed, v_observed, memory_observed = observe_trajectory(
            driver_config, phase2_observation, trajectory_index=0, trajectory=trajectory
        )
        physical_v_observed = v_observed
        physical_memory_observed = memory_observed
        if (
            km_generation_method in {KM_GENERATION_EULER_ORACLE, KM_GENERATION_FINE_BRANCH}
            and float(phase2_observation.get("sigma_obs", 0.0)) > 0.0
        ):
            clean_observation = deepcopy(phase2_observation)
            clean_observation["sigma_obs"] = 0.0
            phase2_clean_t, physical_v_observed, physical_memory_observed = observe_trajectory(
                driver_config, clean_observation, trajectory_index=0, trajectory=trajectory
            )
            if not np.array_equal(phase2_clean_t, t_observed):
                raise ValueError("Clean and noisy phase-2 observation grids must align.")
    phase1_t, phase1_v, phase1_memory = phase1_t_observed, phase1_v_observed, phase1_memory_observed
    phase1_physical_v, phase1_physical_memory = phase1_physical_v_observed, phase1_physical_memory_observed
    t_inference, v_inference, memory_inference = t_observed, v_observed, memory_observed
    physical_v_inference, physical_memory_inference = physical_v_observed, physical_memory_observed
    phase1_horizon = observation_horizon(phase1_t)
    phase2_horizon = observation_horizon(t_inference)
    phase2_estimation = experiment_config["phase2"]["estimation"]
    phase2_estimation.setdefault("t_max", phase2_horizon)
    kernel_horizon = float(phase2_estimation["t_max"])
    if not np.isfinite(kernel_horizon) or kernel_horizon <= 0.0:
        raise ValueError("phase2.estimation.t_max must be finite and positive.")
    horizon_tolerance = 64.0 * np.finfo(float).eps * max(1.0, phase2_horizon)
    if kernel_horizon > phase2_horizon + horizon_tolerance:
        raise ValueError(
            "phase2.estimation.t_max cannot exceed the phase-2 observation horizon: "
            f"t_max={kernel_horizon:.8g}, horizon={phase2_horizon:.8g}."
        )
    logging.info(
        "[DATA] Phase 1 uses L=%d intervals (%d observations) on [0, %.8g]; phase 2 uses L=%d intervals "
        "(%d observations) on [0, %.8g]; kernel reconstruction uses [0, %.8g].",
        len(phase1_t) - 1,
        len(phase1_t),
        phase1_horizon,
        len(t_inference) - 1,
        len(t_inference),
        phase2_horizon,
        kernel_horizon,
    )
    phase1_km_samples = compute_km_samples(
        driver_config,
        phase1_t,
        phase1_v,
        phase1_memory,
        phase1_observation,
        sample_size=driver_config["simulation"]["num_steps"],
        trajectory_index=0,
        random_seed=conditional_sample_seed,
        levy_estimation=experiment_config["phase1"]["estimation"],
        levy_estimate=levy_estimate,
        generation_config=km_generation,
        physical_v_observed=phase1_physical_v,
        physical_memory_observed=phase1_physical_memory,
    )
    if phase1_observation == phase2_observation:
        km_samples = phase1_km_samples
    else:
        km_samples = compute_km_samples(
            driver_config,
            t_inference,
            v_inference,
            memory_inference,
            phase2_observation,
            sample_size=driver_config["simulation"]["num_steps"],
            trajectory_index=0,
            random_seed=conditional_sample_seed,
            levy_estimation=experiment_config["phase1"]["estimation"],
            levy_estimate=(levy_estimate if levy_estimate is not None else phase1_km_samples.get("levy")),
            generation_config=km_generation,
            physical_v_observed=physical_v_inference,
            physical_memory_observed=physical_memory_inference,
        )
    return KramersMoyalProblem(
        config=experiment_config,
        driver_config=driver_config,
        output_dir=output_dir,
        t=t,
        v_true=v_true,
        noise=noise,
        kernel_true=kernel_true,
        noise_components=noise_components,
        t_observed=t_observed,
        v_observed=v_observed,
        t_inference=t_inference,
        v_inference=v_inference,
        memory_inference=memory_inference,
        km_samples=km_samples,
        phase1_t_inference=phase1_t,
        phase1_v_inference=phase1_v,
        phase1_memory_inference=phase1_memory,
        phase1_km_samples=phase1_km_samples,
    )


def solve_problem(problem, *, make_plots=True):
    """Run both inverse stages for one prepared Kramers--Moyal problem."""
    config = problem.config
    km_samples = problem.km_samples
    phase1_km_samples = problem.km_samples if problem.phase1_km_samples is None else problem.phase1_km_samples
    plot_counter = PlotCounter()

    if make_plots:
        plot_kernel(problem.t, problem.kernel_true, problem.output_dir, prefix=plot_counter.get_next())
        plot_traj(
            problem.t_observed,
            problem.t,
            problem.v_observed,
            problem.v_true,
            problem.output_dir,
            prefix=plot_counter.get_next(),
        )
        plot_noise(problem.t, problem.noise, problem.output_dir, prefix=plot_counter.get_next())
        plot_km_samples(
            phase1_km_samples["x"],
            phase1_km_samples["v_next"],
            problem.output_dir,
            num_ref=1,
            max_samples=_DIAGNOSTIC_BRANCH_LIMIT,
            prefix=plot_counter.get_next(),
        )

    phase1_config = deepcopy(problem.driver_config)
    phase1_config["estimation"] = deepcopy(config["phase1"]["estimation"])

    levy_metrics = {}
    levy = None
    if "levy" in phase1_km_samples:
        levy_metrics = Evaluator(phase1_config).evaluate_levy(phase1_km_samples)
        levy = phase1_km_samples["levy"]

    sigma_solver = SigmaSolver(phase1_config)
    sigma_pilot, pilot_loss = sigma_solver.solve(phase1_km_samples["x"], phase1_km_samples["d2"])
    sigma_est = sigma_pilot
    loss_sigma = list(pilot_loss)

    sigma_abs, sigma_rel, sigma_true = Evaluator(phase1_config).evaluate_sigma(phase1_km_samples["x"], sigma_est)
    if make_plots:
        plot_sigma_est(
            phase1_config, phase1_km_samples["x"], sigma_est, problem.output_dir, prefix=plot_counter.get_next()
        )
        plot_loss(loss_sigma, problem.output_dir, prefix=plot_counter.get_next())

    if levy is not None and make_plots:
        plot_levy_parameters(
            levy_metrics["alpha_true"],
            levy_metrics["alpha_est"],
            levy_metrics["kappa_true"],
            levy_metrics["kappa_est"],
            problem.output_dir,
            prefix=plot_counter.get_next(),
        )
        plot_annular_rates(
            levy["bin_edges"],
            levy["empirical_rates"],
            levy["fitted_rates"],
            levy_metrics["true_rates"],
            problem.output_dir,
            prefix=plot_counter.get_next(),
        )

    phase2_config = deepcopy(problem.driver_config)
    phase2_config["estimation"] = deepcopy(config["phase2"]["estimation"])
    kernel_basis = get_basis(phase2_config["estimation"])
    kernel_solver = KernelSolver(kernel_basis, problem.t_inference)
    reference_indices = np.asarray(km_samples["idx"])
    d1_empirical = np.asarray(km_samples["d1"])
    force_reference = np.asarray(km_samples["f"])
    kernel_est, d1_fit, _ = kernel_solver.solve(
        problem.v_inference, reference_indices, d1_empirical, force_reference, problem.t
    )
    logging.info(
        ">>> [Regularization] mode=%s, selected lambda=%.8g", kernel_solver.mode, kernel_solver.selected_lambda
    )

    sigma_est = sigma_solver.predict(km_samples["x"])
    _, _, sigma_true = Evaluator(phase1_config).evaluate_sigma(km_samples["x"], sigma_est, report=False)
    d1_reference, d2_reference = Evaluator.finite_step_reference_moments(km_samples, sigma_true)
    d1_metrics = Evaluator(phase2_config).evaluate_d1(d1_empirical, d1_reference, d1_fit)
    kernel_mask = problem.t <= float(phase2_config["estimation"]["t_max"])
    kernel_metrics = Evaluator(phase2_config).evaluate_kernel(
        problem.t[kernel_mask], kernel_est[kernel_mask], problem.kernel_true[kernel_mask]
    )
    if make_plots:
        plot_d1(
            np.asarray(km_samples["x"]),
            d1_reference,
            d1_empirical,
            d1_fit,
            problem.output_dir,
            prefix=plot_counter.get_next(),
        )
        plot_kernel_est(problem.t, problem.kernel_true, kernel_est, problem.output_dir, prefix=plot_counter.get_next())

    arrays = result_arrays(
        problem.t,
        problem.v_true,
        problem.noise,
        problem.t_observed,
        problem.v_observed,
        km_samples,
        sigma_est,
        sigma_true,
        d1_reference,
        d2_reference,
        d1_fit,
        kernel_solver,
        kernel_est,
        problem.kernel_true,
        reference_indices,
        noise_components=problem.noise_components,
    )
    arrays.update(
        {
            "t_inference": problem.t_inference,
            "v_inference": problem.v_inference,
            "memory_inference": problem.memory_inference,
            "phase1_t_inference": (
                problem.t_inference if problem.phase1_t_inference is None else problem.phase1_t_inference
            ),
            "phase1_v_inference": (
                problem.v_inference if problem.phase1_v_inference is None else problem.phase1_v_inference
            ),
        }
    )
    metrics = result_metrics(
        km_samples, (sigma_abs, sigma_rel), d1_metrics, kernel_metrics, kernel_solver, levy_metrics
    )
    metrics["sigma"]["estimator"] = sigma_solver.metadata()
    metrics["inference_data"] = {
        "phase1": {
            "L": int(
                len(problem.t_inference) - 1
                if problem.phase1_t_inference is None
                else len(problem.phase1_t_inference) - 1
            ),
            "observation_points": int(
                len(problem.t_inference) if problem.phase1_t_inference is None else len(problem.phase1_t_inference)
            ),
            "t_start": float(
                problem.t_inference[0] if problem.phase1_t_inference is None else problem.phase1_t_inference[0]
            ),
            "t_end": float(
                problem.t_inference[-1] if problem.phase1_t_inference is None else problem.phase1_t_inference[-1]
            ),
        },
        "phase2": {
            "L": int(len(problem.t_inference) - 1),
            "observation_points": int(len(problem.t_inference)),
            "t_start": float(problem.t_inference[0]),
            "t_end": float(problem.t_inference[-1]),
        },
    }
    return arrays, metrics


def run_experiment(config, *, experiment_name, output_dir, make_plots=True):
    """Prepare and solve the standard production workflow."""
    problem = prepare_problem(config, experiment_name=experiment_name, data_dir=output_dir, output_dir=output_dir)
    arrays, metrics = solve_problem(problem, make_plots=make_plots)
    return Path(output_dir), arrays, metrics
