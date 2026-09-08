"""Error metrics for Kramers--Moyal parameter estimates."""

import logging

import numpy as np

from gle_inverse.numerics.errors import relative_l2_error
from gle_inverse.numerics.kramers_moyal import brownian_euler_reference
from gle_inverse.numerics.levy import stable_annular_rates
from gle_inverse.simulation.noises import diffusion_profile, parse_noise_config

_SIGMA_L2_QUANTILES = (0.01, 0.99)
_SIGMA_L2_GRID_SIZE = 1001


class Evaluator:
    """Evaluate diffusion, stable-noise, and kernel estimates."""

    def __init__(self, config):
        self.config = config
        self.noise_model = parse_noise_config(config["noise"])

    def evaluate_sigma(self, x_ref, sigma_est, *, report=True):
        """Compare a diffusion-amplitude estimate in the continuous L2 norm."""
        x_ref = np.asarray(x_ref, dtype=float)
        sigma_est = np.asarray(sigma_est, dtype=float)
        sigma_ref = diffusion_profile(x_ref, self.noise_model.diffusion_name())

        if x_ref.ndim != 1 or sigma_est.shape != x_ref.shape:
            raise ValueError("Sigma evaluation requires one-dimensional state and estimate arrays of equal length.")
        if np.any(~np.isfinite(x_ref)) or np.any(~np.isfinite(sigma_est)):
            raise ValueError("Sigma evaluation received nonfinite states or estimates.")

        order = np.argsort(x_ref)
        x_sorted = x_ref[order]
        sigma_sorted = sigma_est[order]
        x_unique, first = np.unique(x_sorted, return_index=True)
        sigma_unique = sigma_sorted[first]
        lower, upper = np.quantile(x_ref, _SIGMA_L2_QUANTILES)
        if not np.isfinite(lower) or not np.isfinite(upper) or not lower < upper:
            raise ValueError("Sigma evaluation requires a nondegenerate occupied state interval.")
        x_eval = np.linspace(float(lower), float(upper), _SIGMA_L2_GRID_SIZE)
        sigma_eval = np.interp(x_eval, x_unique, sigma_unique)
        sigma_ref_eval = diffusion_profile(x_eval, self.noise_model.diffusion_name())
        err_abs, err_rel = relative_l2_error(sigma_eval, sigma_ref_eval, x_eval)

        if report:
            logging.info("=" * 30)
            logging.info("[Evaluation] Sigma")
            logging.info("  -> L2 quadrature points : %d", len(x_eval))
            logging.info("  -> Abs L2 Err : %.12f", err_abs)
            logging.info("  -> Rel L2 Err : %.12f", err_rel)
            logging.info("=" * 30)

        return err_abs, err_rel, sigma_ref

    @staticmethod
    def evaluate_d1(d1_emp, d1_ref, d1_fit):
        """Compare empirical and fitted first conditional moments."""
        d1_emp = np.asarray(d1_emp, dtype=float)
        d1_ref = np.asarray(d1_ref, dtype=float)
        d1_fit = np.asarray(d1_fit, dtype=float)

        norm_ref = float(np.linalg.norm(d1_ref))
        emp_abs = float(np.linalg.norm(d1_emp - d1_ref))
        fit_abs = float(np.linalg.norm(d1_fit - d1_ref))

        emp_rel = emp_abs / norm_ref
        fit_rel = fit_abs / norm_ref

        logging.info("=" * 30)
        logging.info("[Evaluation] First KM coefficient")
        logging.info("  -> Empirical Rel L2 : %.12f", emp_rel)
        logging.info("  -> Fitted Rel L2    : %.12f", fit_rel)
        logging.info("=" * 30)

        return emp_abs, emp_rel, fit_abs, fit_rel

    @staticmethod
    def finite_step_reference_moments(km, sigma_ref):
        """Return the Brownian Euler finite-step comparison targets."""
        drift = np.asarray(km["f"]) - np.asarray(km["mem"])
        return brownian_euler_reference(drift, sigma_ref, km["dt"])

    @staticmethod
    def evaluate_kernel(t_eval, kernel_est, kernel_ref):
        """Return absolute and relative kernel reconstruction errors."""
        t_eval = np.asarray(t_eval, dtype=float)
        kernel_est = np.asarray(kernel_est, dtype=float)
        kernel_ref = np.asarray(kernel_ref, dtype=float)

        err_abs, err_rel = relative_l2_error(kernel_est, kernel_ref, t_eval)

        logging.info("=" * 30)
        logging.info("[Evaluation] Memory kernel")
        logging.info("  -> Abs L2 Err : %.12f", float(err_abs))
        logging.info("  -> Rel L2 Err : %.12f", float(err_rel))
        logging.info("=" * 30)

        return err_abs, err_rel

    def evaluate_levy(self, km):
        """Evaluate stable parameters and annular-rate diagnostics."""
        if "levy" not in km:
            return {}
        levy = km["levy"]
        if self.noise_model.kind != "brownian_levy":
            raise ValueError("Lévy diagnostics require noise.type=brownian_levy.")
        alpha_true, kappa_true = self.noise_model.stable_parameters()
        alpha_est = float(levy["alpha"])
        kappa_est = float(levy["kappa"])
        alpha_rel = abs(alpha_est - alpha_true) / alpha_true
        kappa_rel = abs(kappa_est - kappa_true) / max(kappa_true, 1.0e-14)

        true_rates = stable_annular_rates(alpha_true, kappa_true, levy["epsilon"], levy["q"], levy["K"])
        empirical_rates = np.asarray(levy["empirical_rates"], dtype=float)
        fitted_rates = np.asarray(levy["fitted_rates"], dtype=float)
        rate_rel_true = float(np.linalg.norm(empirical_rates - true_rates) / max(np.linalg.norm(true_rates), 1.0e-14))
        rate_rel_fit = float(
            np.linalg.norm(empirical_rates - fitted_rates) / max(np.linalg.norm(empirical_rates), 1.0e-14)
        )

        logging.info("=" * 30)
        logging.info("[Evaluation] Two-sided alpha-stable component")
        logging.info("  -> alpha true / est : %.8f / %.8f", alpha_true, alpha_est)
        logging.info("  -> kappa true / est : %.8f / %.8f", kappa_true, kappa_est)
        logging.info("  -> alpha Rel Err    : %.12f", alpha_rel)
        logging.info("  -> kappa Rel Err    : %.12f", kappa_rel)
        logging.info("  -> Annular Rel Err  : %.12f", rate_rel_true)
        logging.info("=" * 30)

        return {
            "alpha_true": alpha_true,
            "alpha_est": alpha_est,
            "alpha_rel_error": float(alpha_rel),
            "kappa_true": kappa_true,
            "kappa_est": kappa_est,
            "kappa_rel_error": float(kappa_rel),
            "annular_rate_rel_error_to_truth": rate_rel_true,
            "annular_rate_rel_error_to_fit": rate_rel_fit,
            "true_rates": true_rates,
        }
