"""记忆核、记忆力与噪声参数的误差。"""

import numpy as np

from model.step2_noise import stable_annular_rates


def relative_l2_error(estimate, reference, grid):
    absolute = np.sqrt(np.mean(np.trapezoid((reference - estimate) ** 2, grid, axis=-1)))
    return absolute, absolute / np.sqrt(np.mean(np.trapezoid(reference**2, grid, axis=-1)))


def kernel_error(t, kernel_est, kernel_true, t_max):
    mask = t <= t_max
    absolute, relative = relative_l2_error(kernel_est[mask], kernel_true[mask], t[mask])
    return {"absolute_l2_error": absolute, "relative_l2_error": relative, "t_max": t_max}


def memory_error(memory_est, memory_true, t):
    absolute, relative = relative_l2_error(memory_est, memory_true, t)
    return {"absolute_l2_error": absolute, "relative_l2_error": relative, "t_min": t[0], "t_max": t[-1]}


def sigma_error(x, sigma_est, sigma_true):
    absolute, relative = relative_l2_error(sigma_est, sigma_true, x)
    return {"absolute_l2_error": absolute, "relative_l2_error": relative, "v_min": x[0], "v_max": x[-1]}


def levy_error(alpha, kappa, levy):
    true_rates = stable_annular_rates(alpha, kappa, levy["epsilon"], levy["q"], levy["n_bins"])
    empirical, fitted = levy["empirical_rates"], levy["fitted_rates"]
    alpha_error, kappa_error = abs(levy["alpha"] - alpha) / abs(alpha), abs(levy["kappa"] - kappa) / abs(kappa)
    return {"alpha_true": alpha, "alpha_est": levy["alpha"], "alpha_rel_error": alpha_error,
            "kappa_true": kappa, "kappa_est": levy["kappa"], "kappa_rel_error": kappa_error,
            "joint_rel_error": np.maximum(alpha_error, kappa_error),
            "annular_rate_rel_error_to_truth": np.linalg.norm(empirical - true_rates) / np.linalg.norm(true_rates),
            "annular_rate_rel_error_to_fit": np.linalg.norm(empirical - fitted) / np.linalg.norm(empirical),
            "tail_count": levy["tail_count"], "epsilon": levy["epsilon"],
            "exposure": levy["exposure"], "identified": levy["identified"]}
