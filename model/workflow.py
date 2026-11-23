"""GLE 数据生成、两阶段反演与误差计算。"""

import numpy as np

from model.data import diffusion_profile, get_force, observe, simulate_paths
from model.embedding import kernel_from_coefficients
from model.step1_kernel import estimate_kernel
from model.step2_noise import estimate_noise, predict_sigma, stable_annular_rates
from model.utils.io import save_yaml
from model.utils.metrics import kernel_error, levy_error, memory_error, sigma_error


def prepare(config):
    paths = simulate_paths(config)
    obs = observe(paths, config["observation"]["sigma_obs"], config["simulation"]["seed"])
    return {"config": config, "paths": paths, "obs": obs}


def estimate(obs, config, seed, *, start=0):
    s1, s2 = config["step1"], config["step2"]
    est = estimate_kernel(obs["v"], obs["t"], config["force"], s1, config["moments"], seed, start=start)
    levy, net, loss = estimate_noise(
        obs["v"][:, start:], obs["dt"], est["scale"], est["drift"], config["moments"], s2,
        config["noise"]["type"] == "brownian_levy", seed, config["compute"]["device"],
    )
    return est, levy, net, loss


def invert(p):
    config, paths, obs = p["config"], p["paths"], p["obs"]
    v, t, dt = obs["v"], obs["t"], obs["dt"]
    s1 = config["step1"]
    levy_case = config["noise"]["type"] == "brownian_levy"

    est, levy, net, loss = estimate(obs, config, config["simulation"]["seed"])
    kernel_est = kernel_from_coefficients(est["u"], est["K"], est["a"], t)
    memory_est = est["z"] @ est["u"]

    memory_true = obs["memory_true"][:, :-1]
    x = np.linspace(*config["evaluation"]["sigma_range"], 1001)
    sigma_est = predict_sigma(net, x)
    sigma_true = diffusion_profile(x, config["noise"]["diffusion"])
    metrics = {
        "data": {"num_paths": v.shape[0], "steps_per_path": v.shape[1] - 1, "dt_obs": dt},
        "sigma": sigma_error(x, sigma_est, sigma_true),
        "kernel": kernel_error(t, kernel_est, obs["kernel_true"], s1["t_max"])
        | {"K": est["K"], "a": est["a"], "condition": est["condition"]},
        "moments": {
            "epsilon": config["moments"]["epsilon"], "sigma_state_count": int(np.sum(est["scale"] <= 1.0)),
            "sigma_range": config["evaluation"]["sigma_range"],
        },
        "cross_validation": {"K_grid": s1["K_grid"], "a_grid": s1["a_grid"], "scores": est["scores"]},
        "memory": memory_error(memory_est, memory_true, t[:-1]),
    }
    arrays = {
        "t": t, "t_sim": paths["t_fine"], "v_true": paths["v_fine"], "v_obs": v[0],
        "noise_increment": paths["noise_fine"],
        "kernel_true": obs["kernel_true"], "kernel_est": kernel_est, "kernel_coefficients": est["u"],
        "force_path": get_force(config["force"], v[0, :-1]),
        "memory_true_path": memory_true[0], "memory_est_path": memory_est[0],
        "moment_scale_path": est["scale"][0],
        "sigma_x": x, "sigma_est": sigma_est, "sigma_true": sigma_true, "loss_sigma": loss,
    }
    if levy_case:
        alpha, kappa = config["noise"]["alpha"], config["noise"]["kappa"]
        metrics["levy"] = levy_error(alpha, kappa, levy)
        arrays["levy_bin_edges"] = levy["bin_edges"]
        arrays["levy_empirical_rates"] = levy["empirical_rates"]
        arrays["levy_fitted_rates"] = levy["fitted_rates"]
        arrays["levy_true_rates"] = stable_annular_rates(alpha, kappa, levy["epsilon"], levy["q"], levy["n_bins"])
    return arrays, metrics, est


def run(config, out):
    out.mkdir(parents=True)
    save_yaml(out / "parameters.yaml", config)
    arrays, metrics, _ = invert(prepare(config))
    np.savez_compressed(out / "estimates.npz", **arrays)
    save_yaml(out / "metrics.yaml", metrics)
    return metrics
