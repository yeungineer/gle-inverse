from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import torch.optim as optim
import yaml
from torch import nn

from model.inverse import SplineBasis, brownian_euler_reference, fit_kernel, stable_annular_rates
from model.simulation import diffusion_profile, generate_trajectory, observation_horizon, observe_trajectory, sample_conditional


def fit_sigma(x, d2, config):
    """Fit D2 = softplus(NN(x))² / 2 using AdamW, then L-BFGS."""
    settings = config["estimation"]
    torch.manual_seed(int(config["simulation"]["seed"]))
    layers = [nn.Linear(1, 64), nn.GELU()]
    for _ in range(2):
        layers.extend((nn.Linear(64, 64), nn.GELU()))
    layers.append(nn.Linear(64, 1))
    net = nn.Sequential(*layers).to("cpu").double()
    # Fixed weight decay and L-BFGS tolerances.
    opt = optim.AdamW(net.parameters(), lr=float(settings["lr"]), weight_decay=1e-08)
    x = torch.from_numpy(np.asarray(x, dtype=float)).double().view(-1, 1).to("cpu")
    d2 = torch.from_numpy(np.asarray(d2, dtype=float)).double().view(-1, 1).to("cpu")
    loss_history = []
    net.train()
    for _ in range(int(settings["epochs"])):
        opt.zero_grad()
        sigma = functional.softplus(net(x))
        d2_fit = 0.5 * sigma**2
        loss = torch.mean((d2 - d2_fit) ** 2)
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
    polish = optim.LBFGS(
        net.parameters(), lr=0.8, max_iter=100, tolerance_grad=1e-12, tolerance_change=1e-14, line_search_fn="strong_wolfe"
    )

    def closure():
        polish.zero_grad()
        sigma = functional.softplus(net(x))
        d2_fit = 0.5 * sigma**2
        loss = torch.mean((d2 - d2_fit) ** 2)
        loss.backward()
        return loss

    polish.step(closure)
    with torch.no_grad():
        sigma = functional.softplus(net(x))
        loss_history.append(float(torch.mean((d2 - 0.5 * sigma**2) ** 2).item()))
    net.eval()
    return (net, np.asarray(loss_history))


def predict_sigma(net, x):
    net.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(x, dtype=float)).double().view(-1, 1).to("cpu")
        return functional.softplus(net(x)).cpu().numpy().reshape(-1)


def relative_l2_error(estimate, reference, time):
    estimate = np.asarray(estimate, dtype=float)
    reference = np.asarray(reference, dtype=float)
    time = np.asarray(time, dtype=float)
    absolute = np.sqrt(np.trapezoid((reference - estimate) ** 2, time))
    reference_norm = np.sqrt(np.trapezoid(reference**2, time))
    return (float(absolute), float(absolute / reference_norm))


_SIGMA_L2_QUANTILES = (0.01, 0.99)
_SIGMA_L2_GRID_SIZE = 1001


def evaluate_sigma(config, x_ref, sigma_est):
    x_ref = np.asarray(x_ref, dtype=float)
    sigma_est = np.asarray(sigma_est, dtype=float)
    sigma_ref = diffusion_profile(x_ref, config["noise"]["diffusion"])
    order = np.argsort(x_ref)
    x_sorted = x_ref[order]
    sigma_sorted = sigma_est[order]
    x_unique, first = np.unique(x_sorted, return_index=True)
    sigma_unique = sigma_sorted[first]
    lower, upper = np.quantile(x_ref, _SIGMA_L2_QUANTILES)
    x_eval = np.linspace(float(lower), float(upper), _SIGMA_L2_GRID_SIZE)
    sigma_eval = np.interp(x_eval, x_unique, sigma_unique)
    sigma_ref_eval = diffusion_profile(x_eval, config["noise"]["diffusion"])
    err_abs, err_rel = relative_l2_error(sigma_eval, sigma_ref_eval, x_eval)
    return (err_abs, err_rel, sigma_ref)


def evaluate_d1(d1_emp, d1_ref, d1_fit):
    d1_emp = np.asarray(d1_emp, dtype=float)
    d1_ref = np.asarray(d1_ref, dtype=float)
    d1_fit = np.asarray(d1_fit, dtype=float)
    norm_ref = float(np.linalg.norm(d1_ref))
    emp_abs = float(np.linalg.norm(d1_emp - d1_ref))
    fit_abs = float(np.linalg.norm(d1_fit - d1_ref))
    emp_rel = emp_abs / norm_ref
    fit_rel = fit_abs / norm_ref
    return (emp_abs, emp_rel, fit_abs, fit_rel)


def evaluate_levy(config, km):
    if "levy" not in km:
        return {}
    levy = km["levy"]
    alpha_true, kappa_true = (config["noise"]["alpha"], config["noise"]["kappa"])
    alpha_est = float(levy["alpha"])
    kappa_est = float(levy["kappa"])
    alpha_rel = abs(alpha_est - alpha_true) / alpha_true
    kappa_rel = abs(kappa_est - kappa_true) / kappa_true
    true_rates = stable_annular_rates(alpha_true, kappa_true, levy["epsilon"], levy["q"], levy["K"])
    empirical_rates = np.asarray(levy["empirical_rates"], dtype=float)
    fitted_rates = np.asarray(levy["fitted_rates"], dtype=float)
    rate_rel_true = float(np.linalg.norm(empirical_rates - true_rates) / np.linalg.norm(true_rates))
    rate_rel_fit = float(np.linalg.norm(empirical_rates - fitted_rates) / np.linalg.norm(empirical_rates))
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


def _observations(driver, trajectory, observation, method):
    t, v, mem = observe_trajectory(driver, observation, 0, trajectory=trajectory)
    physical_v, physical_mem = (v, mem)
    if method in ("euler_oracle", "fine_branch") and float(observation.get("sigma_obs", 0.0)) > 0:
        clean = dict(observation, sigma_obs=0.0)
        _, physical_v, physical_mem = observe_trajectory(driver, clean, 0, trajectory=trajectory)
    return (t, v, mem, physical_v, physical_mem)


def prepare(config, *, brownian_base_dt=None, conditional_sample_seed=None, levy_estimate=None):
    """Generate a path and its two observation/conditional-sampling banks."""
    config = deepcopy(config)
    driver = {key: deepcopy(config[key]) for key in ("kernel", "force", "noise")}
    driver["simulation"] = deepcopy(config["generation"]["simulation"])
    method = config["generation"]["km_samples"]
    trajectory = generate_trajectory(driver, 0, brownian_base_dt=brownian_base_dt)
    t, v, noise, K, _, _, components = trajectory
    obs1, obs2 = (config["phase1"]["observation"], config["phase2"]["observation"])
    first = _observations(driver, trajectory, obs1, method)
    second = first if obs1 == obs2 else _observations(driver, trajectory, obs2, method)
    horizon = observation_horizon(second[0])
    settings = config["phase2"]["estimation"]
    settings.setdefault("t_max", horizon)
    assert 0 < settings["t_max"] <= horizon + 64 * np.finfo(float).eps * max(1.0, horizon)
    banks = []
    for observation, observed in ((obs1, first), (obs2, second)):
        if banks and obs1 == obs2:
            banks.append(banks[0])
            break
        ts, vs, mem, physical_v, physical_mem = observed
        tail = levy_estimate if levy_estimate is not None or not banks else banks[0].get("levy")
        banks.append(
            sample_conditional(
                driver, ts, vs, mem, observation, sample_size=driver["simulation"]["num_steps"], trajectory_index=0,
                random_seed=conditional_sample_seed, levy_estimation=config["phase1"]["estimation"], levy_estimate=tail,
                generation_config=method, physical_v_observed=physical_v, physical_memory_observed=physical_mem,
            )
        )
    return {
        "config": config,
        "driver_config": driver,
        "t": t,
        "v_true": v,
        "noise": noise,
        "kernel_true": K,
        "noise_components": components,
        "t_observed": second[0],
        "v_observed": second[1],
        "t_inference": second[0],
        "v_inference": second[1],
        "memory_inference": second[2],
        "km_samples": banks[1],
        "phase1_t_inference": first[0],
        "phase1_v_inference": first[1],
        "phase1_memory_inference": first[2],
        "phase1_km_samples": banks[0],
    }


def solve(p):
    """Conditional moments → diffusion fit → Volterra inversion → errors."""
    config = p["config"]
    km = p["km_samples"]
    km1 = p["phase1_km_samples"]
    c1 = deepcopy(p["driver_config"])
    c1["estimation"] = deepcopy(config["phase1"]["estimation"])
    levy_metrics = evaluate_levy(c1, km1) if "levy" in km1 else {}
    net, loss = fit_sigma(km1["x"], km1["d2"], c1)
    sigma1 = predict_sigma(net, km1["x"])
    sigma_abs, sigma_rel, _ = evaluate_sigma(c1, km1["x"], sigma1)
    settings = config["phase2"]["estimation"]
    basis = SplineBasis(int(settings["num_basis"]), float(settings["t_max"]))
    K, d1_fit, c, selection = fit_kernel(p["v_inference"], p["t_inference"], km["idx"], km["d1"], km["f"], p["t"], basis)
    sigma = predict_sigma(net, km["x"])
    _, _, sigma_true = evaluate_sigma(c1, km["x"], sigma)
    d1_ref, d2_ref = brownian_euler_reference(np.asarray(km["f"]) - np.asarray(km["mem"]), sigma_true, km["dt"])
    d1_errors = evaluate_d1(km["d1"], d1_ref, d1_fit)
    mask = p["t"] <= float(settings["t_max"])
    K_errors = relative_l2_error(K[mask], p["kernel_true"][mask], p["t"][mask])
    arrays = {key: p[key] for key in ("t", "v_true", "kernel_true", "t_inference", "v_inference")}
    arrays.update(
        noise_increment=p["noise"], x_ref=km["x"], kernel_reference_index=km["idx"], force_ref=km["f"],
        d1_reference=d1_ref, d1_empirical=km["d1"], d1_fit=d1_fit,
        d2_finite_step_reference=d2_ref, d2_diffusion_target=km["d2"], sigma_est=sigma, sigma_true=sigma_true,
        kernel_coefficients=c, kernel_est=K, regularization_singular_values=np.asarray(selection["singular_values"], dtype=float),
        phase1_x=km1["x"], phase1_sigma=sigma1, loss_sigma=loss,
        phase1_branch_state=km1["x"][:1], phase1_branch_endpoints=km1["v_next"][0],
    )
    metrics = {
        "sigma": dict(absolute_l2_error=sigma_abs, relative_l2_error=sigma_rel),
        "kernel": dict(zip(("absolute_l2_error", "relative_l2_error"), K_errors)),
        "first_moment": dict(
            zip(
                ("empirical_absolute_l2_error", "empirical_relative_l2_error", "fitted_absolute_l2_error", "fitted_relative_l2_error"),
                d1_errors,
            )
        ),
        "regularization": dict(
            mode="r1gcv", selected_lambda=float(selection["lambda"]), selected_at_search_boundary=bool(selection["selected_at_boundary"])
        ),
    }
    if "levy" in km:
        for key in ("bin_edges", "empirical_rates", "fitted_rates"):
            arrays["levy_" + key] = km["levy"][key]
        metrics["levy"] = {k: v for k, v in levy_metrics.items() if k != "true_rates"}
        for key in ("epsilon", "q", "K", "tail_count", "alpha_standard_error", "log_kappa_standard_error"):
            metrics["levy"][key] = km["levy"][key]
    return (arrays, metrics)


ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    return yaml.safe_load(Path(path).read_text())


def replace(config, overrides):
    config = deepcopy(config)
    for path, value in overrides.items():
        keys = path.split(".")
        target = config
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = deepcopy(value)
    return config


def configured_case(case, parameters):
    return replace(load_config(ROOT / f"{case}.yaml"), parameters)


def run_point(case, parameters):
    p = prepare(configured_case(case, parameters))
    arrays, metrics = solve(p)
    return (p, arrays, metrics)


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for (k, v) in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_yaml(path, value):
    Path(path).write_text(yaml.safe_dump(serializable(value), sort_keys=False, allow_unicode=True))


def float_column(rows, key):
    return np.asarray([np.nan if row.get(key) is None or row.get(key) == "" else float(row[key]) for row in rows], dtype=float)


def string_column(rows, key):
    return np.asarray(["" if row.get(key) is None else str(row[key]) for row in rows], dtype=str)


def save_figure_data(out, number, arrays, metadata):
    np.savez_compressed(out / f"figure_{number}_data.npz", **arrays)
    save_yaml(out / f"figure_{number}_metadata.yaml", metadata)
