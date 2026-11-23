"""GLE 轨迹模拟与观测数据生成。"""

from math import gamma, pi, sqrt

import numpy as np

KERNEL_WEIGHTS = np.array([0.4021, 0.4021, 0.2753, 0.4638, 0.3187])
KERNEL_RATES = np.array([-0.1425 - 0.2987j, -0.1425 + 0.2987j, -0.7814 + 0j, -0.9643 + 0j, -0.4012 + 0j])
POWER_LAW_RATES = -np.logspace(-2.5, 1.5, 48)


def kernel(t, kind):
    if kind == "complex_exp":
        return (np.exp(np.outer(t, KERNEL_RATES)) @ KERNEL_WEIGHTS).real
    if kind == "power_law":
        return (1.0 - 3.0 * t**2) / (1.0 + t**2) ** 3


def exponential_sum(kind):
    if kind == "complex_exp":
        return KERNEL_WEIGHTS.astype(complex), KERNEL_RATES
    t = np.linspace(0.0, 60.0, 60001)
    basis = np.exp(np.outer(t, POWER_LAW_RATES))
    weights = np.linalg.lstsq(basis, kernel(t, kind), rcond=None)[0]
    return weights.astype(complex), POWER_LAW_RATES.astype(complex)


def get_force(force_config, v):
    kind = force_config["type"]
    if kind == "zero":
        return np.zeros_like(v)
    if kind == "linear":
        return -v
    if kind == "bistable":
        return v - v**3
    if kind == "sin":
        return np.sin(2.0 * v)


def diffusion_profile(v, kind):
    if kind == "constant":
        return np.full_like(v, 0.5)
    if kind == "cosine":
        return 0.5 * (np.cos(v) + 1.0)


def brownian_increments(size, dt, rng):
    return np.sqrt(dt) * rng.standard_normal(size)


def stable_increments(size, dt, alpha, rng):
    U = rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size)
    E = rng.exponential(1.0, size)
    S = np.sin(alpha * U) / np.cos(U) ** (1.0 / alpha) * (np.cos((1.0 - alpha) * U) / E) ** ((1.0 - alpha) / alpha)
    return dt ** (1.0 / alpha) * S


def stable_constant(alpha):
    return alpha * 2.0 ** (alpha - 1.0) * gamma((1.0 + alpha) / 2.0) / (sqrt(pi) * gamma(1.0 - alpha / 2.0))


def simulate_paths(config):
    sim, noise_cfg = config["simulation"], config["noise"]
    dt, L, M, r = sim["dt"], sim["L"], sim["num_paths"], sim["substeps"]
    h = dt / r
    rng = np.random.default_rng(sim["seed"])
    c, lam = exponential_sum(config["kernel"]["type"])
    decay = np.exp(lam * h)
    gain = (decay - 1.0) / lam

    t = dt * np.arange(L + 1)
    v = np.zeros((M, L + 1))
    memory = np.zeros((M, L + 1))
    v_fine = np.zeros(L * r + 1)
    noise_fine = np.zeros(L * r)
    y = np.zeros((M, len(lam)), dtype=complex)
    x = rng.standard_normal(M)
    v[:, 0], v_fine[0] = x, x[0]
    for i in range(L * r):
        force_memory = get_force(config["force"], x) - (y @ c).real
        noise = (diffusion_profile(x, noise_cfg["diffusion"]) * brownian_increments(M, h, rng)
                 + noise_cfg["kappa"] * stable_increments(M, h, noise_cfg["alpha"], rng))
        y = y * decay + np.outer(x, gain)
        x = x + force_memory * h + noise
        v_fine[i + 1], noise_fine[i] = x[0], noise[0]
        if (i + 1) % r == 0:
            l = (i + 1) // r
            v[:, l] = x
            memory[:, l] = (y @ c).real
    return {"t": t, "v": v, "memory": memory, "kernel": kernel(t, config["kernel"]["type"]),
            "t_fine": h * np.arange(L * r + 1), "v_fine": v_fine, "noise_fine": noise_fine}


def observe(paths, sigma_obs, seed):
    rng = np.random.default_rng(seed + 1)
    v = paths["v"] + sigma_obs * rng.standard_normal(paths["v"].shape)
    t = paths["t"]
    return {"t": t, "v": v, "dt": t[1] - t[0], "memory_true": paths["memory"], "kernel_true": paths["kernel"]}
