"""Lévy 参数与布朗噪声振幅估计。"""

import numpy as np
import torch
import torch.nn.functional as functional
from torch import nn, optim

from model.data import stable_constant
from model.moments import empirical_second_sample, small_jump_second_moment, theoretical_second_moment


def stable_annular_rates(alpha, kappa, epsilon, q, n_bins):
    k = np.arange(n_bins + 1)
    return (2.0 * stable_constant(alpha) * kappa**alpha / alpha
            * epsilon ** (-alpha) * q ** (-k * alpha) * (1.0 - q ** (-alpha)))


def estimate_levy(increment, dt, scale, epsilon, q, n_bins):
    size = np.abs(increment.reshape(-1))
    tail = size[size >= epsilon]
    alpha = tail.size / np.sum(np.log(tail / epsilon))
    edges = epsilon * q ** np.arange(n_bins + 2)
    counts = np.histogram(size, bins=edges)[0]
    result = {"alpha": alpha, "tail_count": tail.size, "epsilon": epsilon, "q": q,
              "n_bins": n_bins, "bin_edges": edges, "identified": bool(0 < alpha < 2)}
    if not result["identified"]:
        return result | {"kappa": np.nan, "exposure": np.nan, "small_jump_second_moment": np.nan,
                         "empirical_rates": np.full(n_bins + 1, np.nan), "fitted_rates": np.full(n_bins + 1, np.nan)}
    exposure = dt * np.sum(scale ** (-alpha))
    kappa = (tail.size * alpha * epsilon**alpha / (2.0 * stable_constant(alpha) * exposure)) ** (1.0 / alpha)
    return result | {"kappa": kappa, "exposure": exposure, "empirical_rates": counts / exposure,
                     "fitted_rates": stable_annular_rates(alpha, kappa, epsilon, q, n_bins),
                     "small_jump_second_moment": small_jump_second_moment(alpha, kappa, epsilon)}


def network_sigma(net, v):
    return functional.softplus(net(v))


def fit_sigma(v, empirical, jump_second_moment, lr, epochs, batch_size, seed, device="cpu"):
    torch.manual_seed(seed)
    net = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 1)).to(device)
    x = torch.tensor(v.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)
    Y = torch.tensor(empirical.reshape(-1), dtype=torch.float32, device=device).view(-1, 1)

    def objective(index):
        return torch.mean(
            (Y[index] - theoretical_second_moment(network_sigma(net, x[index]), jump_second_moment)) ** 2
        )

    adam = optim.Adam(net.parameters(), lr=lr)
    history = []
    for epoch in range(epochs):
        for group in adam.param_groups:
            group["lr"] = lr * 0.01 ** (epoch / (epochs - 1))
        order = torch.randperm(len(x), device=device)
        for start in range(0, len(x), batch_size):
            adam.zero_grad()
            loss = objective(order[start:start + batch_size])
            loss.backward()
            adam.step()
        with torch.no_grad():
            total = sum(
                (min(start + batch_size, len(x)) - start) * objective(slice(start, start + batch_size)).item()
                for start in range(0, len(x), batch_size)
            )
            history.append(total / len(x))
    return net, np.array(history)


def predict_sigma(net, v):
    if net is None:
        return np.full_like(v, np.nan)
    device = next(net.parameters()).device
    with torch.no_grad():
        return network_sigma(
            net, torch.tensor(v, dtype=torch.float32, device=device).view(-1, 1),
        ).cpu().numpy().reshape(-1)


def estimate_noise(v, dt, scale, drift, moments, settings, levy_case, seed, device="cpu"):
    levy, jump_second_moment = None, 0.0
    epsilon = moments["epsilon"]
    if levy_case:
        levy = estimate_levy(
            np.diff(v, axis=1) / scale, dt, scale, epsilon, settings["levy_q"], settings["levy_bins"],
        )
        if not levy["identified"]:
            return levy, None, np.array([])
        jump_second_moment = levy["small_jump_second_moment"]
    empirical = empirical_second_sample(v, dt, epsilon, scale, drift)
    keep = scale <= 1.0
    net, loss = fit_sigma(v[:, :-1][keep], empirical[keep], jump_second_moment,
                          settings["lr"], settings["epochs"], settings["batch_size"], seed, device)
    return levy, net, loss
