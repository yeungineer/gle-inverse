"""单轨迹反演（图 8）。"""

import numpy as np

from experiments.common import single_thread
from model.data import diffusion_profile, observe, simulate_paths
from model.embedding import kernel_from_coefficients
from model.step2_noise import predict_sigma
from model.utils.io import configured_case, save_yaml
from model.utils.metrics import kernel_error, memory_error, sigma_error
from model.utils.plotting import plot_memory_inset
from model.utils.style import (
    COLORS, add_panel_label, apply_publication_style, load_prepared_data,
    paper_panels, save_publication_figure, style_axis,
)
from model.workflow import estimate

PARAMS = {"seed": 42, "warmup": 200.0, "fit_T": 20000.0, "display_T": 200.0,
          "memory_window": [2.0, 16.0]}


def run(out, workers=1, device="cpu"):
    single_thread()
    seed = PARAMS["seed"]
    config = configured_case("brownian_levy", {"simulation.seed": seed, "simulation.num_paths": 1,
                                               "compute.device": device})
    sim, s1 = config["simulation"], config["step1"]
    dt = sim["dt"]
    start = round(PARAMS["warmup"] / dt)
    sim["L"] = start + round(PARAMS["fit_T"] / dt)
    obs = observe(simulate_paths(config), config["observation"]["sigma_obs"], seed)
    est, _, net, _ = estimate(obs, config, seed, start=start)

    v = obs["v"][0]
    epsilon = config["moments"]["epsilon"]
    exceedances = start + 1 + np.flatnonzero(np.abs(np.diff(v[start:]) / est["scale"][0]) >= epsilon)
    stop = start + round(PARAMS["display_T"] / dt)
    lag = obs["t"][:round(s1["t_max"] / dt) + 1]
    gamma = kernel_from_coefficients(est["u"], est["K"], est["a"], lag)
    gamma_ref = obs["kernel_true"][:len(lag)]
    lo, hi = (round(t / dt) for t in PARAMS["memory_window"])
    window = slice(start + lo, start + hi + 1)
    t_memory = obs["t"][window]
    memory = est["z"][0, lo:hi + 1] @ est["u"]
    memory_ref = obs["memory_true"][0, window]
    local_v = v[start:-1][est["scale"][0] == 1.0]
    left, right = config["evaluation"]["sigma_range"]
    x = np.linspace(max(left, local_v.min()), min(right, local_v.max()), 1001)
    sigma = predict_sigma(net, x)
    sigma_ref = diffusion_profile(x, config["noise"]["diffusion"])

    np.savez_compressed(out / "data.npz", v_obs=v, exceedance_steps=exceedances,
                        lag=lag, kernel_est=gamma, kernel_true=gamma_ref, kernel_coefficients=est["u"],
                        t_memory=t_memory, memory_est=memory, memory_true=memory_ref,
                        sigma_x=x, sigma_est=sigma, sigma_true=sigma_ref)
    save_yaml(out / "metadata.yaml", {
        "simulation": sim, "warmup": PARAMS["warmup"], "fit_T": PARAMS["fit_T"],
        "fit_intervals": sim["L"] - start, "sigma_state_count": local_v.size,
        "display_steps": [start, stop], "display_exceedances": np.count_nonzero(exceedances <= stop),
        "cross_validation": {"scheme": "expanding time blocks", "folds": s1["folds"],
                             "K_grid": s1["K_grid"], "a_grid": s1["a_grid"], "scores": est["scores"]},
        "kernel": kernel_error(lag, gamma, gamma_ref, s1["t_max"]) | {"K": est["K"], "a": est["a"]},
        "memory": memory_error(memory, memory_ref, t_memory),
        "sigma": sigma_error(x, sigma, sigma_ref),
    })


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    d, m = load_prepared_data(out / "data.npz", out / "metadata.yaml")
    figure, axes = paper_panels(1, column_gap=11.0)
    trajectory, kernel, sigma = axes
    start, stop = m["display_steps"]
    dt = m["simulation"]["dt"]
    v = d["v_obs"]
    t = dt * np.arange(start, stop + 1)
    trajectory.plot(t, v[start:stop + 1], color=COLORS["empirical"], linewidth=0.35)
    steps = d["exceedance_steps"]
    shown = steps[(steps > start) & (steps <= stop)]
    if shown.size:
        trajectory.scatter(dt * shown, v[shown], s=13, facecolors="none", edgecolors=COLORS["levy"],
                           linewidths=0.8, label=r"$|\Delta v|/s\geq\varepsilon$", zorder=4)
        trajectory.legend(loc="upper left")
    trajectory.set(xlabel="$t$", ylabel="$v(t)$", xlim=(t[0], t[-1]))

    for axis, key, grid in ((kernel, "kernel", d["lag"]), (sigma, "sigma", d["sigma_x"])):
        error = 100 * m[key]["relative_l2_error"]
        axis.plot(grid, d[f"{key}_true"], color=COLORS["truth"], label="Ref.")
        axis.plot(grid, d[f"{key}_est"], color=COLORS["estimate"], linestyle="--",
                  label=rf"Est. ($\mathrm{{Err.}}={error:.2f}\%$)")
        axis.set_xlim(grid[0], grid[-1])
    kernel.set(xlabel=r"$\tau$", ylabel=r"$\gamma(\tau)$", xticks=[0, 10, 20, 30])
    kernel.legend(loc="upper right")
    plot_memory_inset(kernel, d["t_memory"], d["memory_true"], d["memory_est"], m["memory"]["relative_l2_error"])
    sigma.set(xlabel="$v$", ylabel=r"$\sigma(v)$", ylim=(0, None))
    sigma.legend(loc="lower center")
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    save_publication_figure(figure, out, "Figure_8")
