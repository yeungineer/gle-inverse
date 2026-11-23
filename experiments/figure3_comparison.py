"""记忆核估计方法比较（图 3）。"""

import numpy as np
from matplotlib import ticker as mticker

from baselines.prony_sobolev import estimate_kernel_prony_sobolev
from baselines.vroylandt_em import fit_em, memory_kernel
from model.data import get_force, kernel, observe, simulate_paths
from model.embedding import kernel_from_coefficients
from model.step1_kernel import estimate_kernel
from model.utils.io import configured_case, save_yaml
from model.utils.metrics import relative_l2_error
from model.utils.style import (
    COLORS, COMPARISON_LINESTYLES, add_panel_label, apply_publication_style, paper_panels,
    save_publication_figure, style_axis,
)
from experiments.common import parallel_map

PARAMS = {
    "seed": 42,
    "error_windows": [[0.0, 10.0], [10.0, 20.0], [20.0, 30.0]],
    "t_max": 30.0,
    "grid_count": 2001,
    "prony_sobolev": {"observation_ratio": 70, "correlation_length": 24, "modes": 10,
                      "omega": 0.05, "t_max": 30.0, "knot_count": 30, "grid_count": 2001},
    "vroylandt_em": {"hidden_dimension": 5, "seed": 42, "tolerance": 1e-8, "max_iterations": 2000},
}
METHODS = ("ours", "prony_sobolev", "vroylandt_em")
LABELS = {"ours": "Ours", "prony_sobolev": "Prony--Sobolev", "vroylandt_em": "Likelihood EM"}
METHOD_COLORS = {"ours": COLORS["estimate"], "prony_sobolev": COLORS["blue"], "vroylandt_em": COLORS["green"]}


def ours(v, t, config, grid):
    s1 = config["step1"]
    est = estimate_kernel(v, t, config["force"], s1, config["moments"], PARAMS["seed"])
    return kernel_from_coefficients(est["u"], est["K"], est["a"], grid)


def prony_sobolev(v, t, config, grid):
    return estimate_kernel_prony_sobolev(v, t, config["force"], PARAMS["prony_sobolev"])(grid)


def vroylandt_em(v, t, config, settings):
    return fit_em(v, t[1] - t[0], get_force(config["force"], v), settings)


def window_errors(t, estimate, reference):
    errors = []
    for lo, hi in PARAMS["error_windows"]:
        if lo < t[0] or hi > t[-1]:
            raise ValueError("Saved kernel grid does not cover the evaluation windows")
        grid = np.r_[lo, t[(t > lo) & (t < hi)], hi]
        ref = np.interp(grid, t, reference)
        est = np.interp(grid, t, estimate)
        errors.append(relative_l2_error(est, ref, grid)[1])
    return np.asarray(errors)


def compare_method(job):
    name, v, t, config, grid = job
    if name == "vroylandt_em":
        fit = vroylandt_em(v, t, config, PARAMS["vroylandt_em"])
        return memory_kernel(fit["A"], grid), len(fit["log_likelihood"])
    method = {"ours": ours, "prony_sobolev": prony_sobolev}[name]
    return method(v, t, config, grid), 0


def run(out, workers=1, device="cpu"):
    config = configured_case("brownian", {"simulation.seed": PARAMS["seed"]})
    obs = observe(simulate_paths(config), config["observation"]["sigma_obs"], PARAMS["seed"])
    v, t = obs["v"], obs["t"]
    grid = np.linspace(0.0, PARAMS["t_max"], PARAMS["grid_count"])
    data = {"t": grid, "kernel_true": kernel(grid, config["kernel"]["type"])}
    fits = parallel_map(compare_method, [(name, v, t, config, grid) for name in METHODS], workers)
    for name, (curve, _) in zip(METHODS, fits):
        data[f"kernel_{name}"] = curve
    metadata = {
        "vroylandt_em_iterations": fits[-1][1], "error_windows": PARAMS["error_windows"],
        "simulation": config["simulation"],
    }
    for name in METHODS:
        errors = window_errors(grid, data[f"kernel_{name}"], data["kernel_true"])
        data[f"window_errors_{name}"] = errors
        metadata[name] = {
            "kernel_error": float(relative_l2_error(data[f"kernel_{name}"], data["kernel_true"], grid)[1]),
            "window_relative_errors": errors.tolist(),
        }
    np.savez_compressed(out / "data.npz", **data)
    save_yaml(out / "metadata.yaml", metadata)


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    d = dict(np.load(out / "data.npz"))
    figure, axes = paper_panels(1, column_gap=12.2)
    reconstruction, residual, windows = axes
    intervals = PARAMS["error_windows"]
    positions = np.arange(len(intervals))
    reconstruction.plot(d["t"], d["kernel_true"], color=COLORS["truth"], label="Ref.")
    for index, name in enumerate(METHODS):
        color = METHOD_COLORS[name]
        linestyle = COMPARISON_LINESTYLES[color]
        estimate = d[f"kernel_{name}"]
        error = relative_l2_error(estimate, d["kernel_true"], d["t"])[1]
        reconstruction.plot(d["t"], estimate, color=color, linestyle=linestyle,
                            label=LABELS[name] + "\n" + rf"($\mathrm{{Err.}}={100 * error:.2f}\%$)")
        residual.plot(d["t"], np.abs(estimate - d["kernel_true"]), color=color,
                      linestyle=linestyle, label=LABELS[name])
        windows.plot(positions, window_errors(d["t"], estimate, d["kernel_true"]),
                     color=color, marker="os^"[index], linestyle=linestyle, label=LABELS[name])
    for axis in (reconstruction, residual):
        axis.set_xlim(0.0, PARAMS["t_max"])
        axis.set_xticks([0, 10, 20, 30])
        axis.set_xlabel(r"$\tau$")
    reconstruction.set_ylabel(r"$\gamma(\tau)$")
    residual.set_ylabel(r"$|\widehat\gamma(\tau)-\gamma(\tau)|$")
    windows.set_xticks(positions, [f"$[{lo:g},{hi:g}]$" for lo, hi in intervals])
    windows.margins(x=0.16)
    windows.set_xlabel("Lag interval")
    windows.set_ylabel("Error $\\gamma$")
    windows.set_yscale("log")
    windows.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
    windows.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    windows.yaxis.set_minor_formatter(mticker.NullFormatter())
    for label, axis in zip("abc", axes):
        axis.legend(loc="best")
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    save_publication_figure(figure, out, "Figure_3")
