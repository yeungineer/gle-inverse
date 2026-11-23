"""采样间隔、轨迹长度与轨迹数量（图 4）。"""

import numpy as np
from functools import lru_cache
from matplotlib import ticker as mticker

from experiments.common import SEEDS, parallel_map
from model.data import diffusion_profile, simulate_paths
from model.embedding import kernel_from_coefficients
from model.step2_noise import predict_sigma
from model.utils.io import configured_case, save_yaml
from model.utils.metrics import relative_l2_error, levy_error
from model.utils.plotting import plot_median_iqr
from model.utils.style import (
    COLORS, COMPARISON_LINESTYLES, NOISE_LABELS, add_panel_label, apply_publication_style,
    error_label, load_prepared_data, paper_panels, save_publication_figure, style_axis,
)
from model.workflow import estimate

FORWARD_DT = 0.002
PARAMS = {
    "seeds": SEEDS, "h_values": [stride * FORWARD_DT for stride in (2, 3, 5, 10, 25, 50)],
    "T_values": list(range(15, 61, 5)), "M_values": list(range(250, 2001, 250)),
    "h": 0.01, "T": 30.0, "M": 1000, "forward_dt": FORWARD_DT, "t_max": 15.0,
}
CASES = ("brownian", "brownian_levy")


def sampling_sweep(job):
    case, seed, device = job
    fine = PARAMS["forward_dt"]
    config = configured_case(case, {"simulation.seed": seed, "simulation.num_paths": max(PARAMS["M_values"]),
                                    "simulation.dt": fine, "simulation.substeps": 1,
                                    "simulation.L": round(max(PARAMS["T_values"]) / fine), "compute.device": device})
    paths = simulate_paths(config)
    score_grid = paths["t"][:round(PARAMS["t_max"] / fine) + 1]
    reference_kernel = paths["kernel"][:score_grid.size]
    x = np.linspace(*config["evaluation"]["sigma_range"], 1001)
    reference_sigma = diffusion_profile(x, config["noise"]["diffusion"])
    @lru_cache(maxsize=None)
    def evaluate(h, T, M):
        stride, stop = round(h / fine), round(T / fine)
        index = slice(0, stop + 1, stride)
        obs = {"v": paths["v"][:M, index], "t": paths["t"][index], "dt": h}
        est, levy, net, _ = estimate(obs, config, seed)
        gamma = kernel_from_coefficients(est["u"], est["K"], est["a"], score_grid)
        row = [relative_l2_error(gamma, reference_kernel, score_grid)[1],
               relative_l2_error(predict_sigma(net, x), reference_sigma, x)[1]]
        if case == "brownian_levy":
            row.append(levy_error(config["noise"]["alpha"], config["noise"]["kappa"], levy)["joint_rel_error"])
        return row
    data = {}
    for parameter in ("h", "T", "M"):
        errors = []
        for value in PARAMS[f"{parameter}_values"]:
            setting = {key: PARAMS[key] for key in ("h", "T", "M")}
            setting[parameter] = value
            errors.append(evaluate(**setting))
        data[parameter] = np.array(errors)
    return data


def run(out, workers=1, device="cpu"):
    jobs = [(case, seed, device) for case in CASES for seed in SEEDS]
    results = parallel_map(sampling_sweep, jobs, workers)
    data = {}
    for row, case in enumerate(CASES):
        group = results[row * len(SEEDS):(row + 1) * len(SEEDS)]
        for parameter in ("h", "T", "M"):
            data[f"{case}__{parameter}"] = np.array([result[parameter] for result in group])
    np.savez_compressed(out / "data.npz", seeds=SEEDS, **data)
    save_yaml(out / "metadata.yaml", PARAMS | {"levy_alpha": configured_case("brownian_levy", {})["noise"]["alpha"]})


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    data, settings = load_prepared_data(out / "data.npz", out / "metadata.yaml")
    figure, axes = paper_panels(2, column_gap=11.0, row_gap=15.0)
    metrics = [(COLORS["red"], "o", error_label(r"\gamma")),
               (COLORS["blue"], "s", error_label(r"\sigma")),
               (COLORS["green"], "^", error_label(r"\alpha,\kappa"))]
    for row, case in enumerate(CASES):
        for col, parameter in enumerate(("h", "T", "M")):
            axis = axes[row, col]
            grid = settings[f"{parameter}_values"]
            values = data[f"{case}__{parameter}"]
            for k, (color, marker, label) in enumerate(metrics[:values.shape[-1]]):
                plot_median_iqr(axis, grid, values[:, :, k], color, marker, label,
                                linestyle=COMPARISON_LINESTYLES[color], quantile=np.nanquantile)
            if parameter == "h":
                axis.set_xscale("log")
                ticks = [value for value in grid if value in (grid[0], settings["h"], 0.05, grid[-1])]
            elif parameter == "T":
                ticks = [15, 30, 45, 60]
            else:
                ticks = [250, 500, 1000, 1500, 2000]
            axis.set_xticks(ticks, [f"${value:g}$" for value in ticks])
            axis.xaxis.set_minor_formatter(mticker.NullFormatter())
            axis.set_yscale("log")
            axis.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
            formatter = mticker.FuncFormatter(lambda value, _: f"${100 * value:.1f}$")
            formatter.set_offset_string(r"$\times10^{-2}$")
            axis.yaxis.set_major_formatter(formatter)
            axis.yaxis.set_minor_formatter(mticker.NullFormatter())
            axis.set_xlabel(r"$\Delta t$" if parameter == "h" else f"${parameter}$")
            axis.set_ylabel("Error")
            axis.legend(loc="upper left" if row == 1 and parameter == "h" else "best")
            axis.set_title(NOISE_LABELS[case], fontsize=7.0, y=1.0, pad=6.0)
    for label, axis in zip("abcdef", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    save_publication_figure(figure, out, "Figure_4")
