"""不同 Lévy 参数下的反演误差（图 6）。"""

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D

from experiments.common import SEEDS, parallel_map
from model.utils.io import configured_case, save_yaml
from model.utils.plotting import plot_heatmap
from model.utils.style import (
    COLORS, add_panel_label, apply_publication_style, error_label, paper_panels,
    save_publication_figure, style_axis,
)
from model.workflow import invert, prepare

PARAMS = {"seeds": SEEDS, "map_paths": 500, "threshold": 0.5,
          "alpha_values": [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
          "kappa_values": [0.04, 0.08, 0.12, 0.18, 0.24]}


def parameter_point(job):
    alpha, kappa, seed, device = job
    config = configured_case("brownian_levy", {
        "noise.alpha": alpha, "noise.kappa": kappa, "simulation.num_paths": PARAMS["map_paths"],
        "simulation.seed": seed, "compute.device": device,
    })
    _, metrics, _ = invert(prepare(config))
    levy = metrics["levy"]
    return (metrics["kernel"]["relative_l2_error"], levy["joint_rel_error"],
            metrics["sigma"]["relative_l2_error"], levy["identified"], levy["tail_count"])


def run(out, workers=1, device="cpu"):
    jobs = [
        (alpha, kappa, seed, device)
        for kappa in PARAMS["kappa_values"] for alpha in PARAMS["alpha_values"] for seed in SEEDS
    ]
    results = np.array(parallel_map(parameter_point, jobs, workers)).reshape(
        len(PARAMS["kappa_values"]), len(PARAMS["alpha_values"]), len(SEEDS), 5,
    )
    np.savez_compressed(
        out / "data.npz", seeds=SEEDS, kernel=results[..., 0], levy=results[..., 1],
        sigma=results[..., 2], identified=results[..., 3].astype(bool), tail_count=results[..., 4],
    )
    save_yaml(out / "metadata.yaml", PARAMS | {
        "aggregation": "median of identified repetitions; missing counts shown separately",
        "sigma_range": [-2, 2],
    })


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    data = dict(np.load(out / "data.npz"))
    figure, axes = paper_panels(1, column_gap=11.5)
    failures = np.sum(~data["identified"], axis=-1)
    medians = np.array([np.nanmedian(data[key], axis=-1) for key in ("kernel", "levy", "sigma")])
    norm = LogNorm(np.nanmin(medians), np.nanmax(medians))
    for axis, key, symbol in zip(axes, ("kernel", "levy", "sigma"), (r"\gamma", r"\alpha,\kappa", r"\sigma")):
        values = np.nanmedian(data[key], axis=-1)
        plot_heatmap(axis, values, PARAMS["alpha_values"], PARAMS["kappa_values"],
                     "$\\alpha$", "$\\kappa$", norm=norm)
        bad = np.argwhere(values > PARAMS["threshold"])
        crosses = axis.scatter(bad[:, 1] + .29, bad[:, 0] + .37, marker="x", s=9, color=COLORS["red"], linewidths=.8,
                               label=r"Error $>50\%$")
        if key != "kernel":
            for i, j in np.argwhere(failures > 0):
                axis.texts[i * values.shape[1] + j].set_y(i + .15)
                axis.text(j, i - .20, rf"$\frac{{{failures[i,j]}}}{{10}}$", ha="center", va="center",
                          fontsize=5.5, color=COLORS["red"])
        header = Line2D([], [], linestyle="none", label=error_label(symbol) + r" ($\%$)")
        axis.legend(handles=[header, crosses], ncol=2, loc="lower right", bbox_to_anchor=(1, 1.015),
                    columnspacing=1.0, handlelength=.8, handletextpad=.25,
                    borderaxespad=0, borderpad=0, fontsize=6.0)
        axis.set_xticks(np.arange(0, len(PARAMS["alpha_values"]), 2),
                        [f"${x:g}$" for x in PARAMS["alpha_values"][::2]])
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    figure.align_xlabels(axes)
    save_publication_figure(figure, out, "Figure_6")
