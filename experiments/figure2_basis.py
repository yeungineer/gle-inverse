"""Laguerre 基的选择（图 2）。"""

import numpy as np
from matplotlib.colors import LogNorm

from model.data import get_force, observe, simulate_paths
from model.embedding import embed, kernel_from_coefficients
from model.moments import empirical_first_sample
from model.step1_kernel import estimate_kernel, normal_statistics
from model.utils.io import configured_case
from model.utils.metrics import kernel_error
from model.utils.plotting import plot_heatmap
from model.utils.style import (
    COLORS, NOISE_LABELS, add_panel_label, apply_publication_style, error_label,
    heatmap_legend, paper_panels, save_publication_figure, style_axis,
)
from experiments.common import parallel_map

PARAMS = {
    "seed": 42, "K_values": [2, 4, 6, 8, 10, 12, 14, 16], "a_values": [0.25, 0.5, 1.0, 2.0],
    "display_a": 0.5, "display_K": [2, 4, 16],
}
CASES = ("brownian", "brownian_levy")


def basis_case(case):
    data = {}
    config = configured_case(case, {"simulation.seed": PARAMS["seed"], "step1.K_grid": PARAMS["K_values"],
                                    "step1.a_grid": PARAMS["a_values"]})
    obs = observe(simulate_paths(config), config["observation"]["sigma_obs"], PARAMS["seed"])
    v, t, s1 = obs["v"], obs["t"], config["step1"]
    est = estimate_kernel(v, t, config["force"], s1, config["moments"], PARAMS["seed"])
    scale = est["scale"]
    D1 = empirical_first_sample(v, obs["dt"], config["moments"]["epsilon"], scale)
    force = get_force(config["force"], v[:, :-1])
    errors = np.zeros((len(PARAMS["a_values"]), len(PARAMS["K_values"])))
    for i, a in enumerate(PARAMS["a_values"]):
        z = embed(v, t, max(PARAMS["K_values"]), a)
        gram, rhs, _ = normal_statistics(z / scale[:, :, None], force / scale - D1)
        G, r = gram.sum(axis=0), rhs.sum(axis=0)
        for j, K in enumerate(PARAMS["K_values"]):
            u = np.linalg.solve(G[:K, :K], r[:K])
            gamma = kernel_from_coefficients(u, K, a, t)
            errors[i, j] = kernel_error(t, gamma, obs["kernel_true"], s1["t_max"])["relative_l2_error"]
            if case == "brownian" and a == PARAMS["display_a"] and K in PARAMS["display_K"]:
                data[f"kernel_K{K}"] = gamma
    data |= {
        f"{case}__errors": errors, f"{case}__cv_scores": est["scores"],
        f"{case}__cv_choice": np.array([PARAMS["a_values"].index(est["a"]), PARAMS["K_values"].index(est["K"])]),
    }
    if case == "brownian":
        data |= {
            "t": t, "kernel_true": obs["kernel_true"],
            "kernel_cv": kernel_from_coefficients(est["u"], est["K"], est["a"], t),
            "cv_K": est["K"], "cv_a": est["a"],
        }
    return data


def run(out, workers=1, device="cpu"):
    data = {}
    for result in parallel_map(basis_case, list(CASES), workers):
        data.update(result)
    np.savez_compressed(out / "data.npz", **data)


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    d = dict(np.load(out / "data.npz"))
    figure, axes = paper_panels(1, column_gap=11.5)
    errors_both = np.array([d[f"{case}__errors"] for case in CASES])
    norm = LogNorm(errors_both.min(), errors_both.max())
    for axis, case in zip(axes[:2], CASES):
        errors = d[f"{case}__errors"]
        plot_heatmap(axis, errors, PARAMS["K_values"], PARAMS["a_values"], "$K$", "$a$", norm=norm)
        axis.set_yticks(np.arange(len(PARAMS["a_values"])), [f"${a:.2f}$" for a in PARAMS["a_values"]])
        i, j = d[f"{case}__cv_choice"]
        axis.scatter([j - 0.23], [i - 0.30], marker="s", s=14, facecolors="none",
                     edgecolors=COLORS["estimate"], linewidths=1.1, label="CV")
        i, j = np.unravel_index(np.argmin(errors), errors.shape)
        axis.scatter([j + 0.23], [i - 0.30], marker="o", s=6, color=COLORS["blue"], label="Min.")
        heatmap_legend(axis, NOISE_LABELS[case] + ": " + error_label(r"\gamma") + r" ($\%$)")

    axes[2].plot(d["t"], d["kernel_true"], color=COLORS["truth"], label="Ref.")
    for K, color, linestyle in zip(
        PARAMS["display_K"], (COLORS["blue"], COLORS["green"], COLORS["gray"]), (":", "-.", (0, (5, 2))),
    ):
        axes[2].plot(d["t"], d[f"kernel_K{K}"], color=color, linestyle=linestyle,
                     label=f"Est. $K={K},\\,a={PARAMS['display_a']:g}$")
    axes[2].plot(d["t"], d["kernel_cv"], color=COLORS["estimate"], linestyle="--",
                 label=f"Est. CV\n$K={int(d['cv_K'])},\\,a={float(d['cv_a']):g}$")
    axes[2].set_xlim(0.0, d["t"][-1])
    axes[2].set_xticks([0, 10, 20, 30])
    axes[2].set_xlabel(r"$\tau$")
    axes[2].set_ylabel(r"$\gamma(\tau)$")
    axes[2].legend(loc="upper right")
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    save_publication_figure(figure, out, "Figure_2")
