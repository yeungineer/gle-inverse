"""不同外力与幂律记忆核的反演（图 7）。"""

import numpy as np
from matplotlib import ticker as mticker

from experiments.common import SEEDS, parallel_map
from model.embedding import embed
from model.utils.io import configured_case, save_yaml
from model.utils.plotting import plot_median_iqr
from model.utils.style import (
    COLORS, NOISE_LABELS, add_panel_label, apply_publication_style, error_label, paper_panels,
    save_publication_figure, style_axis,
)
from model.workflow import invert, prepare

PARAMS = {"seeds": SEEDS, "forces": ["zero", "linear", "sin"], "powerlaw_display_t": 10.0,
          "condition_K": 12, "condition_a": 0.5, "rate_bin_count": 6, "rate_bin_ratio": 2.0**0.5}
CASES = ("brownian", "brownian_levy")


def transfer_case(job):
    kernel_kind, case, force, seed, device = job
    config = configured_case(case, {"kernel.type": kernel_kind, "noise.diffusion": "cosine",
                                    "force.type": force, "simulation.seed": seed, "compute.device": device})
    if kernel_kind == "power_law" and case == "brownian_levy":
        config["step2"].update(levy_q=PARAMS["rate_bin_ratio"], levy_bins=PARAMS["rate_bin_count"] - 1)
    p = prepare(config)
    arrays, metrics, est = invert(p)
    if kernel_kind == "complex_exp":
        errors = [metrics[name]["relative_l2_error"] for name in ("kernel", "sigma")]
        if case == "brownian_levy":
            errors.append(metrics["levy"]["joint_rel_error"])
        z = embed(
            p["obs"]["v"], p["obs"]["t"], PARAMS["condition_K"], PARAMS["condition_a"],
        ) / est["scale"][:, :, None]
        Z = z.reshape(-1, PARAMS["condition_K"])
        return {"errors": np.asarray(errors), "condition": np.linalg.cond(Z.T @ Z)}
    return {**{key: arrays[key] for key in ("t", "kernel_true", "kernel_est", "sigma_x", "sigma_true", "sigma_est")},
            **{key: value for key, value in arrays.items() if key.startswith("levy_")}}


def run(out, workers=1, device="cpu"):
    jobs = [
        ("complex_exp", case, force, seed, device)
        for case in CASES for force in PARAMS["forces"] for seed in SEEDS
    ]
    jobs += [("power_law", case, "zero", seed, device) for case in CASES for seed in SEEDS]
    results = parallel_map(transfer_case, jobs, workers)
    data = {}
    count = len(SEEDS)
    for c, case in enumerate(CASES):
        group = results[c * 3 * count:(c + 1) * 3 * count]
        data[f"{case}__force_errors"] = np.array([r["errors"] for r in group]).reshape(3, count, -1)
        data[f"{case}__condition"] = np.array([r["condition"] for r in group]).reshape(3, count)
        group = results[6 * count + c * count:6 * count + (c + 1) * count]
        for key in ("kernel_est", "sigma_est"):
            data[f"{case}__powerlaw_{key}"] = np.array([r[key] for r in group])
        data.update({key: group[0][key] for key in ("t", "kernel_true", "sigma_x", "sigma_true")})
        if case == "brownian_levy":
            data.update({key: group[0][key] for key in ("levy_bin_edges", "levy_true_rates")})
            for key in ("levy_empirical_rates", "levy_fitted_rates"):
                data[key] = np.array([r[key] for r in group])
    np.savez_compressed(out / "data.npz", seeds=SEEDS, **data)
    save_yaml(out / "metadata.yaml", PARAMS | {"diffusion": "cosine in both noise models", "sigma_range": [-2, 2]})


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    data = dict(np.load(out / "data.npz"))
    figure, axes = paper_panels(2, column_gap=12.0, row_gap=13.5)
    position = np.arange(3)
    metric_styles = [(COLORS["red"], "o", error_label(r"\gamma")), (COLORS["blue"], "s", error_label(r"\sigma")),
                     (COLORS["green"], "^", error_label(r"\alpha,\kappa"))]
    case_styles = [("brownian", COLORS["blue"], "o"), ("brownian_levy", COLORS["red"], "s")]
    axes[1, 0].plot(data["t"], data["kernel_true"], color=COLORS["truth"], label="Ref.", zorder=0.5)
    axes[1, 1].plot(data["sigma_x"], data["sigma_true"], color=COLORS["truth"], label="Ref.", zorder=0.5)
    for c, (case, color, marker) in enumerate(case_styles):
        name = NOISE_LABELS[case]
        values = data[f"{case}__force_errors"]
        for k, (metric_color, metric_marker, label) in enumerate(metric_styles[:values.shape[-1]]):
            q25, median, q75 = np.quantile(values[:, :, k], [.25, .5, .75], axis=1)
            axes[0, c].errorbar(position + .14 * (k - .5 * (values.shape[-1] - 1)), median,
                                yerr=[median - q25, q75 - median],
                                color=metric_color, marker=metric_marker, linestyle="none", capsize=2, label=label)
        axes[0, c].set_title(name, fontsize=7.0, y=1.0, pad=6.0)
        axes[0, c].set_ylabel("Error")
        axes[0, c].set_yscale("log")
        q25, median, q75 = np.quantile(data[f"{case}__condition"], [.25, .5, .75], axis=1)
        axes[0, 2].errorbar(position + .18 * (c - .5), median, yerr=[median - q25, q75 - median],
                            color=color, marker=marker, linestyle="none", capsize=2, label=name)
        plot_median_iqr(axes[1, 0], data["t"], data[f"{case}__powerlaw_kernel_est"], color, "", name,
                        linestyle=("-.", "--")[c])
        plot_median_iqr(axes[1, 1], data["sigma_x"], data[f"{case}__powerlaw_sigma_est"], color, "", name,
                        linestyle=("-.", "--")[c])
    for axis in axes[0]:
        axis.set_xticks(position, ["$0$", "$-v$", "$\\sin 2v$"])
        axis.set_xlabel("$F(v)$")
        axis.margins(x=.15, y=.2)
    axes[0, 2].set_ylabel(r"$\mathrm{cond}(G)$")
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylim(top=2700)
    axes[1, 0].set(xlim=(0, PARAMS["powerlaw_display_t"]), xlabel=r"$\tau$", ylabel=r"$\gamma(\tau)$")
    axes[1, 0].set_ylim(top=1.2)
    axes[1, 1].set(xlabel="$v$", ylabel="$\\sigma(v)$")
    centers = np.sqrt(data["levy_bin_edges"][:-1] * data["levy_bin_edges"][1:])
    axes[1, 2].plot(centers, data["levy_true_rates"], color=COLORS["truth"], label="Ref.")
    plot_median_iqr(axes[1, 2], centers, data["levy_fitted_rates"], COLORS["estimate"], "s", "Est.",
                    linestyle="--", zorder=3)
    plot_median_iqr(axes[1, 2], centers, data["levy_empirical_rates"], COLORS["empirical"], "o", "Obs.",
                    linestyle=":")
    axes[1, 2].set(xscale="log", yscale="log", xlabel=r"$\sqrt{r_k r_{k+1}}$", ylabel="Jump rate")
    for axis, exponent in ((axes[0, 0], -2), (axes[0, 1], -2), (axes[0, 2], 3), (axes[1, 2], -3)):
        axis.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
        formatter = mticker.FuncFormatter(lambda value, _, unit=10.0**exponent: f"${value / unit:.1f}$")
        formatter.set_offset_string(rf"$\times10^{{{exponent}}}$")
        axis.yaxis.set_major_formatter(formatter)
        axis.yaxis.set_minor_formatter(mticker.NullFormatter())
    for label, axis in zip("abcdef", axes.ravel()):
        axis.legend(loc="best")
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    for axis, columns in zip(axes[0, :2], (2, 3)):
        axis.legend(loc="lower center", ncol=columns, fontsize=5.8, handlelength=.8, handletextpad=.25,
                    columnspacing=.4, borderaxespad=.25)
    axes[0, 2].legend(loc="upper center", fontsize=5.8, handlelength=1.1, borderaxespad=.25)
    save_publication_figure(figure, out, "Figure_7")
