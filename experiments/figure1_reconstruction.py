"""记忆核与噪声参数的重构（图 1）。"""

import numpy as np
import yaml

from model.data import simulate_paths
from model.moments import empirical_second_sample, observed_scale
from model.step2_noise import fit_sigma, predict_sigma
from model.utils.io import configured_case, save_yaml
from model.utils.metrics import relative_l2_error
from model.utils.plotting import plot_memory_inset
from model.utils.style import (
    COLORS, add_panel_label, apply_publication_style, paper_panels, save_publication_figure,
    style_axis,
)
from model.workflow import invert, prepare
from experiments.common import parallel_map

PARAMS = {"seed": 42, "memory_window": [2.0, 16.0], "display_T": 200.0, "display_seed": 42,
          "display_levy_candidates": 8, "sigma_ylim": [0.40, 0.60]}
CASES = ("brownian", "brownian_levy")
KEYS = (
    "t", "kernel_true", "kernel_est", "sigma_x", "sigma_est", "sigma_true",
    "memory_true_path", "memory_est_path",
)


def display_paths(epsilon):
    data = {}
    metadata = {"seed": PARAMS["display_seed"], "t_max": PARAMS["display_T"], "used_for_inference": False}
    for case in CASES:
        candidates = PARAMS["display_levy_candidates"] if case == "brownian_levy" else 1
        config = configured_case(
            case, {"simulation.seed": PARAMS["display_seed"], "simulation.num_paths": candidates},
        )
        config["simulation"]["L"] = round(PARAMS["display_T"] / config["simulation"]["dt"])
        paths = simulate_paths(config)
        selected = 0
        selection = {"candidate_paths": candidates, "selected_index": selected,
                     "dt": config["simulation"]["dt"], "substeps": config["simulation"]["substeps"]}
        if case == "brownian_levy":
            scale = observed_scale(paths["v"], paths["t"], config["force"], config["moments"])
            counts = np.count_nonzero(np.abs(np.diff(paths["v"], axis=1) / scale) >= epsilon, axis=1)
            eligible = np.flatnonzero(counts > 0)
            selected = int(eligible[0])
            selection.update(selected_index=selected, rule="first_path_with_threshold_exceedance",
                             epsilon=float(epsilon), candidate_exceedance_counts=counts.tolist(),
                             selected_exceedance_count=int(counts[selected]),
                             alpha=config["noise"]["alpha"], kappa=config["noise"]["kappa"])
            data[f"{case}__display_scale"] = scale[selected]
        data[f"{case}__t_display"] = paths["t"]
        data[f"{case}__v_display"] = paths["v"][selected]
        metadata[case] = selection
    return data, metadata


def refresh_display_paths(out):
    with np.load(out / "data.npz", allow_pickle=False) as stored:
        data = dict(stored)
    metadata = yaml.safe_load((out / "metadata.yaml").read_text())
    display, metadata["display"] = display_paths(metadata["brownian_levy"]["epsilon"])
    data.update(display)
    temporary = out / "data_display.npz"
    np.savez_compressed(temporary, **data)
    temporary.replace(out / "data.npz")
    save_yaml(out / "metadata.yaml", metadata)


def reconstruct(job):
    case, device = job
    config = configured_case(case, {"simulation.seed": PARAMS["seed"], "compute.device": device})
    p = prepare(config)
    arrays, metrics, est = invert(p)
    data = {f"{case}__{key}": arrays[key] for key in KEYS}
    metadata = {
        "sigma_error": metrics["sigma"]["relative_l2_error"],
        "kernel_error": metrics["kernel"]["relative_l2_error"],
        "simulation": config["simulation"],
    }
    if case == "brownian_levy":
        s2, v = config["step2"], p["obs"]["v"]
        D2 = empirical_second_sample(v, p["obs"]["dt"], config["moments"]["epsilon"], est["scale"], est["drift"])
        keep = est["scale"] <= 1.0
        net, loss = fit_sigma(
            v[:, :-1][keep], D2[keep], 0.0, s2["lr"], s2["epochs"],
            s2["batch_size"], PARAMS["seed"], device,
        )
        data[f"{case}__sigma_uncorrected"] = predict_sigma(net, arrays["sigma_x"])
        metadata["sigma_uncorrected_error"] = relative_l2_error(
            data[f"{case}__sigma_uncorrected"], arrays["sigma_true"], arrays["sigma_x"],
        )[1]
        metadata |= {"alpha_est": metrics["levy"]["alpha_est"], "kappa_est": metrics["levy"]["kappa_est"],
                     "epsilon": metrics["levy"]["epsilon"]}
    return data, metadata


def run(out, workers=1, device="cpu"):
    data, metadata = {}, {}
    results = parallel_map(reconstruct, [(case, device) for case in CASES], workers)
    for case, (arrays, metrics) in zip(CASES, results):
        data.update(arrays)
        metadata[case] = metrics
    display, metadata["display"] = display_paths(metadata["brownian_levy"]["epsilon"])
    data.update(display)
    np.savez_compressed(out / "data.npz", **data)
    save_yaml(out / "metadata.yaml", metadata)


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    d = dict(np.load(out / "data.npz"))
    m = yaml.safe_load((out / "metadata.yaml").read_text())
    figure, axes = paper_panels(2, column_gap=12.0, row_gap=12.0)
    for row, case in enumerate(CASES):
        t = d[f"{case}__t"]
        trajectory, kernel, sigma = axes[row]

        t_show, v_show = d[f"{case}__t_display"], d[f"{case}__v_display"]
        trajectory.plot(t_show, v_show, color=COLORS["empirical"], linewidth=0.35)
        trajectory.set_xlabel("$t$")
        trajectory.set_ylabel("$v(t)$")
        trajectory.set_xlim(t_show[0], t_show[-1])

        x = d[f"{case}__sigma_x"]
        sigma.plot(x, d[f"{case}__sigma_true"], color=COLORS["truth"], label="Ref.")
        sigma.plot(x, d[f"{case}__sigma_est"], color=COLORS["estimate"], linestyle="--",
                   label=rf"Est. ($\mathrm{{Err.}}={100 * m[case]['sigma_error']:.2f}\%$)")
        sigma.set_xlabel("$v$")
        sigma.set_ylabel("$\\sigma(v)$")

        kernel.plot(t, d[f"{case}__kernel_true"], color=COLORS["truth"], label="Ref.")
        kernel.plot(t, d[f"{case}__kernel_est"], color=COLORS["estimate"], linestyle="--",
                    label=rf"Est. ($\mathrm{{Err.}}={100 * m[case]['kernel_error']:.2f}\%$)")
        kernel.set_xlabel(r"$\tau$")
        kernel.set_ylabel(r"$\gamma(\tau)$")
        kernel.set_xlim(t[0], t[-1])
        kernel.set_xticks([0, 10, 20, 30])
        kernel.legend(loc="upper right")
        window = (t[:-1] >= PARAMS["memory_window"][0]) & (t[:-1] <= PARAMS["memory_window"][1])
        t_memory = t[:-1][window]
        memory_ref = d[f"{case}__memory_true_path"][window]
        memory_est = d[f"{case}__memory_est_path"][window]
        inset_error = relative_l2_error(memory_est, memory_ref, t_memory)[1]
        plot_memory_inset(kernel, t_memory, memory_ref, memory_est, inset_error)

    levy = m["brownian_levy"]
    t, v = d["brownian_levy__t_display"], d["brownian_levy__v_display"]
    exceedances = np.abs(np.diff(v) / d["brownian_levy__display_scale"]) >= levy["epsilon"]
    axes[1, 0].scatter(t[1:][exceedances], v[1:][exceedances], s=13.0, facecolors="none",
                       edgecolors=COLORS["levy"], linewidths=0.8,
                       label=r"$|\Delta v|/s\geq\varepsilon$", zorder=4)
    axes[1, 0].set_ylim(v.min() - 0.3, v.max() + 1.6)
    axes[1, 0].legend(loc="upper left", handletextpad=0.3, borderaxespad=0.25, borderpad=0.2,
                      fontsize=6.0, labelspacing=0.15, alignment="left")
    axes[1, 0].text(0.97, 0.97,
                    rf"$\widehat{{\alpha}}={levy['alpha_est']:.3f}$" + "\n"
                    + rf"$\widehat{{\kappa}}={levy['kappa_est']:.3f}$",
                    transform=axes[1, 0].transAxes, ha="right", va="top", fontsize=6.0, linespacing=1.3)
    axes[1, 2].plot(d["brownian_levy__sigma_x"], d["brownian_levy__sigma_uncorrected"], color=COLORS["uncorrected"],
                    linestyle=(0, (2.0, 1.25)),
                    label=rf"w/o $J_\varepsilon$ ($\mathrm{{Err.}}={100 * levy['sigma_uncorrected_error']:.2f}\%$)")
    axes[0, 2].set_ylim(*PARAMS["sigma_ylim"])
    axes[1, 2].set_ylim(-0.12, 1.08)
    for axis in axes[:, 2]:
        axis.legend(loc="lower center")
    for label, axis in zip("abcdef", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    save_publication_figure(figure, out, "Figure_1")
