"""速度测量噪声对反演误差的影响（图 5）。"""

import numpy as np
from matplotlib import ticker as mticker

from model.data import diffusion_profile, observe, simulate_paths
from model.embedding import kernel_from_coefficients
from model.step2_noise import predict_sigma
from model.utils.io import configured_case
from model.utils.metrics import kernel_error, relative_l2_error, levy_error
from model.utils.plotting import plot_median_iqr
from model.workflow import estimate
from model.utils.style import (
    COLORS, COMPARISON_LINESTYLES, NOISE_LABELS, add_panel_label, apply_publication_style,
    error_label, paper_panels, save_publication_figure, style_axis,
)
from experiments.common import SEEDS, parallel_map

PARAMS = {"seeds": SEEDS, "noise_levels": [0.0, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.125, 0.15]}
CASES = ("brownian", "brownian_levy")


def measurement_sweep(job):
    case, seed, device = job
    config = configured_case(case, {"simulation.seed": seed, "compute.device": device})
    paths = simulate_paths(config)
    x = np.linspace(*config["evaluation"]["sigma_range"], 1001)
    sigma_true = diffusion_profile(x, config["noise"]["diffusion"])
    rows = []
    for noise in PARAMS["noise_levels"]:
        obs = observe(paths, noise, seed)
        est, levy, net, loss = estimate(obs, config, seed)
        sigma = relative_l2_error(predict_sigma(net, x), sigma_true, x)[1]
        gamma = kernel_from_coefficients(est["u"], est["K"], est["a"], obs["t"])
        error = kernel_error(obs["t"], gamma, obs["kernel_true"], config["step1"]["t_max"])["relative_l2_error"]
        alpha, kappa = 0.0, 0.0
        if case == "brownian_levy":
            errors = levy_error(config["noise"]["alpha"], config["noise"]["kappa"], levy)
            alpha, kappa = errors["alpha_rel_error"], errors["kappa_rel_error"]
        rows.append([sigma, error, alpha, kappa])
    return np.array(rows)


def run(out, workers=1, device="cpu"):
    shape = (len(PARAMS["seeds"]), len(PARAMS["noise_levels"]))
    data = {}
    jobs = [(case, seed, device) for case in CASES for seed in SEEDS]
    results = np.array(parallel_map(measurement_sweep, jobs, workers)).reshape(len(CASES), *shape, 4)
    for c, case in enumerate(CASES):
        sigma, kernel, alpha, kappa = (results[c, :, :, k] for k in range(4))
        data |= {f"{case}__sigma": sigma, f"{case}__kernel": kernel}
    data |= {"alpha": alpha, "kappa": kappa}
    np.savez_compressed(out / "data.npz", seeds=PARAMS["seeds"], noise_levels=PARAMS["noise_levels"], **data)


def plot(out, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    d = dict(np.load(out / "data.npz"))
    noise = d.get("noise_levels", np.array([0.0, 0.005, 0.01, 0.02, 0.05]))
    low_noise = noise <= 0.05
    figure, axes = paper_panels(1, column_gap=11.3)
    kernel, sigma, tail = axes
    for case, color, marker in (("brownian", COLORS["brownian"], "o"), ("brownian_levy", COLORS["levy"], "s")):
        plot_median_iqr(kernel, noise, d[f"{case}__kernel"], color, marker,
                        NOISE_LABELS[case], linestyle=COMPARISON_LINESTYLES[color])
        plot_median_iqr(sigma, noise[low_noise], d[f"{case}__sigma"][:, low_noise], color, marker,
                        NOISE_LABELS[case], linestyle=COMPARISON_LINESTYLES[color])
    identified = np.isfinite(d["kappa"])
    for name, color, marker in (("alpha", COLORS["blue"], "o"), ("kappa", COLORS["red"], "s")):
        errors = np.where(identified, d[name], np.nan)
        plot_median_iqr(tail, noise, errors, color, marker, error_label("\\" + name),
                        linestyle=COMPARISON_LINESTYLES[color], quantile=np.nanquantile)
    for axis, ylabel, y_format in zip(
        axes, (r"Error $\gamma$", r"Error $\sigma$", "Error"), ("%.3f", "%.3f", "%.2f")
    ):
        axis.set_xlabel("$\\sigma_{\\rm obs}$")
        axis.set_ylabel(ylabel)
        axis.yaxis.labelpad = 1.0
        axis.set_yscale("log")
        axis.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"${y_format}$"))
        axis.yaxis.set_minor_formatter(mticker.NullFormatter())
        axis.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
        axis.legend(loc="lower right")
    kernel.margins(y=.035)
    kernel.set_yticks([0.025, 0.030, 0.035, 0.040])
    sigma.set_yticks([0.001, 0.010, 0.100, 1.000])
    tail.yaxis.set_major_locator(mticker.LogLocator(subs=(1, 3)))
    sigma.set_xticks([0.0, 0.01, 0.02, 0.03, 0.04, 0.05])
    for axis in (kernel, tail):
        axis.set_xticks(np.linspace(0.0, noise[-1], 4))
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    save_publication_figure(figure, out, "Figure_5")
