"""Plot saved basic experiments with the same style and row height as the paper."""

import matplotlib.pyplot as plt
import numpy as np

from model.simulation import diffusion_profile
from model.style import COLORS, apply_publication_style, save_publication_figure, style_axis
from model.style import SINGLE_PANEL_SIZE as SINGLE_COLUMN


def _save_plot(fig, name, output_dir, prefix):
    for ax in fig.axes:
        style_axis(ax)
    fig.tight_layout(pad=0.6)
    save_publication_figure(fig, output_dir, f"{prefix}_{name}" if prefix else name)
    plt.close(fig)


def plot_kernel(t_raw, gamma, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    mask = (t_raw >= 0) & (t_raw <= min(t_raw[-1], 30))
    ax.plot(t_raw[mask], gamma[mask], color=COLORS["empirical"], linewidth=1.0)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\gamma(t)$")
    ax.set_xlim(0.0, 30.0)
    _save_plot(fig, "kernel", output_dir, prefix)


def plot_noise(t_raw, noise_increment, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    mask = (t_raw >= 0) & (t_raw <= min(t_raw[-1], 600))
    ax.plot(t_raw[mask], np.asarray(noise_increment)[mask], color=COLORS["empirical"], linewidth=0.45, alpha=0.82)
    ax.set_xlabel("$t$")
    ax.set_ylabel("Noise increment")
    _save_plot(fig, "noise", output_dir, prefix)


def plot_traj(t_obs, t_raw, v_obs, v_raw, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    mask_raw = (t_raw >= 0.0) & (t_raw < 200.0)
    mask_obs = (t_obs >= 0.0) & (t_obs < 200.0)
    ax.plot(t_raw[mask_raw], v_raw[mask_raw], color=COLORS["truth"], alpha=0.92, linewidth=1.15, label="Ref. $v(t)$")
    ax.plot(
        t_obs[mask_obs], v_obs[mask_obs], linestyle="none", marker="o", color=COLORS["empirical"], markeredgecolor="none", markersize=1.8,
        alpha=0.68, label="Obs. $\\widetilde{v}(t)$", rasterized=True,
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("$v(t)$")
    ax.set_xlim(0.0, 200.0)
    ax.legend(loc="best")
    _save_plot(fig, "traj", output_dir, prefix)


def plot_km_samples(x, endpoints, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    endpoints = np.asarray(endpoints, dtype=float)[:3000]
    for i, v in enumerate(endpoints):
        ax.plot([0, 1], [x, v], color=COLORS["estimate_band"], linewidth=0.4, alpha=0.2, zorder=1, label="Branch" if i == 0 else None)
    ax.scatter([0], [x], s=18, color=COLORS["estimate"], edgecolors="white", linewidths=0.4, zorder=4, label="Ref. state")
    ax.scatter(
        np.ones(len(endpoints)), endpoints, s=6, color=COLORS["empirical"], edgecolors="none", alpha=0.52, zorder=3, rasterized=True,
        label="Endpoint",
    )
    ax.set_xlim(-0.1, 1.1)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["$v_t$", "$v_{t+1}$"])
    ax.set_ylabel("$v$")
    ax.legend(loc="best")
    _save_plot(fig, "km_nextstep", output_dir, prefix)


def plot_sigma_est(config, x_ref, sigma_est, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    x_ref = np.asarray(x_ref, dtype=float)
    sigma_est = np.asarray(sigma_est, dtype=float)
    idx = np.argsort(x_ref)
    x_plot = x_ref[idx]
    sigma_plot = sigma_est[idx]
    noise_model = config["noise"]
    sigma_ref = diffusion_profile(x_plot, noise_model["diffusion"])
    ax.plot(x_plot, sigma_ref, color=COLORS["truth"], linewidth=1.15, label="Ref. $\\sigma(v)$")
    ax.plot(x_plot, sigma_plot, color=COLORS["estimate"], linestyle="--", linewidth=1.0, label="Est. $\\widehat{\\sigma}(v)$")
    ax.set_xlabel("$v$")
    ax.set_ylabel("$\\sigma(v)$")
    ax.legend()
    _save_plot(fig, "sigma_est", output_dir, prefix)


def plot_d1(x_ref, d1_ref, d1_emp, d1_fit, output_dir, prefix=None):
    x_ref = np.asarray(x_ref, dtype=float)
    d1_ref = np.asarray(d1_ref, dtype=float)
    d1_emp = np.asarray(d1_emp, dtype=float)
    d1_fit = np.asarray(d1_fit, dtype=float)
    idx = np.argsort(x_ref)
    x_plot = x_ref[idx]
    d1_ref_plot = d1_ref[idx]
    d1_emp_plot = d1_emp[idx]
    d1_fit_plot = d1_fit[idx]
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.plot(x_plot, d1_ref_plot, color=COLORS["truth"], linestyle="-", linewidth=1.15, alpha=0.95, label="Ref. $D^{(1)}(v)$", zorder=1)
    ax.scatter(x_plot, d1_emp_plot, s=6, color=COLORS["empirical"], edgecolors="none", alpha=0.48, label="Data", zorder=2, rasterized=True)
    ax.plot(x_plot, d1_fit_plot, color=COLORS["estimate"], linestyle="--", linewidth=1.0, label="Fit", zorder=3)
    ax.set_xlabel("$v$")
    ax.set_ylabel("$D^{(1)}(v)$")
    ax.legend(loc="best")
    _save_plot(fig, "d1_est", output_dir, prefix)


def plot_kernel_est(t_eval, kernel_ref, kernel_est, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    t_eval = np.asarray(t_eval, dtype=float)
    kernel_ref = np.asarray(kernel_ref, dtype=float)
    kernel_est = np.asarray(kernel_est, dtype=float)
    mask = t_eval <= min(float(t_eval[-1]), 30.0)
    ax.plot(t_eval[mask], kernel_ref[mask], color=COLORS["truth"], linewidth=1.15, label="Ref. $\\gamma(t)$")
    ax.plot(t_eval[mask], kernel_est[mask], color=COLORS["estimate"], linestyle="--", linewidth=1.0, label="Est. $\\widehat{\\gamma}(t)$")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$\\gamma(t)$")
    ax.set_xlim(0.0, 30.0)
    ax.legend()
    _save_plot(fig, "kernel_est", output_dir, prefix)


def plot_loss(loss, output_dir, prefix=None):
    loss = np.asarray(loss, dtype=float)
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.plot(loss, color=COLORS["empirical"], linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("$n_{\\rm epoch}$")
    ax.set_ylabel("$\\mathcal{L}$")
    _save_plot(fig, "loss", output_dir, prefix)


def plot_levy_parameters(alpha_true, alpha_est, kappa_true, kappa_est, output_dir, prefix=None):
    fig, axes = plt.subplots(1, 2, figsize=SINGLE_COLUMN)
    for ax, label, truth, estimate in zip(axes, ("$\\alpha$", "$\\kappa$"), (alpha_true, kappa_true), (alpha_est, kappa_est)):
        ax.bar([0], [truth], width=0.58, color=COLORS["truth"], edgecolor="none", label="Ref.")
        ax.bar([1], [estimate], width=0.58, color=COLORS["estimate"], edgecolor="none", label="Est.")
        ax.set_xticks([0, 1], ["Ref.", "Est."])
        ax.set_ylabel(label)
        ax.margins(x=0.1)
        ax.legend(loc="best")
    for ax, truth, estimate in zip(axes, (alpha_true, kappa_true), (alpha_est, kappa_est)):
        upper = 1.12 * max(float(truth), float(estimate))
        ax.set_ylim(0.0, upper if upper > 0.0 else 1.0)
    _save_plot(fig, "levy_parameters", output_dir, prefix)


def plot_annular_rates(bin_edges, empirical_rates, fitted_rates, true_rates, output_dir, prefix=None):
    bin_edges = np.asarray(bin_edges, dtype=float)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.loglog(
        centers, true_rates, color=COLORS["truth"], marker="o", markerfacecolor="white", markeredgewidth=0.7, linewidth=1.15, label="Ref."
    )
    ax.loglog(
        centers, fitted_rates, color=COLORS["estimate"], linestyle="--", marker="s", markerfacecolor="white", markeredgewidth=0.7,
        linewidth=1.0, label="Fit",
    )
    ax.loglog(centers, empirical_rates, color=COLORS["empirical"], linestyle="none", marker="^", markersize=4.2, label="Data")
    ax.set_xlabel("$\\sqrt{r_k r_{k+1}}$")
    ax.set_ylabel("$n_k/(LM\\Delta t)$")
    ax.legend(frameon=False)
    _save_plot(fig, "levy_annular_rates", output_dir, prefix)


def plot_run(root):
    import yaml

    apply_publication_style()
    config = yaml.safe_load((root / "parameters.yaml").read_text())
    with np.load(root / "estimates.npz", allow_pickle=False) as stored:
        a = dict(stored)
    plot_kernel(a["t"], a["kernel_true"], root, prefix="001")
    plot_traj(a["t_inference"], a["t"], a["v_inference"], a["v_true"], root, prefix="002")
    plot_noise(a["t"], a["noise_increment"], root, prefix="003")
    plot_km_samples(a["phase1_branch_state"][0], a["phase1_branch_endpoints"], root, prefix="004")
    plot_sigma_est(config, a["phase1_x"], a["phase1_sigma"], root, prefix="005")
    plot_loss(a["loss_sigma"], root, prefix="006")
    count = 7
    if config["noise"]["type"] == "brownian_levy":
        m = yaml.safe_load((root / "metrics.yaml").read_text())["levy"]
        from model.inverse import stable_annular_rates

        true_rates = stable_annular_rates(m["alpha_true"], m["kappa_true"], m["epsilon"], m["q"], m["K"])
        plot_levy_parameters(m["alpha_true"], m["alpha_est"], m["kappa_true"], m["kappa_est"], root, prefix="007")
        plot_annular_rates(a["levy_bin_edges"], a["levy_empirical_rates"], a["levy_fitted_rates"], true_rates, root, prefix="008")
        count = 9
    plot_d1(a["x_ref"], a["d1_reference"], a["d1_empirical"], a["d1_fit"], root, prefix=f"{count:03d}")
    plot_kernel_est(a["t"], a["kernel_true"], a["kernel_est"], root, prefix=f"{count + 1:03d}")
