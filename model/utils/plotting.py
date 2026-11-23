"""单次反演和论文实验的共用绘图函数。"""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
import numpy as np
import yaml

from model.utils.style import (
    COLORS, IQR_ALPHA, SEQUENTIAL_HEATMAP_COLORS, apply_publication_style,
    save_publication_figure, style_axis, style_inset,
)
from model.utils.style import SINGLE_PANEL_SIZE as SINGLE_COLUMN


def _save_plot(fig, name, output_dir, prefix):
    for ax in fig.axes:
        style_axis(ax)
    fig.tight_layout(pad=0.6)
    save_publication_figure(fig, output_dir, f"{prefix}_{name}" if prefix else name, bbox_inches="tight")
    plt.close(fig)


def plot_kernel(t, gamma, t_max, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    mask = t <= t_max
    ax.plot(t[mask], gamma[mask], color=COLORS["truth"])
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\gamma(\tau)$")
    ax.set_xlim(0.0, t_max)
    _save_plot(fig, "kernel", output_dir, prefix)


def plot_traj(t_obs, t_raw, v_obs, v_raw, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.plot(t_raw, v_raw, color=COLORS["truth"], alpha=0.92, linewidth=1.15, label="Ref. $v(t)$")
    ax.plot(t_obs, v_obs, linestyle="none", marker="o", color=COLORS["empirical"],
            markeredgecolor="none", markersize=1.8,
            alpha=0.68, label="Obs. $\\widetilde{v}(t)$", rasterized=True)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$v(t)$")
    ax.set_xlim(0.0, t_raw[-1])
    ax.legend(loc="best")
    _save_plot(fig, "traj", output_dir, prefix)


def plot_noise(t, noise_increment, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.plot(t[: len(noise_increment)], noise_increment, color=COLORS["empirical"], linewidth=0.45, alpha=0.82)
    ax.set_xlabel("$t$")
    ax.set_ylabel("Noise increment")
    _save_plot(fig, "noise", output_dir, prefix)


def plot_sigma_est(x, sigma_true, sigma_est, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.plot(x, sigma_true, color=COLORS["truth"], label="Ref. $\\sigma(v)$")
    ax.plot(x, sigma_est, color=COLORS["estimate"], linestyle="--", label="Est. $\\widehat{\\sigma}(v)$")
    ax.set_xlabel("$v$")
    ax.set_ylabel("$\\sigma(v)$")
    ax.legend()
    _save_plot(fig, "sigma_est", output_dir, prefix)


def plot_loss(loss, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.plot(loss, color=COLORS["empirical"], linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("$n_{\\rm epoch}$")
    ax.set_ylabel("$\\mathcal{L}$")
    _save_plot(fig, "loss", output_dir, prefix)


def plot_levy_parameters(alpha_true, alpha_est, kappa_true, kappa_est, output_dir, prefix=None):
    fig, axes = plt.subplots(1, 2, figsize=SINGLE_COLUMN)
    for ax, label, truth, estimate in zip(axes, ("$\\alpha$", "$\\kappa$"),
                                          (alpha_true, kappa_true), (alpha_est, kappa_est)):
        ax.bar([0], [truth], width=0.58, color=COLORS["truth"], edgecolor="none", label="Ref.")
        ax.bar([1], [estimate], width=0.58, color=COLORS["estimate"], edgecolor="none", label="Est.")
        ax.set_xticks([0, 1], ["Ref.", "Est."])
        ax.set_ylabel(label)
        ax.margins(x=0.1)
        ax.legend(loc="best")
        ax.set_ylim(0.0, 1.12 * max(truth, estimate))
    _save_plot(fig, "levy_parameters", output_dir, prefix)


def plot_annular_rates(bin_edges, empirical_rates, fitted_rates, true_rates, output_dir, prefix=None):
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    ax.loglog(centers, true_rates, color=COLORS["truth"], marker="o", markerfacecolor="white",
              markeredgewidth=0.7, label="Ref.")
    ax.loglog(centers, fitted_rates, color=COLORS["estimate"], linestyle="--", marker="s", markerfacecolor="white",
              markeredgewidth=0.7, label="Est.")
    ax.loglog(centers, empirical_rates, color=COLORS["empirical"], linestyle="none", marker="^", label="Obs.")
    ax.set_xlabel("$\\sqrt{r_k r_{k+1}}$")
    ax.set_ylabel("$n_k/T_{\\widehat\\alpha}$")
    ax.legend(frameon=False)
    _save_plot(fig, "levy_annular_rates", output_dir, prefix)


def plot_d1(t, d1_ref, d1_fit, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    tt = t[: len(d1_ref)]
    ax.plot(tt, d1_ref, color=COLORS["truth"], alpha=0.95, label="Ref. $D_s^{(1)}(X_t)$", zorder=1)
    ax.plot(tt, d1_fit, color=COLORS["estimate"], linestyle="--", label="Est.", zorder=3)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$D_s^{(1)}(X_t)$")
    ax.set_xlim(0.0, tt[-1])
    ax.legend(loc="best")
    _save_plot(fig, "d1_est", output_dir, prefix)


def plot_kernel_est(t, kernel_ref, kernel_est, t_max, output_dir, prefix=None):
    fig, ax = plt.subplots(figsize=SINGLE_COLUMN)
    mask = t <= t_max
    ax.plot(t[mask], kernel_ref[mask], color=COLORS["truth"], label=r"Ref. $\gamma(\tau)$")
    ax.plot(t[mask], kernel_est[mask], color=COLORS["estimate"], linestyle="--",
            label=r"Est. $\widehat{\gamma}(\tau)$")
    ax.set_xlabel(r"$\tau$")
    ax.set_ylabel(r"$\gamma(\tau)$")
    ax.set_xlim(0.0, t_max)
    ax.legend()
    _save_plot(fig, "kernel_est", output_dir, prefix)


def plot_run(root, *, use_tex=True):
    apply_publication_style(use_tex=use_tex)
    config = yaml.safe_load((root / "parameters.yaml").read_text())
    metrics = yaml.safe_load((root / "metrics.yaml").read_text())
    a = dict(np.load(root / "estimates.npz"))
    t_max = metrics["kernel"]["t_max"]
    plot_kernel(a["t"], a["kernel_true"], t_max, root, prefix="001")
    plot_traj(a["t"], a["t_sim"], a["v_obs"], a["v_true"], root, prefix="002")
    plot_noise(a["t_sim"], a["noise_increment"], root, prefix="003")
    plot_sigma_est(a["sigma_x"], a["sigma_true"], a["sigma_est"], root, prefix="004")
    plot_loss(a["loss_sigma"], root, prefix="005")
    count = 6
    if config["noise"]["type"] == "brownian_levy":
        m = metrics["levy"]
        plot_levy_parameters(m["alpha_true"], m["alpha_est"], m["kappa_true"], m["kappa_est"], root, prefix="006")
        plot_annular_rates(a["levy_bin_edges"], a["levy_empirical_rates"], a["levy_fitted_rates"],
                           a["levy_true_rates"],
                           root, prefix="007")
        count = 8
    plot_d1(a["t"], (a["force_path"] - a["memory_true_path"]) / a["moment_scale_path"],
            (a["force_path"] - a["memory_est_path"]) / a["moment_scale_path"], root, prefix=f"{count:03d}")
    plot_kernel_est(a["t"], a["kernel_true"], a["kernel_est"], t_max, root, prefix=f"{count + 1:03d}")


def plot_memory_inset(axis, t, reference, estimate, error):
    inset = axis.inset_axes([0.48, 0.30, 0.47, 0.35])
    inset.plot(t, reference, color=COLORS["truth"], linewidth=0.6)
    inset.plot(t, estimate, color=COLORS["blue"], linestyle="--", linewidth=0.85)
    inset.set(xlabel="$t$", ylabel=r"$\mathcal{M}$")
    inset.set_title(rf"$\mathrm{{Err.}}={100 * error:.2f}\%$", fontsize=5.5, pad=2.0)
    style_inset(inset)


def plot_median_iqr(axis, x, values, color, marker, label, linestyle="-", *, quantile=np.quantile, zorder=2):
    has_data = np.isfinite(values).any(axis=0)
    quartiles = np.full((3, len(x)), np.nan)
    quartiles[:, has_data] = quantile(values[:, has_data], [0.25, 0.5, 0.75], axis=0)
    q25, median, q75 = quartiles
    axis.fill_between(x, q25, q75, color=color, alpha=IQR_ALPHA, linewidth=0.0)
    axis.plot(x, median, color=color, marker=marker, linestyle=linestyle, label=label, zorder=zorder)
    return median


def plot_heatmap(axis, values, x_labels, y_labels, xlabel, ylabel, norm=None):
    colormap = LinearSegmentedColormap.from_list("sequential", SEQUENTIAL_HEATMAP_COLORS)
    colormap.set_bad("#eeeeee")
    image = axis.imshow(values, origin="lower", aspect="auto", cmap=colormap,
                        norm=norm or LogNorm(np.nanmin(values), np.nanmax(values)))
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            percent = 100 * values[i, j]
            if not np.isfinite(percent):
                label = r"$\mathrm{NA}$"
            elif percent >= 1000:
                mantissa, exponent = f"{percent:.0e}".split("e")
                label = rf"$\mathtt{{{mantissa}e{int(exponent)}}}$"
            else:
                label = f"{percent:.0f}" if percent >= 10 else f"{percent:.1f}"
            axis.text(j, i, label, ha="center", va="center", fontsize=5.0,
                      color="white" if image.norm(values[i, j]) > 0.65 else "black")
    axis.set_xticks(np.arange(len(x_labels)), [f"${x:g}$" for x in x_labels])
    axis.set_yticks(np.arange(len(y_labels)), [f"${y:g}$" for y in y_labels])
    axis.tick_params(length=0)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    return image
