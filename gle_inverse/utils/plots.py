"""Plot styling and diagnostic figures for the inverse workflow."""

import logging
import math
import shutil
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from gle_inverse.simulation.noises import diffusion_profile, parse_noise_config

MM_PER_INCH = 25.4
MIN_ARTWORK_DPI = 500
ARTWORK_DPI_STEP = 50
SINGLE_COLUMN_MIN_PIXELS = 3543
FULL_WIDTH_MIN_PIXELS = 7480
PAPER_HEIGHT = 68.0 / MM_PER_INCH
SINGLE_COLUMN = (90.0 / MM_PER_INCH, PAPER_HEIGHT)
ONE_HALF_COLUMN = (140.0 / MM_PER_INCH, PAPER_HEIGHT)
DOUBLE_COLUMN = (190.0 / MM_PER_INCH, PAPER_HEIGHT)
PAPER_BOTTOM = 0.200
PAPER_TOP = 0.900
SAVE_PADDING_INCHES = 0.10

PAPER_COLORS = {
    "black": "#1F1F1F",
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "pink": "#CC79A7",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#E69F00",
    "violet": "#6F4C9B",
}

LATEX_PREAMBLE = (
    r"\usepackage{CJKutf8}\usepackage{amsmath,amssymb,bm}"
    r"\AtBeginDocument{\begin{CJK}{UTF8}{gbsn}}"
    r"\AtEndDocument{\end{CJK}}"
)

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "text.latex.preamble": LATEX_PREAMBLE,
    "font.size": 7.0,
    "axes.labelsize": 8.0,
    "axes.titlesize": 8.0,
    "axes.linewidth": 0.75,
    "lines.linewidth": 1.0,
    "lines.markersize": 4.0,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "legend.frameon": False,
}


class PlotCounter:
    def __init__(self):
        self.count = 0

    def get_next(self):
        self.count += 1
        return f"{self.count:03d}"


def configure_paper_style(**overrides):
    plt.rcParams.update(PAPER_STYLE | {"text.usetex": shutil.which("latex") is not None} | overrides)


def create_styled_subplots(*args, **kwargs):
    configure_paper_style()
    return plt.subplots(*args, **kwargs)


def publication_dpi(figure, output_width_inches=None):
    width_inches = float(figure.get_size_inches()[0])
    output_width_inches = width_inches if output_width_inches is None else float(output_width_inches)
    minimum_pixels = FULL_WIDTH_MIN_PIXELS if width_inches >= 7.0 else SINGLE_COLUMN_MIN_PIXELS
    required_dpi = minimum_pixels / output_width_inches
    stepped_dpi = math.ceil((required_dpi - 1.0e-9) / ARTWORK_DPI_STEP) * ARTWORK_DPI_STEP
    return max(MIN_ARTWORK_DPI, stepped_dpi)


def save_figure(figure, output_dir, stem, *, clean_axes=True, inward_ticks=False, close=True):
    if clean_axes:
        for axis in figure.axes:
            axis.grid(False)
            if inward_ticks:
                axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)
            legend = axis.get_legend()
            if legend is not None:
                _, labels = axis.get_legend_handles_labels()
                if len(labels) <= 1:
                    legend.remove()
                else:
                    legend.set_frame_on(False)

    figure.canvas.draw()
    tight_bbox = figure.get_tightbbox(figure.canvas.get_renderer())
    output_width_inches = (
        float(tight_bbox.width) + 2.0 * SAVE_PADDING_INCHES
        if tight_bbox is not None
        else float(figure.get_size_inches()[0])
    )
    dpi = publication_dpi(figure, output_width_inches)
    output_path = Path(output_dir) / f"{stem}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gle_figure_") as temporary:
        temporary_path = Path(temporary) / output_path.name
        figure.savefig(
            temporary_path,
            dpi=dpi,
            facecolor="white",
            transparent=False,
            bbox_inches="tight",
            pad_inches=SAVE_PADDING_INCHES,
        )
        shutil.copyfile(temporary_path, output_path)
    logging.info("[PLOT] Saved: %s", output_path)
    if close:
        plt.close(figure)
    return output_path


def _filename(base_name, suffix=None, prefix=None):
    return "_".join(str(value) for value in (prefix, base_name, suffix) if value)


def _save_plot(fig, base_name, output_dir, suffix=None, prefix=None):
    for axis in fig.axes:
        axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_color("#333333")
            spine.set_linewidth(0.75)
        legend = axis.get_legend()
        if legend is not None:
            _, labels = axis.get_legend_handles_labels()
            if len(labels) <= 1:
                legend.remove()
            else:
                legend.set_frame_on(False)

    axes = fig.axes
    horizontal_panels = len(axes) >= 2 and all(
        abs(axis.get_position().y0 - axes[0].get_position().y0) < 1.0e-6 for axis in axes[1:]
    )
    if len(axes) == 1 or horizontal_panels:
        fig.set_size_inches(float(fig.get_size_inches()[0]), PAPER_HEIGHT, forward=True)
    if len(axes) == 1:
        fig.subplots_adjust(left=0.185, right=0.97, bottom=PAPER_BOTTOM, top=PAPER_TOP)
    elif horizontal_panels:
        fig.subplots_adjust(left=0.07, right=0.985, bottom=PAPER_BOTTOM, top=PAPER_TOP, wspace=0.42)
    else:
        fig.subplots_adjust(left=0.145, right=0.975, bottom=0.115, top=0.9, hspace=0.58)
    return save_figure(fig, output_dir, _filename(base_name, suffix, prefix))

COLORS = {
    "true": PAPER_COLORS["black"],
    "estimate": PAPER_COLORS["orange"],
    "empirical": PAPER_COLORS["blue"],
    "secondary": PAPER_COLORS["green"],
    "accent": PAPER_COLORS["pink"],
    "branch": PAPER_COLORS["sky"],
}

LINEWIDTH = {"reference": 1.05, "estimate": 1.15, "data": 0.85}


def plot_kernel(t_raw, gamma, output_dir, suffix=None, prefix=None):
    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    mask = (t_raw >= 0) & (t_raw <= min(t_raw[-1], 30))

    ax.plot(t_raw[mask], gamma[mask], color=COLORS["empirical"], linewidth=LINEWIDTH["data"])
    ax.set_xlabel(r"$t$")
    ax.set_ylabel(r"$\gamma(t)$")
    ax.set_xlim(0.0, 30.0)
    _save_plot(fig, "kernel", output_dir, suffix, prefix)


def draw_noise(axis, t_raw, noise_increment):
    """Plot the stochastic velocity increment used by Euler--Maruyama."""
    mask = (t_raw >= 0) & (t_raw <= min(t_raw[-1], 600))
    axis.plot(t_raw[mask], np.asarray(noise_increment)[mask], color=COLORS["empirical"], linewidth=0.45, alpha=0.82)
    axis.set_xlabel(r"$t$")
    axis.set_ylabel("Noise increment")


def plot_noise(t_raw, noise_increment, output_dir, suffix=None, prefix=None):
    """Save the stochastic-increment diagnostic."""
    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    draw_noise(ax, t_raw, noise_increment)
    _save_plot(fig, "noise", output_dir, suffix, prefix)


def draw_trajectory(axis, t_obs, t_raw, v_obs, v_raw, *, levy_increment=None, time_max=200.0, observation_stride=1):
    """Draw reference and observed trajectories on an existing axis."""
    t_raw = np.asarray(t_raw, dtype=float)
    v_raw = np.asarray(v_raw, dtype=float)
    t_obs = np.asarray(t_obs, dtype=float)
    v_obs = np.asarray(v_obs, dtype=float)
    time_max = float(time_max)
    observation_stride = int(observation_stride)
    if time_max <= 0.0:
        raise ValueError("time_max must be positive.")
    if observation_stride < 1:
        raise ValueError("observation_stride must be at least one.")

    mask_raw = (t_raw >= 0.0) & (t_raw < time_max)
    mask_obs = (t_obs >= 0.0) & (t_obs < time_max)

    axis.plot(
        t_raw[mask_raw],
        v_raw[mask_raw],
        color=COLORS["true"],
        alpha=0.92,
        linewidth=LINEWIDTH["reference"],
        drawstyle="steps-post" if levy_increment is not None else "default",
        label=r"Ref. $v(t)$",
    )
    observed_indices = np.flatnonzero(mask_obs)
    observed_indices = observed_indices[::observation_stride]
    axis.plot(
        t_obs[observed_indices],
        v_obs[observed_indices],
        linestyle="none",
        marker="o",
        color=COLORS["empirical"],
        markeredgecolor="none",
        markersize=1.8,
        alpha=0.68,
        label=r"Obs. $\widetilde{v}(t)$",
        rasterized=True,
    )

    if levy_increment is not None:
        levy_increment = np.asarray(levy_increment, dtype=float)
        if levy_increment.shape != t_raw.shape:
            raise ValueError("The Lévy-increment array must match the raw trajectory.")

    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$v(t)$")
    axis.set_xlim(0.0, time_max)
    axis.legend(loc="best")


def plot_traj(t_obs, t_raw, v_obs, v_raw, output_dir, suffix=None, prefix=None):
    """Save the reference-and-observed trajectory diagnostic."""
    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    draw_trajectory(ax, t_obs, t_raw, v_obs, v_raw)
    _save_plot(fig, "traj", output_dir, suffix, prefix)


def draw_conditional_branches(axis, conditioned_state, endpoints):
    """Draw one conditional one-step ensemble on an existing axis."""
    endpoints = np.asarray(endpoints, dtype=float)
    for sample_index, endpoint in enumerate(endpoints):
        axis.plot(
            [0, 1],
            [conditioned_state, endpoint],
            color=COLORS["branch"],
            linewidth=0.4,
            alpha=0.20,
            zorder=1,
            label="Branch" if sample_index == 0 else None,
        )

    axis.scatter(
        [0],
        [conditioned_state],
        s=18,
        color=COLORS["estimate"],
        edgecolors="white",
        linewidths=0.4,
        zorder=4,
        label="Ref. state",
    )
    axis.scatter(
        np.ones(len(endpoints)),
        endpoints,
        s=6,
        color=COLORS["empirical"],
        edgecolors="none",
        alpha=0.52,
        zorder=3,
        rasterized=True,
        label="Endpoint",
    )

    axis.set_xlim(-0.1, 1.1)
    axis.set_xticks([0, 1])
    axis.set_xticklabels([r"$v_t$", r"$v_{t+1}$"])
    axis.set_ylabel(r"$v$")
    axis.legend(loc="best")
    axis.tick_params(axis="both", which="both", direction="in", top=True, right=True)


def plot_km_samples(x_obs, v_next, output_dir, num_ref=4, max_samples=100, suffix=None, prefix=None):
    x_obs = np.asarray(x_obs, dtype=float)
    v_next = np.asarray(v_next, dtype=float)
    indices = np.arange(len(x_obs))

    num_selected = min(num_ref, len(indices))
    idx_selected = np.linspace(0, len(indices) - 1, num_selected, dtype=int)

    if num_selected == 1:
        figure_size = SINGLE_COLUMN
    elif num_selected == 2:
        figure_size = ONE_HALF_COLUMN
    else:
        figure_size = DOUBLE_COLUMN
    fig, axes = create_styled_subplots(1, num_selected, figsize=figure_size, squeeze=False)
    axes = axes[0]
    for ax, idx in zip(axes, idx_selected):
        samples = v_next[idx, : min(max_samples, v_next.shape[1])]
        draw_conditional_branches(ax, x_obs[idx], samples)

    _save_plot(fig, "km_nextstep", output_dir, suffix, prefix)


def draw_sigma_est(axis, config, x_ref, sigma_est):
    """Draw reference and reconstructed diffusion amplitudes."""
    x_ref = np.asarray(x_ref, dtype=float)
    sigma_est = np.asarray(sigma_est, dtype=float)

    idx = np.argsort(x_ref)
    x_plot = x_ref[idx]
    sigma_plot = sigma_est[idx]
    noise_model = parse_noise_config(config["noise"])
    sigma_ref = diffusion_profile(x_plot, noise_model.diffusion_name())

    axis.plot(x_plot, sigma_ref, color=COLORS["true"], linewidth=LINEWIDTH["reference"], label=r"Ref. $\sigma(v)$")
    axis.plot(
        x_plot,
        sigma_plot,
        color=COLORS["estimate"],
        linestyle="--",
        linewidth=LINEWIDTH["estimate"],
        label=r"Est. $\widehat{\sigma}(v)$",
    )
    axis.set_xlabel(r"$v$")
    axis.set_ylabel(r"$\sigma(v)$")
    axis.legend()


def plot_sigma_est(config, x_ref, sigma_est, output_dir, suffix=None, prefix=None):
    """Save the diffusion-amplitude reconstruction diagnostic."""
    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    draw_sigma_est(ax, config, x_ref, sigma_est)
    _save_plot(fig, "sigma_est", output_dir, suffix, prefix)


def plot_d1(x_ref, d1_ref, d1_emp, d1_fit, output_dir, suffix=None, prefix=None):
    x_ref = np.asarray(x_ref, dtype=float)
    d1_ref = np.asarray(d1_ref, dtype=float)
    d1_emp = np.asarray(d1_emp, dtype=float)
    d1_fit = np.asarray(d1_fit, dtype=float)

    idx = np.argsort(x_ref)
    x_plot = x_ref[idx]
    d1_ref_plot = d1_ref[idx]
    d1_emp_plot = d1_emp[idx]
    d1_fit_plot = d1_fit[idx]

    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    ax.plot(
        x_plot,
        d1_ref_plot,
        color=COLORS["true"],
        linestyle="-",
        linewidth=LINEWIDTH["reference"],
        alpha=0.95,
        label=r"Ref. $D^{(1)}(v)$",
        zorder=1,
    )
    ax.scatter(
        x_plot,
        d1_emp_plot,
        s=6,
        color=COLORS["empirical"],
        edgecolors="none",
        alpha=0.48,
        label="Data",
        zorder=2,
        rasterized=True,
    )
    ax.plot(
        x_plot,
        d1_fit_plot,
        color=COLORS["estimate"],
        linestyle="--",
        linewidth=LINEWIDTH["estimate"],
        label="Fit",
        zorder=3,
    )

    ax.set_xlabel(r"$v$")
    ax.set_ylabel(r"$D^{(1)}(v)$")
    ax.legend(loc="best")

    _save_plot(fig, "d1_est", output_dir, suffix, prefix)


def draw_kernel_est(axis, t_eval, kernel_ref, kernel_est):
    """Draw reference and reconstructed memory kernels."""
    t_eval = np.asarray(t_eval, dtype=float)
    kernel_ref = np.asarray(kernel_ref, dtype=float)
    kernel_est = np.asarray(kernel_est, dtype=float)

    mask = t_eval <= min(float(t_eval[-1]), 30.0)

    axis.plot(
        t_eval[mask],
        kernel_ref[mask],
        color=COLORS["true"],
        linewidth=LINEWIDTH["reference"],
        label=r"Ref. $\gamma(t)$",
    )
    axis.plot(
        t_eval[mask],
        kernel_est[mask],
        color=COLORS["estimate"],
        linestyle="--",
        linewidth=LINEWIDTH["estimate"],
        label=r"Est. $\widehat{\gamma}(t)$",
    )
    axis.set_xlabel(r"$t$")
    axis.set_ylabel(r"$\gamma(t)$")
    axis.set_xlim(0.0, 30.0)
    axis.legend()


def plot_kernel_est(t_eval, kernel_ref, kernel_est, output_dir, suffix=None, prefix=None):
    """Save the memory-kernel reconstruction diagnostic."""
    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    draw_kernel_est(ax, t_eval, kernel_ref, kernel_est)
    _save_plot(fig, "kernel_est", output_dir, suffix, prefix)


def plot_loss(loss, output_dir, suffix=None, prefix=None):
    loss = np.asarray(loss, dtype=float)

    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    ax.plot(loss, color=COLORS["empirical"], linewidth=LINEWIDTH["data"])
    ax.set_yscale("log")
    ax.set_xlabel(r"$n_{\rm epoch}$")
    ax.set_ylabel(r"$\mathcal{L}$")

    _save_plot(fig, "loss", output_dir, suffix, prefix)


def plot_levy_parameters(alpha_true, alpha_est, kappa_true, kappa_est, output_dir, suffix=None, prefix=None):
    fig, axes = create_styled_subplots(1, 2, figsize=SINGLE_COLUMN)
    for ax, label, truth, estimate in zip(
        axes, (r"$\alpha$", r"$\kappa$"), (alpha_true, kappa_true), (alpha_est, kappa_est)
    ):
        ax.bar([0], [truth], width=0.58, color=COLORS["true"], edgecolor="none", label="Ref.")
        ax.bar([1], [estimate], width=0.58, color=COLORS["estimate"], edgecolor="none", label="Est.")
        ax.set_xticks([0, 1], ["Ref.", "Est."])
        ax.set_ylabel(label)
        ax.margins(x=0.10)
        ax.legend(loc="best")
    for ax, truth, estimate in zip(axes, (alpha_true, kappa_true), (alpha_est, kappa_est)):
        upper = 1.12 * max(float(truth), float(estimate))
        ax.set_ylim(0.0, upper if upper > 0.0 else 1.0)

    _save_plot(fig, "levy_parameters", output_dir, suffix, prefix)


def plot_annular_rates(bin_edges, empirical_rates, fitted_rates, true_rates, output_dir, suffix=None, prefix=None):
    bin_edges = np.asarray(bin_edges, dtype=float)
    centers = np.sqrt(bin_edges[:-1] * bin_edges[1:])
    fig, ax = create_styled_subplots(figsize=SINGLE_COLUMN)
    ax.loglog(
        centers,
        true_rates,
        color=COLORS["true"],
        marker="o",
        markerfacecolor="white",
        markeredgewidth=0.7,
        linewidth=LINEWIDTH["reference"],
        label="Ref.",
    )
    ax.loglog(
        centers,
        fitted_rates,
        color=COLORS["estimate"],
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markeredgewidth=0.7,
        linewidth=LINEWIDTH["estimate"],
        label="Fit",
    )
    ax.loglog(
        centers, empirical_rates, color=COLORS["empirical"], linestyle="none", marker="^", markersize=4.2, label="Data"
    )
    ax.set_xlabel(r"$\sqrt{r_k r_{k+1}}$")
    ax.set_ylabel(r"$n_k/(LM\Delta t)$")
    ax.legend(frameon=False)
    _save_plot(fig, "levy_annular_rates", output_dir, suffix, prefix)
