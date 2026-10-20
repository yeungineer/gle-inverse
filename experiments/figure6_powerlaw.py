"""Figure 6: powerlaw."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from model.fit import relative_l2_error, run_point, save_figure_data
from model.style import (
    COLORS, IQR_ALPHA, ROW_FIGURE_SIZE, SEQUENTIAL_HEATMAP_COLORS, add_panel_label, apply_publication_style, load_prepared_data,
    save_publication_figure, style_axis, use_fixed_exponent,
)

PARAMS = {
    "figure": {"stem": "Figure_6", "dpi": 600, "seeds": [10, 11, 12, 13, 14, 15], "representative_seed": 42, "iqr_alpha": 0.07},
    "experiment": {
        "km_samples": "euler_oracle",
        "dt": 0.01,
        "kernel_type": "power_law",
        "force_type": "zero",
        "diffusion_type": "constant",
        "kernel_display_t_max": 20.0,
        "N": 20,
        "M_values": [1000, 2000, 3000, 5000, 8000],
        "L_values": [3000, 4000, 5000, 8000],
        "displayed_pairs": [[2000, 3000], [5000, 4000]],
    },
}


def powerlaw_point(config, seed, M, L):
    s = config["experiment"]
    p, arrays, _ = run_point(
        "brownian",
        {
            "generation.simulation.seed": seed,
            "generation.simulation.dt": float(s["dt"]),
            "generation.simulation.num_steps": M,
            "generation.simulation.L": L,
            "generation.km_samples": s["km_samples"],
            "force.type": s["force_type"],
            "kernel.type": s["kernel_type"],
            "noise.diffusion": s["diffusion_type"],
            "phase2.estimation.num_basis": int(s["N"]),
            "phase2.estimation.t_max": 30.0,
        },
    )
    t = arrays["t"]
    keep = t <= 30.0 + 1e-12
    _, error = relative_l2_error(arrays["kernel_est"][keep], arrays["kernel_true"][keep], t[keep])
    return (arrays, error)


def run(config, out):
    s = config["experiment"]
    rows = []
    for seed in map(int, config["figure"]["seeds"]):
        for M in map(int, s["M_values"]):
            for L in map(int, s["L_values"]):
                print(f"powerlaw: seed={seed}, M={M}, L={L}", flush=True)
                _, error = powerlaw_point(config, seed, M, L)
                rows.append(dict(seed=seed, M=M, L=L, kernel_error=error))
    curves = {}
    for M, L in s["displayed_pairs"]:
        M, L = (int(M), int(L))
        a, _ = powerlaw_point(config, int(config["figure"]["representative_seed"]), M, L)
        curves[f"kernel_t__L{L}"] = np.asarray(a["t"], dtype=float)
        curves[f"kernel_truth__L{L}"] = np.asarray(a["kernel_true"], dtype=float)
        curves[f"kernel_estimate__M{M}__L{L}"] = np.asarray(a["kernel_est"], dtype=float)
    rows.sort(key=lambda r: (r["L"], r["M"], r["seed"]))
    arrays = {k: np.asarray([r[k] for r in rows], dtype=np.int64) for k in ("seed", "M", "L")}
    arrays.update(kernel_error=np.asarray([r["kernel_error"] for r in rows], dtype=float), **curves)
    metadata = dict(
        plot_seeds=list(map(int, config["figure"]["seeds"])), figure_style={"iqr_alpha": float(config["figure"]["iqr_alpha"])}, experiment=s
    )
    save_figure_data(out, 6, arrays, metadata)
    return rows


CURVE_STYLES = ((COLORS["blue"], "--"), (COLORS["red"], (0, (5.0, 1.2, 1.2, 1.2))))


def f6_grid(data, metadata):
    M_values = np.asarray(metadata["experiment"]["M_values"], dtype=int)
    L_values = np.asarray(metadata["experiment"]["L_values"], dtype=int)
    seeds = list(map(int, metadata["plot_seeds"]))
    error = np.full((len(seeds), len(L_values), len(M_values)), np.nan)
    for row in range(len(data["M"])):
        i = int(np.flatnonzero(L_values == data["L"][row])[0])
        j = int(np.flatnonzero(M_values == data["M"][row])[0])
        seed = int(data["seed"][row])
        error[seeds.index(seed), i, j] = float(data["kernel_error"][row])
    if not np.all(np.isfinite(error)):
        grid_points = len(M_values) * len(L_values)
        raise RuntimeError(f"Figure 6(b,c) requires {len(seeds)} complete seeds over all {grid_points} M--L grid points.")
    return (M_values, L_values, error)


def plot(data_path, metadata_path, output_dir, *, stem, dpi):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure = plt.figure(figsize=ROW_FIGURE_SIZE)
    outer = figure.add_gridspec(1, 3, width_ratios=(1.08, 1.0, 1.08), wspace=0.5, left=0.058, right=0.988, bottom=0.205, top=0.915)
    heatmap_b = outer[0, 1].subgridspec(1, 2, width_ratios=(1.0, 0.06), wspace=0.1)
    axes = [figure.add_subplot(outer[0, 0]), figure.add_subplot(heatmap_b[0, 0]), figure.add_subplot(outer[0, 2])]
    colorbar_axis = figure.add_subplot(heatmap_b[0, 1])
    display_max = float(metadata["experiment"]["kernel_display_t_max"])
    first_L = int(metadata["experiment"]["displayed_pairs"][0][1])
    truth_t = data[f"kernel_t__L{first_L}"]
    truth_mask = truth_t <= display_max
    axes[0].plot(
        truth_t[truth_mask], data[f"kernel_truth__L{first_L}"][truth_mask], color=COLORS["truth"], linewidth=1.55, label="Truth", zorder=1
    )
    displayed_pairs = metadata["experiment"]["displayed_pairs"]
    for curve_index, pair in enumerate(displayed_pairs):
        M, L = map(int, pair)
        style = CURVE_STYLES[curve_index]
        t = data[f"kernel_t__L{L}"]
        mask = t <= display_max
        key = f"kernel_estimate__M{M}__L{L}"
        is_top_curve = curve_index == len(displayed_pairs) - 1
        axes[0].plot(
            t[mask], data[key][mask], color=style[0], linestyle=style[1], linewidth=1.1, label=f"$M={M},\\,L={L}$",
            zorder=6 if is_top_curve else 3,
        )
    axes[0].set_xlim(0.0, display_max)
    axes[0].set_xlabel("$t$")
    axes[0].set_ylabel("$\\gamma(t)$")
    axes[0].legend(loc="best", fontsize=7.0)
    M_values, L_values, error_by_seed = f6_grid(data, metadata)
    error = np.nanmedian(error_by_seed, axis=0)
    x = np.arange(len(M_values) + 1) - 0.5
    y = np.arange(len(L_values) + 1) - 0.5
    log_error = np.log10(error)
    color_map = mcolors.LinearSegmentedColormap.from_list("figure6_kernel_error", SEQUENTIAL_HEATMAP_COLORS)
    image = axes[1].pcolormesh(x, y, log_error, shading="flat", cmap=color_map, vmin=-0.75, vmax=-0.65)
    for i in range(len(L_values)):
        for j in range(len(M_values)):
            axes[1].text(j, i, f"${error[i, j]:.3f}$", ha="center", va="center", fontsize=6.8, color="white")
    axes[1].set_xticks(np.arange(len(M_values)))
    axes[1].set_xticklabels([f"${value}$" for value in M_values])
    axes[1].set_yticks(np.arange(len(L_values)))
    axes[1].set_yticklabels([f"${value}$" for value in L_values])
    axes[1].set_xlabel("$M$")
    axes[1].set_ylabel("$L$")
    axes[1].tick_params(axis="both", which="both", length=0)
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_ticks([-0.75, -0.725, -0.7, -0.675, -0.65])
    colorbar.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    colorbar.ax.set_title("$\\log_{10}(E_\\gamma)$", pad=5.0, fontsize=6.7)
    colorbar.ax.tick_params(direction="in", length=2.4, width=0.65, labelsize=6.6)
    M_values, L_values, error = f6_grid(data, metadata)
    M_factors = M_values.astype(float) / float(M_values[0])
    L_factors = L_values.astype(float) / float(L_values[0])
    M_slice = error[:, -1, :]
    L_slice = error[:, :, -1]

    def draw_summary(x, values, color, fill, marker, label):
        q25 = np.nanquantile(values, 0.25, axis=0)
        median = np.nanmedian(values, axis=0)
        q75 = np.nanquantile(values, 0.75, axis=0)
        axes[2].fill_between(x, q25, q75, color=fill, alpha=IQR_ALPHA, linewidth=0.0, zorder=2)
        axes[2].plot(
            x, median, color=color, marker=marker, markerfacecolor="white", markeredgewidth=0.75, linewidth=1.25, label=label, zorder=4
        )

    slopes_M = np.asarray([np.polyfit(np.log(M_factors), np.log(values), 1)[0] for values in M_slice])
    slopes_L = np.asarray([np.polyfit(np.log(L_factors), np.log(values), 1)[0] for values in L_slice])
    slope_M = float(np.median(slopes_M))
    slope_L = float(np.median(slopes_L))
    draw_summary(M_factors, M_slice, COLORS["blue"], COLORS["blue_fill"], "o", f"increase $M$: $\\chi_M={slope_M:.3f}$")
    draw_summary(L_factors, L_slice, COLORS["red"], COLORS["red_fill"], "s", f"increase $L$: $\\chi_L={slope_L:.3f}$")
    axes[2].set_xscale("log")
    factor_ticks = M_factors
    axes[2].set_xticks(factor_ticks)
    axes[2].set_xticklabels([f"${value:g}$" for value in factor_ticks])
    axes[2].xaxis.set_minor_formatter(mticker.NullFormatter())
    axes[2].set_xlabel("Data factor")
    use_fixed_exponent(axes[2], -1)
    axes[2].set_ylabel("$E_\\gamma$")
    axes[2].legend(loc="best", fontsize=7.0)
    for label, axis in zip("abc", axes):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=int(dpi), creator="GLE inverse paper Figure 6 workflow")
    plt.close(figure)
    return outputs
