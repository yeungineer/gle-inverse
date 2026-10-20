"""Figure 3: sampling."""

import matplotlib.pyplot as plt
import numpy as np

from model.fit import float_column, run_point, save_figure_data, string_column
from model.inverse import brownian_euler_error_scales, effective_rank
from model.simulation import diffusion_profile
from model.style import (
    COLORS, IQR_ALPHA, ROW_FIGURE_SIZE, add_panel_label, apply_publication_style, load_prepared_data, save_publication_figure, style_axis,
    use_fixed_exponent,
)

METRIC_FIELDS = (
    "d1_normalized_rmse",
    "d2_normalized_rmse",
    "d1_rmse",
    "d2_rmse",
    "kernel_error",
    "sigma_error",
    "effective_rank",
    "smallest_relative_singular",
    "selected_lambda",
    "regularization_selected_at_boundary",
)


STRING_DIAGNOSTIC_FIELDS = ("regularization_method",)


PARAMS = {
    "figure": {"stem": "Figure_3", "dpi": 600, "seeds": [10, 11, 12, 13, 14, 15], "iqr_alpha": 0.07},
    "experiments": {
        "kernel_reconstruction": {"num_basis": 12, "t_max": 30.0},
        "core": {"dt": 0.01, "reference_rows": 3000, "diffusion_type": "constant"},
        "sampling_M": {"values": [1000, 3000, 5000, 7000]},
        "fixed_L": {
            "curves": [
                {"L": 3000, "M_values": [250, 500, 750, 1000]},
                {"L": 6000, "M_values": [250, 500, 750, 1000]},
                {"L": 9000, "M_values": [250, 500, 750, 1000]},
            ]
        },
    },
}


def expected_points(config):
    experiments = config["experiments"]
    dt = float(experiments["core"]["dt"])
    reference_rows = int(experiments["core"]["reference_rows"])
    cases = []
    for M in map(int, experiments["sampling_M"]["values"]):
        for experiment in ("sampling_M_theory", "kernel_M"):
            cases.append((experiment, M, reference_rows))
    for curve in experiments["fixed_L"]["curves"]:
        for M in map(int, curve["M_values"]):
            cases.append(("fixed_L_brownian", M, int(curve["L"])))
    points = [dict(experiment=name, case="brownian", M=M, L=L, dt=dt, budget=M * L) for name, M, L in cases]
    for point in points:
        point["point_id"] = "__".join(
            (point["experiment"], f"M{point['M']}", f"L{point['L']}", f"dt{point['dt']:.12g}", f"B{point['budget']}")
        ).replace(".", "p")
    return points


def _metrics(problem, arrays, metrics):
    d1_error = arrays["d1_empirical"] - arrays["d1_reference"]
    d2_error = arrays["d2_diffusion_target"] - arrays["d2_finite_step_reference"]
    d1_rmse = float(np.sqrt(np.mean(d1_error**2)))
    d2_rmse = float(np.sqrt(np.mean(d2_error**2)))
    km = problem["km_samples"]
    drift = np.asarray(km["f"]) - np.asarray(km["mem"])
    noise = problem["driver_config"]["noise"]
    diffusion = diffusion_profile(km["x"], noise["diffusion"])
    scales = brownian_euler_error_scales(drift, diffusion, float(km["dt"]), sample_size=1)
    d1_scale = np.sqrt(np.mean(np.asarray(scales["d1_standard_deviation"]) ** 2))
    d2_scale = np.sqrt(np.mean(np.asarray(scales["d2_standard_deviation"]) ** 2))
    singular = np.asarray(arrays["regularization_singular_values"])
    regularization = metrics["regularization"]
    return {
        "d1_rmse": d1_rmse,
        "d2_rmse": d2_rmse,
        "d1_normalized_rmse": float(d1_rmse / d1_scale),
        "d2_normalized_rmse": float(d2_rmse / d2_scale),
        "kernel_error": float(metrics["kernel"]["relative_l2_error"]),
        "sigma_error": float(metrics["sigma"]["relative_l2_error"]),
        "effective_rank": effective_rank(singular),
        "smallest_relative_singular": float(singular[-1] / singular[0]),
        "selected_lambda": float(regularization["selected_lambda"]),
        "regularization_selected_at_boundary": bool(regularization["selected_at_search_boundary"]),
        "regularization_method": str(regularization["mode"]),
    }


def _run_point(config, spec, seed, cache):
    reconstruction = config["experiments"]["kernel_reconstruction"]
    overrides = {
        "generation.simulation.seed": seed,
        "generation.simulation.dt": float(spec["dt"]),
        "generation.simulation.num_steps": int(spec["M"]),
        "generation.simulation.L": int(spec["L"]),
        "generation.km_samples": "euler_oracle",
        "noise.diffusion": str(config["experiments"]["core"]["diffusion_type"]),
        "phase2.estimation.num_basis": int(reconstruction["num_basis"]),
        "phase2.estimation.t_max": float(reconstruction["t_max"]),
    }
    key = (seed, spec["M"], spec["L"], spec["dt"])
    if key not in cache:
        problem, arrays, metrics = run_point("brownian", parameters=overrides)
        cache[key] = _metrics(problem, arrays, metrics)
    values = cache[key]
    row = {**spec, "seed": seed, "success": True}
    fields = (
        ("d1_rmse", "d2_rmse", "d1_normalized_rmse", "d2_normalized_rmse")
        if spec["experiment"] == "sampling_M_theory"
        else (
            "kernel_error",
            "sigma_error",
            "effective_rank",
            "smallest_relative_singular",
            "selected_lambda",
            "regularization_selected_at_boundary",
            "regularization_method",
        )
    )
    row.update({field: values[field] for field in fields})
    return row


def run(config, out):
    points = expected_points(config)
    rows = []
    cache = {}
    for seed in map(int, config["figure"]["seeds"]):
        for spec in points:
            rows.append(_run_point(config, spec, seed, cache))
    arrays, metadata = pack_rows(config, rows)
    save_figure_data(out, 3, arrays, metadata)
    return rows


def pack_rows(config, rows):
    rows.sort(key=lambda row: (row["experiment"], row["budget"], row["dt"], row["M"], row["L"], row["seed"]))
    arrays = {key: string_column(rows, key) for key in ("point_id", "experiment", "case")}
    arrays.update({key: np.asarray([int(row[key]) for row in rows], dtype=np.int64) for key in ("seed", "M", "L", "budget")})
    arrays.update(dt=float_column(rows, "dt"), success=np.ones(len(rows), dtype=bool), included_in_plot=np.ones(len(rows), dtype=bool))
    for field in METRIC_FIELDS:
        arrays[field] = float_column(rows, field)
    for field in STRING_DIAGNOSTIC_FIELDS:
        arrays[field] = string_column(rows, field)
    metadata = {
        "plot_seeds": list(map(int, config["figure"]["seeds"])),
        "figure_style": {"iqr_alpha": float(config["figure"]["iqr_alpha"])},
        "panels": {"c": {"display_L_values": [int(curve["L"]) for curve in config["experiments"]["fixed_L"]["curves"]]}},
        "scientific_configuration": config["experiments"],
    }
    return (arrays, metadata)


PALETTE = {
    "blue": (COLORS["blue"], COLORS["blue_fill"]),
    "red": (COLORS["red"], COLORS["red_fill"]),
    "teal": (COLORS["teal"], COLORS["teal_fill"]),
}


def f3_matching(data, experiment, *, L=None):
    mask = (data["experiment"] == experiment) & data["included_in_plot"].astype(bool) & data["success"].astype(bool)
    if L is not None:
        mask &= data["L"] == int(L)
    return mask


def f3_series_matrix(data, metadata, *, experiment, x_field, x_values, metric, L=None):
    seeds = list(map(int, metadata["plot_seeds"]))
    x_numeric = np.asarray(x_values, dtype=float)
    matrix = np.full((len(seeds), len(x_numeric)), np.nan, dtype=float)
    for row_index in np.flatnonzero(f3_matching(data, experiment, L=L)):
        seed = int(data["seed"][row_index])
        if seed not in seeds:
            continue
        x = float(data[x_field][row_index])
        candidates = np.flatnonzero(np.isclose(x_numeric, x, rtol=0.0, atol=1e-12))
        if len(candidates) != 1:
            raise RuntimeError(f"Could not uniquely place {experiment} {x_field}={x}.")
        i = seeds.index(seed)
        j = int(candidates[0])
        if np.isfinite(matrix[i, j]):
            raise RuntimeError(f"Duplicate prepared value for {experiment}, seed={seed}.")
        matrix[i, j] = float(data[metric][row_index])
    if not np.all(np.isfinite(matrix)):
        raise RuntimeError(f"Figure 3 has an incomplete {experiment}/{metric} seed matrix.")
    return matrix


def f3_plot_seed_summary(axis, x, matrix, *, color, fill, label, marker, iqr_alpha, linestyle="-"):
    q25 = np.quantile(matrix, 0.25, axis=0)
    median = np.median(matrix, axis=0)
    q75 = np.quantile(matrix, 0.75, axis=0)
    axis.fill_between(x, q25, q75, color=fill, alpha=iqr_alpha, linewidth=0.0, zorder=2)
    axis.plot(
        x, median, color=color, linewidth=1.35, linestyle=linestyle, marker=marker, markerfacecolor="white", markeredgecolor=color,
        markeredgewidth=0.75, label=label, zorder=3,
    )
    return median


def f3_linear_x(axis, ticks):
    axis.set_xticks(ticks)
    margin = 0.04 * float(ticks[-1] - ticks[0])
    axis.set_xlim(float(ticks[0]) - margin, float(ticks[-1]) + margin)


def plot(data_path, metadata_path, output_dir, *, stem="Figure_3", dpi=600):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    settings = metadata["scientific_configuration"]
    figure, axes = plt.subplots(1, 3, figsize=ROW_FIGURE_SIZE, squeeze=False)
    panels = axes[0]
    x = np.asarray(settings["sampling_M"]["values"], dtype=float)
    d1 = f3_series_matrix(data, metadata, experiment="sampling_M_theory", x_field="M", x_values=x.tolist(), metric="d1_normalized_rmse")
    d2 = f3_series_matrix(data, metadata, experiment="sampling_M_theory", x_field="M", x_values=x.tolist(), metric="d2_normalized_rmse")
    median = f3_plot_seed_summary(
        panels[0], x, d1, color=PALETTE["blue"][0], fill=PALETTE["blue"][1], label="$\\widehat D^{(1)}$", marker="o", iqr_alpha=IQR_ALPHA
    )
    f3_plot_seed_summary(
        panels[0], x, d2, color=PALETTE["red"][0], fill=PALETTE["red"][1], label="$\\widehat D^{(2)}$", marker="s",
        linestyle=(0, (4.0, 1.4, 1.0, 1.4)), iqr_alpha=IQR_ALPHA,
    )
    guide = median[2] * (x / x[2]) ** (-0.5)
    panels[0].plot(x, guide, color=COLORS["truth"], linewidth=0.9, linestyle=(0, (3.0, 2.0)), label="$M^{-1/2}$", zorder=5)
    f3_linear_x(panels[0], x)
    panels[0].set_xticklabels([f"${int(value)}$" for value in x])
    panels[0].set_yticks([0.01, 0.015, 0.02, 0.025, 0.03, 0.035])
    panels[0].set_ylim(0.0095, 0.038)
    use_fixed_exponent(panels[0], -2)
    panels[0].set_xlabel("$M$")
    panels[0].set_ylabel("NRMSE")
    panels[0].legend(loc="upper center", ncol=3, fontsize=7.0, columnspacing=0.85, handletextpad=0.35)
    x = np.asarray(settings["sampling_M"]["values"], dtype=float)
    matrix = f3_series_matrix(data, metadata, experiment="kernel_M", x_field="M", x_values=x.tolist(), metric="kernel_error")
    median = f3_plot_seed_summary(
        panels[1], x, matrix, color=PALETTE["blue"][0], fill=PALETTE["blue"][1], label="$E_\\gamma$", marker="o", iqr_alpha=IQR_ALPHA
    )
    guide = median[2] * (x / x[2]) ** (-0.5)
    panels[1].plot(x, guide, color=COLORS["truth"], linewidth=0.9, linestyle=(0, (3.0, 2.0)), label="$M^{-1/2}$", zorder=5)
    f3_linear_x(panels[1], x)
    panels[1].set_xticklabels([f"${int(value)}$" for value in x])
    panels[1].set_yticks([0.0, 0.04, 0.08, 0.12])
    panels[1].set_ylim(0.0, 0.14)
    use_fixed_exponent(panels[1], -2)
    panels[1].set_xlabel("$M$")
    panels[1].set_ylabel("$E_\\gamma$")
    panels[1].legend(loc="upper center", ncol=2, fontsize=7.0, columnspacing=0.85, handletextpad=0.35)
    fixed_L_settings = settings["fixed_L"]
    curves = list(fixed_L_settings["curves"])
    display_L_values = set(map(int, metadata["panels"]["c"].get("display_L_values", [])))
    if display_L_values:
        curves = [curve for curve in curves if int(curve["L"]) in display_L_values]
    styles = [
        (PALETTE["blue"][0], PALETTE["blue"][1], "o"),
        (PALETTE["red"][0], PALETTE["red"][1], "D"),
        (PALETTE["teal"][0], PALETTE["teal"][1], "^"),
    ]
    all_M = []
    for curve, (color, fill, marker) in zip(curves, styles):
        L = int(curve["L"])
        x = np.asarray(curve["M_values"], dtype=float)
        all_M.extend(x.tolist())
        matrix = f3_series_matrix(
            data, metadata, experiment="fixed_L_brownian", x_field="M", x_values=x.tolist(), metric="kernel_error", L=L
        )
        f3_plot_seed_summary(panels[2], x, matrix, color=color, fill=fill, label=f"$L={L}$", marker=marker, iqr_alpha=IQR_ALPHA)
    labelled_M = np.asarray(curves[0]["M_values"], dtype=float)
    f3_linear_x(panels[2], labelled_M)
    panels[2].set_xticklabels([f"${int(M)}$" for M in labelled_M], fontsize=7.0)
    margin = 0.04 * (max(all_M) - min(all_M))
    panels[2].set_xlim(min(all_M) - margin, max(all_M) + margin)
    panels[2].set_yscale("log")
    panels[2].set_xlabel("$M$")
    panels[2].set_ylabel("$E_\\gamma$")
    panels[2].legend(loc="upper center", ncol=len(curves), fontsize=6.3, columnspacing=0.55, handlelength=1.0, handletextpad=0.3)
    for label, axis in zip("abc", panels):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    figure.subplots_adjust(left=0.073, right=0.992, bottom=0.22, top=0.92, wspace=0.35)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=dpi, creator="GLE inverse Figure 3 workflow")
    plt.close(figure)
    return outputs
