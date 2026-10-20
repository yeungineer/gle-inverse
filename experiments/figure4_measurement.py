"""Figure 4: measurement."""

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.lines import Line2D

from model.fit import configured_case, float_column, prepare, save_figure_data, solve, string_column
from model.inverse import (
    empirical_conditional_moments, estimate_symmetric_stable, remove_symmetric_jump_second_moment, truncated_conditional_moments,
)
from model.style import (
    COLORS, IQR_ALPHA, ROW_FIGURE_SIZE, add_panel_label, apply_publication_style, load_prepared_data, save_publication_figure, style_axis,
    use_fixed_exponent,
)

ROUTE_SEED = {"endpoint": 401, "start_endpoint": 402, "full_path": 403}


METRIC_FIELDS = (
    "sigma_error",
    "kernel_error",
    "alpha_error",
    "kappa_log_error",
    "alpha_est",
    "kappa_est",
    "epsilon",
    "tail_count",
    "selected_lambda",
)


BOOLEAN_DIAGNOSTIC_FIELDS = ("regularization_selected_at_boundary",)


STRING_DIAGNOSTIC_FIELDS = ("regularization_method",)


PARAMS = {
    "figure": {"stem": "Figure_4", "dpi": 600, "seeds": [10, 11, 12, 13, 14, 15], "iqr_alpha": 0.07},
    "experiment": {
        "km_samples": "euler_oracle",
        "diffusion_types": {"brownian": "constant", "brownian_levy": "cosine"},
        "dt": 0.01,
        "M": 2000,
        "L": 3000,
        "N": 12,
        "noise_levels": [0.0, 0.005, 0.01, 0.02, 0.05, 0.1],
        "routes": ["endpoint", "start_endpoint", "full_path"],
    },
}


def noise_points(config):
    return [
        {
            "experiment": "measurement_noise",
            "case": case,
            "route": route,
            "eta_ratio": float(eta),
            "point_id": f"{case}__{route}__eta{float(eta):.12g}".replace(".", "p"),
        }
        for case in ("brownian", "brownian_levy")
        for route in config["experiment"]["routes"]
        for eta in config["experiment"]["noise_levels"]
    ]


def _apply_endpoint_noise(problem, route, sigma, seed):
    samples = problem["km_samples"]
    if route == "full_path":
        problem["phase1_km_samples"] = samples
        return
    delta = np.asarray(samples["delta_v"])
    rng = np.random.default_rng(np.random.SeedSequence([seed, ROUTE_SEED[route]]))
    start_noise = sigma * rng.standard_normal(delta.shape) if route == "start_endpoint" else 0.0
    end_noise = sigma * rng.standard_normal(delta.shape)
    observed = delta + end_noise - start_noise
    branch_start = np.asarray(samples["branch_start"]) + start_noise
    d1, d2 = empirical_conditional_moments(observed, float(samples["dt"]))
    samples.update(
        {
            "delta_v": observed,
            "branch_start": branch_start,
            "v_next": branch_start + observed,
            "d1": d1,
            "d2": d2,
            "d1_raw": d1,
            "d2_raw": d2,
        }
    )
    if samples["model"] == "brownian_levy":
        epsilon = float(problem["config"]["phase1"]["estimation"]["levy_epsilon"])
        levy = estimate_symmetric_stable(observed, float(samples["dt"]), epsilon=epsilon)
        d1, d2, retained = truncated_conditional_moments(observed, float(samples["dt"]), levy["epsilon"])
        samples.update(
            {
                "d1": d1,
                "d2": remove_symmetric_jump_second_moment(d2, levy["small_jump_second_moment"]),
                "d1_truncated": d1,
                "d2_truncated": d2,
                "retained_fraction": np.mean(retained, axis=1),
                "levy": levy,
            }
        )
    problem["phase1_km_samples"] = samples


def noise_point(config, spec, seed):
    settings = config["experiment"]
    case = str(spec["case"])
    route = str(spec["route"])
    sigma = float(spec["eta_ratio"])
    overrides = {
        "generation.simulation.seed": seed,
        "generation.simulation.dt": float(settings["dt"]),
        "generation.simulation.num_steps": int(settings["M"]),
        "generation.simulation.L": int(settings["L"]),
        "generation.km_samples": str(settings["km_samples"]),
        "noise.diffusion": str(settings["diffusion_types"][case]),
        "phase1.observation.sigma_obs": sigma if route == "full_path" else 0.0,
        "phase2.observation.sigma_obs": sigma if route == "full_path" else 0.0,
        "phase2.estimation.num_basis": int(settings["N"]),
        "phase2.estimation.t_max": min(30.0, settings["L"] * settings["dt"]),
    }
    task = configured_case(case, overrides)
    problem = prepare(task, conditional_sample_seed=None)
    _apply_endpoint_noise(problem, route, sigma, seed)
    _, metrics = solve(problem)
    row = {
        **spec,
        "seed": seed,
        "eta_absolute": sigma,
        "clean_state_scale": 1.0,
        "clean_empirical_std": float(np.std(problem["v_true"])),
        "success": True,
        "sigma_error": float(metrics["sigma"]["relative_l2_error"]),
        "kernel_error": float(metrics["kernel"]["relative_l2_error"]),
        "selected_lambda": float(metrics["regularization"]["selected_lambda"]),
        "regularization_method": str(metrics["regularization"]["mode"]),
        "regularization_selected_at_boundary": bool(metrics["regularization"]["selected_at_search_boundary"]),
    }
    if case == "brownian_levy":
        levy = metrics["levy"]
        row.update(
            {
                "alpha_error": abs(levy["alpha_est"] - levy["alpha_true"]),
                "kappa_log_error": abs(np.log(levy["kappa_est"] / levy["kappa_true"])),
                "alpha_est": float(levy["alpha_est"]),
                "kappa_est": float(levy["kappa_est"]),
                "epsilon": float(problem["km_samples"]["levy"]["epsilon"]),
                "tail_count": int(problem["km_samples"]["levy"]["tail_count"]),
            }
        )
    return row


def run(config, out):
    points = noise_points(config)
    rows = []
    for seed in map(int, config["figure"]["seeds"]):
        for spec in points:
            rows.append(noise_point(config, spec, seed))
    arrays, metadata = pack_noise(config, rows)
    save_figure_data(out, 4, arrays, metadata)
    return rows


def pack_noise(config, rows):
    rows.sort(key=lambda row: (row["case"], row["route"], row["eta_ratio"], row["seed"]))
    arrays = {
        "point_id": string_column(rows, "point_id"),
        "experiment": string_column(rows, "experiment"),
        "case": string_column(rows, "case"),
        "route": string_column(rows, "route"),
        "seed": np.asarray([row["seed"] for row in rows], dtype=np.int64),
        "eta_ratio": float_column(rows, "eta_ratio"),
        "eta_absolute": float_column(rows, "eta_absolute"),
        "clean_state_scale": float_column(rows, "clean_state_scale"),
        "clean_empirical_std": float_column(rows, "clean_empirical_std"),
        "success": np.ones(len(rows), dtype=bool),
    }
    for field in METRIC_FIELDS:
        arrays[field] = float_column(rows, field)
    for field in BOOLEAN_DIAGNOSTIC_FIELDS:
        arrays[field] = np.asarray([bool(row.get(field, False)) for row in rows])
    for field in STRING_DIAGNOSTIC_FIELDS:
        arrays[field] = string_column(rows, field)
    seeds = list(map(int, config["figure"]["seeds"]))
    metadata = {
        "seeds": seeds,
        "required_successes": len(seeds),
        "figure_style": {"iqr_alpha": float(config["figure"]["iqr_alpha"])},
        "scientific_configuration": config["experiment"],
    }
    return (arrays, metadata)


ROUTE_STYLE = {
    "endpoint": {"line": COLORS["blue"], "fill": COLORS["blue_fill"], "label": "Final point"},
    "start_endpoint": {"line": COLORS["red"], "fill": COLORS["red_fill"], "label": "Initial and final points"},
    "full_path": {"line": COLORS["teal"], "fill": COLORS["teal_fill"], "label": "Full trajectory"},
}


CASE_STYLE = {
    "brownian": {"linestyle": "-", "marker": "o", "label": "Brownian"},
    "brownian_levy": {"linestyle": (0, (3.0, 1.8)), "marker": "s", "label": "Brownian--L\\'evy"},
}


METRIC_STYLE = {
    "alpha_error": {"linestyle": "-", "marker": "o", "label": "$|\\widehat\\alpha-\\alpha|$"},
    "kappa_log_error": {"linestyle": (0, (3.0, 1.8)), "marker": "s", "label": "$|\\log(\\widehat\\kappa/\\kappa)|$"},
}


def f4_matrix(data, seeds, levels, *, case, route, metric):
    matrix = np.full((len(seeds), len(levels)), np.nan, dtype=float)
    mask = (data["case"] == case) & (data["route"] == route) & np.isfinite(data[metric])
    for row in np.flatnonzero(mask):
        seed = int(data["seed"][row])
        if seed not in seeds:
            continue
        eta = float(data["eta_ratio"][row])
        matches = np.flatnonzero(np.isclose(levels, eta, rtol=0.0, atol=1e-14))
        if len(matches) != 1:
            raise RuntimeError(f"Could not place {case}/{route}/{metric}/eta={eta}.")
        i = seeds.index(seed)
        j = int(matches[0])
        if np.isfinite(matrix[i, j]):
            raise RuntimeError("Duplicate Figure 4 prepared value.")
        matrix[i, j] = float(data[metric][row])
    return matrix


def f4_plot_summary(axis, x, matrix, *, line, fill, linestyle, marker):
    if not np.any(np.isfinite(matrix)):
        return True
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        q25 = np.nanquantile(matrix, 0.25, axis=0)
        median = np.nanmedian(matrix, axis=0)
        q75 = np.nanquantile(matrix, 0.75, axis=0)
    axis.fill_between(x, q25, q75, color=fill, alpha=IQR_ALPHA, linewidth=0.0, zorder=2)
    axis.plot(
        x, median, color=line, linestyle=linestyle, linewidth=1.25, marker=marker, markerfacecolor="white", markeredgecolor=line,
        markeredgewidth=0.72, zorder=4,
    )
    counts = np.sum(np.isfinite(matrix), axis=0)
    incomplete = counts < matrix.shape[0]
    mark = incomplete & np.isfinite(median)
    if np.any(mark):
        axis.scatter(x[mark], median[mark], marker="x", s=13.0, linewidths=0.75, color=COLORS["neutral"], zorder=6)
    return bool(np.any(incomplete))


def f4_format_axis(axis, x, ylabel):
    positive = np.sort(np.unique(x[x > 0.0]))
    if positive.size == 0:
        raise ValueError("Figure 4 requires at least one positive noise level.")
    linear_threshold = float(positive[0])
    axis.set_xscale("symlog", base=10, linthresh=linear_threshold, linscale=0.8)
    axis.set_xticks(x)
    use_fixed_exponent(axis, -2, axis_name="x")
    axis.tick_params(axis="x", which="major", labelsize=6.9, pad=2.0)
    xmax = float(np.max(x))
    axis.set_xlim(-0.22 * linear_threshold, 1.18 * xmax)
    axis.set_yscale("log")
    axis.set_yticks([0.01, 0.1, 1.0])
    axis.yaxis.set_minor_formatter(mticker.NullFormatter())
    axis.set_xlabel("$\\sigma_{\\rm obs}$")
    axis.set_ylabel(ylabel)


def f4_route_handles(routes):
    return [Line2D([0], [0], color=ROUTE_STYLE[route]["line"], linewidth=1.4, label=ROUTE_STYLE[route]["label"]) for route in routes]


def f4_incomplete_handle(required_successes):
    return Line2D(
        [0], [0], color=COLORS["neutral"], marker="x", linestyle="none", markersize=4.0,
        label=f"$n_{{\\rm success}}<{int(required_successes)}$",
    )


def f4_panel_case_metric(axis, data, metadata, *, metric, ylabel):
    seeds = list(map(int, metadata["seeds"]))
    levels = np.asarray(metadata["scientific_configuration"]["noise_levels"], dtype=float)
    routes = tuple(metadata["scientific_configuration"]["routes"])
    x = levels
    incomplete = False
    for route in routes:
        for case in ("brownian", "brownian_levy"):
            matrix = f4_matrix(data, seeds, levels, case=case, route=route, metric=metric)
            incomplete |= f4_plot_summary(
                axis, x, matrix, line=ROUTE_STYLE[route]["line"], fill=ROUTE_STYLE[route]["fill"], linestyle=CASE_STYLE[case]["linestyle"],
                marker=CASE_STYLE[case]["marker"],
            )
    f4_format_axis(axis, x, ylabel)
    case_handles = [
        Line2D(
            [0], [0], color=COLORS["neutral"], linestyle=CASE_STYLE[case]["linestyle"], marker=CASE_STYLE[case]["marker"],
            markerfacecolor="white", markersize=3.5, label=CASE_STYLE[case]["label"],
        )
        for case in ("brownian", "brownian_levy")
    ]
    if incomplete:
        case_handles.append(f4_incomplete_handle(metadata["required_successes"]))
    axis.legend(handles=case_handles, loc="upper left", fontsize=7.0)


def plot(data_path, metadata_path, output_dir, *, stem="Figure_4", dpi=600):
    apply_publication_style()
    data, metadata = load_prepared_data(data_path, metadata_path)
    figure, axes = plt.subplots(1, 3, figsize=ROW_FIGURE_SIZE, squeeze=False)
    f4_panel_case_metric(axes[0, 0], data, metadata, metric="sigma_error", ylabel="$E_\\sigma$")
    f4_panel_case_metric(axes[0, 1], data, metadata, metric="kernel_error", ylabel="$E_\\gamma$")
    seeds = list(map(int, metadata["seeds"]))
    levels = np.asarray(metadata["scientific_configuration"]["noise_levels"], dtype=float)
    routes = tuple(metadata["scientific_configuration"]["routes"])
    x = levels
    incomplete = False
    for route in routes:
        for metric in ("alpha_error", "kappa_log_error"):
            matrix = f4_matrix(data, seeds, levels, case="brownian_levy", route=route, metric=metric)
            incomplete |= f4_plot_summary(
                axes[0, 2], x, matrix, line=ROUTE_STYLE[route]["line"], fill=ROUTE_STYLE[route]["fill"],
                linestyle=METRIC_STYLE[metric]["linestyle"], marker=METRIC_STYLE[metric]["marker"],
            )
    f4_format_axis(axes[0, 2], x, "L\\'evy parameter error")
    metric_handles = [
        Line2D(
            [0], [0], color=COLORS["neutral"], linestyle=METRIC_STYLE[metric]["linestyle"], marker=METRIC_STYLE[metric]["marker"],
            markerfacecolor="white", markersize=3.5, label=METRIC_STYLE[metric]["label"],
        )
        for metric in ("alpha_error", "kappa_log_error")
    ]
    if incomplete:
        metric_handles.append(f4_incomplete_handle(metadata["required_successes"]))
    axes[0, 2].legend(handles=metric_handles, loc="upper left", fontsize=7.0)
    for label, axis in zip("abc", axes.ravel()):
        style_axis(axis)
        add_panel_label(axis, label, y_offset_points=5.5)
    figure.subplots_adjust(left=0.07, right=0.985, bottom=0.205, top=0.92, wspace=0.34)
    routes = metadata["scientific_configuration"]["routes"]
    for axis in axes.ravel():
        lower, upper = axis.get_ylim()
        axis.set_ylim(lower / 5.0, upper)
        axis.add_artist(axis.get_legend())
        axis.legend(handles=f4_route_handles(routes), loc="lower right", fontsize=6.5)
    outputs = save_publication_figure(figure, output_dir, stem, dpi=dpi, creator="GLE inverse Figure 4 workflow")
    plt.close(figure)
    return outputs
