import numpy as np

from gle_inverse.utils.artifacts import save_run_artifacts


def result_arrays(
    t_raw,
    v_true,
    noise_increment,
    t_obs,
    v_obs,
    km,
    sigma_est,
    sigma_ref,
    d1_reference,
    d2_reference,
    d1_fit,
    kernel_solver,
    kernel_est,
    kernel_raw,
    kernel_reference_index,
    noise_components=None,
):
    branch_count = np.asarray(km["v_next"]).shape[1]
    arrays = {
        "t": t_raw,
        "v_true": v_true,
        "noise_increment": noise_increment,
        "t_obs": t_obs,
        "v_obs": v_obs,
        "branch_state": np.asarray(km["x"][:1]),
        "branch_endpoints": np.asarray(km["v_next"][0, :branch_count]),
        "x_ref": km["x"],
        "reference_index": km["idx"],
        "kernel_reference_index": kernel_reference_index,
        "force_ref": km["f"],
        "memory_ref": km["mem"],
        "d1_reference": d1_reference,
        "d2_finite_step_reference": d2_reference,
        "d1_empirical": km["d1"],
        "d2_diffusion_target": km["d2"],
        "sigma_est": sigma_est,
        "sigma_true": sigma_ref,
        "d1_fit": d1_fit,
        "kernel_coefficients": kernel_solver.coefficients,
        "kernel_est": kernel_est,
        "kernel_true": kernel_raw,
        "selected_lambda": np.asarray([kernel_solver.selected_lambda]),
    }
    if noise_components is not None:
        arrays.update(
            {
                "gaussian_increment": np.asarray(noise_components["gaussian_increment"]),
                "levy_increment": np.asarray(noise_components["levy_increment"]),
            }
        )
    arrays["regularization_singular_values"] = kernel_solver.singular_values
    if "levy" in km:
        arrays.update({key: km[key] for key in ("d1_raw", "d2_raw", "d1_truncated", "d2_truncated")})
        arrays.update(
            {
                "levy_bin_edges": km["levy"]["bin_edges"],
                "levy_counts": km["levy"]["counts"],
                "levy_positive_counts": km["levy"]["positive_counts"],
                "levy_negative_counts": km["levy"]["negative_counts"],
                "levy_empirical_rates": km["levy"]["empirical_rates"],
                "levy_fitted_rates": km["levy"]["fitted_rates"],
            }
        )
    return arrays


def result_metrics(km, sigma_errors, d1_metrics, kernel_metrics, kernel_solver, levy_metrics):
    metrics = {
        "model": km["model"],
        "sigma": {"absolute_l2_error": sigma_errors[0], "relative_l2_error": sigma_errors[1]},
        "first_moment": dict(
            zip(
                (
                    "empirical_absolute_l2_error",
                    "empirical_relative_l2_error",
                    "fitted_absolute_l2_error",
                    "fitted_relative_l2_error",
                ),
                d1_metrics,
            )
        ),
        "kernel": dict(zip(("absolute_l2_error", "relative_l2_error"), kernel_metrics)),
        "regularization": {
            "mode": kernel_solver.mode,
            "selected_lambda": kernel_solver.selected_lambda,
            "selected_at_search_boundary": bool(kernel_solver.gcv["selected_at_boundary"]),
        },
        "conditional_samples": {
            "M": int(np.asarray(km["delta_v"]).shape[1]),
            "generation_method": km.get("generation_method", "euler_oracle"),
            "generation_substeps": int(km.get("generation_substeps", 1)),
            "generation_internal_dt": float(km.get("generation_internal_dt", km["dt"])),
        },
    }
    if levy_metrics:
        metrics["levy"] = {key: value for key, value in levy_metrics.items() if key != "true_rates"}
        metrics["levy"].update(
            {
                "epsilon": km["levy"]["epsilon"],
                "q": km["levy"]["q"],
                "K": km["levy"]["K"],
                "small_jump_second_moment": km["levy"]["small_jump_second_moment"],
                "kappa_bins": km["levy"]["kappa_bins"],
                "tail_count": km["levy"].get("tail_count"),
                "alpha_standard_error": km["levy"].get("alpha_standard_error"),
                "log_kappa_standard_error": km["levy"].get("log_kappa_standard_error"),
                "counts": km["levy"]["counts"],
                "positive_counts": km["levy"]["positive_counts"],
                "negative_counts": km["levy"]["negative_counts"],
            }
        )
    return metrics


def save_kramers_moyal_result(output_dir, config, arrays, metrics):
    return save_run_artifacts(output_dir, config, arrays, metrics, name=f"estimates__{metrics['model']}")
