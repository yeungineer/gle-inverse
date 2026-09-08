"""Conditional-data generators for Kramers--Moyal inference."""

from .common import (
    KM_GENERATION_EULER_ORACLE,
    KM_GENERATION_FINE_BRANCH,
    KM_GENERATION_METHODS,
    KM_GENERATION_REPLAY_BRANCH,
    parse_km_generation_config,
)
from .km_euler_oracle import compute_euler_oracle_km_samples
from .km_fine_branch import compute_fine_branch_km_samples
from .km_replay_branch import compute_replay_branch_km_samples


def compute_km_samples(
    config,
    t_observed,
    v_observed,
    memory_observed,
    observation_config,
    sample_size,
    trajectory_index=0,
    random_seed=None,
    levy_estimation=None,
    levy_estimate=None,
    generation_config=None,
    physical_v_observed=None,
    physical_memory_observed=None,
):
    """Dispatch explicitly to one conditional-data construction."""
    method = KM_GENERATION_EULER_ORACLE if generation_config is None else str(generation_config).lower()
    common = {
        "trajectory_index": trajectory_index,
        "random_seed": random_seed,
        "levy_estimation": levy_estimation,
        "levy_estimate": levy_estimate,
    }
    if method == KM_GENERATION_EULER_ORACLE:
        return compute_euler_oracle_km_samples(
            config,
            t_observed,
            v_observed,
            memory_observed,
            observation_config,
            sample_size,
            physical_v_observed=physical_v_observed,
            physical_memory_observed=physical_memory_observed,
            **common,
        )
    if method == KM_GENERATION_FINE_BRANCH:
        return compute_fine_branch_km_samples(
            config,
            t_observed,
            v_observed,
            memory_observed,
            observation_config,
            sample_size,
            generation_config,
            physical_v_observed=physical_v_observed,
            **common,
        )
    if method == KM_GENERATION_REPLAY_BRANCH:
        return compute_replay_branch_km_samples(
            config,
            t_observed,
            v_observed,
            memory_observed,
            observation_config,
            sample_size,
            generation_config,
            **common,
        )
    raise RuntimeError(f"Unreachable KM generation method: {method}.")


__all__ = [
    "KM_GENERATION_EULER_ORACLE",
    "KM_GENERATION_FINE_BRANCH",
    "KM_GENERATION_REPLAY_BRANCH",
    "KM_GENERATION_METHODS",
    "parse_km_generation_config",
    "compute_euler_oracle_km_samples",
    "compute_fine_branch_km_samples",
    "compute_replay_branch_km_samples",
    "compute_km_samples",
]
