import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gle_inverse.utils.artifacts import ArtifactStore
from gle_inverse.utils.results import save_kramers_moyal_result
from gle_inverse.utils.runtime import load_config, setup_env, setup_logging
from gle_inverse.workflows.kramers_moyal import run_experiment

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "kramers_moyal_brownian.yaml"


def main(config_path=None, *, make_plots=True, output_dir=None):
    if config_path is None:
        parser = argparse.ArgumentParser(description="Run Kramers--Moyal GLE inversion.")
        parser.add_argument("--config", default=None)
        arguments = parser.parse_args()
        config_path = arguments.config

    if config_path is None:
        config_path = DEFAULT_CONFIG

    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    source_config = load_config(config_path)
    experiment_name = config_path.stem
    if output_dir is None:
        config, resolved_output_dir = setup_env(experiment_name, config_path=config_path)
    else:
        config = source_config
        resolved_output_dir = Path(output_dir).expanduser().resolve()
        resolved_output_dir.mkdir(parents=True, exist_ok=False)
        setup_logging(resolved_output_dir, log_name="run.log")
        logging.info("[RUN] Experiment: %s", experiment_name)
        logging.info("[RUN] Run ID: %s", resolved_output_dir.name)
        logging.info("[RUN] Output directory: %s", resolved_output_dir)
        ArtifactStore(resolved_output_dir).save_yaml("config_snapshot.yaml", config)
    resolved_output_dir, arrays, metrics = run_experiment(
        config,
        experiment_name=experiment_name,
        output_dir=resolved_output_dir,
        make_plots=make_plots,
    )
    save_kramers_moyal_result(resolved_output_dir, config, arrays, metrics)
    return resolved_output_dir, arrays, metrics


if __name__ == "__main__":
    main()
