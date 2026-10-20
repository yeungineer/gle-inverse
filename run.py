"""Basic GLE simulation and two-stage inversion, independent of the paper studies."""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from model.fit import ROOT, load_config, prepare, save_yaml, solve


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", type=Path, metavar="RUN")
    parser.add_argument("--config", type=Path, default=ROOT / "brownian.yaml")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.plot:
        from model.diagnostics import plot_run

        plot_run(args.plot)
        print(args.plot)
        return
    config = load_config(args.config)
    out = args.output or ROOT / "results" / f"{args.config.stem}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    out.mkdir(parents=True, exist_ok=False)
    save_yaml(out / "parameters.yaml", config)
    arrays, metrics = solve(prepare(config))
    np.savez_compressed(out / "estimates.npz", **arrays)
    save_yaml(out / "metrics.yaml", metrics)
    print(out)


if __name__ == "__main__":
    main()
