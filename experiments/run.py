"""Run a paper experiment, or plot its saved results with --plot."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import argparse
from copy import deepcopy
from datetime import datetime

from experiments import figure1_reconstruction as f1
from experiments import figure2_comparison as f2
from experiments import figure3_sampling as f3
from experiments import figure4_measurement as f4
from experiments import figure5_robustness as f5
from experiments import figure6_powerlaw as f6
from experiments import figure7_preparation as f7
from model.fit import load_config, save_yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figure", type=int, choices=range(1, 8))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--input", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    n = args.figure
    study = (f1, f2, f3, f4, f5, f6, f7)[n - 1]
    if args.plot:
        source = args.input or ROOT / "results" / f"Figure{n}"
        out = args.output or source
        study.plot(source / f"figure_{n}_data.npz", source / f"figure_{n}_metadata.yaml", out, stem=f"Figure_{n}", dpi=600)
    else:
        out = args.output or ROOT / "results" / f"Figure{n}_{datetime.now():%Y%m%d_%H%M%S_%f}"
        out.mkdir(parents=True, exist_ok=False)
        config = load_config(args.config) if args.config else deepcopy(study.PARAMS)
        save_yaml(out / "parameters.yaml", config)
        study.run(config, out)
    print(out)


if __name__ == "__main__":
    main()
