import argparse
from datetime import datetime
from pathlib import Path

from model.utils.io import ROOT, load_config
from model.utils.plotting import plot_run
from model.workflow import run
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--config", type=Path, default=ROOT / "brownian.yaml")
parser.add_argument("--output", type=Path)
parser.add_argument("--plot", type=Path)
parser.add_argument("--device", default="cpu", help="PyTorch device for diffusion fitting")
parser.add_argument("--threads", type=int, default=1)
args = parser.parse_args()
torch.set_num_threads(args.threads)

if args.plot:
    plot_run(args.plot)
else:
    out = args.output or ROOT / "results" / f"{args.config.stem}_{datetime.now():%Y%m%d_%H%M%S}"
    config = load_config(args.config)
    config["compute"]["device"] = args.device
    run(config, out)
    plot_run(out, use_tex=False)
    print(out)
