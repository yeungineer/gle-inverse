"""计算或重绘论文图 1–8。"""

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FIGURES = {
    1: "figure1_reconstruction",
    2: "figure2_basis",
    3: "figure3_comparison",
    4: "figure4_sampling",
    5: "figure5_measurement",
    6: "figure6_robustness",
    7: "figure7_powerlaw",
    8: "figure8_single_trajectory",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("figure", type=int, nargs="+", choices=sorted(FIGURES))
    parser.add_argument("--plot", action="store_true", help="Replot saved data using LaTeX")
    parser.add_argument("--no-tex", action="store_true", help="Preview saved data without LaTeX")
    parser.add_argument("--workers", type=int, default=1, help="Independent CPU experiment processes")
    parser.add_argument("--device", default="cpu", help="PyTorch device for diffusion fitting, e.g. cpu or cuda:0")
    parser.add_argument("--refresh-display", action="store_true", help="With 1 --plot: refresh the display paths")
    args = parser.parse_args()
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    for number in args.figure:
        module = importlib.import_module(f"experiments.{FIGURES[number]}")
        out = ROOT / "results" / f"Figure{number}"
        if not args.plot:
            out.mkdir(parents=True, exist_ok=True)
            module.run(out, workers=args.workers, device=args.device)
        if args.refresh_display and number == 1:
            module.refresh_display_paths(out)
        module.plot(out, use_tex=args.plot and not args.no_tex)
        print(out, flush=True)


if __name__ == "__main__":
    main()
