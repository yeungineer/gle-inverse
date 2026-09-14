"""Figure 6 command-line entry point."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from paper.code_figure_6.generate_data import generate_all
from paper.code_figure_6.plot_figure_6 import make_figure
from paper.code_figure_6.prepare_data import prepare_figure_data
from paper.utils.figures import FigureWorkflow, run_figure_cli

WORKFLOW = FigureWorkflow(
    number=6,
    default_config=Path(__file__).with_name("figure_6.yaml"),
    generate=generate_all,
    prepare=prepare_figure_data,
    plot=make_figure,
)


def main(argv: list[str] | None = None):
    return run_figure_cli(WORKFLOW, argv)


if __name__ == "__main__":
    main()
