"""Shared paper-figure data, styling, and command-line workflow."""

from __future__ import annotations

import argparse
import logging
import shutil
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import matplotlib as mpl
import numpy as np
import yaml
from matplotlib import ticker as mticker

from gle_inverse.utils.runtime import create_timestamped_run_directory, load_config

MM_PER_INCH = 25.4
DEFAULT_DPI = 600
FULL_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 2.0 * 58.0
PANEL_LABEL_REFERENCE_WIDTH = 7.48
PANEL_LABEL_REFERENCE_SIZE = 9.0
SAVE_PADDING_INCHES = 0.10

# MATLAB classic line colors; markers and dashes provide a second visual distinction.
COLORS = {
    "truth": "#202020",
    "estimate": "#0072BD",
    "estimate_band": "#0072BD",
    "brownian": "#0072BD",
    "levy": "#D95319",
    "empirical": "#0072BD",
    "uncorrected": "#777777",
    "neutral_light": "#BBBBBB",
    "blue": "#0072BD",
    "blue_fill": "#0072BD",
    "gray": "#6B6B6B",
    "gray_fill": "#6B6B6B",
    "red": "#D95319",
    "red_fill": "#D95319",
    "teal": "#77AC30",
    "teal_fill": "#77AC30",
    "neutral": "#444444",
}

# One sequential scale across error maps; each colorbar retains its own units.
SEQUENTIAL_HEATMAP_COLORS = ("#F7F9FB", "#CAD8E2", "#8AA6BA", "#52758F", "#203E57")
IQR_ALPHA = 0.07


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as stored:
        return {key: np.asarray(stored[key]) for key in stored.files}


def load_prepared_data(
    data_path: str | Path, metadata_path: str | Path
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    return load_npz(data_path), load_yaml(metadata_path)


def float_column(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray(
        [np.nan if row.get(key) is None or row.get(key) == "" else float(row[key]) for row in rows], dtype=float
    )


def string_column(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray(["" if row.get(key) is None else str(row[key]) for row in rows], dtype=str)


def resolve_code_path(config_path: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (config_path.parent / path).resolve()


@dataclass(frozen=True)
class FigureWorkflow:
    number: int
    default_config: Path
    generate: Callable
    prepare: Callable
    plot: Callable


def _parser(workflow: FigureWorkflow) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=f"Build paper Figure {workflow.number}.")
    parser.add_argument("--mode", choices=("full", "plot"), default="full")
    parser.add_argument("--config", default=str(workflow.default_config))
    parser.add_argument("--run-dir")
    return parser


def _data_paths(run_dir: Path, number: int) -> tuple[Path, Path]:
    return run_dir / f"figure_{number}_data.npz", run_dir / f"figure_{number}_metadata.yaml"


def _full_run(workflow: FigureWorkflow, config: dict, config_path: Path):
    run_dir = create_timestamped_run_directory(config_path.parent / "output", f"figure_{workflow.number}")
    with TemporaryDirectory(prefix=f"figure_{workflow.number}_") as directory:
        effective = deepcopy(config)
        effective["paths"] = {"data": str(Path(directory)), "output": str(run_dir)}
        workflow.generate(effective, config_path)
        products = workflow.prepare(effective, config_path)
        data_path, metadata_path = _data_paths(run_dir, workflow.number)
        shutil.copy2(products[0], data_path)
        shutil.copy2(products[1], metadata_path)
    return run_dir, data_path, metadata_path


def _plot(workflow: FigureWorkflow, run_dir: str | None):
    if run_dir is None:
        raise ValueError("--run-dir is required in plot mode.")
    root = Path(run_dir).expanduser().resolve()
    data_path, metadata_path = _data_paths(root, workflow.number)
    return root, data_path, metadata_path


def run_figure_cli(workflow: FigureWorkflow, argv: list[str] | None = None):
    args = _parser(workflow).parse_args(argv)
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (workflow.default_config.parent / config_path).resolve()
    config = load_config(config_path)
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    run = _full_run(workflow, config, config_path) if args.mode == "full" else _plot(workflow, args.run_dir)
    run_dir, data_path, metadata_path = run
    settings = config["figure"]
    result = workflow.plot(data_path, metadata_path, run_dir, stem=str(settings["stem"]), dpi=int(settings["dpi"]))
    result.update({"run_root": str(run_dir), "data_source": str(run_dir)})
    return result


def _ensure_latex() -> str:
    executable = shutil.which("latex")
    if executable is None:
        raise RuntimeError("LaTeX is required for paper figures.")
    return executable


def apply_publication_style() -> None:
    """Serif typography and restrained strokes, sized for the 150 mm paper text block."""
    _ensure_latex()
    mpl.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath,amssymb,bm}",
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "font.size": 9.0,
        "axes.labelsize": 10.0,
        "axes.titlesize": 9.0,
        "axes.linewidth": 0.65,
        "axes.labelpad": 2.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": False,
        "ytick.right": False,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "xtick.major.width": 0.65,
        "ytick.major.width": 0.65,
        "xtick.minor.size": 1.5,
        "ytick.minor.size": 1.5,
        "legend.fontsize": 7.5,
        "legend.frameon": True,
        "legend.fancybox": False,
        "legend.framealpha": 0.30,
        "legend.facecolor": "white",
        "legend.edgecolor": "none",
        "legend.shadow": False,
        "legend.handlelength": 2.0,
        "legend.handletextpad": 0.4,
        "legend.columnspacing": 0.7,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.5,
        "patch.linewidth": 0.65,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_axis(axis: mpl.axes.Axes) -> None:
    """Match the production paper style: boxed axes and inward ticks."""
    axis.grid(False)
    axis.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.65)


def use_fixed_exponent(
    axis: mpl.axes.Axes,
    exponent: int,
    *,
    axis_name: str = "y",
    tick_count: int | None = None,
) -> None:
    """Show one shared scientific multiplier beside an axis's tick labels."""
    if axis_name not in {"x", "y"}:
        raise ValueError("axis_name must be either 'x' or 'y'.")
    target = axis.xaxis if axis_name == "x" else axis.yaxis
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((int(exponent), int(exponent)))
    formatter.set_useOffset(False)
    target.set_major_formatter(formatter)
    if tick_count is not None:
        target.set_major_locator(mticker.LinearLocator(int(tick_count)))


def style_inset(axis: mpl.axes.Axes) -> None:
    """Style a compact inset without making it compete with the main panel."""
    style_axis(axis)
    axis.tick_params(axis="both", which="major", labelsize=6.5, length=2.0, width=0.6, pad=1.2)
    axis.xaxis.label.set_size(7.0)
    axis.yaxis.label.set_size(7.0)
    axis.xaxis.labelpad = 1.0
    axis.yaxis.labelpad = 1.0


def add_panel_label(
    axis: mpl.axes.Axes, label: str, *, x_offset_points: float = -3.0, y_offset_points: float = 1.5
) -> None:
    """Match the production labels, with optional offsets for top axes."""
    figure_width = float(axis.figure.get_size_inches()[0])
    fontsize = PANEL_LABEL_REFERENCE_SIZE * figure_width / PANEL_LABEL_REFERENCE_WIDTH
    axis.annotate(
        rf"\textbf{{({label})}}",
        xy=(0.0, 1.0),
        xycoords="axes fraction",
        xytext=(float(x_offset_points), float(y_offset_points)),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=fontsize,
        clip_on=False,
        annotation_clip=False,
        zorder=30,
    )


def save_publication_figure(
    figure: mpl.figure.Figure,
    output_dir: str | Path,
    stem: str,
    *,
    dpi: int = DEFAULT_DPI,
    creator: str = "GLE inverse paper figure workflow",
) -> dict[str, object]:
    dpi = int(dpi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"

    figure.savefig(
        pdf_path, format="pdf", bbox_inches="tight", pad_inches=SAVE_PADDING_INCHES, metadata={"Creator": str(creator)}
    )
    figure.savefig(
        png_path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=SAVE_PADDING_INCHES,
        pil_kwargs={"compress_level": 6},
    )

    return {
        "pdf": {"path": str(pdf_path.resolve())},
        "png": {"path": str(png_path.resolve())},
    }


def full_width_figure_size() -> tuple[float, float]:
    """Return the shared full-width two-row figure size in inches."""
    return (FULL_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)


def full_width_single_row_figure_size() -> tuple[float, float]:
    """Return the shared full-width one-row figure size in inches."""
    return (FULL_WIDTH_MM / MM_PER_INCH, (FIGURE_HEIGHT_MM / 2.0) / MM_PER_INCH)
