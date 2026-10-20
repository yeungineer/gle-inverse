import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml
from matplotlib import ticker as mticker

MM_PER_INCH = 25.4
DEFAULT_DPI = 600
FULL_WIDTH_MM = 180.0
FIGURE_HEIGHT_MM = 2.0 * 58.0
FULL_FIGURE_SIZE = (FULL_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / MM_PER_INCH)
ROW_FIGURE_SIZE = (FULL_WIDTH_MM / MM_PER_INCH, FIGURE_HEIGHT_MM / 2.0 / MM_PER_INCH)
SINGLE_PANEL_SIZE = (FULL_WIDTH_MM / (2.0 * MM_PER_INCH), FIGURE_HEIGHT_MM / (2.0 * MM_PER_INCH))
PANEL_LABEL_REFERENCE_WIDTH = 7.48
PANEL_LABEL_REFERENCE_SIZE = 9.0
SAVE_PADDING_INCHES = 0.1
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
SEQUENTIAL_HEATMAP_COLORS = ("#F7F9FB", "#CAD8E2", "#8AA6BA", "#52758F", "#203E57")
IQR_ALPHA = 0.07


def load_prepared_data(data_path, metadata_path):
    with np.load(data_path, allow_pickle=False) as stored:
        data = dict(stored)
    return data, yaml.safe_load(Path(metadata_path).read_text())


def apply_publication_style():
    if shutil.which("latex") is None:
        raise RuntimeError("LaTeX is required for figures.")
    mpl.rcParams.update(
        {
            "text.usetex": True,
            "text.latex.preamble": "\\usepackage{amsmath,amssymb,bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"],
            "font.size": 9.0,
            "axes.labelsize": 10.0,
            "axes.titlesize": 9.0,
            "axes.linewidth": 0.65,
            "axes.labelpad": 2.0,
            "axes.formatter.min_exponent": 1,
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
            "legend.framealpha": 0.3,
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
        }
    )


def style_axis(axis):
    axis.grid(False)
    axis.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.65)


def use_fixed_exponent(axis, exponent, *, axis_name="y", tick_count=None):
    target = axis.xaxis if axis_name == "x" else axis.yaxis
    formatter = mticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((int(exponent), int(exponent)))
    formatter.set_useOffset(False)
    target.set_major_formatter(formatter)
    if tick_count is not None:
        target.set_major_locator(mticker.LinearLocator(int(tick_count)))


def style_inset(axis):
    style_axis(axis)
    axis.tick_params(axis="both", which="major", labelsize=6.5, length=2.0, width=0.6, pad=1.2)
    axis.xaxis.label.set_size(7.0)
    axis.yaxis.label.set_size(7.0)
    axis.xaxis.labelpad = 1.0
    axis.yaxis.labelpad = 1.0


def add_panel_label(axis, label, *, x_offset_points=-3.0, y_offset_points=1.5):
    figure_width = float(axis.figure.get_size_inches()[0])
    fontsize = PANEL_LABEL_REFERENCE_SIZE * figure_width / PANEL_LABEL_REFERENCE_WIDTH
    axis.annotate(
        f"\\textbf{{({label})}}", xy=(0.0, 1.0), xycoords="axes fraction", xytext=(float(x_offset_points), float(y_offset_points)),
        textcoords="offset points", ha="right", va="bottom", fontsize=fontsize, clip_on=False, annotation_clip=False, zorder=30,
    )


def save_publication_figure(figure, output_dir, stem, *, dpi=DEFAULT_DPI, creator="GLE inverse paper figure workflow"):
    for ax in figure.axes:
        lo, hi = sorted(ax.get_ylim())
        ticks = ax.get_yticks()
        if np.count_nonzero((ticks >= lo) & (ticks <= hi)) >= 4:
            continue
        if ax.get_yscale() == "log":
            span = np.log10(hi / lo)
            subs = (1, 2, 4, 6) if span < 1.5 else (1, 2, 5) if span < 2 else (1, 5)
            ax.yaxis.set_major_locator(mticker.LogLocator(subs=subs))
            power = int(np.floor(np.log10(hi))) if hi < 0.1 or hi >= 1000 else 0
            formatter = mticker.FuncFormatter(lambda y, _, scale=10.0**power: f"${y / scale:g}$")
            formatter.set_offset_string(rf"$\times10^{{{power}}}$" if power else "")
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_minor_formatter(mticker.NullFormatter())
            ax.yaxis.labelpad = 4.0
        elif isinstance(ax.yaxis.get_major_locator(), mticker.MaxNLocator):
            ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5, min_n_ticks=4))
    dpi = int(dpi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    figure.savefig(pdf_path, format="pdf", bbox_inches="tight", pad_inches=SAVE_PADDING_INCHES, metadata={"Creator": str(creator)})
    figure.savefig(png_path, format="png", dpi=dpi, bbox_inches="tight", pad_inches=SAVE_PADDING_INCHES, pil_kwargs={"compress_level": 6})
    return {"pdf": {"path": str(pdf_path.resolve())}, "png": {"path": str(png_path.resolve())}}
