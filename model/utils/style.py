import logging
import shutil
from pathlib import Path

import matplotlib as mpl
import numpy as np
import yaml
from matplotlib import ticker as mticker
from matplotlib.figure import Figure

MM_PER_INCH = 25.4
DEFAULT_DPI = 600
SINGLE_PANEL_SIZE = (75.0 / MM_PER_INCH, 50.0 / MM_PER_INCH)
SAVE_PADDING_INCHES = 0.02
BLACK, RED, BLUE, GREEN, GRAY = "#000000", "#FF0000", "#0000FF", "#008000", "#808080"
COMPARISON_LINESTYLES = {RED: "--", BLUE: "-.", GREEN: ":"}
NOISE_LABELS = {"brownian": "Brownian", "brownian_levy": "Brownian–Lévy"}
COLORS = {
    "truth": BLACK,
    "guide": GRAY,
    "estimate": RED,
    "estimate_band": RED,
    "brownian": BLUE,
    "levy": RED,
    "empirical": BLUE,
    "uncorrected": GRAY,
    "neutral_light": "#C0C0C0",
    "blue": BLUE,
    "blue_fill": BLUE,
    "gray": GRAY,
    "gray_fill": GRAY,
    "red": RED,
    "red_fill": RED,
    "green": GREEN,
    "green_fill": GREEN,
    "neutral": "#404040",
}
SEQUENTIAL_HEATMAP_COLORS = ("#F7FBFF", "#C6DBEF", "#6BAED6", "#2171B5", "#08306B")
IQR_ALPHA = 0.07


class _FontTimestampFilter(logging.Filter):
    def filter(self, record):
        return not (
            record.levelno == logging.WARNING
            and record.getMessage() in {
                "'created' timestamp seems very low; regarding as unix timestamp",
                "'modified' timestamp seems very low; regarding as unix timestamp",
            }
        )


def load_prepared_data(data_path, metadata_path):
    with np.load(data_path, allow_pickle=False) as stored:
        data = dict(stored)
    return data, yaml.safe_load(Path(metadata_path).read_text())


def paper_panels(rows, *, column_gap, row_gap=0.0):
    width, panel_width, panel_height = 147.13, 37.22, 34.85
    right, bottom, top = width - 2.0, 8.5, 6.4
    left = right - 3 * panel_width - 2 * column_gap
    height = bottom + rows * panel_height + (rows - 1) * row_gap + top
    figure = Figure(figsize=(width / MM_PER_INCH, height / MM_PER_INCH))
    axes = figure.subplots(rows, 3, gridspec_kw={
        "left": left / width, "right": right / width,
        "bottom": bottom / height, "top": 1 - top / height,
        "wspace": column_gap / panel_width, "hspace": row_gap / panel_height,
    })
    return figure, axes


def apply_publication_style(*, use_tex=True):
    if use_tex and shutil.which("latex") is None:
        raise RuntimeError("LaTeX is required for standalone publication plots; use --no-tex for a preview.")
    mpl.rcParams.update(
        {
            "text.usetex": bool(use_tex),
            "text.latex.preamble": "\\usepackage{amsmath,amssymb,bm}",
            "font.family": "serif",
            "font.serif": ["Computer Modern Roman"] if use_tex else ["DejaVu Serif"],
            "font.size": 8.0,
            "mathtext.fontset": "cm",
            "axes.labelsize": 8.0,
            "axes.prop_cycle": mpl.cycler(color=[BLUE, RED, GREEN, BLACK, GRAY]),
            "axes.titlesize": 8.0,
            "axes.linewidth": 0.65,
            "axes.labelpad": 2.0,
            "axes.formatter.min_exponent": 1,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": False,
            "ytick.right": False,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "ytick.major.pad": 2.0,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "xtick.minor.size": 1.5,
            "ytick.minor.size": 1.5,
            "legend.fontsize": 6.0,
            "legend.frameon": False,
            "legend.fancybox": False,
            "legend.framealpha": 0.0,
            "legend.facecolor": "none",
            "legend.edgecolor": "none",
            "legend.shadow": False,
            "legend.handlelength": 1.5,
            "legend.handletextpad": 0.4,
            "legend.columnspacing": 0.7,
            "legend.labelspacing": 0.25,
            "legend.borderaxespad": 0.25,
            "lines.linewidth": 1.0,
            "lines.markersize": 3.0,
            "lines.markeredgewidth": 1.0,
            "patch.linewidth": 0.65,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": None,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def style_axis(axis):
    axis.grid(False)
    axis.tick_params(axis="both", which="both", direction="in", top=False, right=False)
    for spine in axis.spines.values():
        spine.set_visible(True)
        spine.set_color(BLACK)
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
    axis.tick_params(axis="both", which="major", labelsize=5.5, length=2.0, width=0.6, pad=1.2)
    axis.xaxis.label.set_size(6.0)
    axis.yaxis.label.set_size(6.0)
    axis.xaxis.labelpad = 1.0
    axis.yaxis.labelpad = 1.0


def error_label(symbol, value=None, *, percent=False):
    suffix = ""
    if value is not None:
        suffix = f"={100 * value:.2f}\\%" if percent else f"={value:g}"
    return rf"$\mathrm{{Err.}}\,{symbol}{suffix}$"


def heatmap_legend(axis, title):
    return axis.legend(
        loc="lower right", bbox_to_anchor=(1.0, 1.015), ncol=2,
        frameon=False, borderaxespad=0.0, borderpad=0.0, labelspacing=0.1,
        handlelength=1.0, handletextpad=0.3, columnspacing=0.8,
        fontsize=6.0, title=title, title_fontsize=6.5,
    )


def add_panel_label(axis, label, *, x_offset_points=-3.0, y_offset_points=1.5):
    axis.annotate(
        (rf"\textbf{{({label})}}" if mpl.rcParams["text.usetex"] else f"({label})"),
        fontweight="bold", xy=(0.0, 1.0), xycoords="axes fraction",
        xytext=(float(x_offset_points), float(y_offset_points)),
        textcoords="offset points", ha="right", va="bottom", fontsize=9.0,
        clip_on=False, annotation_clip=False, zorder=30,
    )


def save_publication_figure(
    figure, output_dir, stem, *, dpi=DEFAULT_DPI, bbox_inches=None,
    creator="GLE inverse paper figure workflow",
):
    for ax in figure.axes:
        if ax.get_label() == "<colorbar>":
            continue
        for name, target in (("x", ax.xaxis), ("y", ax.yaxis)):
            scale = ax.get_xscale() if name == "x" else ax.get_yscale()
            if scale == "linear" and isinstance(target.get_major_formatter(), mticker.ScalarFormatter):
                formatter = mticker.ScalarFormatter(useMathText=True, useOffset=False)
                formatter.set_powerlimits((-2, 3))
                target.set_major_formatter(formatter)
                if isinstance(target.get_major_locator(), mticker.MaxNLocator):
                    target.set_major_locator(mticker.MaxNLocator(nbins=4, min_n_ticks=3))
            elif scale == "log" and isinstance(target.get_major_formatter(), mticker.LogFormatter):
                lo, hi = sorted(ax.get_xlim() if name == "x" else ax.get_ylim())
                span = np.log10(hi / lo)
                if span <= 3:
                    target.set_major_locator(mticker.LogLocator(subs=(1, 2, 5)))
                    power = int(np.floor(np.log10(hi))) if hi < 0.1 or hi >= 1000 else 0
                    formatter = mticker.FuncFormatter(lambda v, _, scale=10.0**power: f"${v / scale:g}$")
                    formatter.set_offset_string(rf"$\times10^{{{power}}}$" if power else "")
                    target.set_major_formatter(formatter)
                else:
                    target.set_major_locator(mticker.LogLocator(numticks=5))
                    target.set_major_formatter(mticker.LogFormatterMathtext())
                target.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_offset_position("left")
    dpi = int(dpi)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    font_logger = logging.getLogger("fontTools.ttLib.tables._h_e_a_d")
    timestamp_filter = _FontTimestampFilter()
    font_logger.addFilter(timestamp_filter)
    try:
        figure.savefig(
            pdf_path, format="pdf", bbox_inches=bbox_inches, pad_inches=SAVE_PADDING_INCHES,
            metadata={"Creator": str(creator)},
        )
    finally:
        font_logger.removeFilter(timestamp_filter)
    figure.savefig(
        png_path, format="png", dpi=dpi, bbox_inches=bbox_inches, pad_inches=SAVE_PADDING_INCHES,
        pil_kwargs={"compress_level": 6},
    )
    return {"pdf": {"path": str(pdf_path.resolve())}, "png": {"path": str(png_path.resolve())}}
