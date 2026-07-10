"""Shared plotting style for publication figures."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, Formatter, NullFormatter
import numpy as np
import seaborn as sns

SINGLE_COLUMN_WIDTH_IN = 6
FULL_TWO_COLUMN_WIDTH_IN = 12

PAPER_FONT_SIZE_PT = 14
LEGEND_FONT_SIZE_PT = 9
LINE_WIDTH_PT = 1.25
REFERENCE_LINE_WIDTH_PT = 0.9
SCATTER_SIZE_PT2 = 16
SCATTER_ALPHA = 0.35
DPI = 300
AXIS_TICK_DECIMALS = 2
AXIS_SCALE_LOW_THRESHOLD = 1e-2
AXIS_SCALE_HIGH_THRESHOLD = 1e3

SINGLE_PANEL_ASPECT = 0.6
TWO_PANEL_STACK_ASPECT = 0.55
THREE_PANEL_STACK_ASPECT = 0.62
SQUARE_PANEL_ASPECT = 0.95

COLORS = {
    "volume": "#0072B2",
    "temperature": "#E69F00",
    "storage": "#D55E00",
    "volume_tendency": "#0072B2",
    "temperature_tendency": "#009E73",
    "advection": "#111111",
    "adiabatic": "#009E73",
    "diabatic": "#D55E00",
    "residual": "#111111",
    "mass": "#0072B2",
    "benchmark": "#4D4D4D",
    "calculated": "#111111",
    "heat_flux": "#D55E00",
    "zero": "#333333",
    "grid": "#E7E7E7",
}

FACE_COLORS = {
    "north": "#0072B2",
    "south": "#56B4E9",
    "east": "#6A5ACD",
    "west": "#009E9E",
    "top": "#E69F00",
    "bottom": "#D55E00",
}

STACKED_FIGURE_LAYOUTS = {
    ("single", 2): {
        "left": 0.18,
        "right": 0.96,
        "bottom": 0.16,
        "top": 0.92,
        "hspace": 0.32,
    },
    ("single", 3): {
        "left": 0.16,
        "right": 0.96,
        "bottom": 0.12,
        "top": 0.94,
        "hspace": 0.32,
    },
    ("full", 2): {
        "left": 0.08,
        "right": 0.96,
        "bottom": 0.13,
        "top": 0.93,
        "hspace": 0.28,
    },
    ("full", 3): {
        "left": 0.08,
        "right": 0.96,
        "bottom": 0.10,
        "top": 0.94,
        "hspace": 0.34,
    },
}

ONE_TO_ONE_FIGURE_LAYOUT = {
    "left": 0.15,
    "right": 0.97,
    "bottom": 0.18,
    "top": 0.88,
}


def apply_theme() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update(
        {
            "font.size": PAPER_FONT_SIZE_PT,
            "axes.labelsize": PAPER_FONT_SIZE_PT,
            "axes.titlesize": PAPER_FONT_SIZE_PT,
            "xtick.labelsize": PAPER_FONT_SIZE_PT,
            "ytick.labelsize": PAPER_FONT_SIZE_PT,
            "legend.fontsize": LEGEND_FONT_SIZE_PT,
            "legend.title_fontsize": LEGEND_FONT_SIZE_PT,
            "figure.titlesize": PAPER_FONT_SIZE_PT,
            "lines.linewidth": LINE_WIDTH_PT,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.minor.width": 0.6,
            "ytick.minor.width": 0.6,
            "savefig.dpi": DPI,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def publication_figsize(
    width: str = "single",
    aspect: float = SINGLE_PANEL_ASPECT,
) -> tuple[int, float]:
    widths = {
        "single": SINGLE_COLUMN_WIDTH_IN,
        "full": FULL_TWO_COLUMN_WIDTH_IN,
    }
    figure_width = widths[width]
    return figure_width, figure_width * aspect


def date_locator_formatter() -> tuple[mdates.AutoDateLocator, mdates.ConciseDateFormatter]:
    locator = mdates.AutoDateLocator(minticks=3, maxticks=6)
    formatter = mdates.ConciseDateFormatter(locator)
    return locator, formatter


def format_time_axis(ax) -> None:
    locator, formatter = date_locator_formatter()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.set_minor_locator(mdates.AutoDateLocator(minticks=8, maxticks=18))
    ax.xaxis.set_minor_formatter(NullFormatter())


def style_axis(ax, *, grid: bool = True) -> None:
    if ax.get_xscale() == "linear" and not _is_date_axis(ax.xaxis):
        ax.xaxis.set_minor_locator(AutoMinorLocator())
    if ax.get_yscale() == "linear":
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    if grid:
        ax.grid(True, axis="y", color=COLORS["grid"], linewidth=0.6)
        ax.grid(True, axis="x", color=COLORS["grid"], linewidth=0.4, alpha=0.45)
    ax.tick_params(axis="both", which="major", length=4, pad=3)
    ax.tick_params(axis="both", which="minor", length=2.5)
    ax.margins(x=0.015)
    ax.yaxis.get_offset_text().set_fontsize(PAPER_FONT_SIZE_PT)
    ax.xaxis.get_offset_text().set_fontsize(PAPER_FONT_SIZE_PT)
    sns.despine(ax=ax)


def style_axes(axes: Iterable) -> None:
    for ax in axes:
        style_axis(ax)


class FixedDecimalScaleFormatter(Formatter):
    """Format ticks with a fixed decimal count after applying an axis scale."""

    def __init__(self, exponent: int = 0, decimals: int = AXIS_TICK_DECIMALS) -> None:
        self.exponent = exponent
        self.decimals = decimals
        self.scale = 10.0**exponent

    def __call__(self, value, pos=None) -> str:
        scaled = value / self.scale
        if abs(scaled) < 0.5 * 10 ** (-self.decimals):
            scaled = 0.0
        return f"{scaled:.{self.decimals}f}"


def _is_date_axis(axis) -> bool:
    formatter = axis.get_major_formatter()
    return isinstance(
        formatter,
        (
            mdates.AutoDateFormatter,
            mdates.ConciseDateFormatter,
            mdates.DateFormatter,
        ),
    )


def _axis_scale_exponent(axis) -> int:
    tick_locations = np.asarray(axis.get_majorticklocs(), dtype=float)
    tick_locations = tick_locations[np.isfinite(tick_locations)]
    if tick_locations.size == 0:
        return 0

    lower, upper = sorted(axis.get_view_interval())
    visible_ticks = tick_locations[
        (tick_locations >= lower) & (tick_locations <= upper)
    ]
    if visible_ticks.size:
        tick_locations = visible_ticks

    nonzero_ticks = tick_locations[np.abs(tick_locations) > 0]
    if nonzero_ticks.size == 0:
        return 0

    largest_tick = np.nanmax(np.abs(nonzero_ticks))
    if largest_tick >= AXIS_SCALE_HIGH_THRESHOLD or largest_tick < AXIS_SCALE_LOW_THRESHOLD:
        return int(np.floor(np.log10(largest_tick)))
    return 0


def _scale_label(label: str, exponent: int) -> str:
    if exponent == 0:
        return label

    scale_label = rf"$\times 10^{{{exponent}}}$"
    close_bracket = label.rfind("]")
    if close_bracket >= 0:
        return f"{label[:close_bracket + 1]} {scale_label}{label[close_bracket + 1:]}"
    if label:
        return f"{label} {scale_label}"
    return scale_label


def _base_axis_label(axis) -> str:
    if not hasattr(axis, "_plot_style_base_label"):
        axis._plot_style_base_label = axis.get_label_text()
    return axis._plot_style_base_label


def use_default_numeric_formatter(axis) -> None:
    axis._plot_style_use_default_numeric_formatter = True


def _format_numeric_axis(ax, axis_name: str) -> None:
    axis = ax.xaxis if axis_name == "x" else ax.yaxis
    if getattr(axis, "_plot_style_use_default_numeric_formatter", False):
        return
    if _is_date_axis(axis):
        return
    if (axis_name == "x" and ax.get_xscale() != "linear") or (
        axis_name == "y" and ax.get_yscale() != "linear"
    ):
        return

    exponent = _axis_scale_exponent(axis)
    axis.set_major_formatter(FixedDecimalScaleFormatter(exponent))
    axis.get_offset_text().set_visible(False)
    axis.set_label_text(_scale_label(_base_axis_label(axis), exponent))


def format_numeric_axes(fig) -> None:
    for ax in fig.axes:
        _format_numeric_axis(ax, "x")
        _format_numeric_axis(ax, "y")


def apply_stacked_layout(fig, nrows: int, *, width: str = "single") -> None:
    fig.subplots_adjust(**STACKED_FIGURE_LAYOUTS[(width, nrows)])


def save_figure(fig, path: str | Path) -> None:
    format_numeric_axes(fig)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=0.04)


def legend_kwargs(**overrides) -> dict:
    kwargs = {
        "frameon": True,
        "fancybox": False,
        "framealpha": 0.88,
        "edgecolor": "#CFCFCF",
        "handlelength": 2.2,
        "borderpad": 0.5,
        "labelspacing": 0.4,
        "fontsize": LEGEND_FONT_SIZE_PT,
    }
    kwargs.update(overrides)
    return kwargs


def inside_legend(
    ax,
    handles,
    labels,
    *,
    loc: str = "upper center",
    ncol: int | None = None,
    **overrides,
) -> None:
    if ncol is None:
        ncol = max(1, len(labels))
    ax.legend(
        handles,
        labels,
        loc=loc,
        ncol=ncol,
        **legend_kwargs(
            columnspacing=0.9,
            handlelength=1.8,
            borderpad=0.35,
            labelspacing=0.25,
            **overrides,
        ),
    )


def zero_line(ax) -> None:
    ax.axhline(0, color=COLORS["zero"], linewidth=REFERENCE_LINE_WIDTH_PT, zorder=1)


def pad_y_limits(ax, fraction: float = 0.08, *, symmetric: bool = False) -> None:
    lower, upper = ax.get_ylim()
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        return
    if symmetric:
        limit = max(abs(lower), abs(upper)) * (1 + fraction)
        if np.isfinite(limit) and limit > 0:
            ax.set_ylim(-limit, limit)
        return
    pad = span * fraction
    ax.set_ylim(lower - pad, upper + pad)


def finite_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x).ravel()
    y_values = np.asarray(y).ravel()
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def one_to_one_limits(
    x: np.ndarray,
    y: np.ndarray,
    *,
    symmetric: bool = False,
    pad_fraction: float = 0.05,
) -> tuple[float, float]:
    values = np.concatenate([x, y])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0

    if symmetric:
        limit = np.nanmax(np.abs(values))
        if not np.isfinite(limit) or limit == 0:
            return -1.0, 1.0
        limit *= 1 + pad_fraction
        return -limit, limit

    lower = np.nanmin(values)
    upper = np.nanmax(values)
    if not np.isfinite(lower) or not np.isfinite(upper):
        return -1.0, 1.0
    pad = max((upper - lower) * pad_fraction, np.abs(upper) * 1e-12, 1e-12)
    return lower - pad, upper + pad


def format_one_to_one_axis(ax) -> None:
    ax.set_aspect("equal", adjustable="box")
    ax.set_box_aspect(1)
    style_axis(ax)


apply_theme()
