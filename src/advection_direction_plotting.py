"""Standalone rendering for the face-resolved advection prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from . import advection_direction, plot_style


WARMING_COLOR = "#D55E00"
COOLING_COLOR = "#0072B2"
NEUTRAL_COLOR = "#8A8A8A"
GROUP_COLORS = {
    "advection_zonal": "#6A5ACD",
    "advection_meridional": "#009E73",
    "advection_horizontal": "#CC79A7",
    "advection_vertical": "#E69F00",
    "advection_face_total": "#111111",
}
GROUP_LABELS = {
    "advection_zonal": "Zonal (west + east)",
    "advection_meridional": "Meridional (south + north)",
    "advection_horizontal": "Horizontal",
    "advection_vertical": "Vertical",
    "advection_face_total": "All faces",
}
LEGEND_HEADROOM_FRACTION = 0.30


def plot_advection_direction_exploration(
    composite: xr.Dataset,
) -> Figure:
    """Return the standalone face-component and glyph figure."""
    required = (
        "advection",
        *(
            advection_direction.stage1_face_variable(face)
            for face in advection_direction.REQUIRED_FACES
        ),
    )
    _require_variables(composite, required)
    diagnostics = xr.merge(
        [composite, advection_direction.grouped_advection_components(composite)],
        compat="override",
    )
    diagnostics.attrs.update(composite.attrs)
    daily = advection_direction.complete_daily_face_means(diagnostics)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=plot_style.publication_figsize(
            "full",
            aspect=plot_style.THREE_PANEL_STACK_ASPECT,
        ),
        gridspec_kw={"height_ratios": (1.0, 1.0, 1.15)},
        constrained_layout=True,
    )
    face_ax, grouped_ax, glyph_ax = axes

    lag_days = np.asarray(diagnostics["lag_hour"].values, dtype=float) / 24.0
    _plot_face_timeseries(face_ax, diagnostics, lag_days)
    _plot_grouped_timeseries(grouped_ax, diagnostics, lag_days)
    _plot_daily_face_glyphs(glyph_ax, daily)

    for ax in axes:
        ax.axvline(
            0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle=":",
            zorder=1,
        )
        plot_style.style_axis(ax)
        ax.set_xlim(float(lag_days.min()) / 1.0, float(lag_days.max()) / 1.0)
    glyph_ax.grid(False)
    glyph_ax.set_xlabel("Days relative to event peak")

    title = _figure_title(composite)
    fig.suptitle(title)
    return fig


def write_advection_direction_exploration_plot(
    composite: xr.Dataset,
    output_path: str | Path,
) -> Path:
    """Write the standalone face-component and glyph figure."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_advection_direction_exploration(composite)
    try:
        plot_style.save_figure(fig, path)
    finally:
        plt.close(fig)
    return path


def _plot_face_timeseries(
    ax,
    ds: xr.Dataset,
    lag_days: np.ndarray,
) -> None:
    for face in advection_direction.available_stage1_faces(ds):
        name = advection_direction.stage1_face_variable(face)
        ax.plot(
            lag_days,
            ds[name].values,
            color=plot_style.FACE_COLORS[face],
            label=face.capitalize(),
        )
    plot_style.zero_line(ax)
    ax.set_ylabel("Face contribution [K hr-1]")
    ax.set_title("Signed face contributions")
    _add_upper_axis_headroom(ax)
    ax.legend(ncol=5, loc="upper center", **plot_style.legend_kwargs())


def _plot_grouped_timeseries(
    ax,
    ds: xr.Dataset,
    lag_days: np.ndarray,
) -> None:
    for name in (
        "advection_zonal",
        "advection_meridional",
        "advection_horizontal",
        "advection_vertical",
        "advection_face_total",
    ):
        ax.plot(
            lag_days,
            ds[name].values,
            color=GROUP_COLORS[name],
            label=GROUP_LABELS[name],
            linestyle="--" if name == "advection_face_total" else "-",
        )
    plot_style.zero_line(ax)
    ax.set_ylabel("Grouped contribution [K hr-1]")
    ax.set_title("Grouped advective contributions")
    _add_upper_axis_headroom(ax)
    ax.legend(ncol=5, loc="upper center", **plot_style.legend_kwargs())


def _add_upper_axis_headroom(
    ax,
    *,
    fraction: float = LEGEND_HEADROOM_FRACTION,
) -> None:
    """Expand only the upper y-limit by a fraction of the autoscaled range."""
    if fraction < 0:
        raise ValueError("fraction must be >= 0.")
    lower, upper = ax.get_ylim()
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        raise ValueError("axis must have a finite positive y-range.")
    ax.set_ylim(lower, upper + fraction * span)


def _plot_daily_face_glyphs(ax, daily: xr.Dataset) -> None:
    centers = np.asarray(daily["lag_day_center"].values, dtype=float)
    face_values = np.concatenate(
        [
            np.asarray(
                daily[advection_direction.stage1_face_variable(face)].values,
                dtype=float,
            )
            for face in advection_direction.REQUIRED_FACES
        ]
    )
    finite = np.abs(face_values[np.isfinite(face_values)])
    max_magnitude = float(finite.max()) if finite.size else 0.0
    scale = max(max_magnitude, np.finfo(float).eps)

    for index, center in enumerate(centers):
        values = {
            face: float(
                daily[advection_direction.stage1_face_variable(face)].isel(
                    daily_window=index
                )
            )
            for face in advection_direction.REQUIRED_FACES
        }
        _draw_face_glyph(
            ax,
            center=center,
            values=values,
            magnitude_scale=scale,
            show_labels=index == 0,
        )

    handles = [
        Line2D([0], [0], color=WARMING_COLOR, linewidth=4, label="Warming"),
        Line2D([0], [0], color=COOLING_COLOR, linewidth=4, label="Cooling"),
    ]
    ax.legend(
        handles=handles,
        ncol=2,
        loc="upper center",
        **plot_style.legend_kwargs(),
    )
    ax.text(
        0.995,
        0.03,
        f"Maximum daily |face contribution| = {max_magnitude:.3g} K hr-1",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
    )
    ax.set_ylim(-0.75, 1.15)
    ax.set_yticks([])
    ax.set_ylabel("24-hour face glyph")
    ax.set_title(
        "Daily mean face contributions "
        "(color = sign, thickness = magnitude, not airflow direction)"
    )


def _draw_face_glyph(
    ax,
    *,
    center: float,
    values: dict[str, float],
    magnitude_scale: float,
    show_labels: bool,
) -> None:
    width = 0.58
    height = 0.62
    left = center - width / 2
    right = center + width / 2
    bottom = -height / 2
    top = height / 2
    ax.add_patch(
        Rectangle(
            (left, bottom),
            width,
            height,
            facecolor="none",
            edgecolor="#B5B5B5",
            linewidth=0.7,
            zorder=1,
        )
    )

    _draw_face_line(
        ax,
        [left, left],
        [bottom, top],
        value=values["west"],
        magnitude_scale=magnitude_scale,
    )
    _draw_face_line(
        ax,
        [right, right],
        [bottom, top],
        value=values["east"],
        magnitude_scale=magnitude_scale,
    )
    _draw_face_line(
        ax,
        [left, right],
        [bottom, bottom],
        value=values["south"],
        magnitude_scale=magnitude_scale,
    )
    _draw_face_line(
        ax,
        [left, right],
        [top, top],
        value=values["north"],
        magnitude_scale=magnitude_scale,
    )

    top_value = values["top"]
    ax.scatter(
        [center],
        [0.67],
        s=18.0 + 115.0 * min(abs(top_value) / magnitude_scale, 1.0),
        color=_contribution_color(top_value),
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    if show_labels:
        label_kwargs = {
            "ha": "center",
            "va": "center",
            "fontsize": plot_style.LEGEND_FONT_SIZE_PT,
            "color": "#333333",
        }
        ax.text(left - 0.09, 0, "W", **label_kwargs)
        ax.text(right + 0.09, 0, "E", **label_kwargs)
        ax.text(center, bottom - 0.13, "S", **label_kwargs)
        ax.text(center, top + 0.13, "N", **label_kwargs)
        ax.text(center, 0.91, "T", **label_kwargs)


def _draw_face_line(
    ax,
    x: list[float],
    y: list[float],
    *,
    value: float,
    magnitude_scale: float,
) -> None:
    linewidth = 0.8 + 6.2 * min(abs(value) / magnitude_scale, 1.0)
    ax.plot(
        x,
        y,
        color=_contribution_color(value),
        linewidth=linewidth,
        solid_capstyle="butt",
        zorder=2,
    )


def _contribution_color(value: float) -> str:
    if not np.isfinite(value) or value == 0:
        return NEUTRAL_COLOR
    return WARMING_COLOR if value > 0 else COOLING_COLOR


def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = sorted(name for name in names if name not in ds)
    if missing:
        raise ValueError(
            "Advection-direction plot is missing required variables: "
            f"{', '.join(missing)}"
        )


def _figure_title(ds: xr.Dataset) -> str:
    region = ds.attrs.get("region", "unknown region")
    n_events = ds.attrs.get("n_events", "unknown")
    pre_days = ds.attrs.get("pre_days", "?")
    post_days = ds.attrs.get("post_days", "?")
    return (
        f"Face-resolved advection composite for {region} "
        f"(n={n_events}, -{pre_days} to +{post_days} days)"
    )
