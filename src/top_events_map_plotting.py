"""Single-panel maps from prepared top-event spatial products."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

from . import plot_style
from .top_events_map import region_bounds, validate_top_event_maps

PNW_REGIONS = ("pnw_hotz", "pnw_bartusek")
DEFAULT_HEIGHT_CONTOUR_INTERVAL = 50.0


def plot_top_event_map(
    ds: xr.Dataset,
    *,
    event_index: int = 0,
    outline_regions: Sequence[str] | None = None,
    temperature_limit: float | None = None,
    height_contour_interval: float = DEFAULT_HEIGHT_CONTOUR_INTERVAL,
) -> plt.Figure:
    """Return one map and one colorbar, using only a validated saved product."""
    validate_top_event_maps(ds)
    if not 0 <= event_index < ds.sizes["event"]:
        raise ValueError("event_index is outside the prepared map product.")
    region = str(ds.attrs["region"])
    outlines = (
        tuple(outline_regions)
        if outline_regions is not None
        else (PNW_REGIONS if region in PNW_REGIONS else (region,))
    )
    if not outlines or len(set(outlines)) != len(outlines):
        raise ValueError("Outline regions must be nonempty and unique.")
    extent = np.asarray(ds.attrs["map_extent"], dtype=float)
    for name in outlines:
        west, east, south, north = region_bounds(name)
        if not (
            extent[0] <= west < east <= extent[1]
            and extent[2] <= south < north <= extent[3]
        ):
            raise ValueError(f"Outline {name!r} falls outside the prepared map extent.")
    if temperature_limit is None:
        temperature_limit = max(
            1.0, float(np.ceil(np.abs(ds.t2m_anomaly.values).max()))
        )
    for name, value in (
        ("temperature_limit", temperature_limit),
        ("height_contour_interval", height_contour_interval),
    ):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    panel = ds.isel(event=event_index)
    plot_style.apply_theme()
    data_crs = ccrs.PlateCarree()
    fig = plt.figure(figsize=plot_style.publication_figsize("full", aspect=0.72))
    ax = fig.add_axes((0.03, 0.18, 0.83, 0.71), projection=data_crs)
    ax.set_extent(extent, crs=data_crs)
    mesh = ax.pcolormesh(
        ds.longitude,
        ds.latitude,
        panel.t2m_anomaly,
        transform=data_crs,
        cmap="RdBu_r",
        shading="auto",
        rasterized=True,
        norm=TwoSlopeNorm(vmin=-temperature_limit, vcenter=0, vmax=temperature_limit),
    )
    heights = panel.z500_anomaly.values
    low, high = float(heights.min()), float(heights.max())
    if (high - low) / height_contour_interval > 201:
        plt.close(fig)
        raise ValueError("Contour interval requests more than 200 levels.")
    first = np.ceil(low / height_contour_interval) * height_contour_interval
    levels = np.arange(first, high, height_contour_interval)
    levels = levels[(levels > low) & (levels < high)]
    levels[levels == 0] = 0.0
    if levels.size > 200:
        plt.close(fig)
        raise ValueError("Contour interval requests more than 200 levels.")
    if levels.size:
        contours = ax.contour(
            ds.longitude,
            ds.latitude,
            heights,
            levels=levels,
            colors="#222222",
            linewidths=plot_style.LINE_WIDTH_PT,
            linestyles=["dashed" if value < 0 else "solid" for value in levels],
            transform=data_crs,
        )
        ax.clabel(
            contours,
            fmt="%g",
            fontsize=plot_style.LEGEND_FONT_SIZE_PT,
            inline_spacing=3,
        )
    _decorate_map(ax, data_crs)
    handles = []
    for name in outlines:
        west, east, south, north = region_bounds(name)
        outline = Rectangle(
            (west, south),
            east - west,
            north - south,
            fill=False,
            edgecolor=plot_style.REGION_COLORS.get(name, "#444444"),
            linewidth=2 * plot_style.LINE_WIDTH_PT,
            transform=data_crs,
            zorder=6,
            label=plot_style.REGION_NAME_MAPPING.get(name, name),
        )
        outline.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=3 * plot_style.LINE_WIDTH_PT, foreground="white"
                ),
                path_effects.Normal(),
            ]
        )
        ax.add_patch(outline)
        handles.append(outline)
    color_axis = fig.add_axes((0.90, 0.24, 0.018, 0.58))
    colorbar = fig.colorbar(mesh, cax=color_axis, extend="both")
    colorbar.set_label("2 m temperature anomaly [K]")
    first_day, last_day = pd.to_datetime(panel.sample_date.values[[0, -1]])
    date_label = _date_range_label(first_day, last_day)
    rank = int(panel.selection_rank.item())
    ax.set_title(
        f"{date_label}\n{plot_style.REGION_NAME_MAPPING.get(region, region)} - event rank {rank}",
        pad=14,
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.46, 0.095),
        ncol=min(2, len(handles)),
        **plot_style.legend_kwargs(),
    )
    peak = pd.Timestamp(panel.peak_time.item()).strftime("%d %b %Y %H:%M UTC")
    baseline = (
        f"{ds.attrs['climatology_start_year']}-{ds.attrs['climatology_end_year']}"
    )
    fig.text(
        0.46,
        0.055,
        f"3-day means relative to {baseline} daily climatology\n"
        f"Z500 anomaly contours [m], spacing {height_contour_interval:g} m; peak {peak}",
        ha="center",
        va="center",
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
    )
    return fig


def _decorate_map(ax, data_crs: ccrs.PlateCarree) -> None:
    ax.coastlines(
        resolution="50m", color="#222222", linewidth=plot_style.REFERENCE_LINE_WIDTH_PT
    )
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        edgecolor="#666666",
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT / 2,
    )
    ax.gridlines(
        crs=data_crs,
        draw_labels=False,
        linewidth=0.4,
        color="#666666",
        alpha=0.3,
        linestyle=":",
    )


def _date_range_label(start: pd.Timestamp, end: pd.Timestamp) -> str:
    if (start.year, start.month) == (end.year, end.month):
        return f"{start.day}-{end.day} {end.strftime('%B %Y')}"
    if start.year == end.year:
        return f"{start.day} {start.strftime('%b')} - {end.day} {end.strftime('%b %Y')}"
    return f"{start.strftime('%d %b %Y')} - {end.strftime('%d %b %Y')}"


def map_filename(ds: xr.Dataset, event_index: int) -> str:
    event = ds.isel(event=event_index)
    peak = pd.Timestamp(event.peak_time.item()).strftime("%Y%m%d")
    return (
        f"top_events_map_{ds.attrs['region']}_rank{int(event.selection_rank.item()):02d}"
        f"_event{int(event.event_id.item())}_{peak}.png"
    )


def write_top_event_maps(
    ds: xr.Dataset,
    output_dir: str | Path,
    **plot_options,
) -> list[Path]:
    """Render all selected events with shared scaling and no file replacement."""
    validate_top_event_maps(ds)
    root = Path(output_dir).expanduser().resolve()
    paths = [root / map_filename(ds, index) for index in range(ds.sizes["event"])]
    for path in paths:
        if path.exists():
            raise FileExistsError(f"Figure exists: {path}")
    root.mkdir(parents=True, exist_ok=True)
    for index, path in enumerate(paths):
        fig = plot_top_event_map(ds, event_index=index, **plot_options)
        temporary_path = path.with_name(f".{path.stem}.{uuid4().hex}.partial.png")
        try:
            fig.savefig(
                temporary_path,
                dpi=plot_style.DPI,
                bbox_inches="tight",
                pad_inches=0.04,
                facecolor="white",
            )
            os.link(temporary_path, path)
        finally:
            plt.close(fig)
            temporary_path.unlink(missing_ok=True)
    return paths
