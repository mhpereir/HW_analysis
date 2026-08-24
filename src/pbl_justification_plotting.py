"""Render the two-panel PBL/700 hPa justification figure."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import config, pbl_justification, plot_style

DOMAIN_OUTLINE_COLOR = "#D62728"
MEAN_COLOR = "#0072B2"
ENVELOPE_COLOR = "#56B4E9"
MAP_MARGIN_DEGREES = 2.5


def plot_product(
    ds: xr.Dataset,
    *,
    map_margin_degrees: float = MAP_MARGIN_DEGREES,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the publication figure for a validated diagnostic product."""
    pbl_justification.validate_product(ds)
    if not np.isfinite(map_margin_degrees) or map_margin_degrees <= 0:
        raise ValueError("map_margin_degrees must be finite and positive.")

    plot_style.apply_theme()
    fig = plt.figure(figsize=plot_style.publication_figsize("full", aspect=0.43))
    grid = fig.add_gridspec(1, 2, width_ratios=(1.08, 1.0), wspace=0.22)
    time_ax = fig.add_subplot(grid[0, 0])
    map_ax = fig.add_subplot(grid[0, 1], projection=ccrs.PlateCarree())

    _plot_time_series(time_ax, ds)
    _plot_map(fig, map_ax, ds, margin=map_margin_degrees)
    time_ax.text(
        0.01,
        0.99,
        "a",
        transform=time_ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
    )
    map_ax.text(
        0.01,
        0.99,
        "b",
        transform=map_ax.transAxes,
        ha="left",
        va="top",
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 1.5},
    )
    return fig


def write_figure(fig: plt.Figure, path: str | Path) -> Path:  # type: ignore[type-arg]
    """Atomically write a nonempty figure without replacing prior output."""
    output_path = Path(path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {output_path}")
    if not output_path.suffix:
        raise ValueError("Figure output path requires a file extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}.partial{output_path.suffix}"
    )
    try:
        plot_style.save_figure(fig, temporary_path)
        if not temporary_path.is_file() or temporary_path.stat().st_size == 0:
            raise RuntimeError("Figure writer did not produce a nonempty file.")
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _plot_time_series(ax, ds: xr.Dataset) -> None:
    lag_days = np.asarray(ds["lag_hour"].values, dtype=float) / 24.0
    mean = np.asarray(ds[pbl_justification.AREA_MEAN_NAME].values) / 100.0
    p05 = np.asarray(ds[pbl_justification.SPATIAL_P05_NAME].values) / 100.0
    p95 = np.asarray(ds[pbl_justification.SPATIAL_P95_NAME].values) / 100.0

    ax.fill_between(
        lag_days,
        p05,
        p95,
        color=ENVELOPE_COLOR,
        alpha=0.32,
        linewidth=0,
        label="Spatial 5th-95th percentile",
    )
    ax.plot(lag_days, mean, color=MEAN_COLOR, label="Area-weighted mean")
    ax.axvline(0.0, color="#555555", linestyle=":", linewidth=0.9)
    ax.axhline(
        float(ds.attrs["upper_boundary_reference_hpa"]),
        color=DOMAIN_OUTLINE_COLOR,
        linestyle="--",
        linewidth=1.1,
        label="700 hPa analysis top",
    )
    _set_pressure_limits(ax, p05, p95, float(ds.attrs["upper_boundary_reference_hpa"]))
    ax.set_xlabel("Days relative to heatwave peak")
    ax.set_ylabel("PBL-top pressure (hPa)\nLower pressure indicates a deeper PBL")
    ax.set_title(
        "Peak-aligned regional PBL depth\n"
        f"{int(ds[pbl_justification.SELECTED_EVENT_COUNT_NAME].item())} heatwave events"
    )
    plot_style.style_axis(ax)
    plot_style.inside_legend(ax, *ax.get_legend_handles_labels(), loc="lower center")


def _set_pressure_limits(
    ax, p05: np.ndarray, p95: np.ndarray, reference: float
) -> None:
    low = min(float(np.nanmin(p05)), reference)
    high = max(float(np.nanmax(p95)), reference)
    padding = max(10.0, 0.04 * (high - low))
    ax.set_ylim(high + padding, low - padding)


def _plot_map(fig: plt.Figure, ax, ds: xr.Dataset, *, margin: float) -> None:
    region = str(ds.attrs["region"])
    lat_bounds, lon_bounds = config.REGIONS[region]
    lat_min, lat_max = sorted((float(lat_bounds.start), float(lat_bounds.stop)))
    lon_min, lon_max = sorted((float(lon_bounds.start), float(lon_bounds.stop)))
    data_crs = ccrs.PlateCarree()
    field = ds[pbl_justification.MAP_NAME] / 100.0
    values = np.asarray(field.values, dtype=float)
    vmin = np.floor(values.min() / 25.0) * 25.0
    vmax = np.ceil(values.max() / 25.0) * 25.0
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0

    ax.set_extent(
        (lon_min - margin, lon_max + margin, lat_min - margin, lat_max + margin),
        crs=data_crs,
    )
    ax.set_facecolor("#EAF2F8")
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#F2EFE8", zorder=0)
    mesh = ax.pcolormesh(
        field["lon"],
        field["lat"],
        field,
        transform=data_crs,
        cmap="viridis_r",
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        zorder=1,
    )
    ax.coastlines(resolution="50m", linewidth=0.7, color="#333333", zorder=3)
    ax.add_feature(
        cfeature.BORDERS.with_scale("50m"),
        linewidth=0.55,
        edgecolor="#555555",
        zorder=3,
    )
    ax.add_patch(
        Rectangle(
            (lon_min, lat_min),
            lon_max - lon_min,
            lat_max - lat_min,
            fill=False,
            edgecolor=DOMAIN_OUTLINE_COLOR,
            linewidth=1.5,
            label=f"{region} analysis domain",
            transform=data_crs,
            zorder=4,
        )
    )
    gridlines = ax.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.35,
        color="#777777",
        alpha=0.55,
        linestyle=":",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": plot_style.PAPER_FONT_SIZE_PT - 2}
    gridlines.ylabel_style = {"size": plot_style.PAPER_FONT_SIZE_PT - 2}
    ax.set_title(
        "Mean daily maximum PBL depth\n"
        f"{int(ds[pbl_justification.SELECTED_DAY_COUNT_NAME].item())} heatwave days"
    )
    colorbar = fig.colorbar(
        mesh, ax=ax, orientation="horizontal", pad=0.09, fraction=0.07
    )
    colorbar.set_label("PBL-top pressure (hPa), lower values indicate deeper PBL")
