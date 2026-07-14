"""Plot stacked positive/negative daily T2m and Z500 anomaly composites."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import plot_style  # noqa: E402


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results/spatial_composites"
    / "dyn_net_daily_spatial_composites_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results/spatial_composites"
    / "dyn_net_daily_t2m_z500_composites_pnw_bartusek_tas_q90_1940_2024.png"
)
GROUPS = ("positive", "negative")
MAP_EXTENT = (-170.0, -40.0, 10.0, 80.0)
PNW_BARTUSEK_BOUNDS = (-130.0, -110.0, 40.0, 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot stacked daily T2m/Z500 composites by I_dyn_net sign."
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--temperature-limit",
        type=float,
        default=None,
        help="Optional symmetric T2m anomaly colour limit in K.",
    )
    parser.add_argument(
        "--height-contour-interval",
        type=float,
        default=None,
        help="Optional Z500 anomaly contour interval in metres.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.temperature_limit is not None and args.temperature_limit <= 0:
        raise ValueError("--temperature-limit must be > 0.")
    if args.height_contour_interval is not None and args.height_contour_interval <= 0:
        raise ValueError("--height-contour-interval must be > 0.")


def main() -> int:
    args = parse_args()
    validate_args(args)
    input_path = args.input_path.expanduser().resolve()
    with xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True) as ds:
        fig = plot_spatial_composites(
            ds.load(),
            temperature_limit=args.temperature_limit,
            height_contour_interval=args.height_contour_interval,
        )
    written = write_figure(fig, args.output_path)
    plt.close(fig)
    print(f"Wrote daily dynamical-sign composite figure: {written}")
    return 0


def validate_composite(ds: xr.Dataset) -> None:
    required = {
        "t2m_anomaly",
        "z500_anomaly",
        "event_count",
        "I_dyn_net_mean",
    }
    missing = sorted(required.difference(ds.data_vars))
    if missing:
        raise ValueError("Composite dataset is missing variables: " + ", ".join(missing))
    for coord in ("dyn_sign", "latitude", "longitude"):
        if coord not in ds.coords:
            raise ValueError(f"Composite dataset is missing coordinate {coord!r}.")
    groups = tuple(str(value) for value in ds["dyn_sign"].values)
    if set(groups) != set(GROUPS):
        raise ValueError(f"Expected dyn_sign groups {GROUPS}; found {groups}.")
    for name in ("t2m_anomaly", "z500_anomaly"):
        values = np.asarray(ds[name].values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Composite variable {name!r} contains non-finite values.")


def plot_spatial_composites(
    ds: xr.Dataset,
    *,
    temperature_limit: float | None = None,
    height_contour_interval: float | None = None,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the two-panel publication figure."""
    validate_composite(ds)
    plot_style.apply_theme()

    temperature_limit = temperature_limit or rounded_symmetric_limit(
        ds["t2m_anomaly"].values,
        step=0.5,
    )
    height_limit = rounded_symmetric_limit(ds["z500_anomaly"].values, step=10.0)
    interval = height_contour_interval or automatic_contour_interval(height_limit)
    contour_limit = max(interval, np.ceil(height_limit / interval) * interval)
    contour_levels = np.arange(-contour_limit, contour_limit + 0.5 * interval, interval)
    norm = TwoSlopeNorm(vmin=-temperature_limit, vcenter=0.0, vmax=temperature_limit)

    projection = ccrs.LambertConformal(
        central_longitude=-105.0,
        central_latitude=45.0,
        standard_parallels=(30.0, 60.0),
    )
    data_crs = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=plot_style.publication_figsize("full", aspect=0.83),
        subplot_kw={"projection": projection},
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, group in zip(axes, GROUPS, strict=True):
        panel = ds.sel(dyn_sign=group)
        mesh = ax.pcolormesh(
            ds["longitude"],
            ds["latitude"],
            panel["t2m_anomaly"],
            transform=data_crs,
            cmap="RdBu_r",
            norm=norm,
            shading="auto",
            rasterized=True,
        )
        line_styles = ["dashed" if level < 0 else "solid" for level in contour_levels]
        line_widths = [1.4 if np.isclose(level, 0.0) else 0.8 for level in contour_levels]
        contours = ax.contour(
            ds["longitude"],
            ds["latitude"],
            panel["z500_anomaly"],
            levels=contour_levels,
            colors="#222222",
            linewidths=line_widths,
            linestyles=line_styles,
            transform=data_crs,
        )
        ax.clabel(contours, fmt="%g", fontsize=8, inline_spacing=2)
        decorate_map(ax, data_crs)
        count = int(panel["event_count"].item())
        dyn_mean = float(panel["I_dyn_net_mean"].item())
        ax.set_title(
            f"{group.capitalize()} $I_{{dyn,net}}$: n = {count}, "
            f"mean = {dyn_mean:.2f} K"
        )

    assert mesh is not None
    colorbar = fig.colorbar(
        mesh,
        ax=axes.tolist(),
        orientation="horizontal",
        fraction=0.045,
        pad=0.045,
        aspect=38,
    )
    colorbar.set_label("2 m temperature anomaly (K)")
    fig.suptitle("Four-day Pre-peak Heatwave Spatial Composites")
    return fig


def decorate_map(ax, data_crs: ccrs.PlateCarree) -> None:
    ax.set_extent(MAP_EXTENT, crs=data_crs)
    ax.coastlines(resolution="50m", linewidth=0.7)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.55)
    gridlines = ax.gridlines(
        crs=data_crs,
        draw_labels=True,
        linewidth=0.35,
        color="#666666",
        alpha=0.55,
        linestyle=":",
        x_inline=False,
        y_inline=False,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 9}
    gridlines.ylabel_style = {"size": 9}
    west, east, south, north = PNW_BARTUSEK_BOUNDS
    ax.add_patch(
        Rectangle(
            (west, south),
            east - west,
            north - south,
            fill=False,
            edgecolor="#E69F00",
            linewidth=1.4,
            transform=data_crs,
            zorder=5,
        )
    )


def rounded_symmetric_limit(values: np.ndarray, *, step: float) -> float:
    maximum = float(np.nanmax(np.abs(np.asarray(values, dtype=float))))
    if not np.isfinite(maximum) or maximum <= 0:
        return step
    return float(np.ceil(maximum / step) * step)


def automatic_contour_interval(height_limit: float) -> float:
    target = height_limit / 5.0
    choices = np.array([5.0, 10.0, 20.0, 25.0, 50.0, 100.0])
    return float(choices[np.argmin(np.abs(choices - target))])


def write_figure(fig: plt.Figure, path: str | Path) -> Path:  # type: ignore[type-arg]
    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=plot_style.DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    return output_path


if __name__ == "__main__":
    raise SystemExit(main())
