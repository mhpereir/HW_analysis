"""Plot positive/negative daily T2m and Z500 anomaly evolution."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

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

from src import plot_style

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
GROUP_DIM = "dyn_sign"
LAG_DIM = "lag"
MAP_EXTENT = (-170.0, -40.0, 10.0, 80.0)
PNW_BARTUSEK_BOUNDS = (-130.0, -110.0, 40.0, 60.0)
DEFAULT_PLOT_LAGS = (-2, 0, 2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot event-relative daily T2m/Z500 composites by I_dyn_net sign."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--plot-lags",
        type=int,
        nargs="+",
        default=DEFAULT_PLOT_LAGS,
        help="Event-relative days to plot (default: -2 0 2).",
    )
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
    if not args.plot_lags:
        raise ValueError("--plot-lags must contain at least one lag.")
    if len(set(args.plot_lags)) != len(args.plot_lags):
        raise ValueError("--plot-lags must not contain duplicates.")
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
            plot_lags=args.plot_lags,
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
    for coord in (GROUP_DIM, LAG_DIM, "latitude", "longitude"):
        if coord not in ds.coords:
            raise ValueError(f"Composite dataset is missing coordinate {coord!r}.")
    groups = tuple(str(value) for value in ds[GROUP_DIM].values)
    if set(groups) != set(GROUPS):
        raise ValueError(f"Expected {GROUP_DIM} groups {GROUPS}; found {groups}.")
    lags = np.asarray(ds[LAG_DIM].values)
    if lags.ndim != 1 or lags.size == 0:
        raise ValueError(f"Coordinate {LAG_DIM!r} must be one-dimensional and nonempty.")
    if not np.issubdtype(lags.dtype, np.integer):
        raise ValueError(f"Coordinate {LAG_DIM!r} must contain integer days.")
    if np.any(np.diff(lags) <= 0):
        raise ValueError(f"Coordinate {LAG_DIM!r} must be strictly increasing.")
    for name in ("t2m_anomaly", "z500_anomaly"):
        expected_dims = (GROUP_DIM, LAG_DIM, "latitude", "longitude")
        if ds[name].dims != expected_dims:
            raise ValueError(
                f"Composite variable {name!r} must have dimensions {expected_dims}; "
                f"found {ds[name].dims}."
            )
        values = np.asarray(ds[name].values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Composite variable {name!r} contains non-finite values.")


def plot_spatial_composites(
    ds: xr.Dataset,
    *,
    plot_lags: Sequence[int] = DEFAULT_PLOT_LAGS,
    temperature_limit: float | None = None,
    height_contour_interval: float | None = None,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return a sign-by-lag publication figure."""
    validate_composite(ds)
    return _render_spatial_composites(
        ds,
        plot_lags=plot_lags,
        temperature_limit=temperature_limit,
        height_contour_interval=height_contour_interval,
        group_metric_label=r"$I_{dyn,net}$",
        figure_title="Heatwave Spatial Composite Evolution Relative to Peak Day",
        mean_variable="I_dyn_net_mean",
    )


def validate_matched_composite(
    ds: xr.Dataset,
    *,
    expected_specification: str | None = None,
) -> None:
    """Validate matched membership, provenance, and spatial fields."""
    validate_composite(ds)
    expected_stage = "daily_matched_idyn_spatial_composites"
    actual_stage = ds.attrs.get("pipeline_stage")
    if actual_stage != expected_stage:
        raise ValueError(
            f"Expected pipeline_stage={expected_stage!r}; got {actual_stage!r}."
        )

    required_attrs = (
        "matching_group_variable",
        "matching_specification",
        "matching_label",
        "matching_variables",
        "matching_caliper_sd",
        "matching_pair_count",
        "matching_source_negative_count",
        "matching_source_positive_count",
        "matching_source_zero_count",
        "matching_unmatched_negative_count",
        "matching_unmatched_positive_count",
        "matching_settings_path",
        "matching_settings_sha256",
        "event_features_path",
        "event_features_sha256",
    )
    missing_attrs = [name for name in required_attrs if name not in ds.attrs]
    if missing_attrs:
        raise ValueError(
            "Matched composite is missing attributes: "
            + ", ".join(missing_attrs)
        )
    if ds.attrs["matching_group_variable"] != "I_dyn_pre":
        raise ValueError("Matched composite must use I_dyn_pre as its group variable.")
    if expected_specification is not None:
        actual = str(ds.attrs["matching_specification"])
        if actual != expected_specification:
            raise ValueError(
                f"Expected matching specification {expected_specification!r}; "
                f"got {actual!r}."
            )

    pair_count = int(ds.attrs["matching_pair_count"])
    if pair_count <= 0:
        raise ValueError("Matched composite must contain at least one pair.")
    caliper = float(ds.attrs["matching_caliper_sd"])
    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError("Matched composite caliper must be finite and positive.")
    counts = np.asarray(ds["event_count"].values, dtype=np.int64)
    if not np.array_equal(counts, np.full(len(GROUPS), pair_count)):
        raise ValueError(
            "Matched composite sign counts must both equal matching_pair_count."
        )
    source_negative = int(ds.attrs["matching_source_negative_count"])
    source_positive = int(ds.attrs["matching_source_positive_count"])
    source_zero = int(ds.attrs["matching_source_zero_count"])
    unmatched_negative = int(ds.attrs["matching_unmatched_negative_count"])
    unmatched_positive = int(ds.attrs["matching_unmatched_positive_count"])
    if min(source_negative, source_positive, source_zero) < 0:
        raise ValueError("Matched source-population counts must be nonnegative.")
    if unmatched_negative != source_negative - pair_count:
        raise ValueError("Unmatched negative count is inconsistent.")
    if unmatched_positive != source_positive - pair_count:
        raise ValueError("Unmatched positive count is inconsistent.")
    if min(unmatched_negative, unmatched_positive) < 0:
        raise ValueError("Matched pair count exceeds a source sign population.")

    required_variables = (
        "I_dyn_pre_mean",
        "event_id",
        "I_dyn_pre",
        "event_dyn_sign",
        "matched_pair_id",
        "matched_pair_distance",
    )
    missing = [name for name in required_variables if name not in ds]
    if missing:
        raise ValueError(
            "Matched composite is missing audit variables: " + ", ".join(missing)
        )
    event_ids = np.asarray(ds["event_id"].values)
    signs = np.asarray(ds["event_dyn_sign"].values, dtype=str)
    pair_ids = np.asarray(ds["matched_pair_id"].values, dtype=np.int64)
    distances = np.asarray(ds["matched_pair_distance"].values, dtype=float)
    expected_events = 2 * pair_count
    audit_arrays = (event_ids, signs, pair_ids, distances)
    if any(values.size != expected_events for values in audit_arrays):
        raise ValueError("Matched event audit variables have inconsistent lengths.")
    if np.unique(event_ids).size != expected_events:
        raise ValueError("Matched event IDs must be unique.")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Matched pair distances must be finite and nonnegative.")
    unique_pairs, pair_frequencies = np.unique(pair_ids, return_counts=True)
    if unique_pairs.size != pair_count or not np.all(pair_frequencies == 2):
        raise ValueError("Each matched pair ID must occur exactly twice.")
    for pair_id in unique_pairs:
        pair_mask = pair_ids == pair_id
        if set(signs[pair_mask]) != set(GROUPS):
            raise ValueError(
                f"Matched pair {pair_id} must contain one event from each sign."
            )
        if not np.allclose(distances[pair_mask], distances[pair_mask][0]):
            raise ValueError(
                f"Matched pair {pair_id} has inconsistent distance values."
            )


def plot_matched_spatial_composites(
    ds: xr.Dataset,
    *,
    plot_lags: Sequence[int] = DEFAULT_PLOT_LAGS,
    temperature_limit: float | None = None,
    height_contour_interval: float | None = None,
    expected_specification: str | None = None,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the matched I_dyn_pre sign-by-lag publication figure."""
    validate_matched_composite(
        ds,
        expected_specification=expected_specification,
    )
    pair_count = int(ds.attrs["matching_pair_count"])
    label = str(ds.attrs["matching_label"])
    caliper = float(ds.attrs["matching_caliper_sd"])
    title = (
        "Matched Heatwave Spatial Composite Evolution Relative to Peak Day\n"
        f"{label}, {caliper:.2f} pooled SD (n = {pair_count} pairs)"
    )
    return _render_spatial_composites(
        ds,
        plot_lags=plot_lags,
        temperature_limit=temperature_limit,
        height_contour_interval=height_contour_interval,
        group_metric_label=r"$I_{dyn,pre}$",
        figure_title=title,
        mean_variable="I_dyn_pre_mean",
    )


def _render_spatial_composites(
    ds: xr.Dataset,
    *,
    plot_lags: Sequence[int],
    temperature_limit: float | None,
    height_contour_interval: float | None,
    group_metric_label: str,
    figure_title: str,
    mean_variable: str,
) -> plt.Figure:  # type: ignore[type-arg]
    """Render validated spatial fields with caller-provided population labels."""
    plot_style.apply_theme()
    lags = select_plot_lags(ds, plot_lags)
    plot_ds = ds.sel({LAG_DIM: lags})

    temperature_limit = temperature_limit or rounded_symmetric_limit(
        plot_ds["t2m_anomaly"].values,
        step=0.5,
    )
    height_limit = rounded_symmetric_limit(
        plot_ds["z500_anomaly"].values,
        step=10.0,
    )
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
        nrows=len(GROUPS),
        ncols=lags.size,
        figsize=plot_style.publication_figsize("full", aspect=0.48),
        subplot_kw={"projection": projection},
        constrained_layout=True,
        squeeze=False,
    )
    mesh = None
    line_styles = ["dashed" if level < 0 else "solid" for level in contour_levels]
    line_widths = [1.2 if np.isclose(level, 0.0) else 0.7 for level in contour_levels]
    for row, group in enumerate(GROUPS):
        group_panel = plot_ds.sel({GROUP_DIM: group})
        count = int(group_panel["event_count"].item())
        dyn_mean = float(group_panel[mean_variable].item())
        for column, lag in enumerate(lags):
            ax = axes[row, column]
            panel = group_panel.sel({LAG_DIM: lag})
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
            ax.clabel(contours, fmt="%g", fontsize=6, inline_spacing=1)
            decorate_map(
                ax,
                data_crs,
                left_labels=column == 0,
                bottom_labels=row == len(GROUPS) - 1,
            )
            if row == 0:
                ax.set_title(format_lag_title(lag))
            if column == 0:
                ax.text(
                    -0.24,
                    0.5,
                    (
                        f"{group.capitalize()} {group_metric_label}\n"
                        f"n = {count}\nmean = {dyn_mean:.2f} K"
                    ),
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=9,
                )

    assert mesh is not None
    colorbar = fig.colorbar(
        mesh,
        ax=axes.ravel().tolist(),
        orientation="horizontal",
        fraction=0.055,
        pad=0.055,
        aspect=55,
    )
    colorbar.set_label("2 m temperature anomaly (K)")
    fig.suptitle(figure_title)
    return fig


def select_plot_lags(ds: xr.Dataset, requested: Sequence[int]) -> np.ndarray:
    if not requested:
        raise ValueError("At least one plot lag is required.")
    lags = np.asarray(requested, dtype=int)
    if np.unique(lags).size != lags.size:
        raise ValueError("Plot lags must be unique.")
    available = set(np.asarray(ds[LAG_DIM].values, dtype=int).tolist())
    missing = [int(lag) for lag in lags if int(lag) not in available]
    if missing:
        raise ValueError(
            "Requested plot lags are absent from the composite: "
            + ", ".join(str(lag) for lag in missing)
        )
    return lags


def decorate_map(
    ax,
    data_crs: ccrs.PlateCarree,
    *,
    left_labels: bool = True,
    bottom_labels: bool = True,
) -> None:
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
    gridlines.left_labels = left_labels
    gridlines.bottom_labels = bottom_labels
    gridlines.xlabel_style = {"size": 7}
    gridlines.ylabel_style = {"size": 7}
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


def format_lag_title(lag: int) -> str:
    if lag == 0:
        return r"$t$ (peak)"
    return rf"$t{lag:+d}$"


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
