"""Plot GMT diurnal-cycle diagnostics from a prepared Stage-1 time series."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from uuid import uuid4

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, diurnal, plot_paths, plot_style
from src.artifact_paths import artifact_root
from src.diurnal import (
    DEFAULT_LOCAL_UTC_OFFSET_HOURS,
    DEFAULT_SEASON_MONTHS,
    DIURNAL_VARIABLES,
    HW_CLASS_DIM,
    HW_CLASS_LABELS,
    SAMPLE_PERCENTILE_PREFIX,
    _validate_local_utc_offset_hours,
    _validate_season_months,
    build_diurnal_composite,
)

# Preserve the original script's public numerical API for existing callers.
DIURNAL_QUANTILES = diurnal.DIURNAL_QUANTILES
LOCAL_HOURS = diurnal.LOCAL_HOURS
utc_to_local_time_values = diurnal.utc_to_local_time_values

PLOT_NAME = "diurnal_cycle"
DEFAULT_OUTPUT_FILENAME = "hw_non_hw_diurnal_cycle_jja_GMT.png"
DEFAULT_OUTPUT_PATH = artifact_root() / f"plots_{PLOT_NAME}" / DEFAULT_OUTPUT_FILENAME
VARIABLE_COLORS = plot_style.VARIABLE_COLORS
CLASS_LINESTYLES = {
    "Heatwave days": "-",
    "Non-heatwave days": "--",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the diurnal-cycle diagnostic."""
    parser = argparse.ArgumentParser(
        description="Plot GMT HW and non-HW diurnal cycles for summer days."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--composite-output-path",
        type=Path,
        default=None,
        help="Optional new NetCDF path for plotted values, sample counts and input hash.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path where the diurnal-cycle PNG will be written.",
    )
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEASON_MONTHS),
        metavar="MONTH",
        help="Native-timestamp calendar months retained before hour grouping (default: 6 7 8).",
    )
    parser.add_argument(
        "--local-utc-offset-hours",
        type=int,
        default=DEFAULT_LOCAL_UTC_OFFSET_HOURS,
        help="Hour-label offset from UTC (default: 0, GMT); stored GMT HW days are unchanged.",
    )
    args = parser.parse_args()
    months = _validate_season_months(args.season_months)
    offset = _validate_local_utc_offset_hours(args.local_utc_offset_hours)
    season_token = (
        "jja"
        if months == DEFAULT_SEASON_MONTHS
        else "months_" + "_".join(map(str, months))
    )
    time_token = "GMT" if offset == 0 else f"UTC{offset:+d}"
    return plot_paths.finalize_stage1_plot_paths(
        args,
        parser,
        plot_name=PLOT_NAME,
        default_output_filename=f"hw_non_hw_diurnal_cycle_{season_token}_{time_token}.png",
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate diurnal-cycle CLI arguments."""
    _validate_season_months(args.season_months)
    _validate_local_utc_offset_hours(args.local_utc_offset_hours)
    if (
        args.composite_output_path is not None
        and args.output_path.resolve() == args.composite_output_path.resolve()
    ):
        raise ValueError("Figure and composite output paths must be different.")


def plot_diurnal_cycle(composite: xr.Dataset) -> Figure:
    """Return a four-panel HW/non-HW diurnal-cycle figure."""
    _validate_plot_composite(composite)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=plot_style.publication_figsize("full", aspect=0.85),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2, ax3 = axes

    _plot_temperature_volume_panel(ax0, composite)
    _plot_single_variable_panel(ax1, composite, "dTdt", ylabel="[K hr-1]")
    _plot_tendency_panel(ax2, composite)
    _plot_lwa_panel(ax3, composite)

    for ax in axes:
        ax.set_xlim(0, 23)
        ax.set_xticks(np.arange(0, 24, 3))
    offset = int(composite.attrs.get("local_utc_offset_hours", 0))
    ax3.set_xlabel(
        "GMT hour (UTC+0)" if offset == 0 else f"Local hour (UTC{offset:+d})"
    )

    fig.suptitle(_figure_title(composite))
    plot_style.style_axes(axes)
    return fig


def write_diurnal_cycle_plot(composite: xr.Dataset, output_path: Path) -> Path:
    """Publish a complete diagnostic figure without replacing an existing path."""
    output_path = _require_fresh_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f".{output_path.stem}.{uuid4().hex}{output_path.suffix}"
    )
    fig = plot_diurnal_cycle(composite)
    try:
        plot_style.save_figure(fig, temporary_path)
        os.link(temporary_path, output_path)
    finally:
        plt.close(fig)
        temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> int:
    """Open the harmonized dataset and write the diurnal-cycle figure."""
    args = parse_args()
    validate_args(args)
    _require_fresh_path(args.output_path)
    if args.composite_output_path is not None:
        _require_fresh_path(args.composite_output_path)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        composite = build_diurnal_composite(
            ds,
            season_months=args.season_months,
            local_utc_offset_hours=args.local_utc_offset_hours,
        )
        if args.composite_output_path is not None:
            with args.input_path.open("rb") as handle:
                source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
            composite.attrs.update(
                source_path=str(args.input_path.resolve()),
                source_sha256=source_hash,
                producer_commit=os.environ.get("EXPECTED_COMMIT", "unrecorded"),
            )
            analysis_io.save_diurnal_composite(composite, args.composite_output_path)
            print(f"Wrote numerical diurnal composite: {args.composite_output_path}")
        written = write_diurnal_cycle_plot(composite, args.output_path)
        print("Wrote HW/non-HW diurnal-cycle figure:")
        print(f"  {_display_path(written)}")
    finally:
        ds.close()
    return 0


def _plot_temperature_volume_panel(ax: Axes, composite: xr.Dataset) -> None:
    """Plot T_mean and volume with separate y axes."""
    _plot_class_lines(ax, composite, "T_mean", color=VARIABLE_COLORS["T_mean"])
    ax.set_ylabel("T_mean [K]", color=VARIABLE_COLORS["T_mean"])
    ax.tick_params(axis="y", labelcolor=VARIABLE_COLORS["T_mean"])

    ax_volume = ax.twinx()
    _plot_class_lines(ax_volume, composite, "volume", color=VARIABLE_COLORS["volume"])
    ax_volume.set_ylabel("volume [m2 Pa]", color=VARIABLE_COLORS["volume"])
    ax_volume.tick_params(axis="y", labelcolor=VARIABLE_COLORS["volume"])

    ax.legend(
        handles=[
            _variable_legend_handle("T_mean"),
            _variable_legend_handle("volume"),
        ],
        loc="upper left",
    )
    _add_class_legend(ax_volume)


def _plot_single_variable_panel(
    ax: Axes,
    composite: xr.Dataset,
    name: str,
    *,
    ylabel: str,
) -> None:
    """Plot one variable by HW class."""
    _plot_class_lines(ax, composite, name, color=VARIABLE_COLORS[name])
    plot_style.zero_line(ax)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[_variable_legend_handle(name)], loc="upper left")


def _plot_tendency_panel(ax: Axes, composite: xr.Dataset) -> None:
    """Plot heat-budget tendency terms by HW class."""
    for name in ("advection", "adiabatic", "diabatic"):
        _plot_class_lines(ax, composite, name, color=VARIABLE_COLORS[name])
    plot_style.zero_line(ax)
    ax.set_ylabel("[K hr-1]")
    _expand_yaxis(ax, factor=1.5)
    ax.legend(
        handles=[
            _variable_legend_handle(name)
            for name in ("advection", "adiabatic", "diabatic")
        ],
        loc="upper left",
        ncol=3,
    )


def _plot_lwa_panel(ax: Axes, composite: xr.Dataset) -> None:
    """Plot LWA_a and LWA_c regional diurnal cycles by HW class."""
    for name in ("lwa_a_region", "lwa_c_region"):
        _plot_class_lines(ax, composite, name, color=VARIABLE_COLORS[name])
    ax.set_ylabel("LWA [m hPa]")
    ax.legend(
        handles=[
            _variable_legend_handle(name) for name in ("lwa_a_region", "lwa_c_region")
        ],
        loc="upper left",
    )


def _plot_class_lines(
    ax: Axes,
    composite: xr.Dataset,
    name: str,
    *,
    color: str,
) -> None:
    """Plot class mean traces plus faint IQR bound lines for one variable."""
    x = composite["local_hour"].values
    for class_label in _class_labels(composite):
        subset = composite.sel({HW_CLASS_DIM: class_label})
        linestyle = _class_linestyle(class_label)
        ax.plot(
            x,
            subset[name].values,
            color=color,
            linestyle=linestyle,
            linewidth=plot_style.LINE_WIDTH_PT,
        )
        _plot_iqr_bound_lines(
            ax,
            x,
            subset,
            name,
            color=color,
            linestyle=linestyle,
        )


def _plot_iqr_bound_lines(
    ax: Axes,
    x: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
    linestyle: str,
) -> None:
    """Draw sample IQR bounds as faint lines."""
    bounds = _sample_percentile_bounds(ds, name)
    if bounds is None:
        return

    for bound in bounds:
        ax.plot(
            x,
            bound.values,
            color=color,
            linestyle=linestyle,
            alpha=0.28,
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        )


def _sample_percentile_bounds(
    ds: xr.Dataset,
    name: str,
) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Return 25th and 75th percentile local-hour traces for one variable."""
    envelope_name = f"{SAMPLE_PERCENTILE_PREFIX}{name}"
    if envelope_name not in ds or "quantile" not in ds[envelope_name].dims:
        return None

    lower = _select_quantile(ds[envelope_name], 0.25)
    upper = _select_quantile(ds[envelope_name], 0.75)
    if lower is None or upper is None:
        return None
    return lower, upper


def _select_quantile(da: xr.DataArray, quantile: float) -> xr.DataArray | None:
    """Return a quantile slice when that quantile is present."""
    quantiles = np.asarray(da["quantile"].values, dtype=float)
    matches = np.flatnonzero(np.isclose(quantiles, quantile))
    if matches.size == 0:
        return None
    return da.isel(quantile=int(matches[0]))


def _add_class_legend(ax: Axes) -> None:
    """Add HW/non-HW linestyle legend plus IQR-bound hint."""
    handles = [
        Line2D(
            [0],
            [0],
            color=plot_style.COLORS["zero"],
            linestyle=_class_linestyle(label),
            label=label,
        )
        for label in HW_CLASS_LABELS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=plot_style.COLORS["zero"],
            linestyle="-",
            alpha=0.28,
            label="IQR bounds",
        )
    )
    ax.legend(handles=handles, loc="upper right")


def _variable_legend_handle(name: str) -> Line2D:
    """Return a solid-line variable legend handle."""
    return Line2D(
        [0],
        [0],
        color=VARIABLE_COLORS[name],
        linestyle="-",
        label=plot_style.VARIABLE_NAME_MAPPING.get(name, name),
    )


def _expand_yaxis(ax: Axes, *, factor: float) -> None:
    """Expand an axis y-range around its current center by a scale factor."""
    lower, upper = ax.get_ylim()
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower) * factor
    ax.set_ylim(center - half_range, center + half_range)


def _figure_title(composite: xr.Dataset) -> str:
    """Return a compact title for the diurnal-cycle diagnostic."""
    region = composite.attrs.get("region", "PNW")
    season = composite.attrs.get("season_months", "6 7 8")
    offset = int(composite.attrs.get("local_utc_offset_hours", 0))
    clock = "GMT (UTC+0)" if offset == 0 else f"UTC{offset:+d}"
    bottom = str(composite.attrs.get("heat_budget_bottom_boundary", "unspecified"))
    top = str(composite.attrs.get("heat_budget_top_boundary", "unspecified"))
    layer = f"{bottom.removesuffix('hPa')}-{top.removesuffix('hPa')} hPa"
    start = composite.attrs.get("start_year", "?")
    end = composite.attrs.get("end_year", "?")
    if "n_hw_days" in composite.attrs and "n_non_hw_days" in composite.attrs:
        counts = f"HW days={composite.attrs['n_hw_days']}, non-HW days={composite.attrs['n_non_hw_days']}"
    else:
        counts = f"HW hours={composite.attrs.get('n_hw_samples', 0)}, non-HW hours={composite.attrs.get('n_non_hw_samples', 0)}"
    return (
        f"{region}, {layer}, {start}-{end}\n"
        f"Diurnal cycle, months {season}, {clock}; {counts}"
    )


def _class_labels(composite: xr.Dataset) -> list[str]:
    """Return class labels from the composite coordinate."""
    return [str(value) for value in composite[HW_CLASS_DIM].values]


def _class_linestyle(label: str) -> str:
    """Return the linestyle for one HW class."""
    return CLASS_LINESTYLES.get(label, "-")


def _validate_plot_composite(composite: xr.Dataset) -> None:
    """Validate the minimum composite contract needed for plotting."""
    for dim in (HW_CLASS_DIM, "local_hour"):
        if dim not in composite.dims:
            raise ValueError(f"Composite is missing dimension {dim!r}.")
    missing = sorted(name for name in DIURNAL_VARIABLES if name not in composite)
    if missing:
        raise ValueError(f"Composite is missing variables: {', '.join(missing)}.")


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _require_fresh_path(path: Path) -> Path:
    """Reject existing files and symlinks before constructing an output."""
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Output already exists: {path}")
    return path


if __name__ == "__main__":
    raise SystemExit(main())
