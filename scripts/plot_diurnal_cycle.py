"""Plot local-time diurnal-cycle diagnostics from the Stage-1 time series."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io


DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "plots_diurnal_cycle"
    / "hw_non_hw_diurnal_cycle_jja_local.png"
)
DEFAULT_SEASON_MONTHS: tuple[int, ...] = (6, 7, 8)
DEFAULT_LOCAL_UTC_OFFSET_HOURS = -7
DIURNAL_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
)
DIURNAL_QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75)
HW_CLASS_DIM = "hw_class"
HW_CLASS_LABELS: tuple[str, ...] = ("Heatwave days", "Non-heatwave days")
LOCAL_HOURS = np.arange(24, dtype=np.int64)
SAMPLE_PERCENTILE_PREFIX = "sample_percentile_"
VARIABLE_COLORS = {
    "T_mean": "tab:red",
    "volume": "tab:blue",
    "dTdt": "tab:purple",
    "advection": "tab:orange",
    "adiabatic": "tab:green",
    "diabatic": "tab:brown",
    "lwa_a_region": "tab:olive",
    "lwa_c_region": "tab:cyan",
}
CLASS_LINESTYLES = {
    "Heatwave days": "-",
    "Non-heatwave days": "--",
}

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
})


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the diurnal-cycle diagnostic."""
    parser = argparse.ArgumentParser(
        description="Plot local-time HW and non-HW diurnal cycles for summer days."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
        help="Path to the saved harmonized Stage-1 regional dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the diurnal-cycle PNG will be written.",
    )
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=list(DEFAULT_SEASON_MONTHS),
        metavar="MONTH",
        help="Local-time calendar months to retain, default: 6 7 8.",
    )
    parser.add_argument(
        "--local-utc-offset-hours",
        type=int,
        default=DEFAULT_LOCAL_UTC_OFFSET_HOURS,
        help="Fixed local-time offset from UTC in hours, default: -7 for PDT.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate diurnal-cycle CLI arguments."""
    _validate_season_months(args.season_months)
    _validate_local_utc_offset_hours(args.local_utc_offset_hours)


def build_diurnal_composite(
    ds: xr.Dataset,
    *,
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
    local_utc_offset_hours: int = DEFAULT_LOCAL_UTC_OFFSET_HOURS,
    variables: Sequence[str] = DIURNAL_VARIABLES,
    quantiles: Sequence[float] = DIURNAL_QUANTILES,
    time_dim: str = "time",
    hw_event_id_name: str = "hw_event_id",
) -> xr.Dataset:
    """Return HW and non-HW local-hour means plus sample IQR traces."""
    months = _validate_season_months(season_months)
    offset = _validate_local_utc_offset_hours(local_utc_offset_hours)
    qs = _validate_quantiles(quantiles)
    _validate_diurnal_inputs(
        ds,
        variables=variables,
        time_dim=time_dim,
        hw_event_id_name=hw_event_id_name,
    )

    # Classify HW/non-HW samples first in the dataset's native time convention
    # (GMT/UTC), then shift the selected samples onto local-hour coordinates.
    native_month = ds[time_dim].dt.month
    season_mask = native_month.isin(months)
    hw_event_id = ds[hw_event_id_name].fillna(0)

    class_masks = (
        season_mask & (hw_event_id > 0),
        season_mask & (hw_event_id == 0),
    )

    # Shift only after the native-time masks have been defined.

    local_times = utc_to_local_time_values(ds[time_dim], offset)
    local_index = pd.DatetimeIndex(local_times)
    local_hour = xr.DataArray(
        np.asarray(local_index.hour, dtype=np.int64),
        dims=(time_dim,),
        coords={time_dim: ds[time_dim]},
        name="local_hour",
    )
    # local_month = xr.DataArray(
    #     np.asarray(local_index.month, dtype=np.int64),
    #     dims=(time_dim,),
    #     coords={time_dim: ds[time_dim]},
    #     name="local_month",
    # )

    # season_mask = local_month.isin(months)
    # hw_event_id = ds[hw_event_id_name].fillna(0)
    # class_masks = (
    #     season_mask & (hw_event_id > 0),
    #     season_mask & (hw_event_id == 0),
    # )

    source = ds[list(variables)].assign_coords(
        local_hour=local_hour,
        local_time=(time_dim, local_times),
    )

    class_composites: list[xr.Dataset] = []
    class_sample_counts: list[int] = []
    for label, mask in zip(HW_CLASS_LABELS, class_masks, strict=True):
        selected = source.where(mask, drop=True)
        sample_count = int(selected.sizes.get(time_dim, 0))
        if sample_count == 0:
            raise ValueError(f"No {label.lower()} samples remain after local-season filtering.")
        class_composites.append(
            _build_one_class_diurnal_composite(
                selected,
                variables=variables,
                quantiles=qs,
                time_dim=time_dim,
            )
        )
        class_sample_counts.append(sample_count)
        count_by_hour = selected["volume"].groupby("local_hour").count()
        print(label)
        print(count_by_hour.values)

    composite = xr.concat(
        class_composites,
        dim=xr.IndexVariable(HW_CLASS_DIM, list(HW_CLASS_LABELS)),
    )
    composite = composite.assign_coords(
        class_sample_count=(
            HW_CLASS_DIM,
            np.asarray(class_sample_counts, dtype=np.int64),
        )
    )
    composite.attrs.update(
        {
            "composite_reduction": "mean by local hour and heatwave class",
            "season_months": " ".join(str(month) for month in months),
            "local_utc_offset_hours": int(offset),
            "local_timezone_label": _utc_offset_label(offset),
            "sample_percentiles": ", ".join(str(float(q)) for q in qs),
            "sample_percentile_prefix": SAMPLE_PERCENTILE_PREFIX,
            "n_hw_samples": int(class_sample_counts[0]),
            "n_non_hw_samples": int(class_sample_counts[1]),
        }
    )
    for attr_name in ("region", "threshold_variable", "quantile", "start_year", "end_year"):
        if attr_name in ds.attrs:
            composite.attrs[attr_name] = ds.attrs[attr_name]
    return composite


def utc_to_local_time_values(
    time: xr.DataArray,
    local_utc_offset_hours: int = DEFAULT_LOCAL_UTC_OFFSET_HOURS,
) -> np.ndarray:
    """Return timestamp values shifted from UTC/GMT to fixed local time."""
    offset = _validate_local_utc_offset_hours(local_utc_offset_hours)
    values = np.asarray(time.values, dtype="datetime64[ns]")
    return values + np.timedelta64(offset, "h")


def plot_diurnal_cycle(composite: xr.Dataset) -> Figure:
    """Return a four-panel HW/non-HW local diurnal-cycle figure."""
    _validate_plot_composite(composite)
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 10),
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
        ax.grid(True, linewidth=0.5, alpha=0.35)
    ax3.set_xlabel("Local hour")

    fig.suptitle(_figure_title(composite), fontsize=18)
    return fig


def write_diurnal_cycle_plot(composite: xr.Dataset, output_path: Path) -> Path:
    """Write the local diurnal-cycle diagnostic figure."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_diurnal_cycle(composite)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def main() -> int:
    """Open the harmonized dataset and write the diurnal-cycle figure."""
    args = parse_args()
    validate_args(args)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        composite = build_diurnal_composite(
            ds,
            season_months=args.season_months,
            local_utc_offset_hours=args.local_utc_offset_hours,
        )
        written = write_diurnal_cycle_plot(composite, args.output_path)
        print("Wrote HW/non-HW local diurnal-cycle figure:")
        print(f"  {_display_path(written)}")
    finally:
        ds.close()
    return 0


def _build_one_class_diurnal_composite(
    selected: xr.Dataset,
    *,
    variables: Sequence[str],
    quantiles: tuple[float, ...],
    time_dim: str,
) -> xr.Dataset:
    """Build one class's local-hour mean and percentile traces."""
    data_vars: dict[str, xr.DataArray] = {}
    for name in variables:
        grouped = selected[name].groupby("local_hour")
        data_vars[name] = grouped.mean(time_dim, skipna=True).reindex(
            local_hour=LOCAL_HOURS
        )
        data_vars[f"{SAMPLE_PERCENTILE_PREFIX}{name}"] = (
            grouped.quantile(quantiles, dim=time_dim, skipna=True)
            .reindex(local_hour=LOCAL_HOURS)
            .transpose("quantile", "local_hour")
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "local_hour": LOCAL_HOURS,
            "quantile": np.asarray(quantiles, dtype=float),
        },
    )


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
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[_variable_legend_handle(name)], loc="upper left")


def _plot_tendency_panel(ax: Axes, composite: xr.Dataset) -> None:
    """Plot heat-budget tendency terms by HW class."""
    for name in ("advection", "adiabatic", "diabatic"):
        _plot_class_lines(ax, composite, name, color=VARIABLE_COLORS[name])
    ax.axhline(0, color="0.2", linewidth=1.0)
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
            _variable_legend_handle(name)
            for name in ("lwa_a_region", "lwa_c_region")
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
            linewidth=1.8,
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
            linewidth=1.0,
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
            color="0.2",
            linestyle=_class_linestyle(label),
            label=label,
        )
        for label in HW_CLASS_LABELS
    ]
    handles.append(
        Line2D([0], [0], color="0.2", linestyle="-", alpha=0.28, label="IQR bounds")
    )
    ax.legend(handles=handles, loc="upper right")


def _variable_legend_handle(name: str) -> Line2D:
    """Return a solid-line variable legend handle."""
    return Line2D([0], [0], color=VARIABLE_COLORS[name], linestyle="-", label=name)


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
    offset = composite.attrs.get("local_timezone_label", "UTC-7")
    hw_count = int(composite.attrs.get("n_hw_samples", 0))
    non_hw_count = int(composite.attrs.get("n_non_hw_samples", 0))
    return (
        f"Local diurnal cycle, {region}, months {season} ({offset}); "
        f"HW n={hw_count//24}, non-HW n={non_hw_count//24}"
    )


def _class_labels(composite: xr.Dataset) -> list[str]:
    """Return class labels from the composite coordinate."""
    return [str(value) for value in composite[HW_CLASS_DIM].values]


def _class_linestyle(label: str) -> str:
    """Return the linestyle for one HW class."""
    return CLASS_LINESTYLES.get(label, "-")


def _validate_diurnal_inputs(
    ds: xr.Dataset,
    *,
    variables: Sequence[str],
    time_dim: str,
    hw_event_id_name: str,
) -> None:
    """Validate required dataset variables and dimensions."""
    if time_dim not in ds.coords:
        raise ValueError(f"Dataset is missing required time coordinate {time_dim!r}.")
    if hw_event_id_name not in ds:
        raise ValueError(f"Dataset is missing event-ID variable {hw_event_id_name!r}.")
    missing = sorted(name for name in variables if name not in ds)
    if missing:
        raise ValueError(f"Dataset is missing requested variables: {', '.join(missing)}.")
    invalid = sorted(name for name in variables if time_dim not in ds[name].dims)
    if invalid:
        raise ValueError(
            "Requested variables must contain the time dimension; invalid: "
            f"{', '.join(invalid)}."
        )
    if time_dim not in ds[hw_event_id_name].dims:
        raise ValueError(f"{hw_event_id_name!r} must contain dimension {time_dim!r}.")


def _validate_plot_composite(composite: xr.Dataset) -> None:
    """Validate the minimum composite contract needed for plotting."""
    for dim in (HW_CLASS_DIM, "local_hour"):
        if dim not in composite.dims:
            raise ValueError(f"Composite is missing dimension {dim!r}.")
    missing = sorted(name for name in DIURNAL_VARIABLES if name not in composite)
    if missing:
        raise ValueError(f"Composite is missing variables: {', '.join(missing)}.")


def _validate_season_months(season_months: Sequence[int]) -> tuple[int, ...]:
    """Return validated month numbers with duplicates removed in input order."""
    if isinstance(season_months, (str, bytes)):
        raise TypeError("season_months must be a sequence of integer month numbers.")

    months: list[int] = []
    for month in season_months:
        if isinstance(month, (bool, np.bool_)) or not isinstance(month, (int, np.integer)):
            raise ValueError("season_months must contain only integer month numbers.")
        month_int = int(month)
        if month_int < 1 or month_int > 12:
            raise ValueError("--season-months values must be between 1 and 12.")
        if month_int not in months:
            months.append(month_int)
    if not months:
        raise ValueError("season_months must contain at least one month.")
    return tuple(months)


def _validate_local_utc_offset_hours(value: int) -> int:
    """Return a validated fixed UTC offset in whole hours."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("--local-utc-offset-hours must be an integer.")
    offset = int(value)
    if offset < -23 or offset > 23:
        raise ValueError("--local-utc-offset-hours must be between -23 and 23.")
    return offset


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Return sorted unique quantile values in [0, 1]."""
    values = tuple(sorted({float(value) for value in quantiles}))
    if not values:
        raise ValueError("quantiles must contain at least one value.")
    invalid = [value for value in values if value < 0.0 or value > 1.0]
    if invalid:
        text = ", ".join(str(value) for value in invalid)
        raise ValueError(f"quantiles must be between 0 and 1; got {text}.")
    return values


def _utc_offset_label(offset: int) -> str:
    """Return display label for a fixed UTC offset."""
    if offset == 0:
        return "UTC+0"
    sign = "+" if offset > 0 else "-"
    return f"UTC{sign}{abs(offset)}"


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
