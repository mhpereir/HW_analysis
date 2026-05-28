"""Build fixed-window event-level feature tables from Stage-1 products."""

from __future__ import annotations

import argparse
import sys
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.event_features import event_feature_config as config
from src import analysis_io, selectors


SURFACE_FLUX_FEATURES = frozenset({"I_sshf_pre", "I_slhf_pre"})


def parse_args() -> argparse.Namespace:
    """Parse fixed-window feature extraction CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build event-level fixed-window features from a Stage-1 dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=config.DEFAULT_INPUT_PATH,
        help="Stage-1 harmonized regional time-series dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=config.DEFAULT_OUTPUT_PATH,
        help="NetCDF feature-table output path.",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=None,
        help="Optional CSV feature-table output path.",
    )
    parser.add_argument(
        "--use-extended-variables",
        action="store_true",
        help="Compute optional land, cloud, PBL, and surface-energy features.",
    )
    parser.add_argument(
        "--allow-missing-extended",
        action="store_true",
        help="Skip unavailable extended variables instead of failing.",
    )
    season = parser.add_mutually_exclusive_group(required=True)
    season.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Calendar months to retain before feature extraction, e.g. 6 7 8.",
    )
    season.add_argument(
        "--all-seasons",
        action="store_true",
        help="Use every event in the Stage-1 event summary table.",
    )
    parser.add_argument(
        "--require-full-event",
        action="store_true",
        help="Require event start/end months to fall within --season-months.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow output files to replace existing files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments that argparse cannot express cleanly."""
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if args.allow_missing_extended and not args.use_extended_variables:
        raise ValueError("--allow-missing-extended requires --use-extended-variables.")
    if args.season_months is not None:
        _validate_season_months(args.season_months)
    _validate_output_path(args.output_path, overwrite=args.overwrite)
    if args.csv_output_path is not None:
        _validate_output_path(args.csv_output_path, overwrite=args.overwrite)


def build_event_features(
    ds: xr.Dataset,
    *,
    use_extended_variables: bool = False,
    allow_missing_extended: bool = False,
    season_months: Sequence[int] | None = None,
    all_seasons: bool = False,
    require_full_event: bool = False,
    input_path: str | Path | None = None,
) -> xr.Dataset:
    """Return one event-level fixed-window feature table."""
    if not all_seasons and season_months is None:
        raise ValueError("Either season_months or all_seasons=True is required.")
    if all_seasons and season_months is not None:
        raise ValueError("Pass season_months or all_seasons=True, not both.")

    ds = ensure_tas_anom(ds)
    event_table = event_summary_table(ds)
    if season_months is not None:
        event_table = selectors.select_events_by_season(
            event_table,
            season_months,
            require_full_event=require_full_event,
        )
    if event_table.sizes.get(config.EVENT_DIM, 0) == 0:
        raise ValueError("No events remain after event-universe selection.")

    feature_spec = active_feature_spec(
        ds,
        use_extended_variables=use_extended_variables,
        allow_missing_extended=allow_missing_extended,
    )
    validate_required_variables(ds, event_table, feature_spec)
    event_table = require_finite_peak_times(event_table)
    event_table, dropped_boundary_events = drop_boundary_events(
        event_table,
        ds,
        active_window_names(feature_spec),
    )

    out = xr.Dataset(coords={config.EVENT_DIM: event_table[config.EVENT_DIM]})
    copy_event_summary_features(out, event_table)
    add_sample_count_features(out, ds, event_table, active_window_names(feature_spec))
    add_integral_features(out, ds, event_table, feature_spec["integral"])
    add_mean_features(out, ds, event_table, feature_spec["mean"])
    add_change_features(out, ds, event_table, feature_spec["change"])
    add_days_from_solstice(out, event_table)
    add_global_attrs(
        out,
        input_path=input_path,
        use_extended_variables=use_extended_variables,
        allow_missing_extended=allow_missing_extended,
        season_months=season_months,
        all_seasons=all_seasons,
        require_full_event=require_full_event,
        dropped_boundary_events=dropped_boundary_events,
        feature_spec=feature_spec,
    )
    return out


def event_summary_table(ds: xr.Dataset, event_dim: str = config.EVENT_DIM) -> xr.Dataset:
    """Return variables that belong only to the event dimension."""
    names = [
        name
        for name, da in ds.data_vars.items()
        if event_dim in da.dims and set(da.dims).issubset({event_dim})
    ]
    if not names:
        raise ValueError("Input dataset contains no event-summary variables.")
    return ds[names]


def ensure_tas_anom(ds: xr.Dataset) -> xr.Dataset:
    """Return a view with tas_anom available for feature extraction."""
    if "tas_anom" in ds:
        return ds
    if "tas_region" not in ds or "tas_climatology" not in ds:
        return ds
    out = ds.copy(deep=False)
    out["tas_anom"] = out["tas_region"] - out["tas_climatology"]
    out["tas_anom"].attrs.update(
        {
            "description": "Derived tas_region minus tas_climatology.",
            "source_variables": "tas_region,tas_climatology",
        }
    )
    if "units" in out["tas_region"].attrs:
        out["tas_anom"].attrs["units"] = out["tas_region"].attrs["units"]
    return out


def active_feature_spec(
    ds: xr.Dataset,
    *,
    use_extended_variables: bool,
    allow_missing_extended: bool,
) -> dict[str, dict[str, str]]:
    """Return active source-variable to window-name feature mappings."""
    spec = {
        "integral": dict(config.DEFAULT_INTEGRAL_FEATURES),
        "mean": dict(config.DEFAULT_MEAN_FEATURES),
        "change": {},
    }
    if not use_extended_variables:
        return spec

    extended = {
        "integral": dict(config.EXTENDED_INTEGRAL_FEATURES),
        "mean": dict(config.EXTENDED_MEAN_FEATURES),
        "change": dict(config.EXTENDED_CHANGE_FEATURES),
    }
    missing = sorted({
        name
        for group in extended.values()
        for name in group
        if name not in ds
    })
    if missing and not allow_missing_extended:
        raise ValueError(
            "Input dataset is missing required extended variables: "
            f"{', '.join(missing)}."
        )
    if missing:
        warnings.warn(
            "Skipping missing extended variables: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )

    for operation, mapping in extended.items():
        for name, window_name in mapping.items():
            if name in ds:
                spec[operation][name] = window_name
    return spec


def validate_required_variables(
    ds: xr.Dataset,
    event_table: xr.Dataset,
    feature_spec: Mapping[str, Mapping[str, str]],
) -> None:
    """Fail clearly when required source or event-summary variables are absent."""
    missing_time = sorted(
        name
        for operation in ("integral", "mean", "change")
        for name in feature_spec[operation]
        if name not in ds
    )
    missing_event = sorted(
        name for name in config.EVENT_SUMMARY_FEATURES if name not in event_table
    )
    missing = []
    if missing_time:
        missing.append("time-indexed variables: " + ", ".join(missing_time))
    if missing_event:
        missing.append("event-summary variables: " + ", ".join(missing_event))
    if missing:
        raise ValueError("Input dataset is missing required " + "; ".join(missing) + ".")


def require_finite_peak_times(event_table: xr.Dataset) -> xr.Dataset:
    """Return event table after verifying every selected event has a peak time."""
    peak = event_table[config.PEAK_TIME_NAME]
    values = np.asarray(peak.compute().values, dtype="datetime64[ns]")
    if np.isnat(values).any():
        raise ValueError("All selected events must have finite peak_time values.")
    return event_table


def drop_boundary_events(
    event_table: xr.Dataset,
    ds: xr.Dataset,
    window_names: Sequence[str],
) -> tuple[xr.Dataset, int]:
    """Drop events whose active feature windows are outside the dataset time range."""
    time_values = np.asarray(ds[config.TIME_DIM].values, dtype="datetime64[ns]")
    if time_values.size == 0:
        raise ValueError("Input dataset has an empty time coordinate.")
    data_start = time_values[0]
    data_end = time_values[-1]
    peak_times = np.asarray(
        event_table[config.PEAK_TIME_NAME].compute().values,
        dtype="datetime64[ns]",
    )

    keep = np.ones(peak_times.shape, dtype=bool)
    for window_name in window_names:
        start_lag, end_lag = config.WINDOWS[window_name]
        starts = peak_times + np.timedelta64(start_lag, "h")
        ends = peak_times + np.timedelta64(end_lag, "h")
        keep &= (starts >= data_start) & (ends <= data_end)

    dropped = int((~keep).sum())
    out = event_table.isel({config.EVENT_DIM: np.flatnonzero(keep)})
    if out.sizes.get(config.EVENT_DIM, 0) == 0:
        raise ValueError("No events have complete required feature windows.")
    return out, dropped


def copy_event_summary_features(out: xr.Dataset, event_table: xr.Dataset) -> None:
    """Copy selected event-summary variables to the output dataset."""
    for name in config.EVENT_SUMMARY_FEATURES:
        da = event_table[name]
        out[name] = da
        out[name].attrs = dict(da.attrs)
        out[name].attrs["operation"] = "copy"


def add_sample_count_features(
    out: xr.Dataset,
    ds: xr.Dataset,
    event_table: xr.Dataset,
    window_names: Sequence[str],
) -> None:
    """Add one timestamp-count diagnostic per active feature window."""
    for window_name in window_names:
        values = [
            int(window_for_peak(ds, peak_time, window_name).sizes.get(config.TIME_DIM, 0))
            for peak_time in event_peak_values(event_table)
        ]
        feature_name = f"n_samples_{window_name}"
        out[feature_name] = (config.EVENT_DIM, np.asarray(values, dtype=np.int64))
        add_feature_attrs(
            out[feature_name],
            source_variable=config.TIME_DIM,
            window_name=window_name,
            operation="count",
            units="samples",
        )


def add_integral_features(
    out: xr.Dataset,
    ds: xr.Dataset,
    event_table: xr.Dataset,
    integral_features: Mapping[str, str],
) -> None:
    """Add hourly-sum integral/exposure features."""
    for source_name, window_name in integral_features.items():
        feature_name = feature_name_for_source(source_name, operation="integral")
        values = [
            nansum_or_nan(window_for_peak(ds, peak_time, window_name)[source_name].values)
            for peak_time in event_peak_values(event_table)
        ]
        out[feature_name] = (config.EVENT_DIM, np.asarray(values, dtype=float))
        source_units = ds[source_name].attrs.get("units")
        if source_units == "K hr-1":
            out[feature_name].attrs["units"] = "K"
        elif source_units is not None:
            out[feature_name].attrs["units"] = f"{source_units} hr"
        add_feature_attrs(
            out[feature_name],
            source_variable=source_name,
            window_name=window_name,
            operation="sum",
        )
        out[feature_name].attrs["integral_method"] = config.INTEGRAL_METHOD
        if source_name in {"lwa_a_region", "lwa_c_region"}:
            out[feature_name].attrs["description"] = "LWA exposure over fixed window."
        if feature_name in SURFACE_FLUX_FEATURES:
            out[feature_name].attrs["sign_convention"] = "native Stage-1/source signs retained"


def add_mean_features(
    out: xr.Dataset,
    ds: xr.Dataset,
    event_table: xr.Dataset,
    mean_features: Mapping[str, str],
) -> None:
    """Add window-mean antecedent-state features."""
    for source_name, window_name in mean_features.items():
        feature_name = feature_name_for_source(source_name, operation="mean")
        values = [
            nanmean_or_nan(window_for_peak(ds, peak_time, window_name)[source_name].values)
            for peak_time in event_peak_values(event_table)
        ]
        out[feature_name] = (config.EVENT_DIM, np.asarray(values, dtype=float))
        if "units" in ds[source_name].attrs:
            out[feature_name].attrs["units"] = ds[source_name].attrs["units"]
        add_feature_attrs(
            out[feature_name],
            source_variable=source_name,
            window_name=window_name,
            operation="mean",
        )


def add_change_features(
    out: xr.Dataset,
    ds: xr.Dataset,
    event_table: xr.Dataset,
    change_features: Mapping[str, str],
) -> None:
    """Add robust end-minus-start change features."""
    for source_name, window_name in change_features.items():
        feature_name = feature_name_for_source(source_name, operation="change")
        values = [
            robust_window_change(ds, peak_time, source_name, window_name)
            for peak_time in event_peak_values(event_table)
        ]
        out[feature_name] = (config.EVENT_DIM, np.asarray(values, dtype=float))
        if "units" in ds[source_name].attrs:
            out[feature_name].attrs["units"] = ds[source_name].attrs["units"]
        add_feature_attrs(
            out[feature_name],
            source_variable=source_name,
            window_name=window_name,
            operation="change",
        )
        out[feature_name].attrs["change_method"] = "final_24h_mean_minus_first_24h_mean"


def add_days_from_solstice(out: xr.Dataset, event_table: xr.Dataset) -> None:
    """Add event peak timing relative to June 21 in each event year."""
    values = []
    for peak_time in event_peak_values(event_table):
        peak_day = peak_time.astype("datetime64[D]")
        year = str(peak_day.astype("datetime64[Y]")).split("-")[0]
        solstice = np.datetime64(
            f"{year}-{config.SOLSTICE_MONTH:02d}-{config.SOLSTICE_DAY:02d}",
            "D",
        )
        values.append(float((peak_day - solstice) / np.timedelta64(1, "D")))
    out["days_from_solstice"] = (config.EVENT_DIM, np.asarray(values, dtype=float))
    out["days_from_solstice"].attrs.update(
        {
            "source_variable": config.PEAK_TIME_NAME,
            "operation": "calendar_day_difference",
            "reference_date": f"{config.SOLSTICE_MONTH:02d}-{config.SOLSTICE_DAY:02d}",
            "units": "days",
        }
    )


def window_for_peak(ds: xr.Dataset, peak_time: np.datetime64, window_name: str) -> xr.Dataset:
    """Return an inclusive timestamp window for one event peak."""
    start_lag, end_lag = config.WINDOWS[window_name]
    start = peak_time + np.timedelta64(start_lag, "h")
    end = peak_time + np.timedelta64(end_lag, "h")
    return ds.sel({config.TIME_DIM: slice(start, end)})


def robust_window_change(
    ds: xr.Dataset,
    peak_time: np.datetime64,
    source_name: str,
    window_name: str,
) -> float:
    """Return final 24 h mean minus first 24 h mean within a configured window."""
    start_lag, end_lag = config.WINDOWS[window_name]
    first_start = peak_time + np.timedelta64(start_lag, "h")
    first_end = first_start + np.timedelta64(24, "h")
    final_end = peak_time + np.timedelta64(end_lag, "h")
    final_start = final_end - np.timedelta64(24, "h")
    first = ds.sel({config.TIME_DIM: slice(first_start, first_end)})[source_name].values
    final = ds.sel({config.TIME_DIM: slice(final_start, final_end)})[source_name].values
    first_mean = nanmean_or_nan(first)
    final_mean = nanmean_or_nan(final)
    if np.isnan(first_mean) or np.isnan(final_mean):
        return np.nan
    return final_mean - first_mean


def event_peak_values(event_table: xr.Dataset) -> np.ndarray:
    """Return selected peak times as datetime64[ns] values."""
    return np.asarray(
        event_table[config.PEAK_TIME_NAME].compute().values,
        dtype="datetime64[ns]",
    )


def active_window_names(feature_spec: Mapping[str, Mapping[str, str]]) -> tuple[str, ...]:
    """Return active window names in config order."""
    active = {
        window_name
        for mapping in feature_spec.values()
        for window_name in mapping.values()
    }
    return tuple(name for name in config.WINDOWS if name in active)


def add_feature_attrs(
    da: xr.DataArray,
    *,
    source_variable: str,
    window_name: str,
    operation: str,
    units: str | None = None,
) -> None:
    """Add common fixed-window feature metadata."""
    start_lag, end_lag = config.WINDOWS[window_name]
    attrs = {
        "source_variable": source_variable,
        "window_name": window_name,
        "window_lag_hours": f"{start_lag},{end_lag}",
        "operation": operation,
        "window_endpoint_inclusion": "inclusive",
    }
    if units is not None:
        attrs["units"] = units
    da.attrs.update(attrs)


def add_global_attrs(
    out: xr.Dataset,
    *,
    input_path: str | Path | None,
    use_extended_variables: bool,
    allow_missing_extended: bool,
    season_months: Sequence[int] | None,
    all_seasons: bool,
    require_full_event: bool,
    dropped_boundary_events: int,
    feature_spec: Mapping[str, Mapping[str, str]],
) -> None:
    """Attach feature-table provenance and method metadata."""
    attrs: dict[str, Any] = {
        "pipeline_stage": config.PIPELINE_STAGE,
        "feature_method": config.FEATURE_METHOD,
        "input_path": "" if input_path is None else str(input_path),
        "extended_variables_used": int(use_extended_variables),
        "allow_missing_extended": int(allow_missing_extended),
        "adaptive_windows_used": 0,
        "integral_method": config.INTEGRAL_METHOD,
        "window_endpoint_inclusion": "inclusive",
        "all_seasons": int(all_seasons),
        "require_full_event": int(require_full_event),
        "dropped_boundary_events": int(dropped_boundary_events),
        "active_windows": ",".join(active_window_names(feature_spec)),
        "active_integral_sources": ",".join(feature_spec["integral"]),
        "active_mean_sources": ",".join(feature_spec["mean"]),
        "active_change_sources": ",".join(feature_spec["change"]),
        "surface_flux_sign_convention": "native Stage-1/source signs retained",
    }
    if season_months is not None:
        attrs["season_months"] = ",".join(str(month) for month in season_months)
    for name, (start_lag, end_lag) in config.WINDOWS.items():
        attrs[f"{name}_window_hours"] = f"{start_lag},{end_lag}"
    out.attrs.update(attrs)


def feature_name_for_source(source_name: str, *, operation: str) -> str:
    """Return the output feature name for a configured source variable."""
    if source_name in config.DEFAULT_FEATURE_NAMES:
        return config.DEFAULT_FEATURE_NAMES[source_name]
    if operation == "change":
        key = f"{source_name}_change"
    else:
        key = source_name
    if key in config.EXTENDED_FEATURE_NAMES:
        return config.EXTENDED_FEATURE_NAMES[key]
    raise KeyError(f"No configured feature name for {source_name!r}.")


def write_feature_outputs(
    features: xr.Dataset,
    output_path: str | Path,
    *,
    csv_output_path: str | Path | None = None,
) -> list[Path]:
    """Write feature table outputs and return paths written."""
    written: list[Path] = []
    netcdf_path = Path(output_path).expanduser().resolve()
    netcdf_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_netcdf(netcdf_path, engine="h5netcdf")
    written.append(netcdf_path)

    if csv_output_path is not None:
        csv_path = Path(csv_output_path).expanduser().resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_dataframe().reset_index().to_csv(csv_path, index=False)
        written.append(csv_path)
    return written


def nansum_or_nan(values: np.ndarray) -> float:
    """Return nansum or NaN for empty/all-NaN values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nansum(arr))


def nanmean_or_nan(values: np.ndarray) -> float:
    """Return nanmean or NaN for empty/all-NaN values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))


def _validate_output_path(path: Path, *, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {path}. Pass --overwrite.")


def _validate_season_months(months: Sequence[int]) -> None:
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        values = ", ".join(str(month) for month in invalid)
        raise ValueError(f"--season-months values must be between 1 and 12; got {values}.")


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    """Open Stage-1 data, build event features, and write requested outputs."""
    args = parse_args()
    validate_args(args)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        features = build_event_features(
            ds,
            use_extended_variables=args.use_extended_variables,
            allow_missing_extended=args.allow_missing_extended,
            season_months=args.season_months,
            all_seasons=args.all_seasons,
            require_full_event=args.require_full_event,
            input_path=args.input_path,
        )
        written = write_feature_outputs(
            features,
            args.output_path,
            csv_output_path=args.csv_output_path,
        )
        print("Wrote event feature table:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
