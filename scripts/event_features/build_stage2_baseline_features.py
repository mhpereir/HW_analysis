"""Build fixed-window baseline-day feature tables from Stage-1 products."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.event_features import event_feature_config as config
from scripts.event_features import fixed_window_features as fixed
from src import analysis_io


BASELINE_DIM = "baseline_day"
REFERENCE_TIME_NAME = "reference_time"
PIPELINE_STAGE = "stage_2_baseline_features"
FEATURE_METHOD = "fixed_windows_relative_to_reference_time"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_baseline_features"
    / "non_event_day_features_fixed_windows.nc"
)
BASELINE_FEATURE_NAMES = {
    **config.DEFAULT_FEATURE_NAMES,
    "lwa_a_region": "I_lwa_a_pre_reference",
    "lwa_c_region": "I_lwa_c_pre_reference",
}


def parse_args() -> argparse.Namespace:
    """Parse fixed-window baseline feature extraction CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Build baseline-day fixed-window features from a Stage-1 dataset."
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
        default=DEFAULT_OUTPUT_PATH,
        help="NetCDF baseline feature-table output path.",
    )
    parser.add_argument(
        "--csv-output-path",
        type=Path,
        default=None,
        help="Optional CSV baseline feature-table output path.",
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
        help="Calendar months to retain, e.g. 6 7 8.",
    )
    season.add_argument(
        "--all-seasons",
        action="store_true",
        help="Use every selected-source non-event day in Stage 1.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow output files to replace existing files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments that argparse cannot express cleanly."""
    if args.allow_missing_extended and not args.use_extended_variables:
        raise ValueError("--allow-missing-extended requires --use-extended-variables.")
    if args.season_months is not None:
        _validate_season_months(args.season_months)
    _validate_output_path(args.output_path, overwrite=args.overwrite)
    if args.csv_output_path is not None:
        _validate_output_path(args.csv_output_path, overwrite=args.overwrite)


def build_baseline_features(
    ds: xr.Dataset,
    *,
    use_extended_variables: bool = False,
    allow_missing_extended: bool = False,
    season_months: Sequence[int] | None = None,
    all_seasons: bool = False,
    input_path: str | Path | None = None,
) -> xr.Dataset:
    """Return one fixed-window feature row per selected-source non-event day."""
    if not all_seasons and season_months is None:
        raise ValueError("Either season_months or all_seasons=True is required.")
    if all_seasons and season_months is not None:
        raise ValueError("Pass season_months or all_seasons=True, not both.")

    ds = fixed.ensure_tas_anom(ds)
    feature_spec = fixed.active_feature_spec(
        ds,
        use_extended_variables=use_extended_variables,
        allow_missing_extended=allow_missing_extended,
    )
    fixed.validate_required_time_variables(ds, feature_spec)
    event_id_source = selected_event_id_source(ds)
    reducer = fixed.WindowReducer(ds)

    reference_times, reference_event_ids, n_calendar_days = daily_reference_rows(
        ds,
        event_id_source,
    )
    non_event = reference_event_ids == 0
    n_non_event_days = int(non_event.sum())
    reference_times = reference_times[non_event]
    if season_months is not None:
        reference_times = reference_times[month_mask(reference_times, season_months)]
    n_selected_before_boundary = int(reference_times.size)
    if n_selected_before_boundary == 0:
        raise ValueError("No selected-source non-event days remain after season selection.")

    window_names = fixed.active_window_names(feature_spec)
    keep = reducer.complete_anchor_mask(reference_times, window_names)
    dropped_boundary_days = int((~keep).sum())
    reference_times = reference_times[keep]
    if reference_times.size == 0:
        raise ValueError("No baseline days have complete required feature windows.")

    out = xr.Dataset(
        coords={BASELINE_DIM: np.arange(reference_times.size, dtype=np.int64)}
    )
    out[REFERENCE_TIME_NAME] = (BASELINE_DIM, reference_times)
    out[REFERENCE_TIME_NAME].attrs.update(
        {
            "description": "First available Stage-1 timestamp for the baseline calendar day.",
            "operation": "first_available_timestamp_per_calendar_day",
        }
    )
    event_adjacent = reducer.any_nonzero(
        event_id_source,
        reference_times,
        window_names,
    )
    out["event_adjacent"] = (BASELINE_DIM, event_adjacent.astype(np.int8))
    out["event_adjacent"].attrs.update(
        {
            "description": (
                "1 when the selected event source is nonzero anywhere in an active "
                "fixed window; 0 otherwise."
            ),
            "event_id_source": event_id_source,
            "flag_values": "0,1",
            "flag_meanings": "clean event_adjacent",
        }
    )

    fixed.add_window_features(
        out,
        ds,
        reducer,
        reference_times,
        row_dim=BASELINE_DIM,
        feature_spec=feature_spec,
        feature_name_for_source=feature_name_for_source,
        sample_count_name_for_window=sample_count_name_for_window,
    )
    relabel_baseline_windows(out)
    fixed.add_days_from_solstice(
        out,
        reference_times,
        row_dim=BASELINE_DIM,
        source_variable=REFERENCE_TIME_NAME,
    )
    add_global_attrs(
        out,
        input_path=input_path,
        event_id_source=event_id_source,
        use_extended_variables=use_extended_variables,
        allow_missing_extended=allow_missing_extended,
        season_months=season_months,
        all_seasons=all_seasons,
        n_calendar_days=n_calendar_days,
        n_non_event_days=n_non_event_days,
        n_selected_before_boundary=n_selected_before_boundary,
        dropped_boundary_days=dropped_boundary_days,
        feature_spec=feature_spec,
    )
    return out


def selected_event_id_source(ds: xr.Dataset) -> str:
    """Return and validate the Stage-1 event-ID source defining baseline days."""
    source = ds.attrs.get("event_id_source")
    if not isinstance(source, str) or not source:
        raise ValueError("Stage-1 dataset is missing required event_id_source metadata.")
    if source not in ds:
        raise ValueError(
            f"Stage-1 event_id_source {source!r} is not present in the dataset."
        )
    if ds[source].dims != (config.TIME_DIM,):
        raise ValueError(
            f"Stage-1 event_id_source {source!r} must have dims "
            f"({config.TIME_DIM!r},); got {ds[source].dims!r}."
        )
    return source


def daily_reference_rows(
    ds: xr.Dataset,
    event_id_source: str,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return first timestamp and validated selected-source event ID for each day."""
    times = np.asarray(ds[config.TIME_DIM].values, dtype="datetime64[ns]")
    days = times.astype("datetime64[D]")
    unique_days, first_idx, counts = np.unique(
        days,
        return_index=True,
        return_counts=True,
    )
    if unique_days.size == 0:
        raise ValueError("Input dataset has no calendar days.")

    event_values = np.asarray(ds[event_id_source].compute().values, dtype=float)
    if not np.isfinite(event_values).all():
        raise ValueError(
            f"Stage-1 event_id_source {event_id_source!r} contains missing values."
        )
    first_values = event_values[first_idx]
    if not np.array_equal(event_values, np.repeat(first_values, counts)):
        raise ValueError(
            f"Stage-1 event_id_source {event_id_source!r} is inconsistent within "
            "at least one calendar day."
        )
    if not np.equal(first_values, np.floor(first_values)).all():
        raise ValueError(
            f"Stage-1 event_id_source {event_id_source!r} must contain integer IDs."
        )
    return times[first_idx], first_values.astype(np.int64), int(unique_days.size)


def month_mask(reference_times: np.ndarray, months: Sequence[int]) -> np.ndarray:
    """Return whether reference timestamps fall in requested calendar months."""
    month_values = (
        np.asarray(reference_times).astype("datetime64[M]").astype(np.int64) % 12
    ) + 1
    return np.isin(month_values, np.asarray(months, dtype=np.int64))


def feature_name_for_source(source_name: str, operation: str) -> str:
    """Return the baseline output feature name for a configured source."""
    if source_name in BASELINE_FEATURE_NAMES:
        return BASELINE_FEATURE_NAMES[source_name]
    key = f"{source_name}_change" if operation == "change" else source_name
    if key in config.EXTENDED_FEATURE_NAMES:
        return config.EXTENDED_FEATURE_NAMES[key]
    raise KeyError(f"No configured baseline feature name for {source_name!r}.")


def sample_count_name_for_window(window_name: str) -> str:
    """Return a baseline sample-count name without event-peak terminology."""
    if window_name == "lwa_pre_peak":
        return "n_samples_lwa_pre_reference"
    return f"n_samples_{window_name}"


def baseline_window_name(window_name: str) -> str:
    """Return a baseline-facing fixed-window name."""
    if window_name == "lwa_pre_peak":
        return "lwa_pre_reference"
    return window_name


def relabel_baseline_windows(out: xr.Dataset) -> None:
    """Replace event-peak terminology in baseline feature metadata."""
    for name in out.data_vars:
        window_name = out[name].attrs.get("window_name")
        if isinstance(window_name, str):
            out[name].attrs["window_name"] = baseline_window_name(window_name)


def add_global_attrs(
    out: xr.Dataset,
    *,
    input_path: str | Path | None,
    event_id_source: str,
    use_extended_variables: bool,
    allow_missing_extended: bool,
    season_months: Sequence[int] | None,
    all_seasons: bool,
    n_calendar_days: int,
    n_non_event_days: int,
    n_selected_before_boundary: int,
    dropped_boundary_days: int,
    feature_spec: Mapping[str, Mapping[str, str]],
) -> None:
    """Attach baseline-table provenance, population, and method metadata."""
    window_names = fixed.active_window_names(feature_spec)
    adjacency_start = min(config.WINDOWS[name][0] for name in window_names)
    adjacency_end = max(config.WINDOWS[name][1] for name in window_names)
    attrs: dict[str, Any] = {
        "pipeline_stage": PIPELINE_STAGE,
        "feature_method": FEATURE_METHOD,
        "input_path": "" if input_path is None else str(input_path),
        "event_id_source": event_id_source,
        "baseline_definition": "selected event_id_source equals zero on reference day",
        "reference_time_method": "first_available_timestamp_per_calendar_day",
        "event_adjacency_definition": (
            "selected event_id_source != 0 anywhere in union of active fixed windows"
        ),
        "event_adjacency_window_hours": f"{adjacency_start},{adjacency_end}",
        "extended_variables_used": int(use_extended_variables),
        "allow_missing_extended": int(allow_missing_extended),
        "adaptive_windows_used": 0,
        "integral_method": config.INTEGRAL_METHOD,
        "window_endpoint_inclusion": "inclusive",
        "all_seasons": int(all_seasons),
        "n_calendar_days": int(n_calendar_days),
        "n_non_event_days": int(n_non_event_days),
        "n_selected_before_boundary": int(n_selected_before_boundary),
        "dropped_boundary_days": int(dropped_boundary_days),
        "n_baseline_days": int(out.sizes[BASELINE_DIM]),
        "n_event_adjacent_days": int(out["event_adjacent"].sum().item()),
        "n_clean_days": int((out["event_adjacent"] == 0).sum().item()),
        "active_windows": ",".join(baseline_window_name(name) for name in window_names),
        "active_integral_sources": ",".join(feature_spec["integral"]),
        "active_mean_sources": ",".join(feature_spec["mean"]),
        "active_change_sources": ",".join(feature_spec["change"]),
        "surface_flux_sign_convention": "native Stage-1/source signs retained",
    }
    if season_months is not None:
        attrs["season_months"] = ",".join(str(month) for month in season_months)
    for name in window_names:
        start_lag, end_lag = config.WINDOWS[name]
        attrs[f"{baseline_window_name(name)}_window_hours"] = f"{start_lag},{end_lag}"
    out.attrs.update(attrs)


def write_feature_outputs(
    features: xr.Dataset,
    output_path: str | Path,
    *,
    csv_output_path: str | Path | None = None,
) -> list[Path]:
    """Write baseline feature table outputs and return paths written."""
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
    """Open Stage-1 data, build baseline features, and write requested outputs."""
    args = parse_args()
    validate_args(args)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        features = build_baseline_features(
            ds,
            use_extended_variables=args.use_extended_variables,
            allow_missing_extended=args.allow_missing_extended,
            season_months=args.season_months,
            all_seasons=args.all_seasons,
            input_path=args.input_path,
        )
        written = write_feature_outputs(
            features,
            args.output_path,
            csv_output_path=args.csv_output_path,
        )
        print("Wrote baseline-day feature table:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
