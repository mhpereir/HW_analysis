"""Build and validate the standalone PBL/700 hPa justification product."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

from . import composites, config, preprocess, selectors

PRODUCT_STAGE = "pbl_700hpa_justification"
PRODUCT_CONTRACT_VERSION = 1
PBL_SOURCE_NAME = "pbl_p"
AREA_MEAN_NAME = "pbl_top_pressure_area_mean"
SPATIAL_P05_NAME = "pbl_top_pressure_spatial_p05"
SPATIAL_P95_NAME = "pbl_top_pressure_spatial_p95"
MAP_NAME = "mean_daily_min_pbl_top_pressure"
EVENT_SAMPLE_COUNT_NAME = "event_sample_count"
SELECTED_EVENT_COUNT_NAME = "selected_event_count"
SELECTED_DAY_COUNT_NAME = "selected_heatwave_day_count"
PRESSURE_LIMIT_PA = 120_000.0
DEFAULT_SEASON_MONTHS: tuple[int, ...] = (6, 7, 8)
DEFAULT_WINDOW_DAYS = 7
SPATIAL_QUANTILES: tuple[float, float] = (0.05, 0.95)
REQUIRED_PRODUCT_VARIABLES = frozenset(
    {
        AREA_MEAN_NAME,
        SPATIAL_P05_NAME,
        SPATIAL_P95_NAME,
        MAP_NAME,
        EVENT_SAMPLE_COUNT_NAME,
        SELECTED_EVENT_COUNT_NAME,
        SELECTED_DAY_COUNT_NAME,
    }
)


def select_full_season_events(
    stage1: xr.Dataset,
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
) -> xr.Dataset:
    """Return Stage-1 events whose complete interval is in the target season."""
    selected = selectors.select_events_by_season(
        stage1,
        season_months,
        require_full_event=True,
    )
    if selected.sizes.get("event", 0) == 0:
        months = ", ".join(str(int(month)) for month in season_months)
        raise ValueError(f"No full events remain in season months: {months}.")
    return selected


def selected_event_days(events: xr.Dataset) -> np.ndarray:
    """Return unique inclusive UTC calendar days occupied by selected events."""
    for name in ("start_time", "end_time"):
        if name not in events:
            raise ValueError(f"Selected event table is missing {name!r}.")
        if events[name].dims != ("event",):
            raise ValueError(f"{name!r} must have dimensions ('event',).")

    starts = np.asarray(events["start_time"].values, dtype="datetime64[D]")
    ends = np.asarray(events["end_time"].values, dtype="datetime64[D]")
    if np.any(np.isnat(starts)) or np.any(np.isnat(ends)):
        raise ValueError("Selected event intervals contain missing timestamps.")
    if np.any(ends < starts):
        raise ValueError("Selected event intervals contain an end before a start.")

    day_blocks = [
        np.arange(
            start.astype(np.int64),
            end.astype(np.int64) + 1,
            dtype=np.int64,
        ).astype("datetime64[D]")
        for start, end in zip(starts, ends, strict=True)
    ]
    return np.unique(np.concatenate(day_blocks))


def event_window_times(
    events: xr.Dataset,
    *,
    pre_days: int = DEFAULT_WINDOW_DAYS,
    post_days: int = DEFAULT_WINDOW_DAYS,
) -> np.ndarray:
    """Return unique hourly timestamps required by all peak-aligned windows."""
    _validate_window_days(pre_days, post_days)
    if "peak_time" not in events or events["peak_time"].dims != ("event",):
        raise ValueError("Selected event table requires peak_time(event).")
    peaks = np.asarray(events["peak_time"].values, dtype="datetime64[ns]")
    if np.any(np.isnat(peaks)):
        raise ValueError("Selected event peak times contain missing timestamps.")
    lag_hours = np.arange(-24 * pre_days, 24 * post_days + 1, dtype=np.int64)
    targets = peaks[:, np.newaxis] + lag_hours[np.newaxis, :].astype("timedelta64[h]")
    return np.unique(targets.reshape(-1))


def heatwave_hour_times(days: np.ndarray) -> np.ndarray:
    """Return all 24 hourly timestamps for each UTC heatwave calendar day."""
    day_values = np.asarray(days, dtype="datetime64[D]")
    if day_values.ndim != 1 or day_values.size == 0:
        raise ValueError("At least one heatwave day is required.")
    if np.any(np.isnat(day_values)):
        raise ValueError("Heatwave days contain missing timestamps.")
    hours = np.arange(24, dtype=np.int64).astype("timedelta64[h]")
    return (day_values.astype("datetime64[h]")[:, np.newaxis] + hours).reshape(-1)


def build_product(
    pbl_top_pressure: xr.DataArray,
    events: xr.Dataset,
    *,
    region: str,
    pre_days: int = DEFAULT_WINDOW_DAYS,
    post_days: int = DEFAULT_WINDOW_DAYS,
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
) -> xr.Dataset:
    """Build the compact peak-aligned and heatwave-day PBL product."""
    _validate_pbl_source(pbl_top_pressure, region=region)
    _validate_window_days(pre_days, post_days)
    if events.sizes.get("event", 0) == 0:
        raise ValueError("At least one selected event is required.")

    window_times = event_window_times(
        events,
        pre_days=pre_days,
        post_days=post_days,
    )
    days = selected_event_days(events)
    map_times = heatwave_hour_times(days)
    _require_exact_source_times(pbl_top_pressure, window_times, label="event windows")
    _require_exact_source_times(pbl_top_pressure, map_times, label="heatwave days")

    window_source = pbl_top_pressure.sel(time=window_times)
    regional_mean = preprocess.compute_region_mean(window_source, region).rename(
        AREA_MEAN_NAME
    )
    regional_quantiles = preprocess.compute_region_weighted_quantiles(
        window_source,
        region,
        SPATIAL_QUANTILES,
    )
    regional = xr.Dataset(
        {
            AREA_MEAN_NAME: regional_mean,
            SPATIAL_P05_NAME: regional_quantiles.sel(quantile=0.05, drop=True).rename(
                SPATIAL_P05_NAME
            ),
            SPATIAL_P95_NAME: regional_quantiles.sel(quantile=0.95, drop=True).rename(
                SPATIAL_P95_NAME
            ),
        }
    )
    stacked = composites.stack_events_centered_on_peak(
        regional,
        events,
        variables=(AREA_MEAN_NAME, SPATIAL_P05_NAME, SPATIAL_P95_NAME),
        pre_days=pre_days,
        post_days=post_days,
        fill_missing=False,
    )
    composite = stacked[[AREA_MEAN_NAME, SPATIAL_P05_NAME, SPATIAL_P95_NAME]].mean(
        "event"
    )
    event_sample_count = stacked[AREA_MEAN_NAME].count("event").astype(np.int32)

    map_source = pbl_top_pressure.sel(time=map_times)
    map_days = np.asarray(map_source["time"].values, dtype="datetime64[D]")
    daily_minimum = (
        map_source.assign_coords(heatwave_day=("time", map_days))
        .groupby("heatwave_day")
        .min("time")
    )
    if daily_minimum.sizes.get("heatwave_day", 0) != days.size:
        raise ValueError(
            "Daily PBL grouping did not preserve every selected heatwave day."
        )
    map_mean = daily_minimum.mean("heatwave_day").rename(MAP_NAME)

    product = xr.Dataset(
        {
            AREA_MEAN_NAME: composite[AREA_MEAN_NAME],
            SPATIAL_P05_NAME: composite[SPATIAL_P05_NAME],
            SPATIAL_P95_NAME: composite[SPATIAL_P95_NAME],
            MAP_NAME: map_mean,
            EVENT_SAMPLE_COUNT_NAME: event_sample_count.rename(EVENT_SAMPLE_COUNT_NAME),
            SELECTED_EVENT_COUNT_NAME: xr.DataArray(np.int32(events.sizes["event"])),
            SELECTED_DAY_COUNT_NAME: xr.DataArray(np.int32(days.size)),
        }
    )
    product["lag_hour"].attrs = {
        "long_name": "hours relative to heatwave peak",
    }
    lat_bounds, lon_bounds = config.REGIONS[region]
    product.attrs.update(
        {
            "pipeline_stage": PRODUCT_STAGE,
            "product_contract_version": PRODUCT_CONTRACT_VERSION,
            "validation_status": "pending",
            "region": region,
            "lat_bounds": f"{lat_bounds.start},{lat_bounds.stop}",
            "lon_bounds": f"{lon_bounds.start},{lon_bounds.stop}",
            "season_months": ",".join(str(int(month)) for month in season_months),
            "selection_require_full_event": 1,
            "heatwave_day_basis": "UTC calendar day from accepted Stage-1 events",
            "pre_days": int(pre_days),
            "post_days": int(post_days),
            "pressure_interpretation": (
                "lower PBL-top pressure means greater PBL height; daily minimum "
                "pressure is daily maximum PBL height in pressure coordinates"
            ),
            "upper_boundary_reference_hpa": 700.0,
        }
    )
    _set_variable_attrs(product)
    return product


def validate_product(ds: xr.Dataset, *, require_complete: bool = True) -> None:
    """Validate the standalone PBL diagnostic product contract."""
    if ds.attrs.get("pipeline_stage") != PRODUCT_STAGE:
        raise ValueError(f"Expected pipeline_stage={PRODUCT_STAGE!r}.")
    if int(ds.attrs.get("product_contract_version", 0)) != PRODUCT_CONTRACT_VERSION:
        raise ValueError("Unsupported PBL justification product contract version.")
    if require_complete and ds.attrs.get("validation_status") != "complete":
        raise ValueError("PBL justification product is not marked complete.")
    missing = sorted(REQUIRED_PRODUCT_VARIABLES.difference(ds.data_vars))
    if missing:
        raise ValueError(f"PBL justification product is missing variables: {missing}.")
    if ds[AREA_MEAN_NAME].dims != ("lag_hour",):
        raise ValueError(f"{AREA_MEAN_NAME} must have dimensions ('lag_hour',).")
    for name in (SPATIAL_P05_NAME, SPATIAL_P95_NAME, EVENT_SAMPLE_COUNT_NAME):
        if ds[name].dims != ("lag_hour",):
            raise ValueError(f"{name} must have dimensions ('lag_hour',).")
    if ds[MAP_NAME].dims != ("lat", "lon"):
        raise ValueError(f"{MAP_NAME} must have dimensions ('lat', 'lon').")
    for name in (SELECTED_EVENT_COUNT_NAME, SELECTED_DAY_COUNT_NAME):
        if ds[name].dims != ():
            raise ValueError(f"{name} must be scalar.")
    for name in (AREA_MEAN_NAME, SPATIAL_P05_NAME, SPATIAL_P95_NAME, MAP_NAME):
        if ds[name].attrs.get("units") != "Pa":
            raise ValueError(f"{name} units must be Pa.")

    region = str(ds.attrs.get("region", ""))
    if region not in config.REGIONS:
        raise ValueError(f"Product region is not configured: {region!r}.")
    lat_bounds, lon_bounds = config.REGIONS[region]
    for coord, bounds in (("lat", lat_bounds), ("lon", lon_bounds)):
        values = np.asarray(ds[coord].values, dtype=float)
        expected_bounds = np.sort([float(bounds.start), float(bounds.stop)])
        actual_bounds = np.array([float(values.min()), float(values.max())])
        if not np.allclose(actual_bounds, expected_bounds, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"Map {coord!r} bounds do not match configured region {region!r}."
            )

    expected_lag = np.arange(
        -24 * int(ds.attrs["pre_days"]),
        24 * int(ds.attrs["post_days"]) + 1,
        dtype=np.int64,
    )
    if not np.array_equal(ds["lag_hour"].values, expected_lag):
        raise ValueError("lag_hour does not match the declared event window.")
    event_count = int(ds[SELECTED_EVENT_COUNT_NAME].item())
    day_count = int(ds[SELECTED_DAY_COUNT_NAME].item())
    if event_count < 1 or day_count < 1:
        raise ValueError("Selected event and heatwave-day counts must be positive.")
    if not np.all(np.asarray(ds[EVENT_SAMPLE_COUNT_NAME].values) == event_count):
        raise ValueError(
            "Every lag must contain the complete selected event population."
        )

    for name in (AREA_MEAN_NAME, SPATIAL_P05_NAME, SPATIAL_P95_NAME, MAP_NAME):
        values = np.asarray(ds[name].values, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values.")
        if float(values.min()) <= 0.0 or float(values.max()) > PRESSURE_LIMIT_PA:
            raise ValueError(f"{name} contains implausible pressure values.")
    if np.any(ds[SPATIAL_P05_NAME].values > ds[SPATIAL_P95_NAME].values):
        raise ValueError("Spatial PBL pressure percentiles are not ordered.")
    for coord in ("lat", "lon"):
        values = np.asarray(ds[coord].values, dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"Map coordinate {coord!r} is invalid.")
        differences = np.diff(values)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"Map coordinate {coord!r} must be strictly monotonic.")


def save_product(ds: xr.Dataset, path: str | Path) -> Path:
    """Validate and atomically publish a complete diagnostic product."""
    validate_product(ds, require_complete=False)
    output_path = Path(path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing product: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    complete = ds.copy(deep=False)
    complete.attrs = dict(ds.attrs)
    complete.attrs["validation_status"] = "complete"
    validate_product(complete)
    temporary_path = output_path.with_name(f".{output_path.name}.{uuid4().hex}.partial")
    encoding = {
        name: {"zlib": True, "complevel": 4, "dtype": "float32"}
        for name in (AREA_MEAN_NAME, SPATIAL_P05_NAME, SPATIAL_P95_NAME, MAP_NAME)
    }
    try:
        complete.to_netcdf(temporary_path, engine="h5netcdf", encoding=encoding)
        reopened = xr.open_dataset(temporary_path, engine="h5netcdf")
        try:
            validate_product(reopened)
        finally:
            reopened.close()
        os.replace(temporary_path, output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def open_product(path: str | Path) -> xr.Dataset:
    """Open and validate a complete PBL justification product."""
    input_path = Path(path).expanduser().resolve()
    ds = xr.open_dataset(input_path, engine="h5netcdf")
    try:
        validate_product(ds)
    except BaseException:
        ds.close()
        raise
    return ds


def annual_source_paths(
    root: str | Path,
    *,
    region: str,
    years: Sequence[int],
) -> tuple[Path, ...]:
    """Return the exact required annual PBL files or fail on missing years."""
    source_root = Path(root).expanduser().resolve()
    requested = tuple(int(year) for year in years)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Requested PBL years must be nonempty and unique.")
    paths = tuple(
        source_root / region / f"ERA5_ARCO_pbl_p_{year}.nc" for year in requested
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing annual PBL files: {missing}")
    return paths


def inventory_sha256(paths: Sequence[Path]) -> str:
    """Hash resolved names and sizes for a compact annual-file inventory."""
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        record = f"{path.resolve()}\t{stat.st_size}\n"
        digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for one provenance input."""
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_pbl_source(pbl: xr.DataArray, *, region: str) -> None:
    if not isinstance(pbl, xr.DataArray):
        raise TypeError("PBL source must be an xarray.DataArray.")
    if pbl.name != PBL_SOURCE_NAME:
        raise ValueError(f"PBL source variable must be named {PBL_SOURCE_NAME!r}.")
    if pbl.dims != ("time", "lat", "lon"):
        raise ValueError("PBL source must have dimensions ('time', 'lat', 'lon').")
    if pbl.attrs.get("units") != "Pa":
        raise ValueError("PBL source units must be Pa.")
    if region not in config.REGIONS:
        raise ValueError(f"Unknown configured region: {region!r}.")
    lat_bounds, lon_bounds = config.REGIONS[region]
    for coord, bounds in (("lat", lat_bounds), ("lon", lon_bounds)):
        values = np.asarray(pbl[coord].values, dtype=float)
        if values.size < 2 or not np.all(np.isfinite(values)):
            raise ValueError(f"PBL source coordinate {coord!r} is invalid.")
        differences = np.diff(values)
        if not (np.all(differences > 0) or np.all(differences < 0)):
            raise ValueError(f"PBL source coordinate {coord!r} must be monotonic.")
        expected = np.sort([float(bounds.start), float(bounds.stop)])
        actual = np.array([float(values.min()), float(values.max())])
        if not np.allclose(actual, expected, atol=1e-8, rtol=0.0):
            raise ValueError(
                f"PBL source {coord!r} bounds {actual.tolist()} do not match "
                f"configured bounds {expected.tolist()}."
            )
    index = pd.DatetimeIndex(np.asarray(pbl["time"].values, dtype="datetime64[ns]"))
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("PBL source time must be unique and strictly increasing.")


def _require_exact_source_times(
    pbl: xr.DataArray,
    required: np.ndarray,
    *,
    label: str,
) -> None:
    source_index = pd.DatetimeIndex(
        np.asarray(pbl["time"].values, dtype="datetime64[ns]")
    )
    required_index = pd.DatetimeIndex(np.asarray(required, dtype="datetime64[ns]"))
    missing = required_index[source_index.get_indexer(required_index) < 0]
    if not missing.empty:
        preview = ", ".join(timestamp.isoformat() for timestamp in missing[:5])
        raise ValueError(
            f"PBL source is missing {len(missing)} {label} timestamps: {preview}"
        )


def _validate_window_days(pre_days: int, post_days: int) -> None:
    for name, value in (("pre_days", pre_days), ("post_days", post_days)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer.")


def _set_variable_attrs(ds: xr.Dataset) -> None:
    common = {"units": "Pa", "source_variable": PBL_SOURCE_NAME}
    ds[AREA_MEAN_NAME].attrs.update(
        common
        | {
            "long_name": "peak-aligned event-mean regional PBL-top pressure",
            "spatial_reduction": "cosine-latitude area-weighted mean",
            "event_reduction": "mean across selected events",
        }
    )
    for name, percentile in ((SPATIAL_P05_NAME, 5), (SPATIAL_P95_NAME, 95)):
        ds[name].attrs.update(
            common
            | {
                "long_name": f"peak-aligned event-mean spatial PBL-top pressure p{percentile:02d}",
                "spatial_reduction": "cosine-latitude area-weighted quantile",
                "spatial_percentile": percentile,
                "event_reduction": "mean across selected events",
            }
        )
    ds[MAP_NAME].attrs.update(
        common
        | {
            "long_name": "mean heatwave-day minimum PBL-top pressure",
            "daily_reduction": "minimum across 24 UTC hours",
            "day_reduction": "mean across selected heatwave days",
            "height_interpretation": "daily maximum PBL height in pressure coordinates",
        }
    )
    ds[EVENT_SAMPLE_COUNT_NAME].attrs["long_name"] = (
        "number of selected events contributing at each lag"
    )
    ds[SELECTED_EVENT_COUNT_NAME].attrs["long_name"] = "selected event count"
    ds[SELECTED_DAY_COUNT_NAME].attrs["long_name"] = "selected heatwave-day count"
