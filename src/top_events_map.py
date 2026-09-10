"""Three-day T2m/Z500 anomaly products for individually ranked heatwaves."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, data_io, selectors

PIPELINE_STAGE = "top_events_map"
CONTRACT_VERSION = 1
WINDOW_DAYS = (-1, 0, 1)
DEFAULT_EXTENT = (-170.0, -40.0, 10.0, 80.0)
DEFAULT_RANK_METRIC = "tas_peak"


def region_bounds(region: str) -> tuple[float, float, float, float]:
    """Return west, east, south, north from the shared region configuration."""
    if region not in config.REGIONS:
        raise ValueError(
            f"Unknown region {region!r}; choose from {sorted(config.REGIONS)}."
        )
    lat, lon = config.REGIONS[region]
    return float(lon.start), float(lon.stop), float(lat.start), float(lat.stop)


def select_events(
    features: xr.Dataset,
    *,
    region: str,
    top_n: int = 1,
    rank_metric: str = DEFAULT_RANK_METRIC,
    peak_year: int | None = None,
) -> xr.Dataset:
    """Use the existing event selector without redefining peaks or events."""
    region_bounds(region)
    if features.attrs.get("pipeline_stage") != "stage_2_event_features":
        raise ValueError("Expected pipeline_stage='stage_2_event_features'.")
    # Existing Stage-2 products do not always carry region metadata.
    stored_region = features.attrs.get("region")
    if stored_region is not None and stored_region != region:
        raise ValueError(
            f"Event-table region {stored_region!r} disagrees with {region!r}."
        )
    for name in ("event_id", "peak_time", rank_metric):
        if name not in features or features[name].dims != ("event",):
            raise ValueError(
                f"Stage-2 variable {name!r} must have dimension ('event',)."
            )
    ids = np.asarray(features.event_id.values)
    if not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != ids.size:
        raise ValueError("Event IDs must be unique integers.")
    if not np.issubdtype(features.peak_time.dtype, np.datetime64):
        raise ValueError("peak_time must contain decoded Gregorian timestamps.")
    peaks = pd.DatetimeIndex(features.peak_time.values)
    if peaks.hasnans:
        raise ValueError("All event peak_time values must be finite.")
    population = features
    if peak_year is not None:
        population = features.isel(event=np.flatnonzero(peaks.year == peak_year))
    selected = selectors.select_top_n_events(population, rank_metric, top_n)
    if selected.sizes["event"] == 0:
        raise ValueError("No finite-ranked events remain in the requested population.")
    selected.attrs = dict(selected.attrs)
    selected.attrs.update(
        {
            "rank_metric": rank_metric,
            "requested_top_n": top_n,
            "peak_year_filter": "all" if peak_year is None else str(peak_year),
            "candidate_event_count": population.sizes["event"],
            "finite_rank_event_count": int(np.isfinite(population[rank_metric]).sum()),
            "region_metadata_source": "event_table"
            if stored_region is not None
            else "explicit_argument",
        }
    )
    return selected


def build_top_event_maps(
    features: xr.Dataset,
    *,
    event_features_path: str | Path,
    region: str,
    daily_dir: str | Path,
    climatology_path: str | Path,
    climatology_years: tuple[int, int],
    source_commit: str,
    top_n: int = 1,
    rank_metric: str = DEFAULT_RANK_METRIC,
    peak_year: int | None = None,
    extent: tuple[float, float, float, float] = DEFAULT_EXTENT,
) -> xr.Dataset:
    """Build reusable maps from selected daily fields and month/day climatology."""
    events = select_events(
        features,
        region=region,
        top_n=top_n,
        rank_metric=rank_metric,
        peak_year=peak_year,
    )
    west, east, south, north = extent
    if not -180 <= west < east <= 180 or not -90 <= south < north <= 90:
        raise ValueError("Invalid map extent (west, east, south, north).")
    rw, re, rs, rn = region_bounds(region)
    if not (west <= rw < re <= east and south <= rs < rn <= north):
        raise ValueError("Map extent must contain the selected event region.")
    start_year, end_year = climatology_years
    if not 1 <= start_year <= end_year <= 9999:
        raise ValueError("Invalid climatology baseline years.")
    peaks = pd.DatetimeIndex(events.peak_time.values).normalize()
    sample_dates = peaks.values[:, None] + np.array(WINDOW_DAYS, dtype="timedelta64[D]")
    dates = pd.DatetimeIndex(np.unique(sample_dates))
    daily_paths = [
        Path(daily_dir).expanduser().resolve() / f"ERA5_daily_t2m_z500_{year}.nc"
        for year in sorted(set(dates.year))
    ]
    source_paths = [Path(event_features_path), *daily_paths, Path(climatology_path)]
    provenance = [file_identity(path) for path in source_paths]
    fields = []
    for path, year in zip(daily_paths, sorted(set(dates.year)), strict=True):
        fields.append(
            data_io.load_daily_spatial_fields(
                path,
                dates[dates.year == year],
                lat_bounds=(south, north),
                lon_bounds=(west, east),
            )
        )
    raw = xr.concat(fields, dim="time", join="exact", combine_attrs="drop_conflicts")
    climate = data_io.load_daily_spatial_fields(
        climatology_path,
        dates,
        lat_bounds=(south, north),
        lon_bounds=(west, east),
        climatology=True,
    )
    for name, expected in (
        ("climatology_start_year", start_year),
        ("climatology_end_year", end_year),
    ):
        if name in climate.attrs and int(climate.attrs[name]) != expected:
            raise ValueError(
                f"{name} disagrees with the supplied climatology baseline."
            )
    raw, climate = xr.align(raw, climate, join="exact")
    indexer = xr.DataArray(
        sample_dates,
        dims=("event", "window_day"),
        coords={
            "event": events.event,
            "window_day": list(WINDOW_DAYS),
        },
    )
    actual_mean = raw.sel(time=indexer).mean("window_day", skipna=False)
    climate_mean = climate.sel(time=indexer).mean("window_day", skipna=False)
    out = xr.Dataset(
        coords={
            "event": events.event,
            "window_day": list(WINDOW_DAYS),
            "latitude": raw.latitude,
            "longitude": raw.longitude,
        }
    )
    for name, units in (("t2m", "K"), ("z500", "m")):
        out[f"{name}_event_mean"] = actual_mean[name]
        out[f"{name}_climatology_mean"] = climate_mean[name]
        out[f"{name}_anomaly"] = actual_mean[name] - climate_mean[name]
        for suffix in ("event_mean", "climatology_mean", "anomaly"):
            out[f"{name}_{suffix}"].attrs.update(
                units=units, cell_methods="window_day: mean"
            )
    for name in ("event_id", "peak_time", "selection_rank"):
        out[name] = events[name]
    out["rank_value"] = events[rank_metric]
    out["rank_value"].attrs = dict(
        events[rank_metric].attrs, source_variable=rank_metric
    )
    out["sample_date"] = indexer
    out["window_day"].attrs["long_name"] = "UTC calendar day relative to peak date"
    out.attrs.update(
        {
            "pipeline_stage": PIPELINE_STAGE,
            "top_events_map_contract_version": CONTRACT_VERSION,
            "region": region,
            "region_bounds": [rw, re, rs, rn],
            "map_extent": list(extent),
            "climatology_start_year": start_year,
            "climatology_end_year": end_year,
            "climatology_matching": "month_day",
            "window_definition": "UTC peak calendar day plus days -1 and +1; equal daily weight",
            "temperature_statistic": "daily_mean_2m_air_temperature",
            "geopotential_to_height_m_s2": config.G_M_S2,
            "event_features_path": str(
                Path(event_features_path).expanduser().resolve()
            ),
            "event_features_sha256": sha256_file(event_features_path),
            "climatology_path": str(Path(climatology_path).expanduser().resolve()),
            "source_files": json.dumps(provenance, sort_keys=True),
            "source_commit": source_commit,
        }
    )
    for name in (
        "rank_metric",
        "requested_top_n",
        "peak_year_filter",
        "candidate_event_count",
        "finite_rank_event_count",
        "region_metadata_source",
    ):
        out.attrs[name] = events.attrs[name]
    if provenance != [file_identity(path) for path in source_paths]:
        raise ValueError("A source file changed while building the map product.")
    validate_top_event_maps(out)
    return out


def validate_top_event_maps(ds: xr.Dataset) -> None:
    """Validate the saved product without access to its upstream inputs."""
    if ds.attrs.get("pipeline_stage") != PIPELINE_STAGE:
        raise ValueError(f"Expected pipeline_stage={PIPELINE_STAGE!r}.")
    if ds.attrs.get("top_events_map_contract_version") != CONTRACT_VERSION:
        raise ValueError("Unsupported top_events_map contract version.")
    for name in (
        "region",
        "map_extent",
        "region_bounds",
        "rank_metric",
        "source_commit",
        "event_features_path",
        "event_features_sha256",
        "source_files",
        "climatology_path",
        "climatology_start_year",
        "climatology_end_year",
    ):
        if name not in ds.attrs:
            raise ValueError(f"Map product is missing metadata {name!r}.")
    if not np.array_equal(
        ds.attrs["region_bounds"], region_bounds(str(ds.attrs["region"]))
    ):
        raise ValueError("Stored region bounds disagree with src/config.py.")
    extent = np.asarray(ds.attrs["map_extent"], dtype=float)
    if extent.shape != (4,) or not np.isfinite(extent).all():
        raise ValueError("Map extent must contain four finite bounds.")
    west, east, south, north = extent
    rw, re, rs, rn = ds.attrs["region_bounds"]
    if not (
        -180 <= west <= rw < re <= east <= 180
        and -90 <= south <= rs < rn <= north <= 90
    ):
        raise ValueError("Map extent must contain the configured event region.")
    if (
        not 1
        <= ds.attrs["climatology_start_year"]
        <= ds.attrs["climatology_end_year"]
        <= 9999
    ):
        raise ValueError("Invalid climatology baseline years in map product.")
    for name, expected in (
        ("climatology_matching", "month_day"),
        ("temperature_statistic", "daily_mean_2m_air_temperature"),
        ("geopotential_to_height_m_s2", config.G_M_S2),
    ):
        if ds.attrs.get(name) != expected:
            raise ValueError(f"Map metadata {name!r} must equal {expected!r}.")
    if ds.sizes.get("event", 0) < 1:
        raise ValueError("Map product contains no events.")
    if "window_day" not in ds.coords or not np.array_equal(ds.window_day, WINDOW_DAYS):
        raise ValueError("Map product requires window_day=(-1, 0, 1).")
    for name in ("latitude", "longitude"):
        if name not in ds.coords or ds[name].dims != (name,):
            raise ValueError(f"Map product needs a one-dimensional {name} coordinate.")
        values = np.asarray(ds[name].values)
        if (
            values.size < 2
            or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0)
        ):
            raise ValueError(f"Map product has an invalid {name} grid.")
    for name in ("event_id", "peak_time", "selection_rank", "rank_value"):
        if name not in ds or ds[name].dims != ("event",):
            raise ValueError(f"Map audit variable {name!r} needs dimension ('event',).")
    ids = ds.event_id.values
    if not np.issubdtype(ids.dtype, np.integer) or np.unique(ids).size != ids.size:
        raise ValueError("Map event IDs must be unique integers.")
    if not np.isfinite(ds.rank_value.values).all():
        raise ValueError("Map rank values must be finite.")
    if np.any(np.diff(ds.rank_value.values) > 0):
        raise ValueError("Map rank values must be in descending order.")
    if not np.array_equal(ds.selection_rank.values, np.arange(1, ids.size + 1)):
        raise ValueError("Map selection ranks must run from 1 through the event count.")
    if not np.issubdtype(ds.peak_time.dtype, np.datetime64):
        raise ValueError("Map peak_time must contain decoded timestamps.")
    peaks = pd.DatetimeIndex(ds.peak_time.values)
    expected_dates = peaks.normalize().values[:, None] + np.array(
        WINDOW_DAYS, dtype="timedelta64[D]"
    )
    if (
        peaks.hasnans
        or "sample_date" not in ds
        or ds.sample_date.dims != ("event", "window_day")
    ):
        raise ValueError(
            "Map product needs finite peaks and sample_date(event, window_day)."
        )
    if not np.array_equal(ds.sample_date.values, expected_dates):
        raise ValueError(
            "sample_date must contain exactly the three days around peak_time."
        )
    for name, units in (("t2m", "K"), ("z500", "m")):
        for suffix in ("event_mean", "climatology_mean", "anomaly"):
            variable = f"{name}_{suffix}"
            if variable not in ds or ds[variable].dims != (
                "event",
                "latitude",
                "longitude",
            ):
                raise ValueError(f"Invalid dimensions for {variable}.")
            if (
                ds[variable].attrs.get("units") != units
                or not np.isfinite(ds[variable]).all()
            ):
                raise ValueError(f"Invalid units or non-finite values in {variable}.")
        expected = ds[f"{name}_event_mean"] - ds[f"{name}_climatology_mean"]
        if not np.array_equal(expected.values, ds[f"{name}_anomaly"].values):
            raise ValueError(f"{name} anomaly does not equal event minus climatology.")


def file_identity(path: str | Path) -> dict[str, str | int]:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return {
        "path": str(resolved),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def sha256_file(path: str | Path) -> str:
    with Path(path).expanduser().open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()
