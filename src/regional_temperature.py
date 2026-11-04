"""Preparation and validation of native-grid regional hourly ERA5 T2m."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import config, preprocess

PIPELINE_STAGE = "era5_regional_hourly_t2m"
CONTRACT_VERSION = 1
GRID_SPACING_DEGREES = 0.25


def sha256(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def expected_hours(start_year: int, end_year: int) -> pd.DatetimeIndex:
    if start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    return pd.date_range(
        f"{start_year}-01-01", f"{end_year + 1}-01-01", freq="h", inclusive="left"
    )


def validate_hours(time: xr.DataArray, start_year: int, end_year: int) -> None:
    if time.dims != ("time",) or not np.issubdtype(time.dtype, np.datetime64):
        raise ValueError("A one-dimensional Gregorian UTC time coordinate is required.")
    actual = pd.DatetimeIndex(time.values)
    if not actual.equals(expected_hours(start_year, end_year)):
        raise ValueError(
            "Every UTC hour of each requested calendar year must occur exactly once, in order."
        )


def validate_hourly_variable(da: xr.DataArray, *, region: str) -> None:
    """Validate a cache field, including when copied onto a Stage-1 axis."""
    if region not in config.REGIONS:
        raise ValueError(f"Unknown regional hourly T2m region {region!r}.")
    if da.dims != ("time",) or da.attrs.get("units") != "K":
        raise ValueError("Regional hourly T2m must be one-dimensional in time, in K.")
    if da.attrs.get("native_time_resolution") != "hourly":
        raise ValueError("Regional T2m must be native hourly data.")
    if da.attrs.get("region") != region:
        raise ValueError("Regional hourly T2m has incompatible region metadata.")
    for key, bounds in zip(
        ("lat_bounds", "lon_bounds"), config.REGIONS[region], strict=True
    ):
        actual = np.asarray(da.attrs.get(key), dtype=float)
        if actual.shape != (2,) or not np.allclose(
            actual, [bounds.start, bounds.stop], rtol=0, atol=1e-8
        ):
            raise ValueError(f"Regional hourly T2m has incompatible {key}.")
    if not np.isfinite(da.values).all():
        raise ValueError("Regional hourly T2m contains nonfinite values.")


def validate_product(ds: xr.Dataset, *, region: str | None = None) -> None:
    if ds.attrs.get("pipeline_stage") != PIPELINE_STAGE:
        raise ValueError(f"Expected pipeline_stage={PIPELINE_STAGE!r}.")
    if ds.attrs.get("regional_hourly_t2m_contract_version") != CONTRACT_VERSION:
        raise ValueError("Unsupported regional hourly T2m contract version.")
    stored_region = ds.attrs.get("region")
    if stored_region not in config.REGIONS or (
        region is not None and stored_region != region
    ):
        raise ValueError("Regional hourly T2m product has an incompatible region.")
    if "tas_region" not in ds or "time" not in ds.coords:
        raise ValueError("Regional hourly T2m product requires tas_region(time).")
    validate_hours(ds.time, int(ds.attrs["start_year"]), int(ds.attrs["end_year"]))
    validate_hourly_variable(ds.tas_region, region=stored_region)
    if ds.tas_region.attrs.get("source_grid_spacing_degrees") != GRID_SPACING_DEGREES:
        raise ValueError(
            "Regional hourly T2m must identify its native 0.25-degree grid."
        )


def prepare_annual_region(
    source: xr.Dataset, *, region: str, year: int, time_chunk: int | None = None
) -> tuple[xr.Dataset, dict]:
    """Reduce bounded regional blocks; never materialize a global annual field."""
    if region not in config.REGIONS:
        raise ValueError(f"Unknown region {region!r}.")
    if "t2m" not in source or set(source.t2m.dims) != {"time", "lat", "lon"}:
        raise ValueError("Source t2m must have only time, lat and lon dimensions.")
    if source.t2m.attrs.get("units") != "K":
        raise ValueError("Native ERA5 t2m source units must be K.")
    validate_hours(source.time, year, year)
    for coord in ("lat", "lon"):
        values = np.asarray(source[coord].values, dtype=float)
        difference = np.diff(values)
        if (
            source[coord].dims != (coord,)
            or values.size < 2
            or not np.isfinite(values).all()
            or not (np.all(difference > 0) or np.all(difference < 0))
            or not np.allclose(
                np.abs(difference), GRID_SPACING_DEGREES, rtol=0, atol=1e-8
            )
        ):
            raise ValueError(
                f"Source {coord} must be a unique regular 0.25-degree coordinate."
            )
    selected = preprocess._select_region(
        source.t2m, region, lat_dim="lat", lon_dim="lon"
    )
    selected = selected.transpose("time", "lat", "lon")
    for coord, bounds in zip(("lat", "lon"), config.REGIONS[region], strict=True):
        expected = np.arange(
            bounds.start, bounds.stop + GRID_SPACING_DEGREES / 2, GRID_SPACING_DEGREES
        )
        actual = np.sort(selected[coord].values)
        if actual.shape != expected.shape or not np.allclose(
            actual, expected, rtol=0, atol=1e-8
        ):
            raise ValueError(
                f"Source does not contain the complete {region} {coord} grid."
            )
    if time_chunk is None:
        chunks = source.t2m.encoding.get("chunksizes")
        time_chunk = int(chunks[source.t2m.dims.index("time")]) if chunks else 168
    if (
        isinstance(time_chunk, bool)
        or not isinstance(time_chunk, int)
        or time_chunk < 1
    ):
        raise ValueError("time_chunk must be a positive integer.")
    values_hash = hashlib.sha256()
    means = []
    missing_value = source.t2m.attrs.get("GRIB_missingValue")
    if missing_value is not None:
        missing_value = np.asarray(missing_value, dtype=source.t2m.dtype).item()
    for start in range(0, source.sizes["time"], time_chunk):
        block = (
            selected.isel(time=slice(start, start + time_chunk))
            .load()
            .astype(np.float64)
        )
        if not np.isfinite(block.values).all() or (
            missing_value is not None and (block.values == missing_value).any()
        ):
            raise ValueError(
                f"Missing/nonfinite regional source cells in {region}, {year}, hour block {start}."
            )
        values_hash.update(np.asarray(block.values, dtype="<f8", order="C").tobytes())
        means.append(preprocess.compute_region_mean(block, region).values)
    lat, lon = config.REGIONS[region]
    result = xr.Dataset(
        {"tas_region": ("time", np.concatenate(means))},
        coords={"time": source.time.values},
        attrs={
            "pipeline_stage": PIPELINE_STAGE,
            "regional_hourly_t2m_contract_version": CONTRACT_VERSION,
            "region": region,
            "start_year": year,
            "end_year": year,
        },
    )
    result.tas_region.attrs.update(
        units="K",
        long_name="Regional native hourly 2 metre air temperature",
        standard_name="air_temperature",
        native_time_resolution="hourly",
        source_variable="t2m",
        region=region,
        lat_bounds=(lat.start, lat.stop),
        lon_bounds=(lon.start, lon.stop),
        source_grid_spacing_degrees=GRID_SPACING_DEGREES,
        time_basis="UTC",
        spatial_mean="cosine-latitude weighted mean",
        cell_selection="inclusive grid centers; land and ocean",
        missing_data_policy="require every selected grid cell finite at every hour",
    )
    result.time.attrs["time_basis"] = "UTC"
    validate_product(result, region=region)
    coordinate_hashes = {
        name: hashlib.sha256(
            np.asarray(selected[name].values, dtype="<f8").tobytes()
        ).hexdigest()
        for name in ("lat", "lon")
    }
    coordinate_hashes["time_ns"] = hashlib.sha256(
        source.time.values.astype("datetime64[ns]").astype("<i8").tobytes()
    ).hexdigest()
    return result, {
        "year": year,
        "hours": source.sizes["time"],
        "time_chunk": time_chunk,
        "regional_shape": [selected.sizes["lat"], selected.sizes["lon"]],
        "source_variable": "t2m",
        "decoded_source_float64_sha256": values_hash.hexdigest(),
        "coordinate_sha256": coordinate_hashes,
        "source_expver": [str(value) for value in np.unique(source.expver.values)]
        if "expver" in source.coords
        else [],
        "min_K": float(result.tas_region.min()),
        "max_K": float(result.tas_region.max()),
    }
