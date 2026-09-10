"""Raw data access for the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: raw data access within top-level Stage 1.

Responsibilities:
- Open raw source datasets.
- Handle source-specific filename conventions.
- Standardize variable names where practical.
- Preserve xarray-native lazy loading.

Out of scope:
- Major preprocessing or transformations.
- Event logic.
- Composite generation.
- Plotting.
"""

from __future__ import annotations

import glob
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd
import xarray as xr

from . import config

DEFAULT_TAS_CHUNKS: dict[str, int] = {"time": 365}
DEFAULT_LWA_CHUNKS: dict[str, int] = {"time": 3650, "lat": 35, "lon": 180}
DEFAULT_THRESHOLD_CHUNKS: dict[str, int] = {"dayofyear": 365}
DEFAULT_HEAT_BUDGET_CHUNKS: dict[str, int] = {"time": 512}
ChunkSpec: TypeAlias = Mapping[str, int] | str
DEFAULT_GLOBAL_HOURLY_CHUNKS: str = "auto"
DEFAULT_REGIONAL_HOURLY_CHUNKS: dict[str, int] = {"time": 24 * 31}
DEFAULT_PBL_CHUNKS: str = "auto"

CLOUD_COVER_LAYOUT_GLOBAL: str = "global-hourly-grid"
CLOUD_COVER_LAYOUT_LEGACY_REGIONAL: str = "legacy-regional"
CLOUD_COVER_LAYOUTS: tuple[str, ...] = (
    CLOUD_COVER_LAYOUT_GLOBAL,
    CLOUD_COVER_LAYOUT_LEGACY_REGIONAL,
)


def load_daily_spatial_fields(
    path: str | Path,
    dates: pd.DatetimeIndex,
    *,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
    climatology: bool = False,
) -> xr.Dataset:
    """Load only requested days/cells from prepared daily ERA5 T2m and Z500.

    Climatology uses month/day keys, then receives the requested actual dates.
    Returned arrays are in memory and own no open file resources.
    """
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Missing daily spatial input: {path}")
    if dates.empty or dates.hasnans or not dates.equals(dates.normalize()):
        raise ValueError("Requested dates must be nonempty finite UTC midnights.")
    if dates.tz is not None or dates.has_duplicates:
        raise ValueError("Requested dates must be unique and timezone-naive UTC.")
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as source:
        ds = _normalize_daily_spatial_fields(source, lat_bounds, lon_bounds)
        time = pd.DatetimeIndex(ds.time.values)
        if time.hasnans or time.has_duplicates or not time.equals(time.normalize()):
            raise ValueError(f"{path}: expected unique finite midnight timestamps.")
        if climatology:
            keys = list(zip(time.month, time.day, strict=True))
            if len(keys) != 366 or len(set(keys)) != 366:
                raise ValueError(f"{path}: climatology needs 366 unique month/day keys.")
            lookup = {key: index for index, key in enumerate(keys)}
            requested = list(zip(dates.month, dates.day, strict=True))
        else:
            lookup = {value: index for index, value in enumerate(time)}
            requested = list(dates)
        missing = [key for key in requested if key not in lookup]
        if missing:
            raise ValueError(f"{path}: missing required daily timestamps/keys: {missing}")
        selected = ds.isel(time=[lookup[key] for key in requested]).load()
        out = xr.Dataset(coords={
            "time": dates.values,
            "latitude": selected.latitude,
            "longitude": selected.longitude,
        }, attrs=dict(source.attrs))
        out.latitude.attrs["units"] = "degrees_north"
        out.longitude.attrs["units"] = "degrees_east"
        for source_name, name, units, divisor in (
            ("t2m", "t2m", "K", 1.0),
            ("z", "z500", "m", config.G_M_S2),
        ):
            values = np.asarray(selected[source_name].values, dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"{path}: {source_name} contains non-finite selected fields.")
            out[name] = (("time", "latitude", "longitude"), values / divisor)
            out[name].attrs["units"] = units
    return out


def _normalize_daily_spatial_fields(
    source: xr.Dataset,
    lat_bounds: tuple[float, float],
    lon_bounds: tuple[float, float],
) -> xr.Dataset:
    """Validate units/coordinates and crop lazily before any spatial reads."""
    rename = {}
    for canonical, aliases in (
        ("time", ("time", "valid_time")),
        ("latitude", ("latitude", "lat")),
        ("longitude", ("longitude", "lon")),
    ):
        found = [name for name in aliases if name in source.coords]
        if len(found) != 1 or source[found[0]].dims != (found[0],):
            raise ValueError(f"Expected one one-dimensional {canonical} coordinate.")
        rename[found[0]] = canonical
    ds = source.rename(rename)
    for name, allowed in (
        ("t2m", {"k", "kelvin"}),
        ("z", {"m**2s**-2", "m^2s^-2", "m2s-2", "m2/s2", "m^2/s^2"}),
    ):
        if name not in ds:
            raise ValueError(f"Daily spatial input is missing {name!r}.")
        units = str(ds[name].attrs.get("units", "")).lower().replace(" ", "")
        if units not in allowed:
            raise ValueError(f"Unexpected {name} units: {ds[name].attrs.get('units')!r}.")
    levels = [name for name in ("pressure_level", "level", "isobaricInhPa") if name in ds.coords]
    if len(levels) != 1:
        raise ValueError("Daily Z500 needs an unambiguous 500 hPa pressure coordinate.")
    level_name = levels[0]
    level = ds[level_name]
    pressure_units = str(level.attrs.get("units", "hPa" if level_name == "isobaricInhPa" else "")).lower()
    if pressure_units not in {"pa", "hpa", "millibars", "mbar"}:
        raise ValueError(f"Unsupported pressure units: {pressure_units!r}.")
    values = np.asarray(level.values).reshape(-1)
    expected = 50000.0 if pressure_units == "pa" else 500.0
    if values.size != 1 or not np.isclose(values[0], expected):
        raise ValueError("Daily spatial input must contain only 500 hPa geopotential.")
    if level_name in ds.dims:
        ds = ds.isel({level_name: 0}, drop=True)
    for name in ("t2m", "z"):
        if set(ds[name].dims) != {"time", "latitude", "longitude"}:
            raise ValueError(f"{name} must have only time, latitude, longitude dimensions.")
    if not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError("Daily spatial input needs decoded Gregorian timestamps.")
    lon = (np.asarray(ds.longitude.values, dtype=float) + 180.0) % 360.0 - 180.0
    ds = ds.assign_coords(longitude=("longitude", lon)).sortby(["latitude", "longitude"])
    for name, bounds, limits in (
        ("latitude", lat_bounds, (-90.0, 90.0)),
        ("longitude", lon_bounds, (-180.0, 180.0)),
    ):
        if not limits[0] <= bounds[0] < bounds[1] <= limits[1]:
            raise ValueError(f"Invalid {name} bounds: {bounds}.")
        axis = np.asarray(ds[name].values, dtype=float)
        if not np.isfinite(axis).all() or axis.size < 2 or np.any(np.diff(axis) <= 0):
            raise ValueError(f"{name} grid must be finite, unique, and increasing.")
        if axis[0] > bounds[0] or axis[-1] < bounds[1]:
            raise ValueError(f"Input {name} grid does not cover requested bounds {bounds}.")
        ds = ds.sel({name: slice(*bounds)})
        if ds.sizes[name] < 2:
            raise ValueError(f"Requested {name} bounds select fewer than two cells.")
    return ds[["t2m", "z"]].transpose("time", "latitude", "longitude")


SURFACE_DIAGNOSTIC_ROOTS: dict[str, str] = {
    "nslr": config.ERA5_NSLR_ROOT,
    "nssr": config.ERA5_NSSR_ROOT,
    "slhf": config.ERA5_SLHF_ROOT,
    "sshf": config.ERA5_SSHF_ROOT,
    "soil_moisture": config.ERA5_SOIL_MOISTURE_ROOT,
}

#file name stems
SURFACE_DIAGNOSTIC_FILE_STEMS: dict[str, str] = {
    "nslr": "nslr_hour_ERA5",
    "nssr": "nssr_hour_ERA5",
    "slhf": "slhf_hour_ERA5",
    "sshf": "sshf_hour_ERA5",
    "soil_moisture": "soil_moisture_hour_ERA5",
}


def open_era5_tas(
    *,
    years: Sequence[int] | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open ERA5 daily surface air temperature on the standard grid."""
    pattern = f"{config.ERA5_TAS_ROOT}/tas_daily_ERA5_*_2x2_bil.nc"
    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=chunks or DEFAULT_TAS_CHUNKS,
    )
    return _standardize_common_structure(ds)


def open_era5_lwa(
    *,
    zg_level: int = 500,
    years: Sequence[int] | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open ERA5 daily LWA fields for a single pressure level."""
    pattern = f"{config.ERA5_LWA_ROOT}/z{zg_level}/LWA_day_ERA5_2deg.{zg_level}.nc"
    path = _glob_required(pattern)[0]
    ds = _open_single_dataset(path, chunks=chunks or DEFAULT_LWA_CHUNKS)
    ds = _standardize_common_structure(ds)
    return _filter_dataset_time_years(ds, years)


def open_era5_lwa_threshold(
    *,
    region: str,
    quantile: str | float,
    zg_level: int = 500,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open ERA5 day-of-year LWA thresholds for a region and quantile."""
    q_token = _normalize_quantile_token(quantile)
    pattern = (
        f"{config.LWA_THRESH_ROOT}/{region}/ERA5/q{q_token}/"
        f"ERA5_LWAthresh_block_1970_2014_q{q_token}_{region}.{zg_level}.nc"
    )
    path = _glob_required(pattern)[0]
    ds = _open_single_dataset(path, chunks=chunks or DEFAULT_THRESHOLD_CHUNKS)
    return _standardize_common_structure(ds)


def open_era5_hw_threshold(
    *,
    region: str,
    quantile: str | float,
    method: str = "evolving",
    years: Sequence[int] | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open ERA5 heatwave thresholds for a region and quantile."""
    if method != "evolving":
        raise ValueError(
            "Unsupported HW threshold method. "
            "Only 'evolving' is implemented in data_io.py."
        )

    q_token = _normalize_quantile_token(quantile)
    pattern = (
        f"{config.HW_THRESH_ROOT}/{region}/ERA5/{method}/q{q_token}/"
        f"ERA5_HWthresh_{method}_1940_2024_tas_q{q_token}_{region}.nc"
    )
    path = _glob_required(pattern)[0]
    ds = _open_single_dataset(path, chunks=chunks or DEFAULT_THRESHOLD_CHUNKS)
    ds = _standardize_common_structure(ds)
    return _filter_dataset_year_coord(ds, years)


def open_era5_heat_budget(
    *,
    years: Sequence[int] | None = None,
    heat_budget_root: str | Path | None = None,
    region: str = "pnw_bartusek",
    bottom_boundary: str | int = "surface",
    top_boundary: str | int = 700,
    start_year_ehb: int = 1940,
    end_year_ehb: int = 2025,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open hourly regional ERA5 heat-budget diagnostics."""
    if heat_budget_root is None:
        heat_budget_root = era5_heat_budget_annual_root(
            region=region,
            bottom_boundary=bottom_boundary,
            top_boundary=top_boundary,
            start_year_ehb=start_year_ehb,
            end_year_ehb=end_year_ehb,
        )
    pattern = str(Path(heat_budget_root) / "heat_budget_*.nc")
    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=chunks or DEFAULT_HEAT_BUDGET_CHUNKS,
    )
    return _standardize_common_structure(ds)


def era5_heat_budget_annual_root(
    *,
    region: str,
    bottom_boundary: str | int,
    top_boundary: str | int,
    start_year_ehb: int,
    end_year_ehb: int,
) -> Path:
    """Return the annual-file directory for a saved Eulerian heat-budget run."""
    return (
        Path(config.ERA5_HEAT_BUDGET_SAVED_RESULTS_ROOT)
        / era5_heat_budget_run_name(
            region=region,
            bottom_boundary=bottom_boundary,
            top_boundary=top_boundary,
            start_year_ehb=start_year_ehb,
            end_year_ehb=end_year_ehb,
        )
        / "annual"
    )


def era5_heat_budget_run_name(
    *,
    region: str,
    bottom_boundary: str | int,
    top_boundary: str | int,
    start_year_ehb: int,
    end_year_ehb: int,
) -> str:
    """Return the saved-results directory name for a heat-budget run."""
    if start_year_ehb > end_year_ehb:
        raise ValueError("start_year_ehb must be less than or equal to end_year_ehb.")
    return (
        f"{region}_"
        f"{normalize_heat_budget_bottom_boundary(bottom_boundary)}_"
        f"{normalize_heat_budget_top_boundary(top_boundary)}_"
        f"{start_year_ehb}_{end_year_ehb}"
    )


def normalize_heat_budget_bottom_boundary(boundary: str | int) -> str:
    """Return the canonical token for the heat-budget bottom boundary."""
    token = str(boundary).strip()
    if token.lower() == "surface":
        return "surface"
    return normalize_heat_budget_top_boundary(boundary)


def normalize_heat_budget_top_boundary(boundary: str | int) -> str:
    """Return the canonical pressure-boundary token, for example ``700hPa``."""
    token = str(boundary).strip()
    if not token:
        raise ValueError("Heat-budget pressure boundary cannot be empty.")

    pressure = token[:-3] if token.lower().endswith("hpa") else token
    if not pressure.isdigit():
        raise ValueError(
            "Heat-budget pressure boundaries must be integer hPa values, "
            f"got {boundary!r}."
        )
    return f"{int(pressure)}hPa"


def open_era5_surface_diagnostic(
    name: str,
    *,
    years: Sequence[int] | None = None,
    chunks: ChunkSpec | None = None,
) -> xr.Dataset:
    """Open a local hourly gridded ERA5 surface diagnostic."""
    try:
        root = SURFACE_DIAGNOSTIC_ROOTS[name]
        stem = SURFACE_DIAGNOSTIC_FILE_STEMS[name]
    except KeyError as exc:
        valid = ", ".join(sorted(SURFACE_DIAGNOSTIC_ROOTS))
        raise ValueError(
            f"Unsupported ERA5 surface diagnostic {name!r}. Expected one of: {valid}."
        ) from exc

    pattern = f"{root}/{stem}_*.nc"
    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=chunks or DEFAULT_GLOBAL_HOURLY_CHUNKS,
    )
    return _standardize_common_structure(ds)


def open_era5_pbl_p(
    *,
    region: str,
    years: Sequence[int] | None = None,
    chunks: ChunkSpec | None = None,
) -> xr.Dataset:
    """Open ad hoc hourly ARCO PBL top-pressure fields for one region.

    PBL fields are not inputs to the active Stage-1 production contract.
    """
    pattern = f"{config.ERA5_PBL_P_ROOT}/{region}/ERA5_ARCO_pbl_p_*.nc"
    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=chunks or DEFAULT_PBL_CHUNKS,
    )
    return _standardize_common_structure(ds)


def open_era5_total_cloud_cover(
    *,
    years: Sequence[int] | None = None,
    chunks: ChunkSpec | None = None,
    source_layout: str = CLOUD_COVER_LAYOUT_GLOBAL,
    root: str | Path | None = None,
    region: str | None = None,
) -> xr.Dataset:
    """Open hourly ERA5 cloud cover from an explicit supported source layout."""
    if source_layout not in CLOUD_COVER_LAYOUTS:
        valid = ", ".join(CLOUD_COVER_LAYOUTS)
        raise ValueError(
            f"Unsupported cloud-cover source layout {source_layout!r}. "
            f"Expected one of: {valid}."
        )

    if source_layout == CLOUD_COVER_LAYOUT_GLOBAL:
        if region is not None:
            raise ValueError(
                "region must be omitted for the global-hourly-grid cloud-cover layout."
            )
        source_root = Path(root or config.ERA5_CLOUD_COVER_ROOT)
        pattern = str(source_root / "cloud_cover_hour_ERA5_*.nc")
        source_chunks = chunks or DEFAULT_GLOBAL_HOURLY_CHUNKS
    else:
        if root is None:
            raise ValueError(
                "root is required for the temporary legacy-regional "
                "cloud-cover layout."
            )
        if region is None:
            raise ValueError(
                "region is required for the legacy-regional cloud-cover layout."
            )
        source_root = Path(root)
        pattern = str(
            source_root / f"ERA5_ARCO_total_cloud_cover_{region}_*.nc"
        )
        source_chunks = chunks or DEFAULT_REGIONAL_HOURLY_CHUNKS

    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=source_chunks,
    )
    ds = _standardize_common_structure(ds)

    if source_layout == CLOUD_COVER_LAYOUT_GLOBAL:
        if "tcc" not in ds:
            raise ValueError(
                "ERA5 cloud-cover dataset is missing required variable 'tcc'."
            )
        ds = ds.rename({"tcc": "total_cloud_cover"})
        source_variable = "tcc"
        source_region = ""
        preaggregated = False
    else:
        if "total_cloud_cover" not in ds:
            raise ValueError(
                "Legacy regional cloud-cover dataset is missing required "
                "variable 'total_cloud_cover'."
            )
        source_variable = "total_cloud_cover"
        source_region = str(region)
        preaggregated = True

    provenance = {
        "source_variable": source_variable,
        "source_layout": source_layout,
        "source_root": str(source_root),
        "source_region": source_region,
        "spatially_preaggregated": int(preaggregated),
    }
    ds.attrs.update(provenance)
    ds["total_cloud_cover"].attrs.update(provenance)
    return ds


def _normalize_quantile_token(quantile: str | float) -> str:
    """Return the quantile token used in filenames."""
    if isinstance(quantile, str):
        token = quantile.strip()
        if not token:
            raise ValueError("Quantile token cannot be empty.")
        return token.removeprefix("q")

    if isinstance(quantile, int):
        return str(quantile)

    if isinstance(quantile, float):
        if quantile.is_integer():
            return str(int(quantile))
        return format(quantile, "g").replace(".", "p")

    raise TypeError(f"Unsupported quantile type: {type(quantile)!r}")


def _glob_required(pattern: str) -> list[str]:
    """Return sorted glob matches or fail with the attempted pattern."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return paths


def _filter_yearly_files(
    paths: Sequence[str],
    years: Sequence[int] | None,
) -> list[str]:
    """Keep only files that contain one of the requested 4-digit years."""
    if years is None:
        return list(paths)

    year_tokens = {str(year) for year in years}
    selected = []
    matched_years = set()
    for path in paths:
        match = re.search(r"(?<!\d)(\d{4})(?!\d)", Path(path).name)
        if match and match.group(1) in year_tokens:
            selected.append(path)
            matched_years.add(match.group(1))

    if not selected:
        requested = ", ".join(sorted(year_tokens))
        raise FileNotFoundError(
            f"No files matched requested years ({requested}) in provided paths."
        )
    missing_years = sorted(year_tokens.difference(matched_years))
    if missing_years:
        requested = ", ".join(missing_years)
        raise FileNotFoundError(
            f"No files matched requested years ({requested}) in provided paths."
        )
    return selected


def _filter_dataset_time_years(
    ds: xr.Dataset,
    years: Sequence[int] | None,
    *,
    time_dim: str = "time",
) -> xr.Dataset:
    """Restrict a time-indexed dataset to requested years after opening."""
    if years is None:
        return ds

    if time_dim not in ds.coords:
        raise ValueError(f"Dataset is missing required time coordinate {time_dim!r}.")

    out = ds.where(ds[time_dim].dt.year.isin(years), drop=True)
    if out.sizes.get(time_dim, 0) == 0:
        requested = ", ".join(str(year) for year in sorted(set(years)))
        raise ValueError(f"No {time_dim!r} values matched requested years ({requested}).")
    return out


def _filter_dataset_year_coord(
    ds: xr.Dataset,
    years: Sequence[int] | None,
    *,
    year_dim: str = "year",
) -> xr.Dataset:
    """Restrict a dataset with a year coordinate to requested years."""
    if years is None or year_dim not in ds.coords:
        return ds

    out = ds.where(ds[year_dim].isin(years), drop=True)
    if out.sizes.get(year_dim, 0) == 0:
        requested = ", ".join(str(year) for year in sorted(set(years)))
        raise ValueError(f"No {year_dim!r} values matched requested years ({requested}).")
    return out


def _open_single_dataset(
    path: str,
    *,
    chunks: ChunkSpec | None = None,
) -> xr.Dataset:
    """Open a single NetCDF dataset lazily."""
    kwargs: dict[str, Any] = {"engine": "h5netcdf"}
    if chunks is not None:
        kwargs["chunks"] = _normalize_chunks(chunks)
    return xr.open_dataset(path, **kwargs)


def _open_multiple_datasets(
    paths: Sequence[str],
    *,
    combine: str,
    chunks: ChunkSpec | None = None,
) -> xr.Dataset:
    """Open multiple NetCDF files lazily using xarray's multi-file loader."""
    kwargs: dict[str, Any] = {
        "combine": combine,
        "data_vars": "all",
        "parallel": True,
        "engine": "h5netcdf",
    }
    if chunks is not None:
        kwargs["chunks"] = _normalize_chunks(chunks)
    return xr.open_mfdataset(list(paths), **kwargs)


def _normalize_chunks(chunks: ChunkSpec) -> Mapping[str, int] | str:
    """Return chunks in a form accepted by xarray open functions."""
    if isinstance(chunks, str):
        return chunks
    return dict(chunks)


def _standardize_common_structure(ds: xr.Dataset) -> xr.Dataset:
    """Apply only shared structural cleanup across source datasets."""
    rename_map: dict[str, str] = {}
    if "valid_time" in ds.dims or "valid_time" in ds.coords:
        rename_map["valid_time"] = "time"
    if "latitude" in ds.dims or "latitude" in ds.coords:
        rename_map["latitude"] = "lat"
    if "longitude" in ds.dims or "longitude" in ds.coords:
        rename_map["longitude"] = "lon"

    if rename_map:
        ds = ds.rename(rename_map)

    drop_names = [
        name
        for name in ("valid_time_bnds", "bnds")
        if name in ds.variables or name in ds.coords
    ]
    if drop_names:
        ds = ds.drop_vars(drop_names, errors="ignore")

    return ds
