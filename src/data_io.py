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
from typing import Any

import xarray as xr

from . import config


DEFAULT_TAS_CHUNKS: dict[str, int] = {"time": 365}
DEFAULT_LWA_CHUNKS: dict[str, int] = {"time": 3650, "lat": 35, "lon": 180}
DEFAULT_THRESHOLD_CHUNKS: dict[str, int] = {"dayofyear": 365}
DEFAULT_HEAT_BUDGET_CHUNKS: dict[str, int] = {"time": 512}


def open_era5_tas(
    *,
    years: Sequence[int] | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open ERA5 daily surface air temperature on the standard grid."""
    pattern = f"{config.ERA5_TAS_ROOT}/tas_daily_ERA_*_2x2_bil.nc"
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
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open ERA5 daily LWA fields for a single pressure level."""
    pattern = f"{config.ERA5_LWA_ROOT}/z{zg_level}/LWA_day_ERA5_2deg.{zg_level}.nc"
    path = _glob_required(pattern)[0]
    ds = _open_single_dataset(path, chunks=chunks or DEFAULT_LWA_CHUNKS)
    return _standardize_common_structure(ds)


def open_era5_lwa_threshold(
    *,
    region: str,
    quantile: str | int | float,
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
    quantile: str | int | float,
    method: str = "evolving",
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
        f"ERA5_HWthresh_{method}_1950_2024_tas_q{q_token}_{region}.nc"
    )
    path = _glob_required(pattern)[0]
    ds = _open_single_dataset(path, chunks=chunks or DEFAULT_THRESHOLD_CHUNKS)
    return _standardize_common_structure(ds)


def open_era5_heat_budget(
    *,
    years: Sequence[int] | None = None,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open hourly regional ERA5 heat-budget diagnostics."""
    pattern = f"{config.ERA5_HEAT_BUDGET_ROOT}/heat_budget_*.nc"
    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=chunks or DEFAULT_HEAT_BUDGET_CHUNKS,
    )
    return _standardize_common_structure(ds)


def _normalize_quantile_token(quantile: str | int | float) -> str:
    """Return the quantile token used in filenames."""
    if isinstance(quantile, str):
        token = quantile.strip()
        if not token:
            raise ValueError("Quantile token cannot be empty.")
        return token[1:] if token.startswith("q") else token

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
    for path in paths:
        match = re.search(r"(?<!\d)(\d{4})(?!\d)", path)
        if match and match.group(1) in year_tokens:
            selected.append(path)

    if not selected:
        requested = ", ".join(sorted(year_tokens))
        raise FileNotFoundError(
            f"No files matched requested years ({requested}) in provided paths."
        )
    return selected


def _open_single_dataset(
    path: str,
    *,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open a single NetCDF dataset lazily."""
    kwargs: dict[str, Any] = {"engine": "h5netcdf"}
    if chunks is not None:
        kwargs["chunks"] = dict(chunks)
    return xr.open_dataset(path, **kwargs)


def _open_multiple_datasets(
    paths: Sequence[str],
    *,
    combine: str,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open multiple NetCDF files lazily using xarray's multi-file loader."""
    kwargs: dict[str, Any] = {
        "combine": combine,
        "data_vars": "all",
        "parallel": True,
        "engine": "h5netcdf",
    }
    if chunks is not None:
        kwargs["chunks"] = dict(chunks)
    return xr.open_mfdataset(list(paths), **kwargs)


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


__all__ = [
    "open_era5_tas",
    "open_era5_lwa",
    "open_era5_lwa_threshold",
    "open_era5_hw_threshold",
    "open_era5_heat_budget",
]
