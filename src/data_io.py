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

import xarray as xr

from . import config


DEFAULT_TAS_CHUNKS: dict[str, int] = {"time": 365}
DEFAULT_LWA_CHUNKS: dict[str, int] = {"time": 3650, "lat": 35, "lon": 180}
DEFAULT_THRESHOLD_CHUNKS: dict[str, int] = {"dayofyear": 365}
DEFAULT_HEAT_BUDGET_CHUNKS: dict[str, int] = {"time": 512}
ChunkSpec: TypeAlias = Mapping[str, int] | str
DEFAULT_GLOBAL_HOURLY_CHUNKS: str = "auto"
DEFAULT_REGIONAL_HOURLY_CHUNKS: str = "auto"
DEFAULT_PBL_CHUNKS: str = "auto"


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
    years: Sequence[int] | None = None,
    chunks: ChunkSpec | None = None,
) -> xr.Dataset:
    """Open local hourly ARCO PBL top pressure fields."""
    pattern = f"{config.ERA5_PBL_P_ROOT}/ERA5_ARCO_pbl_p_*.nc"
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
    region: str,
    years: Sequence[int] | None = None,
    chunks: ChunkSpec | None = None,
) -> xr.Dataset:
    """Open local hourly ARCO total cloud cover regional time series."""
    pattern = (
        f"{config.ERA5_CLOUD_COVER_ROOT}/"
        f"ERA5_ARCO_total_cloud_cover_{region}_*.nc"
    )
    paths = _glob_required(pattern)
    paths = _filter_yearly_files(paths, years)
    ds = _open_multiple_datasets(
        paths,
        combine="by_coords",
        chunks=chunks or DEFAULT_REGIONAL_HOURLY_CHUNKS,
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
