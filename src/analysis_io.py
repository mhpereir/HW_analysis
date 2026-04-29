"""I/O for internal analysis products in the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: Stage 1/Stage 2 handoff for saved analysis-ready artifacts.

Responsibilities:
- Save the harmonized analysis-ready regional dataset produced by Stage 1.
- Reopen saved analysis-ready datasets for Stage 2 workflows.
- Validate expected metadata and dataset conventions on read.
- Manage stable internal filenames or paths for reusable pipeline products.

Out of scope:
- Raw source data loading.
- Source-specific filename handling.
- Scientific preprocessing or harmonization logic.
- Event logic, composites, or plotting.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HARMONIZED_TIMESERIES_PATH = (
    REPO_ROOT / "results" / "stage1" / "harmonized_regional_timeseries.nc"
)
EXPECTED_PIPELINE_STAGE = "stage_1_harmonized_regional_timeseries"
DEFAULT_TIME_DIM = "time"
REQUIRED_HARMONIZED_VARIABLES: frozenset[str] = frozenset(
    {
        "T_mean",
        "volume",
        "dTdt",
        "advection",
        "adiabatic",
        "diabatic",
        "tas_region",
        "tas_climatology",
        "hw_threshold",
        "hw_flag",
        "hw_event_id",
        "lwa_a_region",
        "lwa_a_threshold",
        "lwa_a_flag",
        "lwa_a_event_id",
        "lwa_c_region",
        "lwa_c_threshold",
        "lwa_c_flag",
        "lwa_c_event_id",
    }
)


def save_harmonized_timeseries(
    ds: xr.Dataset,
    path: str | Path = DEFAULT_HARMONIZED_TIMESERIES_PATH,
) -> Path:
    """Save the Stage-1 harmonized regional time-series dataset."""
    _validate_harmonized_timeseries(ds)

    output_path = Path(path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    netcdf_ds = _prepare_for_netcdf(ds)
    netcdf_ds.to_netcdf(output_path, engine="h5netcdf")
    return output_path


def open_harmonized_timeseries(
    path: str | Path = DEFAULT_HARMONIZED_TIMESERIES_PATH,
    *,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open a saved Stage-1 harmonized regional time-series dataset."""
    input_path = Path(path).expanduser().resolve()
    kwargs: dict[str, Any] = {"engine": "h5netcdf"}
    if chunks is not None:
        kwargs["chunks"] = dict(chunks)

    ds = xr.open_dataset(input_path, **kwargs)
    _validate_harmonized_timeseries(ds)
    return ds


def _validate_harmonized_timeseries(ds: xr.Dataset) -> None:
    """Validate the Stage-1 dataset contract shared by save and open."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(ds).__name__}.")

    pipeline_stage = ds.attrs.get("pipeline_stage")
    if pipeline_stage != EXPECTED_PIPELINE_STAGE:
        raise ValueError(
            "Expected harmonized Stage-1 dataset with "
            f"pipeline_stage={EXPECTED_PIPELINE_STAGE!r}; got {pipeline_stage!r}."
        )

    time_dim = str(ds.attrs.get("time_axis", DEFAULT_TIME_DIM))
    if time_dim not in ds.coords:
        raise ValueError(f"Harmonized dataset is missing time coordinate {time_dim!r}.")

    missing = sorted(REQUIRED_HARMONIZED_VARIABLES.difference(ds.data_vars))
    if missing:
        raise ValueError(
            "Harmonized dataset is missing required variables: "
            f"{', '.join(missing)}."
        )


def _prepare_for_netcdf(ds: xr.Dataset) -> xr.Dataset:
    """Return a NetCDF-safe view of the dataset without mutating the caller."""
    out = ds.copy(deep=False)
    out.attrs = _normalize_attrs(out.attrs)

    for name in out.variables:
        out[name].attrs = _normalize_attrs(out[name].attrs)

    for name in out.data_vars:
        if out[name].dtype == bool:
            out[name] = out[name].astype(np.int8)
            out[name].attrs = _normalize_attrs(out[name].attrs)

    return out


def _normalize_attrs(attrs: Mapping[Any, Any]) -> dict[Any, Any]:
    """Convert Python booleans in attrs to NetCDF-compatible integers."""
    normalized: dict[Any, Any] = {}
    for key, value in attrs.items():
        if isinstance(value, bool):
            normalized[key] = int(value)
        elif isinstance(value, np.bool_):
            normalized[key] = int(value)
        else:
            normalized[key] = value
    return normalized
