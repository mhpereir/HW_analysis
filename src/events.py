"""Event construction utilities for the heatwave analysis pipeline.

Pipeline role:
- Convert selection masks into first-class event definitions.

Responsibilities:
- Build first-class daily event-ID products from threshold exceedance inputs.
- Convert boolean masks into contiguous event IDs.
- Filter events by duration.
- Identify event peaks.
- Assign event ranks.
- Extract event-level summary metadata.

Out of scope:
- Combining, ranking, or filtering events for a specific analysis selection.
- Composite computation.
- Plotting.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import xarray as xr

from . import preprocess


def mask_to_event_ids(
    mask: xr.DataArray,
    *,
    time_dim: str = "time",
    min_duration: int = 1,
    name: str = "event_id",
) -> xr.DataArray:
    """Convert a 1D boolean event mask into contiguous integer event IDs.

    CanESM/member-aware event IDs are intentionally deferred. When added, each
    ensemble member should use its own event ID scheme; ensemble-wide ranking
    should happen downstream.
    """
    _validate_event_id_inputs(mask, time_dim, min_duration)

    values = _label_1d_events(mask.values, min_duration=min_duration)
    return xr.DataArray(
        values,
        dims=(time_dim,),
        coords={time_dim: mask[time_dim]},
        name=name,
        attrs={
            "description": "Contiguous event IDs; 0 indicates non-event days.",
            "min_duration": min_duration,
        },
    )


def build_hw_event_ids(
    tas: xr.DataArray,
    hw_threshold: xr.DataArray,
    *,
    region: str,
    min_duration: int = 1,
) -> dict[str, xr.DataArray]:
    """Build daily heatwave threshold mask and event IDs from regional tas.

    This is an ERA5-style regional time-series helper. It keeps event definition
    logic in this module while leaving later event combinations/ranking to
    ``selectors.py``.
    """
    tas_region = preprocess.compute_region_mean(tas, region)
    tas_region = preprocess.floor_daily_time(tas_region)
    threshold_time = preprocess.threshold_to_time(
        hw_threshold,
        tas_region["time"],
        name="hw_threshold",
    )
    mask = preprocess.exceedance_mask(
        tas_region,
        threshold_time,
        mode="above",
        name="hw_exceedance_mask",
    )
    event_id = mask_to_event_ids(
        mask,
        min_duration=min_duration,
        name="hw_event_id",
    )
    return {
        "tas_region": tas_region,
        "hw_threshold": threshold_time,
        "hw_exceedance_mask": mask,
        "hw_event_id": event_id,
    }


def build_lwa_event_ids(
    lwa: xr.DataArray,
    lwa_threshold: xr.DataArray,
    *,
    region: str,
    variable: str = "LWA_a",
    years: Sequence[int] | None = None,
    min_duration: int = 1,
) -> dict[str, xr.DataArray]:
    """Build daily LWA-family threshold mask and event IDs.

    ``variable`` controls output names and keys for LWA variants such as
    ``LWA``, ``LWA_a``, and ``LWA_c``. The input arrays should already be the
    matching source and threshold variables.
    """
    key = _lwa_variable_key(variable)
    lwa = _select_years(lwa, years)
    lwa_region = preprocess.compute_region_mean(lwa, region)
    lwa_region = preprocess.floor_daily_time(lwa_region)
    threshold_time = preprocess.threshold_to_time(
        lwa_threshold,
        lwa_region["time"],
        name=f"{key}_threshold",
    )
    mask = preprocess.exceedance_mask(
        lwa_region,
        threshold_time,
        mode="above",
        name=f"{key}_exceedance_mask",
    )
    event_id = mask_to_event_ids(
        mask,
        min_duration=min_duration,
        name=f"{key}_event_id",
    )
    return {
        f"{key}_region": lwa_region,
        f"{key}_threshold": threshold_time,
        f"{key}_exceedance_mask": mask,
        f"{key}_event_id": event_id,
    }


def _label_1d_events(mask_1d: np.ndarray, min_duration: int) -> np.ndarray:
    """Label contiguous true runs in a 1D mask, filtering short events."""
    mask_array = np.asarray(mask_1d)
    mask_bool = np.where(mask_array == True, True, False)
    event_ids = np.zeros(mask_bool.shape, dtype=np.int64)

    next_event_id = 1
    idx = 0
    n = mask_bool.size
    while idx < n:
        if not mask_bool[idx]:
            idx += 1
            continue

        start = idx
        while idx < n and mask_bool[idx]:
            idx += 1
        stop = idx

        if stop - start >= min_duration:
            event_ids[start:stop] = next_event_id
            next_event_id += 1

    return event_ids


def _validate_event_id_inputs(
    mask: xr.DataArray,
    time_dim: str,
    min_duration: int,
) -> None:
    """Validate inputs for 1D event ID construction."""
    if time_dim not in mask.dims:
        raise ValueError(f"Mask is missing required time dimension {time_dim!r}.")

    if mask.dims != (time_dim,):
        raise ValueError(
            "mask_to_event_ids currently accepts only 1D masks with dims "
            f"({time_dim!r},); got dims {mask.dims!r}."
        )

    if min_duration < 1:
        raise ValueError("min_duration must be >= 1.")


def _select_years(da: xr.DataArray, years: Sequence[int] | None) -> xr.DataArray:
    """Restrict a time-indexed DataArray to requested years when provided."""
    if years is None:
        return da

    return da.where(da["time"].dt.year.isin(years), drop=True)


def _lwa_variable_key(variable: str) -> str:
    """Return the lower-case output key prefix for an LWA-family variable."""
    valid = {"LWA", "LWA_a", "LWA_c"}
    if variable not in valid:
        available = ", ".join(sorted(valid))
        raise ValueError(f"Unsupported LWA variable {variable!r}. Expected one of: {available}.")

    return variable.lower()

