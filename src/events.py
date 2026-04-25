"""Event construction utilities for the heatwave analysis pipeline.

Pipeline role:
- Convert selection masks into first-class event definitions.

Responsibilities:
- Convert boolean masks into contiguous event IDs.
- Filter events by duration.
- Identify event peaks.
- Assign event ranks.
- Extract event-level summary metadata.

Out of scope:
- Building selection masks.
- Composite computation.
- Plotting.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


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
