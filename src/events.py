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

#future to do:
# add event-summary for "LWA_a" events (similar to current tas-defined events)

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
    hw_climatology: xr.DataArray,
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
    climatology_time = preprocess.threshold_to_time(
        hw_climatology,
        tas_region["time"],
        name="tas_climatology",
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
    products = {
        "tas_region": tas_region,
        "tas_climatology": climatology_time,
        "hw_threshold": threshold_time,
        "hw_exceedance_mask": mask,
        "hw_event_id": event_id,
    }
    return products


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



def build_event_summary_table(
    ds: xr.Dataset,
    event_id: str | xr.DataArray,
    *,
    time_dim: str = "time",
    peak_variable: str = "tas_region",
    tas_name: str = "tas_region",
    tas_climatology_name: str = "tas_climatology",
    tas_anom_name: str | None = None,
    hw_threshold_name: str = "hw_threshold",
    lwa_a_name: str = "lwa_a_region",
    lwa_c_name: str = "lwa_c_region",
) -> xr.Dataset:
    """Return one row per nonzero event ID with event-level summary metrics.

    The returned table uses an ``event`` dimension. ``start_time`` and
    ``end_time`` are actual timestamps from the input time axis, so they can be
    passed directly to ``ds.sel(time=slice(start_time, end_time))``.

    Metric calculations are performed once per calendar day. This avoids
    over-counting daily-native quantities that have been projected onto an
    hourly Stage-1 analysis axis. For a truly daily input, this is equivalent
    to using the original time axis.

    Parameters
    ----------
    ds:
        Time-indexed analysis dataset containing event IDs and candidate metric
        variables.
    event_id:
        Name of the event-ID variable in ``ds`` or a 1D event-ID DataArray.
        Non-event times must be labelled ``0``.
    time_dim:
        Name of the time dimension.
    peak_variable:
        Variable used to define ``peak_time`` and ``peak_value``. By default
        this is ``tas_region``, appropriate for heatwave-defined events.
    tas_name, tas_climatology_name, tas_anom_name, hw_threshold_name,
    lwa_a_name, lwa_c_name:
        Variable names used to compute the requested event-level metrics.
        Missing optional variables are reported as NaN-valued metrics.

    Returns
    -------
    xr.Dataset
        Event summary table with variables: ``event_id``, ``start_time``,
        ``end_time``, ``duration``, ``peak_time``, ``peak_value``,
        ``tas_peak``, ``tas_anom_peak``, ``tas_excess_peak``,
        ``tas_excess_integral``, ``lwa_a_peak``, and ``lwa_c_peak``.
    """
    _validate_event_summary_inputs(ds, event_id, time_dim, peak_variable)

    event_da = ds[event_id] if isinstance(event_id, str) else event_id
    event_da = event_da.transpose(time_dim)

    time_values = event_da[time_dim].values
    event_values = event_da.values.astype(np.int64)
    day_values = _as_calendar_days(time_values)

    event_ids = np.array(sorted(int(value) for value in np.unique(event_values) if value > 0))
    n_events = event_ids.size

    columns: dict[str, list[object]] = {
        "event_id": [],
        "start_time": [],
        "end_time": [],
        "duration": [],
        "peak_time": [],
        "peak_value": [],
        "tas_peak": [],
        "tas_anom_peak": [],
        "tas_excess_peak": [],
        "tas_excess_integral": [],
        "lwa_a_peak": [],
        "lwa_c_peak": [],
    }

    for eid in event_ids:
        idx_full = np.flatnonzero(event_values == eid)
        idx_daily = _first_index_per_calendar_day(idx_full, day_values)

        peak_idx, peak_value = _event_peak_index_and_value(
            ds[peak_variable],
            idx_daily,
            time_dim=time_dim,
        )

        tas_values = _values_for_indices(ds, tas_name, idx_daily, time_dim=time_dim)
        tas_anom_values = _event_tas_anomaly_values(
            ds,
            idx_daily,
            tas_name=tas_name,
            tas_climatology_name=tas_climatology_name,
            tas_anom_name=tas_anom_name,
            time_dim=time_dim,
        )
        tas_excess_values = _event_tas_excess_values(
            ds,
            idx_daily,
            tas_name=tas_name,
            hw_threshold_name=hw_threshold_name,
            time_dim=time_dim,
        )
        lwa_a_values = _values_for_indices(ds, lwa_a_name, idx_daily, time_dim=time_dim)
        lwa_c_values = _values_for_indices(ds, lwa_c_name, idx_daily, time_dim=time_dim)

        columns["event_id"].append(eid)
        columns["start_time"].append(time_values[idx_full[0]])
        columns["end_time"].append(time_values[idx_full[-1]])
        columns["duration"].append(idx_daily.size)
        columns["peak_time"].append(time_values[peak_idx] if peak_idx is not None else np.datetime64("NaT"))
        columns["peak_value"].append(peak_value)
        columns["tas_peak"].append(_nanmax_or_nan(tas_values))
        columns["tas_anom_peak"].append(_nanmax_or_nan(tas_anom_values))
        columns["tas_excess_peak"].append(_nanmax_or_nan(tas_excess_values))
        columns["tas_excess_integral"].append(_nansum_or_nan(tas_excess_values)) #nansum != nanmax
        columns["lwa_a_peak"].append(_nanmax_or_nan(lwa_a_values))
        columns["lwa_c_peak"].append(_nanmax_or_nan(lwa_c_values))

    event_coord = np.arange(n_events, dtype=np.int64)
    out = xr.Dataset(
        data_vars={
            "event_id": ("event", np.asarray(columns["event_id"], dtype=np.int64)),
            "start_time": ("event", np.asarray(columns["start_time"], dtype="datetime64[ns]")),
            "end_time": ("event", np.asarray(columns["end_time"], dtype="datetime64[ns]")),
            "duration": ("event", np.asarray(columns["duration"], dtype=np.int64)),
            "peak_time": ("event", np.asarray(columns["peak_time"], dtype="datetime64[ns]")),
            "peak_value": ("event", np.asarray(columns["peak_value"], dtype=float)),
            "tas_peak": ("event", np.asarray(columns["tas_peak"], dtype=float)),
            "tas_anom_peak": ("event", np.asarray(columns["tas_anom_peak"], dtype=float)),
            "tas_excess_peak": ("event", np.asarray(columns["tas_excess_peak"], dtype=float)),
            "tas_excess_integral": ("event", np.asarray(columns["tas_excess_integral"], dtype=float)),
            "lwa_a_peak": ("event", np.asarray(columns["lwa_a_peak"], dtype=float)),
            "lwa_c_peak": ("event", np.asarray(columns["lwa_c_peak"], dtype=float)),
        },
        coords={"event": event_coord},
        attrs={
            "event_id_source": event_id if isinstance(event_id, str) else event_da.name,
            "peak_variable": peak_variable,
            "metric_time_basis": "one sample per calendar day",
            "slice_semantics": f"Use ds.sel({time_dim}=slice(start_time, end_time)).",
        },
    )
    out["duration"].attrs["units"] = "days"
    out["tas_excess_integral"].attrs["description"] = (
        "Sum over event days of max(tas - hw_threshold, 0)."
    )
    return out


def _validate_event_summary_inputs(
    ds: xr.Dataset,
    event_id: str | xr.DataArray,
    time_dim: str,
    peak_variable: str,
) -> None:
    """Validate inputs for event summary table construction."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(ds).__name__}.")

    if time_dim not in ds.coords:
        raise ValueError(f"Dataset is missing required time coordinate {time_dim!r}.")

    if peak_variable not in ds:
        raise ValueError(f"Dataset is missing peak variable {peak_variable!r}.")

    if isinstance(event_id, str):
        if event_id not in ds:
            raise ValueError(f"Dataset is missing event-ID variable {event_id!r}.")
        event_da = ds[event_id]
    elif isinstance(event_id, xr.DataArray):
        event_da = event_id
    else:
        raise TypeError("event_id must be a variable name or an xarray.DataArray.")

    if event_da.dims != (time_dim,):
        raise ValueError(
            "event_id must be 1D with dims "
            f"({time_dim!r},); got {event_da.dims!r}."
        )



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


def _as_calendar_days(time_values: np.ndarray) -> np.ndarray:
    """Return datetime64[D] calendar days from datetime64-like values."""
    try:
        return np.asarray(time_values).astype("datetime64[D]")
    except TypeError as exc:
        raise TypeError(
            "Event summary tables currently require numpy datetime64 time coordinates."
        ) from exc


def _first_index_per_calendar_day(idx_full: np.ndarray, day_values: np.ndarray) -> np.ndarray:
    """Return the first input index for each calendar day represented in idx_full."""
    event_days = day_values[idx_full]
    _, first_positions = np.unique(event_days, return_index=True)
    return idx_full[np.sort(first_positions)]


def _event_peak_index_and_value(
    da: xr.DataArray,
    indices: np.ndarray,
    *,
    time_dim: str,
) -> tuple[int | None, float]:
    """Return the absolute time index and value of the event peak."""
    values = _values_from_dataarray_for_indices(da, indices, time_dim=time_dim)
    if values.size == 0 or np.all(np.isnan(values)):
        return None, np.nan

    local_idx = int(np.nanargmax(values))
    return int(indices[local_idx]), float(values[local_idx])


def _values_for_indices(
    ds: xr.Dataset,
    name: str,
    indices: np.ndarray,
    *,
    time_dim: str,
) -> np.ndarray:
    """Return variable values at absolute positional indices, or NaNs if missing."""
    if name not in ds:
        return np.full(indices.size, np.nan, dtype=float)
    return _values_from_dataarray_for_indices(ds[name], indices, time_dim=time_dim)


def _values_from_dataarray_for_indices(
    da: xr.DataArray,
    indices: np.ndarray,
    *,
    time_dim: str,
) -> np.ndarray:
    """Extract 1D float values from a time-indexed DataArray by positional index."""
    if da.dims != (time_dim,):
        raise ValueError(
            f"Metric variable {da.name!r} must be 1D with dims ({time_dim!r},); "
            f"got {da.dims!r}."
        )
    return np.asarray(da.isel({time_dim: indices}).values, dtype=float)


def _event_tas_anomaly_values(
    ds: xr.Dataset,
    indices: np.ndarray,
    *,
    tas_name: str,
    tas_climatology_name: str,
    tas_anom_name: str | None,
    time_dim: str,
) -> np.ndarray:
    """Return tas anomaly values when available; otherwise NaNs."""
    if tas_anom_name is not None and tas_anom_name in ds:
        return _values_for_indices(ds, tas_anom_name, indices, time_dim=time_dim)

    if tas_name in ds and tas_climatology_name in ds:
        tas = _values_for_indices(ds, tas_name, indices, time_dim=time_dim)
        clim = _values_for_indices(ds, tas_climatology_name, indices, time_dim=time_dim)
        return tas - clim

    return np.full(indices.size, np.nan, dtype=float)


def _event_tas_excess_values(
    ds: xr.Dataset,
    indices: np.ndarray,
    *,
    tas_name: str,
    hw_threshold_name: str,
    time_dim: str,
) -> np.ndarray:
    """Return non-negative tas threshold-excess values when available."""
    if tas_name not in ds or hw_threshold_name not in ds:
        return np.full(indices.size, np.nan, dtype=float)

    tas = _values_for_indices(ds, tas_name, indices, time_dim=time_dim)
    threshold = _values_for_indices(ds, hw_threshold_name, indices, time_dim=time_dim)
    return np.maximum(tas - threshold, 0.0)


def _nanmax_or_nan(values: np.ndarray) -> float:
    """Return nanmax or NaN for empty/all-NaN inputs."""
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nanmax(values))


def _nansum_or_nan(values: np.ndarray) -> float:
    """Return nansum or NaN for empty/all-NaN inputs."""
    if values.size == 0 or np.all(np.isnan(values)):
        return np.nan
    return float(np.nansum(values))
