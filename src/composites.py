"""Composite generation for the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: composite generation within top-level Stage 2.

Responsibilities:
- Build peak-aligned composites.
- Extract event-centered windows.
- Compute ensemble member composites.
- Compute event-mean and ensemble-mean reductions.
- Compute percentile envelopes across events or members.
- Extract top-event traces or subsets.

Out of scope:
- Raw data loading.
- Event selection logic.
- Plot rendering.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


DEFAULT_TIME_DIM = "time"
DEFAULT_EVENT_DIM = "event"
DEFAULT_LAG_DIM = "lag_hour"
DEFAULT_PEAK_TIME_NAME = "peak_time"
DEFAULT_EVENT_ID_NAME = "event_id"


def stack_events_centered_on_peak(
    ds: xr.Dataset,
    event_table: xr.Dataset,
    *,
    variables: Sequence[str] | None = None,
    pre_days: int = 5,
    post_days: int = 5,
    time_dim: str = DEFAULT_TIME_DIM,
    event_dim: str = DEFAULT_EVENT_DIM,
    lag_dim: str = DEFAULT_LAG_DIM,
    peak_time_name: str = DEFAULT_PEAK_TIME_NAME,
    event_id_name: str = DEFAULT_EVENT_ID_NAME,
    fill_missing: bool = True,
) -> xr.Dataset:
    """Stack event-centered windows from ``ds`` around each event peak time.

    This is the core Stage-2 composite preparation routine. It expects an event
    summary table, potentially filtered by ``selectors.select_top_n_events`` or
    ``selectors.select_event_quantile_bin``. For each selected event, it extracts
    the same relative-time window centered on the event's ``peak_time``.

    Parameters
    ----------
    ds:
        Harmonized Stage-1 dataset containing time-indexed analysis variables.
    event_table:
        Event summary table. This may be the full table or a filtered table
        returned by selector functions.
    variables:
        Optional variable names to extract. If omitted, all 1D variables with
        ``time_dim`` are used and event-table variables are skipped.
    pre_days, post_days:
        Number of days before and after the peak time to include. The output
        lag coordinate is hourly and includes both endpoints. For example,
        ``pre_days=5`` and ``post_days=5`` yields lags -120, ..., 0, ..., +120.
    time_dim:
        Time dimension in ``ds``.
    event_dim:
        Event dimension in ``event_table`` and output.
    lag_dim:
        Relative-time dimension in the output. The default stores lag in hours.
    peak_time_name:
        Event-table variable containing peak timestamps.
    event_id_name:
        Event-table variable containing integer event IDs.
    fill_missing:
        If True, windows that extend beyond the available data range are padded
        with NaNs. If False, missing timestamps raise an error.

    Returns
    -------
    xr.Dataset
        Dataset with dimensions ``event`` and ``lag_hour``. The ``event``
        coordinate is the original event ID, not the row number in the event
        table. Event metadata variables from the table are attached along the
        ``event`` dimension.
    """
    _validate_stack_inputs(
        ds,
        event_table,
        time_dim=time_dim,
        event_dim=event_dim,
        peak_time_name=peak_time_name,
        event_id_name=event_id_name,
        pre_days=pre_days,
        post_days=post_days,
    )

    variable_names = _resolve_time_variables(ds, variables, time_dim=time_dim)
    if not variable_names:
        raise ValueError("No time-indexed variables were selected for event stacking.")

    lag_hours = np.arange(-24 * pre_days, 24 * post_days + 1, dtype=np.int64)
    lag_offsets = pd.to_timedelta(lag_hours, unit="h")

    windows: list[xr.Dataset] = []
    event_ids  = np.asarray(event_table[event_id_name].values, dtype=np.int64)
    peak_times = np.asarray(event_table[peak_time_name].values, dtype="datetime64[ns]")

    source = ds[list(variable_names)]
    for event_id, peak_time in zip(event_ids, peak_times, strict=True):
        if np.isnat(peak_time):
            window = _empty_event_window(
                source,
                lag_hours,
                lag_dim=lag_dim,
                time_dim=time_dim,
            )
        else:
            target_times = peak_time + lag_offsets.to_numpy().astype("timedelta64[ns]")
            window = _extract_window_at_times(
                source,
                target_times,
                lag_hours,
                time_dim=time_dim,
                lag_dim=lag_dim,
                fill_missing=fill_missing,
            )
        windows.append(window.expand_dims({event_dim: [event_id]}))

    stacked = xr.concat(windows, dim=event_dim)
    stacked = _attach_event_metadata(
        stacked,
        event_table,
        event_ids=event_ids,
        event_dim=event_dim,
        event_id_name=event_id_name,
    )
    stacked.attrs.update(
        {
            "composite_alignment": peak_time_name,
            "pre_days": int(pre_days),
            "post_days": int(post_days),
            "lag_units": "hours relative to event peak time",
            "n_events": int(event_ids.size),
        }
    )
    return stacked


def peak_aligned_composite(
    ds: xr.Dataset,
    event_table: xr.Dataset,
    *,
    variables: Sequence[str] | None = None,
    pre_days: int = 5,
    post_days: int = 5,
    time_dim: str = DEFAULT_TIME_DIM,
    event_dim: str = DEFAULT_EVENT_DIM,
    lag_dim: str = DEFAULT_LAG_DIM,
    peak_time_name: str = DEFAULT_PEAK_TIME_NAME,
    event_id_name: str = DEFAULT_EVENT_ID_NAME,
    skipna: bool = True,
) -> xr.Dataset:
    """Return the event-mean composite centered on event peak times.

    This is a convenience wrapper around ``stack_events_centered_on_peak`` plus
    a mean over the event dimension. Use ``stack_events_centered_on_peak`` when
    individual-event traces are needed for top-event plots.
    """
    stacked = stack_events_centered_on_peak(
        ds,
        event_table,
        variables=variables,
        pre_days=pre_days,
        post_days=post_days,
        time_dim=time_dim,
        event_dim=event_dim,
        lag_dim=lag_dim,
        peak_time_name=peak_time_name,
        event_id_name=event_id_name,
    )
    composite = stacked.mean(event_dim, skipna=skipna)
    composite.attrs.update(stacked.attrs)
    composite.attrs["composite_reduction"] = "mean over selected events"
    return composite


def event_percentile_envelope(
    stacked: xr.Dataset,
    *,
    q: Sequence[float] = (0.05, 0.5, 0.95),
    event_dim: str = DEFAULT_EVENT_DIM,
    quantile_dim: str = "quantile",
    skipna: bool = True,
) -> xr.Dataset:
    """Compute percentile envelopes across stacked event windows.

    Parameters
    ----------
    stacked:
        Output from ``stack_events_centered_on_peak``.
    q:
        Quantiles in [0, 1]. Defaults to 5th, 50th, and 95th percentiles.
    event_dim:
        Dimension over which events are indexed.
    quantile_dim:
        Name of the output quantile dimension.
    skipna:
        Whether to ignore NaNs during quantile calculation.
    """
    if event_dim not in stacked.dims:
        raise ValueError(f"stacked dataset is missing event dimension {event_dim!r}.")
    _validate_quantiles(q)

    out = stacked.quantile(q, dim=event_dim, skipna=skipna)
    if "quantile" in out.dims and quantile_dim != "quantile":
        out = out.rename({"quantile": quantile_dim})
    out.attrs.update(stacked.attrs)
    out.attrs["composite_reduction"] = f"quantiles over {event_dim}"
    return out


def extract_event_window_by_id(
    stacked: xr.Dataset,
    event_id: int,
    *,
    event_dim: str = DEFAULT_EVENT_DIM,
) -> xr.Dataset:
    """Return one already-stacked event window by original event ID."""
    if event_dim not in stacked.coords:
        raise ValueError(f"stacked dataset is missing event coordinate {event_dim!r}.")
    return stacked.sel({event_dim: int(event_id)})


def _validate_stack_inputs(
    ds: xr.Dataset,
    event_table: xr.Dataset,
    *,
    time_dim: str,
    event_dim: str,
    peak_time_name: str,
    event_id_name: str,
    pre_days: int,
    post_days: int,
) -> None:
    """Validate inputs for peak-aligned event stacking."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError(f"ds must be an xarray.Dataset, got {type(ds).__name__}.")
    if not isinstance(event_table, xr.Dataset):
        raise TypeError(
            f"event_table must be an xarray.Dataset, got {type(event_table).__name__}."
        )
    if time_dim not in ds.coords:
        raise ValueError(f"ds is missing required time coordinate {time_dim!r}.")
    if event_dim not in event_table.dims:
        raise ValueError(f"event_table is missing event dimension {event_dim!r}.")
    if peak_time_name not in event_table:
        raise ValueError(f"event_table is missing peak-time variable {peak_time_name!r}.")
    if event_id_name not in event_table:
        raise ValueError(f"event_table is missing event-ID variable {event_id_name!r}.")
    if pre_days < 0 or post_days < 0:
        raise ValueError("pre_days and post_days must be non-negative.")
    if event_table.sizes[event_dim] == 0:
        raise ValueError("event_table contains no selected events.")


def _resolve_time_variables(
    ds: xr.Dataset,
    variables: Sequence[str] | None,
    *,
    time_dim: str,
) -> list[str]:
    """Return valid time-indexed variables to include in event windows."""
    if variables is None:
        return [name for name, da in ds.data_vars.items() if da.dims == (time_dim,)] # type: ignore

    missing = sorted(name for name in variables if name not in ds)
    if missing:
        raise ValueError(f"Dataset is missing requested variables: {', '.join(missing)}.")

    invalid = sorted(name for name in variables if time_dim not in ds[name].dims)
    if invalid:
        raise ValueError(
            "Requested variables must contain the time dimension; invalid: "
            f"{', '.join(invalid)}."
        )
    return list(variables)


def _extract_window_at_times(
    source: xr.Dataset,
    target_times: np.ndarray,
    lag_hours: np.ndarray,
    *,
    time_dim: str,
    lag_dim: str,
    fill_missing: bool,
) -> xr.Dataset:
    """Extract one event window at exact timestamps and rename time to lag."""
    if fill_missing:
        window = source.reindex({time_dim: target_times})
    else:
        window = source.sel({time_dim: target_times})

    window = window.assign_coords({time_dim: lag_hours}).rename({time_dim: lag_dim})
    window[lag_dim].attrs["units"] = "hours since event peak"
    return window


def _empty_event_window(
    source: xr.Dataset,
    lag_hours: np.ndarray,
    *,
    lag_dim: str,
    time_dim: str,
) -> xr.Dataset:
    """Return an all-NaN event window with the expected lag coordinate."""
    template = source.isel({time_dim: slice(0, lag_hours.size)}).copy(deep=False)
    template = template.reindex({time_dim: np.arange(lag_hours.size)})
    template = template.assign_coords({time_dim: lag_hours}).rename({time_dim: lag_dim})
    for name in template.data_vars:
        template[name] = xr.full_like(template[name], np.nan, dtype=float)
    template[lag_dim].attrs["units"] = "hours since event peak"
    return template


def _attach_event_metadata(
    stacked: xr.Dataset,
    event_table: xr.Dataset,
    *,
    event_ids: np.ndarray,
    event_dim: str,
    event_id_name: str,
) -> xr.Dataset:
    """Attach 1D event-table variables to a stacked event-window dataset."""
    out = stacked.assign_coords({event_dim: event_ids})
    for name, da in event_table.data_vars.items():
        if da.dims != (event_dim,):
            continue
        if name == event_id_name:
            continue
        out[name] = (event_dim, da.values)
        out[name].attrs.update(da.attrs)

    out[event_id_name] = (event_dim, event_ids)
    out[event_id_name].attrs.update(event_table[event_id_name].attrs)
    return out


def _validate_quantiles(q: Sequence[float]) -> None:
    """Validate quantile values."""
    if len(q) == 0:
        raise ValueError("At least one quantile is required.")
    for value in q:
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError("All quantiles must lie within [0, 1].")