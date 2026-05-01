"""Reusable selection logic for the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: selection logic within top-level Stage 2.

Responsibilities:
- Compute heatwave threshold exceedance masks.
- Compute LWA threshold exceedance masks.
- Compute compound selection masks.
- Apply seasonal filtering so selected events stay within the target season.
- Support quantile-based selection modes.
- Support top-N event selection modes.

Out of scope:
- Converting masks into event objects or event IDs.
- Composite computation.
- Plotting.
"""


from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import xarray as xr


DEFAULT_EVENT_DIM = "event"
DEFAULT_EVENT_ID_NAME = "event_id"


def select_events_by_metric(
    event_table: xr.Dataset,
    metric: str,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
    event_dim: str = DEFAULT_EVENT_DIM,
    inclusive: str = "both",
    drop: bool = True,
) -> xr.Dataset:
    """Select events whose metric values fall within a numeric range.

    Parameters
    ----------
    event_table:
        Event summary table, usually produced by ``events.build_event_summary_table``.
    metric:
        Name of the event-level metric to threshold, for example
        ``"tas_excess_integral"`` or ``"tas_peak"``.
    min_value, max_value:
        Optional lower and upper metric bounds. At least one must be provided.
    event_dim:
        Dimension indexing events.
    inclusive:
        Bound inclusion policy. One of ``"both"``, ``"left"``, ``"right"``,
        or ``"neither"``.
    drop:
        If True, return only selected event rows. If False, preserve the full
        event dimension and mask non-selected rows.

    Returns
    -------
    xr.Dataset
        Filtered event table. The original table attributes are preserved and
        selection metadata are added to the returned dataset attrs.
    """
    _validate_event_metric_table(event_table, metric, event_dim=event_dim)
    if min_value is None and max_value is None:
        raise ValueError("At least one of min_value or max_value must be provided.")

    metric_da = _numeric_metric_dataarray(event_table[metric])
    selected = _finite_metric_mask(metric_da)

    if min_value is not None:
        selected = selected & _lower_bound_mask(
            metric_da,
            min_value,
            inclusive=inclusive in {"both", "left"},
        )
    if max_value is not None:
        selected = selected & _upper_bound_mask(
            metric_da,
            max_value,
            inclusive=inclusive in {"both", "right"},
        )

    out = event_table.where(selected, drop=drop)
    attrs = {
        "selection_type": "metric_range",
        "selection_metric": metric,
        "selection_min_value": np.nan if min_value is None else float(min_value),
        "selection_max_value": np.nan if max_value is None else float(max_value),
        "selection_inclusive": inclusive,
        "n_selected_events": _count_true(selected),
    }
    metric_units = _selection_metric_units(event_table[metric])
    if metric_units is not None:
        attrs["selection_metric_units"] = metric_units
    out.attrs.update(attrs)
    return out


def select_top_n_events(
    event_table: xr.Dataset,
    metric: str,
    n: int,
    *,
    event_dim: str = DEFAULT_EVENT_DIM,
    largest: bool = True,
    dropna: bool = True,
    keep_order: str = "ranked",
) -> xr.Dataset:
    """Select the top ``n`` events ranked by an event-level metric.

    Parameters
    ----------
    event_table:
        Event summary table.
    metric:
        Metric used for ranking.
    n:
        Number of events to retain. Must be positive.
    event_dim:
        Dimension indexing events.
    largest:
        If True, select largest metric values. If False, select smallest values.
    dropna:
        If True, ignore NaN metric values. If False, NaNs are sorted to the end.
    keep_order:
        ``"ranked"`` returns rows ordered by metric rank. ``"event"`` returns
        selected rows in their original event-table order.

    Returns
    -------
    xr.Dataset
        Event table containing the selected events, with an added
        ``selection_rank`` variable on ``event_dim``.
    """
    _validate_event_metric_table(event_table, metric, event_dim=event_dim)
    if n < 1:
        raise ValueError("n must be >= 1.")
    if keep_order not in {"ranked", "event"}:
        raise ValueError("keep_order must be either 'ranked' or 'event'.")

    metric_values = _metric_values(event_table[metric])
    candidate_idx = np.arange(metric_values.size)
    finite = np.isfinite(metric_values)
    if dropna:
        candidate_idx = candidate_idx[finite]
        candidate_values = metric_values[finite]
    else:
        candidate_values = metric_values.copy()
        fill_value = -np.inf if largest else np.inf
        candidate_values = np.where(np.isnan(candidate_values), fill_value, candidate_values)

    if candidate_idx.size == 0:
        return _empty_ranked_selection(event_table, metric, event_dim=event_dim, selection_type="top_n")

    order = np.argsort(candidate_values, kind="mergesort")
    if largest:
        order = order[::-1]

    selected_idx = candidate_idx[order[:n]]
    ranked_idx = selected_idx.copy()
    if keep_order == "event":
        selected_idx = np.sort(selected_idx)

    out = event_table.isel({event_dim: selected_idx})
    rank_values = _rank_values_for_selected_indices(selected_idx, ranked_idx)
    out["selection_rank"] = (event_dim, rank_values)
    out.attrs.update(
        {
            "selection_type": "top_n",
            "selection_metric": metric,
            "selection_n": int(n),
            "selection_largest": int(largest),
            "selection_dropna": int(dropna),
            "selection_order": keep_order,
            "n_selected_events": int(selected_idx.size),
        }
    )
    return out


def select_event_quantile_bin(
    event_table: xr.Dataset,
    metric: str,
    *,
    qmin: float,
    qmax: float,
    event_dim: str = DEFAULT_EVENT_DIM,
    inclusive: str = "left",
    drop: bool = True,
) -> xr.Dataset:
    """Select events whose metric lies within a quantile bin.

    Parameters
    ----------
    event_table:
        Event summary table.
    metric:
        Metric used to define the quantile bin.
    qmin, qmax:
        Quantile bounds in the closed interval [0, 1]. ``qmin`` must be less
        than or equal to ``qmax``.
    event_dim:
        Dimension indexing events.
    inclusive:
        Bound inclusion policy after converting quantiles to metric values.
        The default, ``"left"``, is useful for adjacent bins such as
        [0, 1/3), [1/3, 2/3), [2/3, 1]. For the final bin, prefer
        ``inclusive="both"``.
    drop:
        If True, return only selected event rows.

    Returns
    -------
    xr.Dataset
        Filtered event table with quantile-bound metadata added to attrs.
    """
    _validate_event_metric_table(event_table, metric, event_dim=event_dim)
    _validate_quantile_bounds(qmin, qmax)

    metric_da = event_table[metric]
    values = _metric_values(metric_da)
    metric_units = _selection_metric_units(metric_da)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return _empty_quantile_selection(
            event_table,
            metric,
            qmin=qmin,
            qmax=qmax,
            event_dim=event_dim,
            metric_units=metric_units,
        )

    lower = float(np.nanquantile(finite_values, qmin))
    upper = float(np.nanquantile(finite_values, qmax))

    out = select_events_by_metric(
        event_table,
        metric,
        min_value=lower,
        max_value=upper,
        event_dim=event_dim,
        inclusive=inclusive,
        drop=drop,
    )
    attrs = {
        "selection_type": "metric_quantile_bin",
        "selection_metric": metric,
        "selection_qmin": float(qmin),
        "selection_qmax": float(qmax),
        "selection_lower_value": lower,
        "selection_upper_value": upper,
        "selection_inclusive": inclusive,
    }
    if metric_units is not None:
        attrs["selection_metric_units"] = metric_units
    out.attrs.update(attrs)
    return out


def select_events_by_season(
    event_table: xr.Dataset,
    season_months: Sequence[int],
    *,
    time_name: str = "peak_time",
    event_dim: str = DEFAULT_EVENT_DIM,
    require_full_event: bool = False,
    start_time_name: str = "start_time",
    end_time_name: str = "end_time",
    drop: bool = True,
) -> xr.Dataset:
    """Select events whose peak time, or full event interval, falls within a season.

    Parameters
    ----------
    event_table:
        Event summary table. It may also be a harmonized dataset containing
        event-level variables alongside time-indexed variables.
    season_months:
        Calendar months in the target season, using 1=January through
        12=December. Cross-year seasons such as DJF can be passed as
        ``[12, 1, 2]``.
    time_name:
        Event-level timestamp used for peak-time selection when
        ``require_full_event`` is False.
    event_dim:
        Dimension indexing events.
    require_full_event:
        If True, select only events where every calendar month touched by the
        inclusive ``[start_time, end_time]`` interval is in ``season_months``.
    start_time_name, end_time_name:
        Event-level interval endpoints used when ``require_full_event`` is True.
    drop:
        If True, return only selected event rows. If False, preserve the full
        event dimension and mask only variables that contain ``event_dim``.

    Returns
    -------
    xr.Dataset
        Filtered event table with season-selection metadata added to attrs.
    """
    months = _validate_season_months(season_months)
    _validate_event_table(event_table, event_dim=event_dim)

    if require_full_event:
        _validate_event_time_variable(event_table, start_time_name, event_dim=event_dim)
        _validate_event_time_variable(event_table, end_time_name, event_dim=event_dim)
        selected = _full_event_season_mask(
            event_table[start_time_name],
            event_table[end_time_name],
            months,
            event_dim=event_dim,
        )
    else:
        _validate_event_time_variable(event_table, time_name, event_dim=event_dim)
        selected = _event_time_month_mask(event_table[time_name], months)

    out = _apply_event_selection(event_table, selected, event_dim=event_dim, drop=drop)
    out.attrs.update(
        {
            "selection_type": "season",
            "selection_months": ",".join(str(month) for month in months),
            "selection_time_name": time_name,
            "selection_require_full_event": int(require_full_event),
            "n_selected_events": _count_true(selected),
        }
    )
    return out


def selected_event_ids(
    event_table: xr.Dataset,
    *,
    event_id_name: str = DEFAULT_EVENT_ID_NAME,
    event_dim: str = DEFAULT_EVENT_DIM,
) -> np.ndarray:
    """Return selected nonzero event IDs from an event table as ``int64`` values."""
    if event_id_name not in event_table:
        raise ValueError(f"event_table is missing event-ID variable {event_id_name!r}.")
    if event_dim not in event_table[event_id_name].dims:
        raise ValueError(
            f"{event_id_name!r} must contain event dimension {event_dim!r}; "
            f"got {event_table[event_id_name].dims!r}."
        )

    ids = np.asarray(event_table[event_id_name].values, dtype=np.int64)
    return ids[ids > 0]


def event_id_mask(
    event_id: xr.DataArray,
    selected_ids: Sequence[int] | np.ndarray,
    *,
    name: str = "selected_event_mask",
) -> xr.DataArray:
    """Return a time mask for samples whose event ID is in ``selected_ids``.

    This bridges event-level selections back to the harmonized time-series
    dataset. For example::

        top = select_top_n_events(event_table, "tas_excess_integral", 10)
        mask = event_id_mask(ds["hw_event_id"], selected_event_ids(top))
        ds_top_event_hours = ds.where(mask, drop=True)
    """
    ids = np.asarray(list(selected_ids), dtype=np.int64)
    mask = event_id.isin(ids)
    mask = mask.fillna(False).astype(bool)
    mask.name = name
    mask.attrs.update(
        {
            "selected_event_ids": ",".join(str(int(eid)) for eid in ids),
            "event_id_source": event_id.name,
        }
    )
    return mask


def filter_dataset_to_events(
    ds: xr.Dataset,
    selected_events: xr.Dataset | Sequence[int] | np.ndarray,
    *,
    event_id_name: str = "hw_event_id",
    summary_event_id_name: str = DEFAULT_EVENT_ID_NAME,
    event_dim: str = DEFAULT_EVENT_DIM,
    drop: bool = True,
) -> xr.Dataset:
    """Filter a time-indexed dataset to samples belonging to selected events.

    ``selected_events`` may be a filtered event table or a sequence of integer
    event IDs. This is a convenience wrapper around ``event_id_mask``.
    """
    if event_id_name not in ds:
        raise ValueError(f"Dataset is missing event-ID variable {event_id_name!r}.")

    if isinstance(selected_events, xr.Dataset):
        ids = selected_event_ids(
            selected_events,
            event_id_name=summary_event_id_name,
            event_dim=event_dim,
        )
    else:
        ids = np.asarray(list(selected_events), dtype=np.int64)

    mask = event_id_mask(ds[event_id_name], ids)
    out = ds.where(mask, drop=drop)
    out.attrs.update(
        {
            "event_filter_source": event_id_name,
            "event_filter_ids": ",".join(str(int(eid)) for eid in ids),
            "n_event_filter_ids": int(ids.size),
        }
    )
    return out


def _validate_event_metric_table(
    event_table: xr.Dataset,
    metric: str,
    *,
    event_dim: str,
) -> None:
    """Validate common event-table inputs."""
    _validate_event_table(event_table, event_dim=event_dim)
    if metric not in event_table:
        raise ValueError(f"event_table is missing metric variable {metric!r}.")
    if event_table[metric].dims != (event_dim,):
        raise ValueError(
            f"Metric {metric!r} must be 1D with dims ({event_dim!r},); "
            f"got {event_table[metric].dims!r}."
        )


def _validate_event_table(event_table: xr.Dataset, *, event_dim: str) -> None:
    """Validate common event-table shape inputs."""
    if not isinstance(event_table, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(event_table).__name__}.")
    if event_dim not in event_table.dims:
        raise ValueError(f"event_table is missing required dimension {event_dim!r}.")


def _validate_event_time_variable(event_table: xr.Dataset, name: str, *, event_dim: str) -> None:
    """Validate that a timestamp variable is 1D on the event dimension."""
    if name not in event_table:
        raise ValueError(f"event_table is missing time variable {name!r}.")
    da = event_table[name]
    if da.dims != (event_dim,):
        raise ValueError(
            f"Time variable {name!r} must be 1D with dims ({event_dim!r},); "
            f"got {da.dims!r}."
        )
    if not np.issubdtype(da.dtype, np.datetime64):
        raise TypeError(f"Time variable {name!r} must have datetime64 dtype.")


def _validate_season_months(season_months: Sequence[int]) -> tuple[int, ...]:
    """Return validated month numbers with duplicates removed in input order."""
    if isinstance(season_months, (str, bytes)):
        raise TypeError("season_months must be a sequence of integer month numbers.")

    months: list[int] = []
    for month in season_months:
        if isinstance(month, (bool, np.bool_)) or not isinstance(month, (int, np.integer)):
            raise ValueError("season_months must contain only integer month numbers.")
        month_int = int(month)
        if month_int < 1 or month_int > 12:
            raise ValueError("season_months values must be between 1 and 12.")
        if month_int not in months:
            months.append(month_int)

    if not months:
        raise ValueError("season_months must contain at least one month.")
    return tuple(months)


def _validate_quantile_bounds(qmin: float, qmax: float) -> None:
    """Validate quantile bounds."""
    if not 0.0 <= qmin <= 1.0:
        raise ValueError("qmin must satisfy 0 <= qmin <= 1.")
    if not 0.0 <= qmax <= 1.0:
        raise ValueError("qmax must satisfy 0 <= qmax <= 1.")
    if qmin > qmax:
        raise ValueError("qmin must be less than or equal to qmax.")


def _finite_metric_mask(metric_da: xr.DataArray) -> xr.DataArray:
    """Return finite-value mask for a metric DataArray."""
    return xr.apply_ufunc(np.isfinite, metric_da).fillna(False).astype(bool)


def _lower_bound_mask(metric_da: xr.DataArray, value: float, *, inclusive: bool) -> xr.DataArray:
    """Return lower-bound mask."""
    return metric_da >= value if inclusive else metric_da > value


def _upper_bound_mask(metric_da: xr.DataArray, value: float, *, inclusive: bool) -> xr.DataArray:
    """Return upper-bound mask."""
    return metric_da <= value if inclusive else metric_da < value


def _numeric_metric_dataarray(metric_da: xr.DataArray) -> xr.DataArray:
    """Return a numeric metric view for range, ranking, and quantile selection."""
    if np.issubdtype(metric_da.dtype, np.datetime64):
        raise TypeError(
            f"Metric {metric_da.name!r} is datetime64 and cannot be used for "
            "numeric metric selection."
        )
    if np.issubdtype(metric_da.dtype, np.timedelta64):
        out = metric_da / np.timedelta64(1, "D")
        out.name = metric_da.name
        return out
    if np.issubdtype(metric_da.dtype, np.number):
        return metric_da
    raise TypeError(
        f"Metric {metric_da.name!r} must be numeric or timedelta64; "
        f"got dtype {metric_da.dtype}."
    )


def _selection_metric_units(metric_da: xr.DataArray) -> str | None:
    """Return selection units added by metric normalization, if any."""
    if np.issubdtype(metric_da.dtype, np.timedelta64):
        return "days"
    return None


def _event_time_month_mask(time_da: xr.DataArray, months: Sequence[int]) -> xr.DataArray:
    """Return a boolean event mask based on event timestamp month."""
    mask = time_da.dt.month.isin(months)
    return mask.fillna(False).astype(bool)


def _full_event_season_mask(
    start_time: xr.DataArray,
    end_time: xr.DataArray,
    months: Sequence[int],
    *,
    event_dim: str,
) -> xr.DataArray:
    """Return a mask for events whose inclusive interval touches only season months."""
    start_values = np.asarray(start_time.compute().values, dtype="datetime64[ns]")
    end_values = np.asarray(end_time.compute().values, dtype="datetime64[ns]")
    allowed = set(int(month) for month in months)
    selected = np.asarray(
        [
            _interval_months_are_in_season(start, end, allowed)
            for start, end in zip(start_values, end_values)
        ],
        dtype=bool,
    )
    return xr.DataArray(selected, dims=(event_dim,), coords={event_dim: start_time[event_dim]})


def _interval_months_are_in_season(
    start: np.datetime64,
    end: np.datetime64,
    allowed_months: set[int],
) -> bool:
    """Return True when every calendar month touched by an interval is allowed."""
    if np.isnat(start) or np.isnat(end) or end < start:
        return False

    start_month = start.astype("datetime64[M]")
    end_month = end.astype("datetime64[M]")
    month_values = np.arange(start_month, end_month + np.timedelta64(1, "M"), dtype="datetime64[M]")
    month_numbers = (month_values.astype(int) % 12) + 1
    return bool(np.isin(month_numbers, list(allowed_months)).all())


def _apply_event_selection(
    event_table: xr.Dataset,
    selected: xr.DataArray,
    *,
    event_dim: str,
    drop: bool,
) -> xr.Dataset:
    """Apply an event-dimension mask without broadcasting across other dimensions."""
    if drop:
        selected_values = np.asarray(selected.compute().values, dtype=bool)
        selected_idx = np.flatnonzero(selected_values)
        return event_table.isel({event_dim: selected_idx})

    out = event_table.copy(deep=False)
    for name, da in event_table.data_vars.items():
        if event_dim in da.dims:
            out[name] = da.where(selected)
    return out


def _metric_values(metric_da: xr.DataArray) -> np.ndarray:
    """Return metric values as a realized 1D float NumPy array."""
    numeric_metric = _numeric_metric_dataarray(metric_da)
    return np.asarray(numeric_metric.compute().values, dtype=float)


def _count_true(mask: xr.DataArray) -> int:
    """Return the number of true values in a boolean DataArray."""
    return int(mask.sum().compute().item())


def _rank_values_for_selected_indices(selected_idx: np.ndarray, ranked_idx: np.ndarray) -> np.ndarray:
    """Return 1-based rank values aligned to ``selected_idx`` order."""
    rank_lookup = {int(idx): rank for rank, idx in enumerate(ranked_idx, start=1)}
    return np.asarray([rank_lookup[int(idx)] for idx in selected_idx], dtype=np.int64)


def _empty_ranked_selection(
    event_table: xr.Dataset,
    metric: str,
    *,
    event_dim: str,
    selection_type: str,
) -> xr.Dataset:
    """Return an empty event-table selection with rank metadata."""
    out = event_table.isel({event_dim: slice(0, 0)})
    out["selection_rank"] = (event_dim, np.asarray([], dtype=np.int64))
    out.attrs.update(
        {
            "selection_type": selection_type,
            "selection_metric": metric,
            "n_selected_events": 0,
        }
    )
    return out


def _empty_quantile_selection(
    event_table: xr.Dataset,
    metric: str,
    *,
    qmin: float,
    qmax: float,
    event_dim: str,
    metric_units: str | None = None,
) -> xr.Dataset:
    """Return an empty quantile-bin selection."""
    out = event_table.isel({event_dim: slice(0, 0)})
    attrs = {
        "selection_type": "metric_quantile_bin",
        "selection_metric": metric,
        "selection_qmin": float(qmin),
        "selection_qmax": float(qmax),
        "n_selected_events": 0,
    }
    if metric_units is not None:
        attrs["selection_metric_units"] = metric_units
    out.attrs.update(attrs)
    return out
