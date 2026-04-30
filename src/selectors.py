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

    metric_da = event_table[metric]
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
    out.attrs.update(
        {
            "selection_type": "metric_range",
            "selection_metric": metric,
            "selection_min_value": np.nan if min_value is None else float(min_value),
            "selection_max_value": np.nan if max_value is None else float(max_value),
            "selection_inclusive": inclusive,
            "n_selected_events": _count_true(selected),
        }
    )
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

    values = _metric_values(event_table[metric])
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return _empty_quantile_selection(
            event_table,
            metric,
            qmin=qmin,
            qmax=qmax,
            event_dim=event_dim,
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
    out.attrs.update(
        {
            "selection_type": "metric_quantile_bin",
            "selection_metric": metric,
            "selection_qmin": float(qmin),
            "selection_qmax": float(qmax),
            "selection_lower_value": lower,
            "selection_upper_value": upper,
            "selection_inclusive": inclusive,
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
    if not isinstance(event_table, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(event_table).__name__}.")
    if event_dim not in event_table.dims:
        raise ValueError(f"event_table is missing required dimension {event_dim!r}.")
    if metric not in event_table:
        raise ValueError(f"event_table is missing metric variable {metric!r}.")
    if event_table[metric].dims != (event_dim,):
        raise ValueError(
            f"Metric {metric!r} must be 1D with dims ({event_dim!r},); "
            f"got {event_table[metric].dims!r}."
        )


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


def _metric_values(metric_da: xr.DataArray) -> np.ndarray:
    """Return metric values as a realized 1D float NumPy array."""
    return np.asarray(metric_da.compute().values, dtype=float)


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
) -> xr.Dataset:
    """Return an empty quantile-bin selection."""
    out = event_table.isel({event_dim: slice(0, 0)})
    out.attrs.update(
        {
            "selection_type": "metric_quantile_bin",
            "selection_metric": metric,
            "selection_qmin": float(qmin),
            "selection_qmax": float(qmax),
            "n_selected_events": 0,
        }
    )
    return out