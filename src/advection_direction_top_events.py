"""Shared preparation for ranked top-event face-advection figures."""

from __future__ import annotations

import xarray as xr

from . import advection_direction, composites, selectors

DEFAULT_TOP_N = 10
DEFAULT_WINDOW_DAYS = 7
DEFAULT_RANK_METRIC = "tas_peak"
SUPPORTED_DATA_REPRESENTATIONS = frozenset({"absolute", "climatological_anomaly"})


def validate_options(
    *,
    top_n: int,
    window_days: int,
    smoothing_window: int,
    season_months: list[int] | None,
    require_full_event: bool,
) -> None:
    """Validate shared top-event selection and display options."""
    if top_n < 1:
        raise ValueError("--top-n must be >= 1.")
    if window_days < 1:
        raise ValueError("--window-days must be >= 1.")
    if smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1.")
    if require_full_event and season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if season_months is not None:
        invalid = [month for month in season_months if month < 1 or month > 12]
        if invalid:
            values = ", ".join(str(month) for month in invalid)
            raise ValueError(f"Season months must be between 1 and 12; got {values}.")


def top_event_variables(stage1: xr.Dataset) -> tuple[str, ...]:
    """Return the total and available face-advection variables."""
    faces = advection_direction.available_stage1_faces(stage1)
    variables = (
        "advection",
        *(advection_direction.stage1_face_variable(face) for face in faces),
    )
    missing = [name for name in variables if name not in stage1]
    if missing:
        raise ValueError(
            f"Top-event source is missing required variables: {', '.join(missing)}."
        )
    return variables


def select_reference_events(
    stage1: xr.Dataset,
    *,
    season_months: list[int] | None,
    require_full_event: bool,
) -> xr.Dataset:
    """Return the absolute event population used for rank and reference."""
    selection_source = stage1.copy(deep=False)
    selection_source.attrs = dict(stage1.attrs)
    if season_months is None:
        return selection_source
    selected = selectors.select_events_by_season(
        selection_source,
        season_months,
        require_full_event=require_full_event,
    )
    if selected.sizes.get("event", 0) == 0:
        raise ValueError("No events remain after seasonal filtering.")
    return selected


def select_top_tas_events(
    event_table: xr.Dataset,
    *,
    n: int = DEFAULT_TOP_N,
) -> xr.Dataset:
    """Select heatwave events by descending absolute Stage-1 peak tas."""
    return selectors.select_top_n_events(
        event_table,
        DEFAULT_RANK_METRIC,
        n,
        largest=True,
        keep_order="ranked",
    )


def build_top_event_inputs(
    plot_source: xr.Dataset,
    absolute_stage1: xr.Dataset,
    *,
    data_representation: str,
    top_n: int = DEFAULT_TOP_N,
    window_days: int = DEFAULT_WINDOW_DAYS,
    season_months: list[int] | None = None,
    require_full_event: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build an all-event mean and ranked event windows from one population.

    Event membership and rank always come from ``absolute_stage1``. The values
    plotted and composited come from ``plot_source``, which is either Stage 1
    itself or its timestamp-matched climatological anomaly.
    """
    if data_representation not in SUPPORTED_DATA_REPRESENTATIONS:
        supported = ", ".join(sorted(SUPPORTED_DATA_REPRESENTATIONS))
        raise ValueError(
            f"Unsupported top-event data representation {data_representation!r}; "
            f"expected one of: {supported}."
        )

    variables = top_event_variables(absolute_stage1)
    plot_variables = top_event_variables(plot_source)
    if plot_variables != variables:
        raise ValueError(
            "Top-event plot source and absolute Stage 1 must contain identical "
            "face-advection variables."
        )

    event_table = select_reference_events(
        absolute_stage1,
        season_months=season_months,
        require_full_event=require_full_event,
    )
    selected_events = select_top_tas_events(event_table, n=top_n)
    if selected_events.sizes.get("event", 0) == 0:
        raise ValueError("No finite events are available for top-event ranking.")

    reference = composites.all_event_peak_aligned_composite(
        plot_source,
        event_table=event_table,
        variables=variables,
        pre_days=window_days,
        post_days=window_days,
        event_percentiles=None,
    )
    event_windows = composites.stack_events_centered_on_peak(
        plot_source,
        selected_events,
        variables=variables,
        pre_days=window_days,
        post_days=window_days,
    )

    selection_attrs = {
        "data_representation": data_representation,
        "top_event_rank_metric": DEFAULT_RANK_METRIC,
        "top_event_rank_largest": 1,
        "top_event_requested_count": int(top_n),
        "top_event_selected_count": int(event_windows.sizes["event"]),
        "top_event_reference_event_count": int(reference.attrs["n_events"]),
    }
    reference.attrs = {
        **dict(plot_source.attrs),
        **dict(reference.attrs),
        **selection_attrs,
    }
    event_windows.attrs = {
        **dict(plot_source.attrs),
        **dict(event_windows.attrs),
        **selection_attrs,
    }
    return reference, event_windows
