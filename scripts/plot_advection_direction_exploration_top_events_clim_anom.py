"""Plot ranked top-event face-advection climatological-anomaly overlays."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import (
    advection_direction,
    advection_direction_plotting,
    analysis_io,
    climatology,
    composites,
    plot_paths,
    selectors,
)

PLOT_NAME = "advection_direction_exploration_top_events_clim_anom"
DEFAULT_TOP_N = 10
DEFAULT_WINDOW_DAYS = 7
DEFAULT_RANK_METRIC = "tas_peak"


def parse_args() -> argparse.Namespace:
    """Parse Stage-1, climatology, event-selection, and output arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot ranked top-event face-advection climatological anomalies "
            "against the all-event anomaly mean."
        )
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument("--climatology-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=DEFAULT_TOP_N)
    parser.add_argument("--window-days", type=int, default=DEFAULT_WINDOW_DAYS)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=advection_direction_plotting.DEFAULT_SMOOTHING_WINDOW,
    )
    parser.add_argument("--season-months", type=int, nargs="+", default=None)
    parser.add_argument("--require-full-event", action="store_true")
    args = plot_paths.finalize_stage1_plot_paths(
        parser.parse_args(),
        parser,
        plot_name=PLOT_NAME,
    )
    if args.climatology_path is None:
        args.climatology_path = analysis_io.default_regional_hourly_climatology_path(
            region=args.region,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    args.climatology_path = args.climatology_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate top-event anomaly plotting arguments."""
    if args.top_n < 1:
        raise ValueError("--top-n must be >= 1.")
    if args.window_days < 1:
        raise ValueError("--window-days must be >= 1.")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1.")
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if args.season_months is not None:
        invalid = [month for month in args.season_months if month < 1 or month > 12]
        if invalid:
            values = ", ".join(str(month) for month in invalid)
            raise ValueError(f"Season months must be between 1 and 12; got {values}.")


def select_reference_events(
    stage1: xr.Dataset,
    *,
    season_months: list[int] | None,
    require_full_event: bool,
) -> xr.Dataset:
    """Return the event population used for ranking and the reference mean."""
    if season_months is None:
        return stage1
    selected = selectors.select_events_by_season(
        stage1,
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


def build_top_event_anomaly_inputs(
    stage1: xr.Dataset,
    climate: xr.Dataset,
    *,
    top_n: int = DEFAULT_TOP_N,
    window_days: int = DEFAULT_WINDOW_DAYS,
    season_months: list[int] | None = None,
    require_full_event: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the all-event anomaly mean and ranked event anomaly windows."""
    faces = advection_direction.available_stage1_faces(stage1)
    variables = (
        "advection",
        *(advection_direction.stage1_face_variable(face) for face in faces),
    )
    event_table = select_reference_events(
        stage1,
        season_months=season_months,
        require_full_event=require_full_event,
    )
    selected_events = select_top_tas_events(event_table, n=top_n)
    if selected_events.sizes.get("event", 0) == 0:
        raise ValueError("No finite events are available for top-event ranking.")

    anomaly_source = climatology.apply_regional_hourly_climatology(
        stage1,
        climate,
        variables=variables,
    )
    reference = composites.all_event_peak_aligned_composite(
        anomaly_source,
        event_table=event_table,
        variables=variables,
        pre_days=window_days,
        post_days=window_days,
        event_percentiles=None,
    )
    event_windows = composites.stack_events_centered_on_peak(
        anomaly_source,
        selected_events,
        variables=variables,
        pre_days=window_days,
        post_days=window_days,
    )

    selection_attrs = {
        "top_event_rank_metric": DEFAULT_RANK_METRIC,
        "top_event_rank_largest": 1,
        "top_event_requested_count": int(top_n),
        "top_event_selected_count": int(event_windows.sizes["event"]),
        "top_event_reference_event_count": int(reference.attrs["n_events"]),
    }
    reference.attrs = {
        **dict(anomaly_source.attrs),
        **dict(reference.attrs),
        **selection_attrs,
    }
    event_windows.attrs = {
        **dict(anomaly_source.attrs),
        **dict(event_windows.attrs),
        **selection_attrs,
    }
    return reference, event_windows


def main() -> int:
    """Build anomaly overlays and write hourly and smoothed top-event figures."""
    args = parse_args()
    validate_args(args)
    _require_new_output_dir(args.output_dir)
    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    climate = analysis_io.open_regional_hourly_climatology(args.climatology_path)
    try:
        reference, event_windows = build_top_event_anomaly_inputs(
            stage1,
            climate,
            top_n=args.top_n,
            window_days=args.window_days,
            season_months=args.season_months,
            require_full_event=args.require_full_event,
        )
        reference.attrs["climatology_path"] = str(args.climatology_path)
        event_windows.attrs["climatology_path"] = str(args.climatology_path)
        written = advection_direction_plotting.write_top_event_advection_direction_exploration_outputs(
            reference,
            event_windows,
            args.output_dir,
            smoothing_window=args.smoothing_window,
        )
    finally:
        climate.close()
        stage1.close()

    print(
        f"Wrote {len(written)} top-event face-advection climatological-anomaly plots:"
    )
    for path in written:
        print(f"  {path}")
    return 0


def _require_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Top-event anomaly output directory already exists: {output_dir}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
