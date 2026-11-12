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
    advection_direction_plotting,
    advection_direction_top_events,
    analysis_io,
    climatology,
    plot_paths,
    season_selection,
)

PLOT_NAME = "advection_direction_exploration_top_events_clim_anom"
DEFAULT_TOP_N = advection_direction_top_events.DEFAULT_TOP_N
DEFAULT_WINDOW_DAYS = advection_direction_top_events.DEFAULT_WINDOW_DAYS
DEFAULT_RANK_METRIC = advection_direction_top_events.DEFAULT_RANK_METRIC
select_reference_events = advection_direction_top_events.select_reference_events
select_top_tas_events = advection_direction_top_events.select_top_tas_events


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
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=advection_direction_plotting.DEFAULT_SMOOTHING_WINDOW,
    )
    season_selection.add_season_arguments(parser, default_full_event=True)
    args = plot_paths.finalize_stage1_plot_paths(
        season_selection.parse_args(parser),
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
    advection_direction_top_events.validate_options(
        top_n=args.top_n,
        window_days=args.window_days,
        smoothing_window=args.smoothing_window,
        season_months=args.season_months,
        require_full_event=args.require_full_event,
        all_seasons=args.all_seasons,
    )


def build_top_event_anomaly_inputs(
    stage1: xr.Dataset,
    climate: xr.Dataset,
    *,
    top_n: int = DEFAULT_TOP_N,
    window_days: int = DEFAULT_WINDOW_DAYS,
    season_months: list[int] | None = None,
    require_full_event: bool | None = None,
    all_seasons: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the all-event anomaly mean and ranked event anomaly windows."""
    variables = advection_direction_top_events.top_event_variables(stage1)
    anomaly_source = climatology.apply_regional_hourly_climatology(
        stage1,
        climate,
        variables=variables,
    )
    return advection_direction_top_events.build_top_event_inputs(
        anomaly_source,
        stage1,
        data_representation="climatological_anomaly",
        top_n=top_n,
        window_days=window_days,
        season_months=season_months,
        require_full_event=require_full_event,
        all_seasons=all_seasons,
    )


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
            **season_selection.season_kwargs(args),
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
