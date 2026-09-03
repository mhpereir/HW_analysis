"""Plot face-resolved climatological-anomaly advection composites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

PLOT_NAME = "advection_direction_exploration_clim_anom"
DEFAULT_OUTPUT_FILENAME = "advection_face_contributions_clim_anom.png"


def parse_args() -> argparse.Namespace:
    """Parse Stage-1, climatology, selection, and output arguments."""
    parser = argparse.ArgumentParser(
        description="Plot peak-aligned face-advection climatological anomalies."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument("--climatology-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--season-months", type=int, nargs="+", default=None)
    parser.add_argument("--require-full-event", action="store_true")
    args = plot_paths.finalize_stage1_plot_paths(
        parser.parse_args(),
        parser,
        plot_name=PLOT_NAME,
        default_output_filename=DEFAULT_OUTPUT_FILENAME,
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
    return args


def validate_args(args: argparse.Namespace) -> None:
    if args.window_days < 1:
        raise ValueError("--window-days must be >= 1.")
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if args.season_months is not None:
        invalid = [month for month in args.season_months if month < 1 or month > 12]
        if invalid:
            values = ", ".join(str(month) for month in invalid)
            raise ValueError(f"Season months must be between 1 and 12; got {values}.")


def main() -> int:
    """Build timestamp anomalies before stacking and write the two-panel figure."""
    args = parse_args()
    validate_args(args)
    if args.output_path.exists():
        raise FileExistsError(f"Anomaly plot already exists: {args.output_path}.")
    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    climate = analysis_io.open_regional_hourly_climatology(args.climatology_path)
    try:
        variables = (
            "advection",
            *(
                advection_direction.stage1_face_variable(face)
                for face in advection_direction.available_stage1_faces(stage1)
            ),
        )
        anomaly_source = climatology.apply_regional_hourly_climatology(
            stage1,
            climate,
            variables=variables,
        )
        event_table = stage1
        if args.season_months is not None:
            event_table = selectors.select_events_by_season(
                stage1,
                args.season_months,
                require_full_event=args.require_full_event,
            )
            if event_table.sizes.get("event", 0) == 0:
                raise ValueError("No events remain after seasonal filtering.")
        composite = composites.all_event_peak_aligned_composite(
            anomaly_source,
            event_table=event_table,
            variables=variables,
            pre_days=args.window_days,
            post_days=args.window_days,
            event_percentiles=None,
        )
        composite.attrs.update(anomaly_source.attrs)
        composite.attrs["climatology_path"] = str(args.climatology_path)
        path = advection_direction_plotting.write_advection_direction_exploration_plot(
            composite,
            args.output_path,
        )
    finally:
        climate.close()
        stage1.close()
    print(f"Wrote advection-direction climatological-anomaly plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
