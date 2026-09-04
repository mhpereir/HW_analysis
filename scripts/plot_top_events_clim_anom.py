"""Plot top-event climatological-anomaly diagnostics from Stage 1."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import plot_top_events as absolute_plot
from src import analysis_io, climatology, plot_paths, plotting

PLOT_NAME = "top_events_clim_anom"
PRESENTATION_PLOT_NAME = f"{PLOT_NAME}_presentation"
FILENAME_TAG = "clim_anom"


def parse_args() -> argparse.Namespace:
    """Parse Stage-1, climatology, ranking, and output arguments."""
    parser = argparse.ArgumentParser(
        description="Plot climatological-anomaly diagnostics for top HW events."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--climatology-path",
        type=Path,
        default=None,
        help="Regional hourly climatology companion NetCDF path.",
    )
    absolute_plot.add_top_event_plot_arguments(parser)
    parsed = parser.parse_args()
    args = plot_paths.finalize_stage1_plot_paths(
        parsed,
        parser,
        plot_name=_default_plot_name(parsed.layout),
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
    """Apply the absolute top-event workflow's argument validation."""
    absolute_plot.validate_args(args)


def main() -> int:
    """Rank absolute events, anomalize timestamp data, and write their figures."""
    args = parse_args()
    validate_args(args)
    _require_new_output_dir(args.output_dir)

    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    climate = analysis_io.open_regional_hourly_climatology(args.climatology_path)
    try:
        absolute_plot.describe_harmonized_dataset(stage1)
        selected_events = absolute_plot.select_top_tas_events(stage1, n=args.top_n)
        variables = absolute_plot._top_event_variables(
            args.plot_extended_variables,
            args.layout,
        )
        anomaly_source = climatology.apply_regional_hourly_climatology(
            stage1,
            climate,
            variables=variables,
        )
        anomaly_source.attrs["climatology_path"] = str(args.climatology_path)
        written = absolute_plot.write_top_event_plots(
            anomaly_source,
            selected_events,
            output_dir=args.output_dir,
            event_table=stage1,
            window_days=args.window_days,
            smoothing_window=args.smoothing_window,
            plot_extended_variables=args.plot_extended_variables,
            layout=args.layout,
            filename_tag=FILENAME_TAG,
        )
        print(f"Wrote {len(written)} top-event climatological-anomaly figures:")
        for path in written:
            print(f"  {absolute_plot._display_path(path)}")
    finally:
        climate.close()
        stage1.close()
    return 0


def _default_plot_name(layout: str) -> str:
    """Return the isolated anomaly output namespace for one layout."""
    if layout == plotting.PRESENTATION_COMPOSITE_LAYOUT:
        return PRESENTATION_PLOT_NAME
    return PLOT_NAME


def _require_new_output_dir(output_dir: Path) -> None:
    """Refuse to merge anomaly figures into an existing output directory."""
    if output_dir.exists():
        raise FileExistsError(
            f"Top-event anomaly output directory already exists: {output_dir}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
