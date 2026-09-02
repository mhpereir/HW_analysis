"""Plot peak-aligned climatological-anomaly composites for all HW events."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import plot_composite_timeseries_all as absolute_plot
from src import analysis_io, climatology, composites, plot_paths, plotting, selectors

PLOT_NAME = "composite_timeseries_all_clim_anom"
DEFAULT_OUTPUT_FILENAME = "hw_all_events_composite_clim_anom.png"
PRESENTATION_PLOT_NAME = f"{PLOT_NAME}_presentation"
PRESENTATION_DEFAULT_OUTPUT_FILENAME = (
    "hw_all_events_composite_clim_anom_presentation.png"
)


def parse_args() -> argparse.Namespace:
    """Parse Stage-1, climatology, selection, and output arguments."""
    parser = argparse.ArgumentParser(
        description="Plot peak-aligned climatological anomalies for all HW events."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument("--climatology-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument(
        "--window-days", type=int, default=absolute_plot.DEFAULT_WINDOW_DAYS
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=absolute_plot.DEFAULT_SMOOTHING_WINDOW,
    )
    parser.add_argument("--season-months", type=int, nargs="+", default=None)
    parser.add_argument("--require-full-event", action="store_true")
    parser.add_argument("--plot-extended-variables", action="store_true")
    parser.add_argument(
        "--layout",
        choices=plotting.COMPOSITE_LAYOUTS,
        default=plotting.PAPER_COMPOSITE_LAYOUT,
    )
    parsed = parser.parse_args()
    plot_name, default_output_filename = _default_plot_destination(parsed.layout)
    args = plot_paths.finalize_stage1_plot_paths(
        parsed,
        parser,
        plot_name=plot_name,
        default_output_filename=default_output_filename,
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
    """Apply the absolute workflow's selection and window validation."""
    absolute_plot.validate_args(args)


def main() -> int:
    """Build anomalies before stacking and write raw and smoothed figures."""
    args = parse_args()
    validate_args(args)
    _require_new_outputs(args.output_path)
    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    climate = analysis_io.open_regional_hourly_climatology(args.climatology_path)
    try:
        variables = absolute_plot._composite_variables(
            args.plot_extended_variables,
            args.layout,
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
                months = " ".join(str(month) for month in args.season_months)
                raise ValueError(
                    f"No events remain after filtering to season months: {months}."
                )
        composite = composites.all_event_peak_aligned_composite(
            anomaly_source,
            event_table=event_table,
            variables=variables,
            pre_days=args.window_days,
            post_days=args.window_days,
            event_percentiles=(0.25, 0.5, 0.75),
        )
        composite.attrs.update(anomaly_source.attrs)
        composite.attrs["climatology_path"] = str(args.climatology_path)
        written = plotting.write_composite_timeseries_outputs(
            composite,
            args.output_path,
            smoothed_output_path=_smoothed_output_path(args.output_path),
            smoothing_window=args.smoothing_window,
            smoothed_variables=absolute_plot._smoothed_variables(
                args.plot_extended_variables,
                args.layout,
            ),
            plot_extended_variables=args.plot_extended_variables,
            layout=args.layout,
        )
        print("Wrote HW all-event climatological-anomaly figures:")
        for path in written:
            print(f"  {path}")
    finally:
        climate.close()
        stage1.close()
    return 0


def _smoothed_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_smoothed{output_path.suffix}")


def _default_plot_destination(layout: str) -> tuple[str, str]:
    """Return the output namespace and filename for one figure layout."""
    if layout == plotting.PRESENTATION_COMPOSITE_LAYOUT:
        return PRESENTATION_PLOT_NAME, PRESENTATION_DEFAULT_OUTPUT_FILENAME
    return PLOT_NAME, DEFAULT_OUTPUT_FILENAME


def _require_new_outputs(output_path: Path) -> None:
    existing = [
        path
        for path in (output_path, _smoothed_output_path(output_path))
        if path.exists()
    ]
    if existing:
        paths = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Anomaly plot output already exists: {paths}.")


if __name__ == "__main__":
    raise SystemExit(main())
