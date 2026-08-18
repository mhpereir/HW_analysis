"""Plot split peak-aligned climatological-anomaly HW composites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from scripts import plot_composite_timeseries_split as absolute_plot
from src import analysis_io, climatology, plot_paths, plotting, selectors


PLOT_NAME = "composite_timeseries_split_clim_anom"
DEFAULT_OUTPUT_FILENAME = "hw_events_composite_clim_anom.png"


def parse_args() -> argparse.Namespace:
    """Parse the split workflow plus its climatology companion."""
    parser = argparse.ArgumentParser(
        description="Plot split peak-aligned climatological-anomaly HW composites."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument("--climatology-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--window-days", type=int, default=absolute_plot.DEFAULT_WINDOW_DAYS)
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=absolute_plot.DEFAULT_SMOOTHING_WINDOW,
    )
    parser.add_argument("--split-variable", required=True)
    parser.add_argument("--split-quantiles", type=float, nargs="+")
    parser.add_argument("--split-years", type=int, nargs="+")
    parser.add_argument("--season-months", type=int, nargs="+", default=None)
    parser.add_argument("--require-full-event", action="store_true")
    parser.add_argument("--plot-extended-variables", action="store_true")
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


def main() -> int:
    """Build timestamp anomalies, retain absolute event splits, and write figures."""
    args = parse_args()
    absolute_plot.validate_args(args)
    output_path = absolute_plot._split_output_path(
        args.output_path,
        args.split_variable,
    )
    _require_new_outputs(output_path)
    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    climate = analysis_io.open_regional_hourly_climatology(args.climatology_path)
    try:
        variables = absolute_plot._composite_variables(args.plot_extended_variables)
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
        composite_kwargs: dict[str, object] = {
            "variables": variables,
            "pre_days": args.window_days,
            "post_days": args.window_days,
            "event_percentiles": (0.25, 0.5, 0.75),
        }
        if args.split_variable == "peak_time":
            composite = absolute_plot.build_split_year_composite(
                anomaly_source,
                event_table=event_table,
                split_years=args.split_years,
                composite_kwargs=composite_kwargs,
            )
        else:
            composite = absolute_plot.build_split_quantile_composite(
                anomaly_source,
                event_table=event_table,
                split_variable=args.split_variable,
                split_quantiles=args.split_quantiles,
                composite_kwargs=composite_kwargs,
            )
        composite.attrs.update(anomaly_source.attrs)
        composite.attrs["climatology_path"] = str(args.climatology_path)
        written = plotting.write_split_composite_timeseries_outputs(
            composite,
            output_path,
            smoothed_output_path=_smoothed_output_path(output_path),
            smoothing_window=args.smoothing_window,
            smoothed_variables=absolute_plot._smoothed_variables(
                args.plot_extended_variables
            ),
            plot_extended_variables=args.plot_extended_variables,
        )
        print("Wrote HW split climatological-anomaly figures:")
        for path in written:
            print(f"  {path}")
    finally:
        climate.close()
        stage1.close()
    return 0


def _smoothed_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_smoothed{output_path.suffix}")


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
