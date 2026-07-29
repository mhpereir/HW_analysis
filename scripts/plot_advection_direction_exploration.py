"""Plot face-resolved peak-aligned advection diagnostics from enhanced Stage 1."""

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
    composites,
    data_io,
    plot_paths,
    selectors,
)


PLOT_NAME = "advection_direction_exploration"
DEFAULT_OUTPUT_FILENAME = "advection_face_contributions_two_panel.png"
EXPLORATION_SUBDIR = "advection_direction_exploration"


def parse_args() -> argparse.Namespace:
    """Parse standalone advection-direction plot arguments."""
    parser = argparse.ArgumentParser(
        description="Plot peak-aligned face and grouped advection tendencies."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Standalone PNG output path.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=7,
        help="Number of complete days on each side of event peak time.",
    )
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Optional calendar months to retain, e.g. 6 7 8.",
    )
    parser.add_argument(
        "--require-full-event",
        action="store_true",
        help="Require complete event intervals within --season-months.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing standalone exploration PNG.",
    )
    args = parser.parse_args()
    try:
        return finalize_args(args)
    except ValueError as exc:
        parser.error(str(exc))


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Validate options and fill enhanced Stage-1 and plot paths."""
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")
    if args.window_days < 1:
        raise ValueError("--window-days must be >= 1.")
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if args.season_months is not None:
        invalid = [month for month in args.season_months if month < 1 or month > 12]
        if invalid:
            values = ", ".join(str(month) for month in invalid)
            raise ValueError(
                "--season-months values must be between 1 and 12; "
                f"got {values}."
            )

    args.bottom_boundary = data_io.normalize_heat_budget_bottom_boundary(
        args.bottom_boundary
    )
    args.top_boundary = data_io.normalize_heat_budget_top_boundary(args.top_boundary)
    default_stage1 = analysis_io.default_harmonized_timeseries_path(
        region=args.region,
        bottom_boundary=args.bottom_boundary,
        top_boundary=args.top_boundary,
        threshold_variable=args.threshold_variable,
        quantile=args.quantile,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if args.input_path is None:
        args.input_path = (
            default_stage1.parent / EXPLORATION_SUBDIR / default_stage1.name
        )
    if args.output_path is None:
        args.output_path = (
            plot_paths.default_plot_output_dir(
                plot_name=PLOT_NAME,
                region=args.region,
                bottom_boundary=args.bottom_boundary,
                top_boundary=args.top_boundary,
                start_year=args.start_year,
                end_year=args.end_year,
            )
            / DEFAULT_OUTPUT_FILENAME
        )
    args.input_path = args.input_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    return args


def build_composite(ds, args: argparse.Namespace):
    """Build the selected all-event face-advection composite."""
    event_table = ds
    if args.season_months is not None:
        event_table = selectors.select_events_by_season(
            ds,
            args.season_months,
            require_full_event=args.require_full_event,
        )
        if event_table.sizes.get("event", 0) == 0:
            months = " ".join(str(month) for month in args.season_months)
            raise ValueError(
                f"No events remain after filtering to season months: {months}."
            )

    variables = (
        "advection",
        *(
            advection_direction.stage1_face_variable(face)
            for face in advection_direction.available_stage1_faces(ds)
        ),
    )
    composite = composites.all_event_peak_aligned_composite(
        ds,
        event_table=event_table,
        variables=variables,
        pre_days=args.window_days,
        post_days=args.window_days,
        event_percentiles=None,
    )
    for name in (
        "region",
        "threshold_variable",
        "quantile",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "start_year",
        "end_year",
    ):
        if name in ds.attrs:
            composite.attrs[name] = ds.attrs[name]
    composite.attrs["season_months"] = (
        "" if args.season_months is None else ",".join(map(str, args.season_months))
    )
    composite.attrs["require_full_event"] = int(args.require_full_event)
    return composite


def main() -> int:
    """Build and write the standalone advection-direction figure."""
    args = parse_args()
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Exploration plot already exists: {args.output_path}. "
            "Pass --overwrite to replace it."
        )

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        composite = build_composite(ds, args)
        path = advection_direction_plotting.write_advection_direction_exploration_plot(
            composite,
            args.output_path,
        )
    finally:
        ds.close()
    print(f"Wrote advection-direction exploration plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
