#!/usr/bin/env python3
"""Build an isolated Stage-1 copy with face-resolved advection tendencies."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import advection_direction, analysis_io, data_io, plot_paths

EXPLORATION_SUBDIR = "advection_direction_exploration"
DEFAULT_CHUNK_HOURS = 24 * 31


def parse_args() -> argparse.Namespace:
    """Parse Stage-1 exploration builder arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Add normalized EHB face contributions to an isolated copy of "
            "an existing Stage-1 dataset."
        )
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "Enhanced Stage-1 output. Defaults to the "
            f"results/stage1/{EXPLORATION_SUBDIR}/ subfolder."
        ),
    )
    parser.add_argument(
        "--start-year-ehb",
        type=int,
        default=1940,
        help="First year token in the EHB saved-results path.",
    )
    parser.add_argument(
        "--end-year-ehb",
        type=int,
        default=2025,
        help="Last year token in the EHB saved-results path.",
    )
    parser.add_argument(
        "--heat-budget-root",
        type=Path,
        default=None,
        help="Optional explicit directory containing annual heat_budget_*.nc files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing exploration output only.",
    )
    args = parser.parse_args()
    try:
        return finalize_args(args)
    except ValueError as exc:
        parser.error(str(exc))


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize run tokens and fill input and exploration output paths."""
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")
    if args.start_year_ehb > args.end_year_ehb:
        raise ValueError("--start-year-ehb must be less than or equal to --end-year-ehb.")
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
        args.input_path = default_stage1
    if args.output_path is None:
        args.output_path = default_stage1.parent / EXPLORATION_SUBDIR / default_stage1.name
    if args.heat_budget_root is None:
        args.heat_budget_root = data_io.era5_heat_budget_annual_root(
            region=args.region,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
            start_year_ehb=args.start_year_ehb,
            end_year_ehb=args.end_year_ehb,
        )

    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if input_path == output_path:
        raise ValueError(
            "Exploration output must differ from the existing Stage-1 input path."
        )
    args.input_path = input_path
    args.output_path = output_path
    args.heat_budget_root = args.heat_budget_root.expanduser().resolve()
    return args


def build_exploration_dataset(args: argparse.Namespace) -> Path:
    """Build and save the isolated enhanced Stage-1 dataset."""
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Exploration output already exists: {args.output_path}. "
            "Pass --overwrite to replace it."
        )

    years = list(range(args.start_year, args.end_year + 1))
    stage1 = analysis_io.open_harmonized_timeseries(
        args.input_path,
        chunks={"time": DEFAULT_CHUNK_HOURS},
    )
    heat_budget = data_io.open_era5_heat_budget(
        years=years,
        heat_budget_root=args.heat_budget_root,
    )
    try:
        enhanced = advection_direction.add_face_advection_tendencies(
            stage1,
            heat_budget,
        )
        enhanced.attrs.update(
            {
                "advection_direction_exploration": 1,
                "advection_direction_parent_stage1": str(args.input_path),
                "advection_direction_heat_budget_root": str(args.heat_budget_root),
            }
        )
        return analysis_io.save_harmonized_timeseries(enhanced, args.output_path)
    finally:
        stage1.close()
        heat_budget.close()


def main() -> int:
    """Build the isolated enhanced Stage-1 product."""
    args = parse_args()
    path = build_exploration_dataset(args)
    print(f"Wrote enhanced Stage-1 exploration dataset: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
