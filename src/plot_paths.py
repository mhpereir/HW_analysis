"""Shared CLI path helpers for Stage-1-based plotting scripts."""

from __future__ import annotations

import argparse
from pathlib import Path

from . import analysis_io, data_io


REPO_ROOT = Path(__file__).resolve().parents[1]


def add_stage1_path_arguments(parser: argparse.ArgumentParser) -> None:
    """Add required Stage-1 run tokens and optional input override to a parser."""
    parser.add_argument(
        "--region",
        required=True,
        help="Region token used in the harmonized Stage-1 filename.",
    )
    parser.add_argument(
        "--bottom-boundary",
        required=True,
        help="Heat-budget bottom boundary token, either surface or integer hPa.",
    )
    parser.add_argument(
        "--top-boundary",
        required=True,
        help="Heat-budget top boundary token, as an integer hPa value.",
    )
    parser.add_argument(
        "--threshold-variable",
        required=True,
        choices=["tas", "lwa", "lwa_a", "lwa_c"],
        help="Threshold variable token used in the harmonized Stage-1 filename.",
    )
    parser.add_argument(
        "--quantile",
        required=True,
        help="Threshold quantile token used in the harmonized Stage-1 filename.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="First analysis year token used in the harmonized Stage-1 filename.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
        help="Last analysis year token used in the harmonized Stage-1 filename.",
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help=(
            "Optional explicit path to the saved harmonized Stage-1 regional "
            "dataset. If omitted, it is constructed from the Stage-1 run tokens."
        ),
    )


def finalize_stage1_plot_paths(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    *,
    plot_name: str,
    default_output_filename: str | None = None,
) -> argparse.Namespace:
    """Normalize Stage-1 run tokens and fill default input/output paths."""
    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year.")

    try:
        args.bottom_boundary = data_io.normalize_heat_budget_bottom_boundary(
            args.bottom_boundary
        )
        args.top_boundary = data_io.normalize_heat_budget_top_boundary(
            args.top_boundary
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.input_path is None:
        args.input_path = analysis_io.default_harmonized_timeseries_path(
            region=args.region,
            threshold_variable=args.threshold_variable,
            quantile=args.quantile,
            start_year=args.start_year,
            end_year=args.end_year,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
        )

    output_dir = default_plot_output_dir(
        plot_name=plot_name,
        region=args.region,
        bottom_boundary=args.bottom_boundary,
        top_boundary=args.top_boundary,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    if hasattr(args, "output_path") and args.output_path is None:
        if default_output_filename is None:
            raise ValueError("default_output_filename is required for output_path.")
        args.output_path = output_dir / default_output_filename
    if hasattr(args, "output_dir") and args.output_dir is None:
        args.output_dir = output_dir
    return args


def default_plot_output_dir(
    *,
    plot_name: str,
    region: str,
    bottom_boundary: str | int,
    top_boundary: str | int,
    start_year: int,
    end_year: int,
) -> Path:
    """Return the nested default output directory for one plotting script."""
    bottom = data_io.normalize_heat_budget_bottom_boundary(bottom_boundary)
    top = data_io.normalize_heat_budget_top_boundary(top_boundary)
    return (
        REPO_ROOT
        / "results"
        / f"plots_{filename_token(plot_name)}"
        / f"region_{filename_token(region)}"
        / f"boundary_{filename_token(bottom)}_{filename_token(top)}"
        / f"time_range_{start_year}_{end_year}"
    )


def filename_token(value: object) -> str:
    """Return a conservative path token while preserving hPa-style casing."""
    token = str(value).strip()
    for old, new in (("/", "-"), ("\\", "-"), (" ", "-")):
        token = token.replace(old, new)
    if not token:
        raise ValueError("Path token cannot be empty.")
    return token
