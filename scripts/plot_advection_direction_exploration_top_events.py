#!/usr/bin/env python3
"""Plot ranked top-event face-advection values against an all-event mean."""

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
    plot_paths,
)

PLOT_NAME = "advection_direction_exploration_top_events"
DEFAULT_TOP_N = advection_direction_top_events.DEFAULT_TOP_N
DEFAULT_WINDOW_DAYS = advection_direction_top_events.DEFAULT_WINDOW_DAYS
DEFAULT_RANK_METRIC = advection_direction_top_events.DEFAULT_RANK_METRIC


def parse_args() -> argparse.Namespace:
    """Parse Stage-1, event-selection, and output arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot ranked top-event face-advection values against the all-event mean."
        )
    )
    plot_paths.add_stage1_path_arguments(parser)
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
    parser.add_argument("--season-months", type=int, nargs="+", default=None)
    parser.add_argument("--require-full-event", action="store_true")
    args = plot_paths.finalize_stage1_plot_paths(
        parser.parse_args(),
        parser,
        plot_name=PLOT_NAME,
    )
    args.input_path = args.input_path.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def validate_args(args: argparse.Namespace) -> None:
    """Validate raw top-event plotting arguments."""
    advection_direction_top_events.validate_options(
        top_n=args.top_n,
        window_days=args.window_days,
        smoothing_window=args.smoothing_window,
        season_months=args.season_months,
        require_full_event=args.require_full_event,
    )


def build_top_event_inputs(
    stage1: xr.Dataset,
    *,
    top_n: int = DEFAULT_TOP_N,
    window_days: int = DEFAULT_WINDOW_DAYS,
    season_months: list[int] | None = None,
    require_full_event: bool = False,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Build the all-event mean and ranked raw Stage-1 event windows."""
    return advection_direction_top_events.build_top_event_inputs(
        stage1,
        stage1,
        data_representation="absolute",
        top_n=top_n,
        window_days=window_days,
        season_months=season_months,
        require_full_event=require_full_event,
    )


def main() -> int:
    """Build raw overlays and write hourly and smoothed top-event figures."""
    args = parse_args()
    validate_args(args)
    _require_new_output_dir(args.output_dir)
    stage1 = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        reference, event_windows = build_top_event_inputs(
            stage1,
            top_n=args.top_n,
            window_days=args.window_days,
            season_months=args.season_months,
            require_full_event=args.require_full_event,
        )
        written = advection_direction_plotting.write_top_event_advection_direction_exploration_outputs(
            reference,
            event_windows,
            args.output_dir,
            smoothing_window=args.smoothing_window,
        )
    finally:
        stage1.close()

    print(f"Wrote {len(written)} top-event face-advection absolute plots:")
    for path in written:
        print(f"  {path}")
    return 0


def _require_new_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"Top-event absolute output directory already exists: {output_dir}."
        )


if __name__ == "__main__":
    raise SystemExit(main())
