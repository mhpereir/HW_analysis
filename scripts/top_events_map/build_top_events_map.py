"""Prepare individual ranked-event maps using the daily spatial ERA5 holdings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, config, top_events_map


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--event-features-path", type=Path, required=True)
    parser.add_argument("--region", choices=sorted(config.REGIONS), required=True)
    parser.add_argument("--daily-dir", type=Path, required=True)
    parser.add_argument("--climatology-path", type=Path, required=True)
    parser.add_argument("--climatology-start-year", type=int, required=True)
    parser.add_argument("--climatology-end-year", type=int, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--top-n", type=int, default=1)
    parser.add_argument("--rank-metric", default=top_events_map.DEFAULT_RANK_METRIC)
    parser.add_argument(
        "--peak-year", type=int, help="Filter peak year before ranking, e.g. 2021."
    )
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        default=top_events_map.DEFAULT_EXTENT,
        metavar=("WEST", "EAST", "SOUTH", "NORTH"),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_path.exists():
        raise FileExistsError(f"Output exists: {args.output_path}")
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            [
                "git",
                "-C",
                str(REPO_ROOT),
                "status",
                "--porcelain",
                "--untracked-files=normal",
            ],
            text=True,
        ).strip()
    )
    with xr.open_dataset(
        args.event_features_path, engine="h5netcdf", decode_timedelta=True
    ) as features:
        product = top_events_map.build_top_event_maps(
            features,
            event_features_path=args.event_features_path,
            region=args.region,
            daily_dir=args.daily_dir,
            climatology_path=args.climatology_path,
            climatology_years=(args.climatology_start_year, args.climatology_end_year),
            source_commit=commit,
            top_n=args.top_n,
            rank_metric=args.rank_metric,
            peak_year=args.peak_year,
            extent=tuple(args.extent),
        )
    product.attrs["source_tree_dirty"] = int(dirty)
    written = analysis_io.save_top_event_maps(product, args.output_path)
    print(f"Wrote {product.sizes['event']} top-event map field(s): {written}")
    for event in range(product.sizes["event"]):
        row = product.isel(event=event)
        print(
            f"  rank={row.selection_rank.item()} event_id={row.event_id.item()} "
            f"peak_time={row.peak_time.values} sample_dates={row.sample_date.values}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
