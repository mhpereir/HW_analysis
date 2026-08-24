"""Build the standalone PBL-top-pressure justification product."""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, data_io, pbl_justification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a peak-aligned PBL-top-pressure time series and heatwave-day map."
        )
    )
    parser.add_argument("--stage1-path", type=Path, required=True)
    parser.add_argument("--pbl-root", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--region", default="pnw_bartusek")
    parser.add_argument("--start-year", type=int, default=1940)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--threshold-variable", default="tas")
    parser.add_argument("--quantile", default="90")
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=pbl_justification.DEFAULT_SEASON_MONTHS,
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=pbl_justification.DEFAULT_WINDOW_DAYS,
    )
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year.")
    if args.window_days < 0:
        parser.error("--window-days must be nonnegative.")
    if not args.source_commit.strip():
        parser.error("--source-commit must not be empty.")
    return args


def main() -> int:
    args = parse_args()
    stage1_path = args.stage1_path.expanduser().resolve()
    pbl_root = args.pbl_root.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not stage1_path.is_file():
        raise FileNotFoundError(f"Stage-1 product does not exist: {stage1_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing product: {output_path}")

    years = tuple(range(args.start_year, args.end_year + 1))
    pbl_paths = pbl_justification.annual_source_paths(
        pbl_root,
        region=args.region,
        years=years,
    )
    stage1 = analysis_io.open_harmonized_timeseries(stage1_path)
    try:
        _validate_stage1_identity(stage1, args)
        selected = pbl_justification.select_full_season_events(
            stage1,
            args.season_months,
        )
        events = selected[["event_id", "start_time", "end_time", "peak_time"]].load()
    finally:
        stage1.close()

    pbl_ds = data_io.open_era5_pbl_p(
        region=args.region,
        years=years,
        chunks={"time": 200},
        root=pbl_root,
    )
    try:
        product = pbl_justification.build_product(
            pbl_ds[pbl_justification.PBL_SOURCE_NAME],
            events,
            region=args.region,
            pre_days=args.window_days,
            post_days=args.window_days,
            season_months=args.season_months,
        )
        product.attrs.update(
            {
                "source_commit": args.source_commit,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "stage1_path": str(stage1_path),
                "stage1_sha256": pbl_justification.file_sha256(stage1_path),
                "stage1_threshold_variable": args.threshold_variable,
                "stage1_quantile": str(args.quantile),
                "analysis_start_year": args.start_year,
                "analysis_end_year": args.end_year,
                "pbl_source_root": str(pbl_root),
                "pbl_source_resolved_root": str(pbl_root.resolve()),
                "pbl_source_file_count": len(pbl_paths),
                "pbl_source_inventory_sha256": (
                    pbl_justification.inventory_sha256(pbl_paths)
                ),
                "pbl_source_inventory_definition": (
                    "SHA-256 of resolved annual paths and byte sizes"
                ),
            }
        )
        product.load()
    finally:
        pbl_ds.close()

    written = pbl_justification.save_product(product, output_path)
    print(f"Selected events: {int(product.selected_event_count.item())}")
    print(
        f"Selected UTC heatwave days: {int(product.selected_heatwave_day_count.item())}"
    )
    print(f"Annual PBL source files: {len(pbl_paths)}")
    print(f"Wrote PBL justification product: {written}")
    return 0


def _validate_stage1_identity(stage1, args: argparse.Namespace) -> None:
    expected = {
        "region": str(args.region),
        "threshold_variable": str(args.threshold_variable),
        "quantile": str(args.quantile),
        "start_year": str(args.start_year),
        "end_year": str(args.end_year),
    }
    for name, expected_value in expected.items():
        if name not in stage1.attrs:
            raise ValueError(f"Stage-1 product is missing identity attribute {name!r}.")
        actual = str(stage1.attrs[name])
        if actual != expected_value:
            raise ValueError(f"Stage-1 {name}={actual!r}; expected {expected_value!r}.")


if __name__ == "__main__":
    raise SystemExit(main())
