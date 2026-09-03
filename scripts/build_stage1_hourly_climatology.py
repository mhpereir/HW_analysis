"""Build the regional calendar-hour climatology companion from Stage 1."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, climatology, data_io, plot_paths

DEFAULT_CHUNK_HOURS = 24 * 31


def parse_args() -> argparse.Namespace:
    """Parse Stage-1 source and climatology output arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Build a calendar-month/day/UTC-hour climatology companion from "
            "a canonical Stage-1 contract-version-2 product."
        )
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Regional hourly climatology NetCDF output path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of an existing climatology output.",
    )
    args = parser.parse_args()
    try:
        return finalize_args(args)
    except ValueError as exc:
        parser.error(str(exc))


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    """Normalize run tokens and fill Stage-1 and climatology paths."""
    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")
    args.bottom_boundary = data_io.normalize_heat_budget_bottom_boundary(
        args.bottom_boundary
    )
    args.top_boundary = data_io.normalize_heat_budget_top_boundary(args.top_boundary)
    if args.input_path is None:
        args.input_path = analysis_io.default_harmonized_timeseries_path(
            region=args.region,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
            threshold_variable=args.threshold_variable,
            quantile=args.quantile,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    if args.output_path is None:
        args.output_path = analysis_io.default_regional_hourly_climatology_path(
            region=args.region,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    args.input_path = args.input_path.expanduser().resolve()
    args.output_path = args.output_path.expanduser().resolve()
    if args.input_path == args.output_path:
        raise ValueError("Climatology output must differ from its Stage-1 input.")
    return args


def build_climatology_product(args: argparse.Namespace) -> Path:
    """Build and save one canonical regional hourly climatology product."""
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Climatology output already exists: {args.output_path}. "
            "Pass --overwrite to replace it."
        )
    source_sha256 = file_sha256(args.input_path)
    stage1 = analysis_io.open_harmonized_timeseries(
        args.input_path,
        chunks={"time": DEFAULT_CHUNK_HOURS},
    )
    try:
        require_canonical_climatology_source(stage1)
        product = climatology.build_regional_hourly_climatology(
            stage1,
            variables=climatology_variables_for_source(stage1),
        )
        product.attrs.update(
            {
                "source_stage1_path": str(args.input_path),
                "source_stage1_sha256": source_sha256,
            }
        )
        return analysis_io.save_regional_hourly_climatology(
            product,
            args.output_path,
        )
    finally:
        stage1.close()


def require_canonical_climatology_source(ds) -> None:
    """Require the standard Stage-1 provenance used by anomaly workflows."""
    contract_version = int(ds.attrs.get("stage1_contract_version", 1))
    if contract_version < 2:
        raise ValueError("Climatology source must use Stage-1 contract version 2.")
    source_layout = ds.attrs.get("cloud_cover_source_layout")
    if source_layout != data_io.CLOUD_COVER_LAYOUT_GLOBAL:
        raise ValueError(
            "Climatology source must use canonical global-hourly-grid cloud cover; "
            f"got {source_layout!r}."
        )
    missing = sorted(name for name in climatology.DEFAULT_VARIABLES if name not in ds)
    if missing:
        raise ValueError(
            "Climatology source is missing required variables: " + ", ".join(missing)
        )


def climatology_variables_for_source(ds) -> tuple[str, ...]:
    """Return standard variables plus an available lower-boundary face."""
    optional = ("advection_bottom",) if "advection_bottom" in ds else ()
    return (*climatology.DEFAULT_VARIABLES, *optional)


def file_sha256(path: str | Path, *, chunk_bytes: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest of one input product without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Build and report the regional hourly climatology companion."""
    args = parse_args()
    path = build_climatology_product(args)
    print(f"Wrote regional hourly climatology: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
