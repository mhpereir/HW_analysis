"""Prepare a compact native-grid regional hourly T2m cache from annual ERA5."""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, config, data_io, regional_temperature


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--region", required=True, choices=sorted(config.REGIONS))
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--source-root", type=Path, default=Path(config.ERA5_HOURLY_T2M_ROOT)
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "New directory for NetCDF and manifest; normally "
            "$HWA_ARTIFACT_ROOT/regional_hourly_t2m/<region>/<attempt>/."
        ),
    )
    parser.add_argument(
        "--time-chunk",
        type=int,
        help="Hours per read; default follows the source storage chunk.",
    )
    args = parser.parse_args(argv)
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year.")
    if args.time_chunk is not None and args.time_chunk < 1:
        parser.error("--time-chunk must be positive.")
    return args


def file_identity(path: Path) -> dict:
    stat = path.stat()
    return {
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "inode": stat.st_ino,
        "device": stat.st_dev,
    }


def execution_provenance() -> dict:
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), *args], text=True
        ).strip()

    return {
        "checkout": str(REPO_ROOT),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
        "script_sha256": regional_temperature.sha256(Path(__file__)),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "hostname": socket.gethostname(),
        "python": sys.version,
        "interpreter": sys.executable,
        "numpy": np.__version__,
        "xarray": xr.__version__,
    }


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    source_root = args.source_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().absolute()
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(f"Refusing existing output directory: {output_dir}")
    years = range(args.start_year, args.end_year + 1)
    paths = [source_root / f"2mT_hour_ERA5_{year}.nc" for year in years]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing annual ERA5 source: {path}")
    execution = execution_provenance()
    output_dir.mkdir(parents=True, exist_ok=False)
    products, sources = [], []
    for year, path in zip(years, paths, strict=True):
        identity = file_identity(path)
        with data_io.open_era5_hourly_t2m_year(year, root=source_root) as source:
            product, checks = regional_temperature.prepare_annual_region(
                source,
                region=args.region,
                year=year,
                time_chunk=args.time_chunk,
            )
        if file_identity(path) != identity:
            raise ValueError(f"Raw source changed during preparation: {path}")
        products.append(product)
        sources.append({"path": str(path.resolve()), **identity, **checks})
        print(
            f"Prepared {args.region} {year}: {checks['hours']} hours, "
            f"grid {checks['regional_shape']}, block {checks['time_chunk']} hours",
            flush=True,
        )
    combined = xr.concat(
        products, dim="time", data_vars="minimal", coords="minimal", compat="equals"
    )
    combined.attrs.update(
        start_year=args.start_year,
        end_year=args.end_year,
        producer_commit=execution["commit"],
        manifest_filename="manifest.json",
    )
    filename = (
        f"tas_hourly_{args.region}_ERA5_025deg_{args.start_year}_{args.end_year}.nc"
    )
    saved = analysis_io.save_regional_hourly_t2m(combined, output_dir / filename)
    readme = output_dir / "README.md"
    readme.write_text(
        f"# Regional hourly ERA5 T2m: {args.region}\n\n"
        f"Prepared {args.start_year}-{args.end_year}; native 0.25-degree grid, "
        "inclusive regional grid centers, cosine-latitude weighted mean, UTC hours.\n\n"
        "The prepared variable is tas_region(time), matching the Stage-1 regional name. "
        "It records source_variable=t2m for the raw ERA5 input; Stage 1 stores the "
        "optional hourly copy as tas_region_hourly alongside its daily tas_region.\n\n"
        "Retain this immutable cache and manifest together. The manifest records the "
        "command, source paths and identities, regional input hashes, and output checksum. "
        "Regional input hashes cover decoded selected cells, not complete global files.\n\n"
        "This hourly cache is separate from the 2-degree daily event-selection temperature. "
        "Use it directly with the temperature diagnostic, or through Stage 1's "
        "--hourly-t2m-path option.\n"
    )
    manifest = {
        "pipeline_stage": regional_temperature.PIPELINE_STAGE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": execution,
        "argv": sys.argv[1:] if argv is None else argv,
        "region": args.region,
        "start_year": args.start_year,
        "end_year": args.end_year,
        "source_root": str(source_root),
        "sources": sources,
        "hash_definition": "Regional decoded Kelvin float64 C-order time/lat/lon bytes, little-endian; coordinates hashed separately. Not global-file hashes.",
        "output": {
            "variable": "tas_region",
            "path": str(saved.resolve()),
            "sha256": regional_temperature.sha256(saved),
            "hours": combined.sizes["time"],
            "first_hour_utc": str(combined.time.values[0]),
            "last_hour_utc": str(combined.time.values[-1]),
        },
        "validation": {
            "complete_years": True,
            "complete_regional_grid": True,
            "all_cells_finite": True,
            "netcdf_roundtrip_exact": True,
        },
    }
    with (output_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved {saved}", flush=True)
    return saved


if __name__ == "__main__":
    main()
