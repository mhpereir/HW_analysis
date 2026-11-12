"""Validate a saved production GMT diurnal composite against its source."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io
from src.diurnal_validation import validate_gmt_diurnal


def file_hash(path: Path) -> str:
    """Return the SHA-256 of an existing artifact."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input-path", "composite-path", "figure-path", "output-path"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    args = parser.parse_args()
    if args.output_path.exists() or args.output_path.is_symlink():
        raise FileExistsError(args.output_path)
    source_hash = file_hash(args.input_path)
    if source_hash != args.expected_input_sha256:
        raise ValueError("Source checksum differs from the accepted input.")
    with (
        analysis_io.open_harmonized_timeseries(args.input_path) as source,
        xr.open_dataset(args.composite_path, engine="h5netcdf") as saved,
    ):
        if saved.attrs["source_sha256"] != source_hash:
            raise ValueError("Saved composite source checksum mismatch.")
        if saved.attrs["source_path"] != str(args.input_path.resolve()):
            raise ValueError("Saved composite source path mismatch.")
        commit = os.environ["EXPECTED_COMMIT"]
        if saved.attrs["producer_commit"] != commit:
            raise ValueError("Saved composite producer commit mismatch.")
        report = validate_gmt_diurnal(source, saved)
    report.update(
        validated_at=datetime.now(timezone.utc).isoformat(),
        producer_commit=commit,
        python=sys.executable,
        artifacts={
            str(p): file_hash(p)
            for p in (args.input_path, args.composite_path, args.figure_path)
        },
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x") as handle:
        json.dump(
            report,
            handle,
            indent=2,
            allow_nan=False,
            default=lambda value: value.item(),
        )
        handle.write("\n")
    print(f"Independent numerical validation passed: {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
