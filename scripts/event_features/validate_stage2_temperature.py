"""Independently validate rebuilt Stage-2 products against immutable inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from contextlib import ExitStack
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xarray as xr

from src.stage2_temperature_validation import validate_temperature_product


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "input-path",
        "climatology-path",
        "event-path",
        "baseline-path",
        "output-path",
    ):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(args.output_path)
    paths = {
        "stage1": args.input_path,
        "climatology": args.climatology_path,
        "events": args.event_path,
        "baseline": args.baseline_path,
    }
    with ExitStack() as stack:
        datasets = {
            name: stack.enter_context(
                xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True)
            )
            for name, path in paths.items()
        }
        report = {
            "expected_commit": args.expected_commit,
            "independent_direct_source_validation": True,
            **{
                name: validate_temperature_product(
                    datasets["stage1"], datasets["climatology"], datasets[name]
                )
                for name in ("events", "baseline")
            },
        }
    report["files"] = {}
    for name, path in paths.items():
        with path.open("rb") as handle:
            checksum = hashlib.file_digest(handle, "sha256").hexdigest()
        report["files"][name] = {"path": str(path.resolve()), "sha256": checksum}
    args.output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
