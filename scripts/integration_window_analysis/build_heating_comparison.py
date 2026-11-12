"""Build and independently validate one region's heating-rank sweep."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xarray as xr

from src import analysis_io
from src.integration_window_analysis import build_heating_comparison
from src.integration_window_validation import validate_against_sources


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--climatology-path", type=Path, required=True)
    parser.add_argument(
        "--reference-dir",
        type=Path,
        required=True,
        help="Accepted regional 4d/7d/14d/21d directories.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-days", type=int, default=4)
    parser.add_argument("--max-days", type=int, default=31)
    parser.add_argument("--target-peak", default="2021-06-29T00:00:00")
    parser.add_argument("--target-event-id", type=int)
    args = parser.parse_args()
    if not 1 <= args.min_days <= args.max_days:
        parser.error("Require 1 <= min-days <= max-days.")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    reference_paths = {
        day: args.reference_dir / f"{day}d/event_features.nc" for day in (4, 7, 14, 21)
    }
    paths = {
        "stage1": args.input_path,
        "climatology": args.climatology_path,
        **{f"stage2_{day}d": path for day, path in reference_paths.items()},
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    with ExitStack() as stack:
        source = stack.enter_context(
            analysis_io.open_harmonized_timeseries(args.input_path)
        )
        climate = stack.enter_context(
            analysis_io.open_regional_hourly_climatology(args.climatology_path)
        )
        if climate.attrs.get("source_stage1_sha256") != hashes["stage1"]:
            raise ValueError("Climatology source checksum does not match Stage 1.")
        references = {
            day: stack.enter_context(
                xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True)
            )
            for day, path in reference_paths.items()
        }
        table = build_heating_comparison(
            source,
            references[4],
            climate,
            integration_days=tuple(range(args.min_days, args.max_days + 1)),
            target_peak=args.target_peak,
            target_event_id=args.target_event_id,
        )
        metadata = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "generating_commit": os.environ["EXPECTED_COMMIT"],
            "pbs_job_id": os.environ["PBS_JOBID"],
            "project_root": str(REPO_ROOT),
            "python": sys.executable,
            "input_paths": json.dumps(
                {name: str(path.resolve()) for name, path in paths.items()},
                sort_keys=True,
            ),
            "input_sha256": json.dumps(hashes, sort_keys=True),
        }
        table.attrs.update(metadata)
        args.output_dir.mkdir(parents=True, exist_ok=False)
        path = args.output_dir / "heating_comparison.nc"
        analysis_io.save_integration_window_ranks(table, path)
        with analysis_io.open_integration_window_ranks(path) as saved:
            validation = validate_against_sources(saved, source, climate, references)
            saved.to_dataframe().reset_index().to_csv(
                args.output_dir / "all_events.csv", index=False, mode="x"
            )
            target = saved.where(
                saved.event_id == saved.attrs["target_event_id"], drop=True
            )
            target.to_dataframe().reset_index().to_csv(
                args.output_dir / "target_event.csv", index=False, mode="x"
            )
            excluded = saved.where(saved.common_cohort == 0, drop=True)
            excluded.to_dataframe().reset_index().to_csv(
                args.output_dir / "excluded_events.csv", index=False, mode="x"
            )
        if any(sha256(path) != hashes[name] for name, path in paths.items()):
            raise ValueError("An input changed during the run.")
        with (args.output_dir / "validation.json").open("x") as stream:
            json.dump(validation, stream, indent=2)
        manifest = {
            **metadata,
            "input_paths": {name: str(path.resolve()) for name, path in paths.items()},
            "input_sha256": hashes,
            "region": table.attrs["region"],
            "output_sha256": {
                path.name: sha256(path)
                for path in sorted(args.output_dir.iterdir())
                if path.is_file()
            },
            "validation": validation,
        }
        with (args.output_dir / "manifest.json").open("x") as stream:
            json.dump(manifest, stream, indent=2)
    print(json.dumps(validation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
