"""Build, independently validate and plot one Stage-2 region/window pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xarray as xr
from PIL import Image

from scripts.event_features.build_stage2_baseline_features import (
    build_baseline_features,
)
from scripts.event_features.build_stage2_event_features import (
    build_event_features,
    write_feature_outputs,
)
from scripts.event_features.event_feature_config import integration_windows
from src import analysis_io
from src.stage2_validation import validate_core_pair


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--integration-hours", type=int, required=True)
    args = parser.parse_args()
    windows = integration_windows(args.integration_hours)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    source_hash = sha256(args.input_path)
    common = {
        "input_path": args.input_path,
        "season_months": [6, 7, 8],
        "integration_hours": args.integration_hours,
    }
    with analysis_io.open_harmonized_timeseries(args.input_path) as source:
        events = build_event_features(source, require_full_event=True, **common)
        baseline = build_baseline_features(source, **common)
        for kind, table in (("event", events), ("baseline", baseline)):
            write_feature_outputs(
                table,
                args.output_dir / f"{kind}_features.nc",
                csv_output_path=args.output_dir / f"{kind}_features.csv",
            )
        with (
            xr.open_dataset(
                args.output_dir / "event_features.nc",
                engine="h5netcdf",
                decode_timedelta=True,
            ) as saved_events,
            xr.open_dataset(
                args.output_dir / "baseline_features.nc",
                engine="h5netcdf",
                decode_timedelta=True,
            ) as saved_baseline,
        ):
            validation = validate_core_pair(
                source, saved_events, saved_baseline, args.integration_hours
            )
        source_attrs = dict(source.attrs)
    with (args.output_dir / "validation.json").open("x") as stream:
        json.dump(validation, stream, indent=2)
    print(json.dumps(validation, indent=2), flush=True)

    for layout in ("full", "presentation"):
        output = args.output_dir / f"event_vs_clean_baseline_{layout}.png"
        subprocess.run(
            [
                sys.executable,
                str(
                    REPO_ROOT
                    / "scripts/event_features/plot_adiabatic_advection_comparison_baseline.py"
                ),
                "--input-path",
                str(args.output_dir / "baseline_features.nc"),
                "--event-input-path",
                str(args.output_dir / "event_features.nc"),
                "--output-path",
                str(output),
                "--layout",
                layout,
            ],
            check=True,
            cwd=REPO_ROOT,
        )
        with Image.open(output) as picture:
            picture.load()
            if min(picture.size) < 1000 or all(
                a == b for a, b in picture.convert("RGB").getextrema()
            ):
                raise ValueError(f"Blank or undersized figure: {output}")

    if sha256(args.input_path) != source_hash:
        raise ValueError("Stage-1 source changed during the run.")
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": os.environ["EXPECTED_COMMIT"],
        "project_root": str(REPO_ROOT),
        "job_id": os.environ["PBS_JOBID"],
        "python": sys.executable,
        "input_path": str(args.input_path),
        "input_sha256": source_hash,
        "source_attributes": source_attrs,
        "windows": windows,
        "output_dir": str(args.output_dir),
        "sha256": {
            p.name: sha256(p) for p in sorted(args.output_dir.iterdir()) if p.is_file()
        },
        "validation": validation,
        "acceptance": "Numerical checks passed; scheduler, logs and visual review remain required.",
    }
    with (args.output_dir / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, default=str)
    print(f"Completed {args.output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
