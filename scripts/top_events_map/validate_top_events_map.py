"""Check prepared map fields against independent direct sums from source files."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io
from src.top_events_map import sha256_file
from src.top_events_map_validation import validate_against_sources


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_path.exists():
        raise FileExistsError(f"Validation report exists: {args.output_path}")
    with analysis_io.open_top_event_maps(args.input_path) as product:
        report = validate_against_sources(product)
        report["product_commit"] = product.attrs["source_commit"]
    report.update(
        {
            "product_sha256": sha256_file(args.input_path),
            "validator_commit": subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
            ).strip(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(
        f"Independent daily-source checks passed for {len(report['events'])} events: {args.output_path}"
    )
    print(json.dumps(report["max_absolute_errors"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
