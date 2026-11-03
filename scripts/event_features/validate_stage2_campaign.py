"""Independently audit saved core Stage-2 pairs from an explicit campaign manifest."""

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

from src import analysis_io
from src.artifact_paths import artifact_root
from src.stage2_validation import validate_core_pair


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    root = artifact_root()
    if not path.is_absolute() or not path.resolve().is_relative_to(root):
        raise ValueError(f"Expected an absolute path beneath {root}: {path}")
    if path.resolve() == root:
        raise ValueError("The artifact root itself is not a product path.")
    return path.resolve()


def load_pairs(manifest: dict) -> list[dict]:
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("Manifest requires a nonempty pairs list.")
    identities = set()
    products = set()
    resolved = []
    for entry in pairs:
        pair = dict(entry)
        identity = pair["id"]
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Pair IDs must be nonempty and unique.")
        identities.add(identity)
        if not isinstance(pair["require_full_event"], bool):
            raise TypeError(
                "Each pair requires an explicit boolean population policy."
            )
        hours = pair["integration_hours"]
        if isinstance(hours, bool) or not isinstance(hours, int) or hours <= 0:
            raise ValueError("integration_hours must be a positive integer.")
        for key in ("input", "events", "baseline"):
            pair[key] = artifact_path(pair[key])
        for key in ("events", "baseline"):
            path = pair[key]
            if path in products or path == pair["input"]:
                raise ValueError("Each product must have a distinct path.")
            products.add(path)
        resolved.append(pair)
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = artifact_path(args.manifest)
    manifest_hash = sha256(manifest_path)
    pairs = load_pairs(json.loads(manifest_path.read_text()))
    destination = artifact_path(args.output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    inputs = {pair[key] for pair in pairs for key in ("input", "events", "baseline")}
    hashes = {str(path): sha256(path) for path in sorted(inputs)}
    results = []
    for index, pair in enumerate(pairs):
        with ExitStack() as stack:
            source = stack.enter_context(
                analysis_io.open_harmonized_timeseries(pair["input"])
            )
            events = stack.enter_context(
                analysis_io.open_stage2_features(
                    pair["events"], require_full_event=pair["require_full_event"]
                )
            )
            baseline = stack.enter_context(
                analysis_io.open_stage2_features(
                    pair["baseline"], expected_stage="stage_2_baseline_features"
                )
            )
            for table in (events, baseline):
                if Path(table.attrs["input_path"]).resolve() != pair["input"]:
                    raise ValueError(f"Source provenance mismatch: {pair['id']}")
            result = validate_core_pair(
                source,
                events,
                baseline,
                pair["integration_hours"],
                require_full_event=pair["require_full_event"],
            )
        record = {"pair": pair, "validation": result}
        with (destination / f"pair_{index:03d}.json").open("x") as stream:
            json.dump(record, stream, indent=2, default=str)
        results.append(record)
        print(json.dumps({"id": pair["id"], **result}), flush=True)
    for path, expected in hashes.items():
        if sha256(Path(path)) != expected:
            raise ValueError(f"Input changed during validation: {path}")
    if sha256(manifest_path) != manifest_hash:
        raise ValueError("Manifest changed during validation.")
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "commit": os.environ["EXPECTED_COMMIT"],
        "job_id": os.environ["PBS_JOBID"],
        "python": sys.executable,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
        "input_sha256": hashes,
        "pairs": results,
        "numerical_validation": "passed",
        "remaining_acceptance": "PBS status, logs, provenance and figure review",
    }
    with (destination / "validation.json").open("x") as stream:
        json.dump(report, stream, indent=2, default=str)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
