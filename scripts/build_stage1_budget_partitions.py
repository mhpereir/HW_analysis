"""Add explicitly named boundary-flux terms to an immutable Stage-1 v2 product."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, budget_partition


def build_product(input_path: Path, output_path: Path) -> Path:
    input_path, output_path = input_path.resolve(), output_path.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    with input_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    with analysis_io.open_harmonized_timeseries(input_path) as source:
        if source.attrs.get("stage1_contract_version") != 2:
            raise ValueError("Boundary-flux migration requires Stage-1 contract v2.")
        product = budget_partition.add_boundary_flux_fields(source)
        product.attrs.update(
            partition_source_stage1_path=str(input_path),
            partition_source_stage1_sha256=digest,
            partition_generating_commit=os.environ.get("EXPECTED_COMMIT", "unrecorded"),
            partition_pbs_job_id=os.environ.get("PBS_JOBID", "local"),
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".partition-", dir=output_path.parent
        ) as temporary:
            staged = Path(temporary) / output_path.name
            analysis_io.save_harmonized_timeseries(product, staged)
            with analysis_io.open_harmonized_timeseries(staged) as saved:
                budget_partition.validate_stage1(saved)
            with input_path.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != digest:
                    raise ValueError("Source Stage 1 changed during migration.")
            os.link(staged, output_path)
    return output_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    print(build_product(args.input_path, args.output_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
