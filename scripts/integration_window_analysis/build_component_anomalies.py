"""Build, independently validate and plot one accepted population's anomaly budget."""

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

from scripts.event_features.plot_adiabatic_advection_comparison_baseline import (
    write_tendency_scatter_plot,
)
from src import analysis_io
from src.stage2_component_anomalies import build_component_anomaly_pair
from src.stage2_component_anomaly_validation import validate_anomalies_against_sources


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_references(paths: dict[str, Path], hashes: dict[str, str]) -> None:
    """Verify the exact accepted Stage-1 and Stage-2 reference files."""
    manifest = json.loads(paths["reference_manifest"].read_text())
    if manifest["input_sha256"] != hashes["stage1"]:
        raise ValueError("Reference manifest Stage-1 checksum mismatch.")
    for name in ("event", "baseline"):
        if manifest["sha256"][paths[name].name] != hashes[name]:
            raise ValueError(f"Accepted {name} reference checksum mismatch.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--climatology-path", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--integration-hours", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.integration_hours <= 0:
        parser.error("integration-hours must be positive")
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    paths = {
        "stage1": args.input_path.resolve(),
        "climatology": args.climatology_path.resolve(),
        "event": (args.reference_dir / "event_features.nc").resolve(),
        "baseline": (args.reference_dir / "baseline_features.nc").resolve(),
        "reference_manifest": (args.reference_dir / "manifest.json").resolve(),
    }
    hashes = {name: sha256(path) for name, path in paths.items()}
    verify_references(paths, hashes)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "generating_commit": os.environ["EXPECTED_COMMIT"],
        "pbs_job_id": os.environ["PBS_JOBID"],
        "project_root": str(REPO_ROOT),
        "python": sys.executable,
        "input_paths": json.dumps(
            {name: str(path) for name, path in paths.items()}, sort_keys=True
        ),
        "input_sha256": json.dumps(hashes, sort_keys=True),
    }
    with ExitStack() as stack:
        source = stack.enter_context(
            analysis_io.open_harmonized_timeseries(paths["stage1"])
        )
        climate = stack.enter_context(
            analysis_io.open_regional_hourly_climatology(paths["climatology"])
        )
        if climate.attrs.get("source_stage1_sha256") != hashes["stage1"]:
            raise ValueError("Climatology source checksum does not match Stage 1.")
        refs = {
            name: stack.enter_context(
                xr.open_dataset(paths[name], engine="h5netcdf", decode_timedelta=True)
            )
            for name in ("event", "baseline")
        }
        for ref in refs.values():
            if ref.attrs.get("input_path") != str(paths["stage1"]):
                raise ValueError(
                    "Accepted reference uses a different Stage-1 source path."
                )
            if (
                ref.attrs.get("heat_budget_pre_window_hours")
                != f"{-args.integration_hours},0"
            ):
                raise ValueError(
                    "Requested window does not match the accepted reference."
                )
        events, baseline = build_component_anomaly_pair(
            source, climate, refs["event"], refs["baseline"]
        )
        args.output_dir.mkdir(parents=True, exist_ok=False)
        for name, table in (("event", events), ("baseline", baseline)):
            table.attrs.update(metadata)
            path = args.output_dir / f"{name}_features_clim_anom.nc"
            analysis_io.save_component_anomalies(table, path)
            table.to_dataframe().reset_index().to_csv(
                path.with_suffix(".csv"), index=False, mode="x"
            )
        with (
            analysis_io.open_component_anomalies(
                args.output_dir / "event_features_clim_anom.nc"
            ) as saved_events,
            analysis_io.open_component_anomalies(
                args.output_dir / "baseline_features_clim_anom.nc"
            ) as saved_baseline,
        ):
            validation = validate_anomalies_against_sources(
                source,
                climate,
                saved_events,
                saved_baseline,
                refs["event"],
                refs["baseline"],
            )
            for layout in ("full", "presentation"):
                for extension in ("png", "pdf"):
                    write_tendency_scatter_plot(
                        saved_baseline,
                        saved_events,
                        args.output_dir
                        / f"event_vs_clean_baseline_clim_anom_{layout}.{extension}",
                        layout=layout,
                    )
        if any(sha256(path) != hashes[name] for name, path in paths.items()):
            raise ValueError("An input changed during the run.")
        with (args.output_dir / "validation.json").open("x") as stream:
            json.dump(validation, stream, indent=2)
        readme = (
            "# Component-anomaly event and clean-baseline budgets\n\n"
            f"Region: {events.attrs['region']}. Integration window: {args.integration_hours / 24:g} days. "
            f"Climatology: {events.attrs['climatology_start_year']}-{events.attrs['climatology_end_year']}, all observations.\n\n"
            "The total and all plotted components are integrated tendency anomalies. "
            "Raw totals were not recalculated. Reference populations and non-budget features are preserved.\n\n"
            "Contains event/baseline NetCDF and CSV tables, full/presentation PNG and PDF figures, "
            "independent validation and immutable input/output hashes.\n\n"
            f"Generating commit: {metadata['generating_commit']}. PBS job: {metadata['pbs_job_id']}.\n\n"
            "Scientific definition: docs/products/stage2_component_anomalies.md. "
            "Entrypoint: scripts/integration_window_analysis/build_component_anomalies.py. "
            "Scheduler: schedulers/schedule_stage2_component_anomalies.sh.\n\n"
            "Retain these outputs and their source references independently of checkout retirement. "
            "Scheduler/log and original-resolution visual acceptance remain required.\n"
        )
        (args.output_dir / "README.md").write_text(readme)
        manifest = {
            **metadata,
            "input_paths": {name: str(path) for name, path in paths.items()},
            "input_sha256": hashes,
            "region": events.attrs["region"],
            "integration_hours": args.integration_hours,
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
