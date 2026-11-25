"""Regenerate and validate both budget partitions against accepted 96-hour references."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import xarray as xr
from PIL import Image

from scripts.build_stage1_budget_partitions import build_product
from scripts.build_stage1_hourly_climatology import build_climatology_product
from scripts.event_features.build_stage2_baseline_features import (
    build_baseline_features,
)
from scripts.event_features.build_stage2_event_features import (
    build_event_features,
    write_feature_outputs,
)
from scripts.event_features.plot_adiabatic_advection_comparison_baseline import (
    write_tendency_scatter_plot,
)
from scripts.integration_window_analysis.build_component_anomalies import (
    sha256,
    verify_references,
)
from src import analysis_io, budget_partition, selectors
from src.budget_partition_validation import validate_products
from src.stage2_component_anomalies import build_component_anomaly_pair
from src.stage2_component_anomaly_validation import validate_anomalies_against_sources


def render_temporal(stage1: Path, climate: Path, output: Path, attrs: dict) -> None:
    """Exercise the four public CLI consumers over the complete split matrix."""
    common = [
        "--input-path",
        str(stage1),
        "--region",
        str(attrs["region"]),
        "--bottom-boundary",
        str(attrs["heat_budget_bottom_boundary"]).removesuffix("hPa"),
        "--top-boundary",
        str(attrs["heat_budget_top_boundary"]).removesuffix("hPa"),
        "--threshold-variable",
        "tas",
        "--quantile",
        "90",
        "--start-year",
        str(attrs["start_year"]),
        "--end-year",
        str(attrs["end_year"]),
        "--season-months",
        "6",
        "7",
        "8",
        "--require-full-event",
        "--plot-extended-variables",
        "--window-days",
        "7",
        "--smoothing-window",
        "24",
    ]
    for partition in budget_partition.CHOICES:
        for representation in ("absolute", "clim_anom"):
            suffix = "_clim_anom" if representation == "clim_anom" else ""
            extra = ["--climatology-path", str(climate)] if suffix else []
            for family in ("all", "split"):
                variants = (
                    (None,)
                    if family == "all"
                    else (
                        "duration",
                        "tas_anom_peak",
                        "tas_excess_integral",
                        "tas_excess_peak",
                        "tas_peak",
                        "peak_time",
                    )
                )
                for variant in variants:
                    directory = (
                        output / partition / "temporal" / family / representation
                    )
                    path = directory / f"{family}.png"
                    command = [
                        sys.executable,
                        str(
                            REPO_ROOT
                            / f"scripts/plot_composite_timeseries_{family}{suffix}.py"
                        ),
                        *common,
                        *extra,
                        "--budget-partition",
                        partition,
                        "--output-path",
                        str(path),
                    ]
                    if variant:
                        command += ["--split-variable", variant]
                        command += (
                            ["--split-years", "1982"]
                            if variant == "peak_time"
                            else ["--split-quantiles", "0.75" if suffix else "0.90"]
                        )
                    print(json.dumps({"command": command}), flush=True)
                    subprocess.run(command, check=True, cwd=REPO_ROOT)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(output)
    paths = {
        "stage1": args.input_path.resolve(),
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
        "input_paths": {name: str(path) for name, path in paths.items()},
        "input_sha256": hashes,
        "integration_hours": 96,
        "require_full_event": False,
        "temporal_require_full_event": True,
        "partitions": list(budget_partition.CHOICES),
    }
    with ExitStack() as stack:
        original = stack.enter_context(
            analysis_io.open_harmonized_timeseries(paths["stage1"])
        )
        references = {
            kind: stack.enter_context(
                analysis_io.open_stage2_features(
                    paths[kind],
                    expected_stage=f"stage_2_{kind}_features",
                    require_full_event=False,
                )
            )
            for kind in ("event", "baseline")
        }
        if original.attrs.get("event_id_source") != "hw_event_id":
            raise ValueError("This pilot requires the TAS-selected Stage-1 product.")
        for table in references.values():
            if (
                table.attrs.get("input_path") != str(paths["stage1"])
                or table.attrs.get("heat_budget_pre_window_hours") != "-96,0"
            ):
                raise ValueError(
                    "References must describe this exact Stage 1 and 96-hour window."
                )
        output.mkdir(parents=True, exist_ok=False)
        (output / "request.json").write_text(json.dumps(metadata, indent=2))
        products = output / "products"
        stage1_path = build_product(paths["stage1"], products / "stage1.nc")
        climate_path = build_climatology_product(
            argparse.Namespace(
                input_path=stage1_path,
                output_path=products / "climatology.nc",
                overwrite=False,
            )
        )
        extended = stack.enter_context(
            analysis_io.open_harmonized_timeseries(stage1_path)
        )
        climate = stack.enter_context(
            analysis_io.open_regional_hourly_climatology(climate_path)
        )
        source_attrs = dict(extended.attrs)
        saved = {}
        for kind, builder in (
            ("event", build_event_features),
            ("baseline", build_baseline_features),
        ):
            ref = references[kind]
            table = builder(
                extended,
                input_path=stage1_path,
                integration_hours=96,
                use_extended_variables=bool(ref.attrs["extended_variables_used"]),
                allow_missing_extended=bool(
                    ref.attrs.get("allow_missing_extended", False)
                ),
            )
            table.attrs.update(
                {
                    key: metadata[key]
                    for key in ("generating_commit", "pbs_job_id", "python")
                }
            )
            path = products / f"{kind}_features.nc"
            write_feature_outputs(table, path, csv_output_path=path.with_suffix(".csv"))
            saved[kind] = stack.enter_context(
                analysis_io.open_stage2_features(path, require_full_event=False)
            )
        validation = validate_products(
            original, extended, climate, saved["event"], saved["baseline"], references
        )
        anomalies = build_component_anomaly_pair(
            extended,
            climate,
            saved["event"],
            saved["baseline"],
            require_full_event=False,
        )
        for kind, table in zip(("event", "baseline"), anomalies, strict=True):
            table.attrs.update(
                source_stage1_sha256=sha256(stage1_path),
                climatology_sha256=sha256(climate_path),
                source_stage1_path=str(stage1_path),
                climatology_path=str(climate_path),
            )
            path = products / f"{kind}_features_clim_anom.nc"
            analysis_io.save_component_anomalies(table, path)
            table.to_dataframe().reset_index().to_csv(
                path.with_suffix(".csv"), index=False, mode="x"
            )
            saved[f"{kind}_anomaly"] = stack.enter_context(
                analysis_io.open_component_anomalies(path)
            )
        validation["anomalies"] = validate_anomalies_against_sources(
            extended,
            climate,
            saved["event_anomaly"],
            saved["baseline_anomaly"],
            saved["event"],
            saved["baseline"],
        )
        full_jja = selectors.select_events_by_season(
            extended, [6, 7, 8], require_full_event=True
        )
        original_jja = selectors.select_events_by_season(
            original, [6, 7, 8], require_full_event=True
        )
        xr.testing.assert_equal(full_jja.event, original_jja.event)
        validation["temporal_full_jja_events"] = int(full_jja.sizes["event"])
        for partition in budget_partition.CHOICES:
            for representation, suffix in (("absolute", ""), ("clim_anom", "_anomaly")):
                for layout in ("full", "presentation"):
                    for extension in ("png", "pdf"):
                        path = (
                            output
                            / partition
                            / "integration_96h"
                            / representation
                            / f"event_vs_clean_baseline_{layout}.{extension}"
                        )
                        write_tendency_scatter_plot(
                            saved[f"baseline{suffix}"],
                            saved[f"event{suffix}"],
                            path,
                            layout=layout,
                            partition=partition,
                        )
    render_temporal(stage1_path, climate_path, output, source_attrs)
    figures = list(output.rglob("*.png"))
    if len(figures) != 64 or len(list(output.rglob("*.pdf"))) != 8:
        raise ValueError("Incomplete pilot figure matrix.")
    for path in figures:
        with Image.open(path) as picture:
            picture.verify()
    if any(sha256(path) != hashes[name] for name, path in paths.items()):
        raise ValueError("An immutable reference changed during the pilot.")
    validation["figure_counts"] = {"png": 64, "pdf": 8}
    (output / "validation.json").write_text(json.dumps(validation, indent=2))
    (output / "README.md").write_text(
        "# PNW boundary-flux partition pilot\n\n"
        "Both named partitions are stored in the products. Original arrays and accepted 96-hour event/baseline rows are retained. "
        "The new fields regroup signed boundary fluxes; they are not local-gradient tendencies. "
        "Ten-panel all/split hourly and smoothed figures use complete JJA events. Integration figures use peak-in-JJA events and 97 inclusive hourly samples.\n\n"
        f"Generating commit: {metadata['generating_commit']}. PBS job: {metadata['pbs_job_id']}.\n\n"
        "See request.json, validation.json and manifest.json for exact paths, hashes and numerical evidence. "
        "Scheduler/log and original-resolution visual checks are separate acceptance requirements. "
        "Retain these products independently of source checkout retirement.\n"
    )
    manifest = {
        **metadata,
        "validation": validation,
        "output_sha256": {
            str(path.relative_to(output)): sha256(path)
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(validation, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
