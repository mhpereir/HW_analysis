"""Assemble and validate a pressure-layer Stage 1 from an accepted reference."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, harmonize, pressure_layer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--reference-sha256", required=True)
    parser.add_argument("--heat-budget-manifest", type=Path, required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument(
        "--threshold-variable", choices=pressure_layer.SELECTIONS, required=True
    )
    parser.add_argument("--quantile", default="90")
    parser.add_argument("--bottom-hpa", type=int, required=True)
    parser.add_argument("--top-hpa", type=int, required=True)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument("--validate-all-source-years", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")
    return args


def write_json(path: Path, value: dict) -> None:
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main() -> None:
    args = parse_args()
    args.reference = args.reference.resolve()
    args.heat_budget_manifest = args.heat_budget_manifest.resolve()
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing existing output directory: {args.output_dir}")
    reference_hash = pressure_layer.sha256(args.reference)
    if reference_hash != args.reference_sha256:
        raise ValueError("Reference SHA-256 differs from the accepted reference hash.")
    manifest = json.loads(args.heat_budget_manifest.read_text())
    pressure_layer.validate_manifest(
        manifest, region=args.region, bottom=args.bottom_hpa, top=args.top_hpa
    )
    first_source, last_source = (
        manifest["production_start_year"],
        manifest["production_end_year"],
    )
    if not first_source <= args.start_year <= args.end_year <= last_source:
        raise ValueError("Analysis years must lie inside the EHB campaign.")
    years = list(range(args.start_year, args.end_year + 1))
    validation_years = (
        list(range(first_source, last_source + 1))
        if args.validate_all_source_years
        else years
    )
    annual_root = Path(manifest["annual_dir"])
    annual_validation = []
    budgets = []
    for year in validation_years:
        path = annual_root / f"heat_budget_{year}.nc"
        with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as source:
            source.load()
            checks = pressure_layer.validate_annual_budget(
                source,
                year=year,
                bottom=args.bottom_hpa,
                top=args.top_hpa,
            )
            annual_validation.append(
                {"path": str(path), "sha256": pressure_layer.sha256(path), **checks}
            )
            if year in years:
                product_sources = list(harmonize.HEAT_BUDGET_VARIABLE_MAP)
                product_sources.extend(
                    f"flux_contribution_{face}" for face in pressure_layer.FACES
                )
                budgets.append(source[product_sources])
        print(f"Validated EHB {year}: {checks}", flush=True)
    budget = xr.concat(
        budgets, dim="time", data_vars="minimal", coords="minimal", compat="equals"
    )
    with analysis_io.open_harmonized_timeseries(args.reference) as source_reference:
        reference = pressure_layer.select_reference(
            source_reference,
            years=years,
            region=args.region,
            threshold_variable=args.threshold_variable,
            quantile=args.quantile,
        ).load()
    product = harmonize.replace_heat_budget(reference, budget)
    commit = subprocess.check_output(
        ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    product.attrs.update(
        {
            "assembly_method": "accepted_regional_reference_with_new_pressure_layer",
            "regional_reference_path": str(args.reference),
            "regional_reference_sha256": reference_hash,
            "producer_commit": commit,
            "heat_budget_bottom_boundary": f"{args.bottom_hpa}hPa",
            "heat_budget_top_boundary": f"{args.top_hpa}hPa",
            "heat_budget_root": str(annual_root),
            "heat_budget_manifest": str(args.heat_budget_manifest),
            "heat_budget_manifest_sha256": pressure_layer.sha256(
                args.heat_budget_manifest
            ),
            "heat_budget_producer_commit": manifest["git"]["commit"],
            "start_year": args.start_year,
            "end_year": args.end_year,
            "start_year_ehb": first_source,
            "end_year_ehb": last_source,
        }
    )
    pressure_layer.validate_product(product, reference, budget)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    write_json(
        args.output_dir / "source_validation.json",
        {"annual_sources": annual_validation},
    )
    output_path = (
        args.output_dir
        / analysis_io.default_harmonized_timeseries_path(
            region=args.region,
            bottom_boundary=args.bottom_hpa,
            top_boundary=args.top_hpa,
            threshold_variable=args.threshold_variable,
            quantile=args.quantile,
            start_year=args.start_year,
            end_year=args.end_year,
        ).name
    )
    analysis_io.save_harmonized_timeseries(product, output_path)
    with analysis_io.open_harmonized_timeseries(output_path) as reopened:
        validation = pressure_layer.validate_product(reopened.load(), reference, budget)
        for key, expected in product.attrs.items():
            if reopened.attrs.get(key) != expected:
                raise ValueError(f"Stored metadata differs for {key}.")
    provenance = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "validated",
        "producer_commit": commit,
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "python": sys.executable,
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        "reference": {"path": str(args.reference), "sha256": reference_hash},
        "ehb_manifest": {
            "path": str(args.heat_budget_manifest),
            "sha256": pressure_layer.sha256(args.heat_budget_manifest),
            "content": manifest,
        },
        "source_validation": "source_validation.json",
        "validation": validation,
        "output": {
            "path": str(output_path),
            "sha256": pressure_layer.sha256(output_path),
        },
    }
    write_json(args.output_dir / "manifest.json", provenance)
    print(json.dumps(provenance, indent=2), flush=True)


if __name__ == "__main__":
    main()
