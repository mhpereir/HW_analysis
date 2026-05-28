"""Build initial ERA5 regional time-series and daily event-ID products.

This script opens the currently supported ERA5 products, builds the independent
daily LWA_a and heatwave event-ID arrays, and reports what was constructed.
Selection logic that combines or ranks events belongs downstream in
``src.selectors``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import xarray as xr

from src import analysis_io, data_io, events, harmonize

FULL_DIAGNOSTIC_DATASET_KEYS: tuple[str, ...] = (
    "pbl_p",
    "nslr",
    "nssr",
    "slhf",
    "sshf",
    "soil_moisture",
    "cloud_cover",
)

EVENT_SUMMARY_VARIABLES: dict[str, tuple[str, str]] = {
    "tas": ("hw_event_id", "tas_region"),
    "lwa": ("lwa_event_id", "lwa_region"),
    "lwa_a": ("lwa_a_event_id", "lwa_a_region"),
    "lwa_c": ("lwa_c_event_id", "lwa_c_region"),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the ERA5 input loader."""
    parser = argparse.ArgumentParser(
        description="Load the currently configured ERA5 inputs for HW_analysis."
    )
    parser.add_argument(
        "--region",
        default="pnw_bartusek",
        help="Region key used by threshold products and heat-budget source paths.",
    )
    parser.add_argument(
        "--bottom-boundary",
        default="surface",
        help=(
            "Eulerian heat-budget bottom boundary used in the saved-results path. "
            "Use 'surface' or an integer hPa value."
        ),
    )
    parser.add_argument(
        "--top-boundary",
        default="700",
        help=(
            "Eulerian heat-budget top boundary used in the saved-results path. "
            "Use an integer hPa value."
        ),
    )
    parser.add_argument(
        "--start-year-ehb",
        type=int,
        default=1940,
        help="First year token in the Eulerian heat-budget saved-results path.",
    )
    parser.add_argument(
        "--end-year-ehb",
        type=int,
        default=2025,
        help="Last year token in the Eulerian heat-budget saved-results path.",
    )
    parser.add_argument(
        "--quantile",
        default="90",
        help="Threshold quantile token, for example 95 or 97p5.",
    )
    parser.add_argument(
        "--zg-level",
        type=int,
        default=500,
        help="Pressure level used for the LWA products.",
    )
    parser.add_argument(
        "--start-year",
        type=int,
        required=True,
        help="First analysis year to load, inclusive.",
    )
    parser.add_argument(
        "--end-year",
        type=int,
        required=True,
        help="Last analysis year to load, inclusive.",
    )
    parser.add_argument(
        "--threshold-variable",
        default="tas",
        help="Variable used for thresholding.",
        choices=["tas", "lwa", "lwa_a", "lwa_c"],
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path where the harmonized Stage-1 regional dataset will be saved.",
    )
    parser.add_argument(
        "--add-full-diagnostics",
        action="store_true",
        help=(
            "Add optional local ARCO/ERA5 surface diagnostics, PBL statistics, "
            "cloud cover, and approximate heating-rate variables."
        ),
    )
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must be less than or equal to --end-year.")
    if args.start_year_ehb > args.end_year_ehb:
        parser.error("--start-year-ehb must be less than or equal to --end-year-ehb.")

    try:
        args.bottom_boundary = data_io.normalize_heat_budget_bottom_boundary(
            args.bottom_boundary
        )
        args.top_boundary = data_io.normalize_heat_budget_top_boundary(
            args.top_boundary
        )
    except ValueError as exc:
        parser.error(str(exc))

    args.analysis_years = list(range(args.start_year, args.end_year + 1))
    args.heat_budget_root = data_io.era5_heat_budget_annual_root(
        region=args.region,
        bottom_boundary=args.bottom_boundary,
        top_boundary=args.top_boundary,
        start_year_ehb=args.start_year_ehb,
        end_year_ehb=args.end_year_ehb,
    )
    if args.output_path is None:
        args.output_path = analysis_io.default_harmonized_timeseries_path(
            region=args.region,
            bottom_boundary=args.bottom_boundary,
            top_boundary=args.top_boundary,
            threshold_variable=args.threshold_variable,
            quantile=args.quantile,
            start_year=args.start_year,
            end_year=args.end_year,
        )
    return args


def load_era5_inputs(args: argparse.Namespace) -> dict[str, object]:
    """Open all currently supported ERA5 inputs."""
    datasets: dict[str, object] = {
        "tas": data_io.open_era5_tas(years=args.analysis_years),
        "lwa": data_io.open_era5_lwa(
            zg_level=args.zg_level,
            years=args.analysis_years,
        ),
        "lwa_threshold": data_io.open_era5_lwa_threshold(
            region=args.region,
            quantile=args.quantile,
            zg_level=args.zg_level,
        ),
        "hw_threshold": data_io.open_era5_hw_threshold(
            region=args.region,
            quantile=args.quantile,
            method="evolving",
            years=args.analysis_years,
        ),
        "heat_budget": data_io.open_era5_heat_budget(
            years=args.analysis_years,
            heat_budget_root=args.heat_budget_root,
        ),
    }
    if args.add_full_diagnostics:
        datasets.update(load_full_diagnostic_inputs(args))
    return datasets


def load_full_diagnostic_inputs(args: argparse.Namespace) -> dict[str, xr.Dataset]:
    """Open optional local ARCO/ERA5 full-diagnostic inputs."""
    return {
        "pbl_p": data_io.open_era5_pbl_p(years=args.analysis_years),
        "nslr": data_io.open_era5_surface_diagnostic(
            "nslr",
            years=args.analysis_years,
        ),
        "nssr": data_io.open_era5_surface_diagnostic(
            "nssr",
            years=args.analysis_years,
        ),
        "slhf": data_io.open_era5_surface_diagnostic(
            "slhf",
            years=args.analysis_years,
        ),
        "sshf": data_io.open_era5_surface_diagnostic(
            "sshf",
            years=args.analysis_years,
        ),
        "soil_moisture": data_io.open_era5_surface_diagnostic(
            "soil_moisture",
            years=args.analysis_years,
        ),
        "cloud_cover": data_io.open_era5_total_cloud_cover(
            region=args.region,
            years=args.analysis_years,
        ),
    }


def describe_dataset(name: str, ds: object) -> None:
    """Print a compact summary of one loaded dataset."""
    if not hasattr(ds, "sizes") or not hasattr(ds, "data_vars"):
        print(f"{name}: unexpected object type {type(ds)!r}")
        return

    dims_str = ", ".join(f"{dim}={size}" for dim, size in ds.sizes.items())  # type: ignore
    vars_str = ", ".join(ds.data_vars)  # type: ignore
    print(f"{name}:")
    print(f"  dims: {dims_str}")
    print(f"  vars: {vars_str}")


def describe_dataarray(name: str, da: xr.DataArray) -> None:
    """Print a compact summary of one DataArray."""
    dims_str = ", ".join(f"{dim}={size}" for dim, size in da.sizes.items())
    print(f"{name}:")
    print(f"  dims: {dims_str}")
    print(f"  name: {da.name}")
    if "region" in da.attrs:
        print(f"  region: {da.attrs['region']}")


def describe_event_ids(name: str, mask: xr.DataArray, event_id: xr.DataArray) -> None:
    """Print event-mask and event-ID counts."""
    event_days = int(mask.sum().compute().item())
    n_events = int(event_id.max().compute().item())
    print(f"{name}:")
    print(f"  event days: {event_days}")
    print(f"  events: {n_events}")


def describe_analysis_dataset(ds: xr.Dataset) -> None:
    """Print a compact summary of the assembled hourly analysis dataset."""
    dims_str = ", ".join(f"{dim}={size}" for dim, size in ds.sizes.items())
    vars_str = ", ".join(ds.data_vars) #type: ignore
    print("Harmonized Stage-1 regional dataset:")
    print(f"  dims: {dims_str}")
    print(f"  vars: {vars_str}")


def describe_event_summary_table(ds: xr.Dataset) -> None:
    """Print a compact summary of the event summary table."""
    dims_str = ", ".join(f"{dim}={size}" for dim, size in ds.sizes.items())
    vars_str = ", ".join(ds.data_vars) #type: ignore
    print("Event summary table:")
    print(f"  dims: {dims_str}")
    print(f"  vars: {vars_str}")


def append_event_summary_table(
    ds: xr.Dataset,
    *,
    threshold_variable: str = "tas",
) -> xr.Dataset:
    """Return the harmonized dataset with the requested event summary table attached."""
    event_id_name, peak_variable = event_summary_variables(threshold_variable)
    event_summary = events.build_event_summary_table(
        ds,
        event_id_name,
        peak_variable=peak_variable,
    )
    describe_event_summary_table(event_summary)
    out = xr.merge([ds, event_summary])
    out.attrs.update(event_summary.attrs)
    return out


def append_hw_event_summary_table(ds: xr.Dataset) -> xr.Dataset:
    """Return the harmonized dataset with the heatwave event summary table attached."""
    return append_event_summary_table(ds, threshold_variable="tas")


def event_summary_variables(threshold_variable: str) -> tuple[str, str]:
    """Return event-ID and peak-variable names for a threshold variable."""
    try:
        return EVENT_SUMMARY_VARIABLES[threshold_variable]
    except KeyError as exc:
        valid = ", ".join(sorted(EVENT_SUMMARY_VARIABLES))
        raise ValueError(
            f"Unsupported threshold variable {threshold_variable!r}. "
            f"Expected one of: {valid}."
        ) from exc


def require_dataset(value: Any) -> xr.Dataset:
    if not isinstance(value, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(value).__name__}")
    return value


def full_diagnostic_datasets(datasets: dict[str, object]) -> dict[str, xr.Dataset]:
    """Return optional full-diagnostic datasets from the loaded input mapping."""
    return {
        key: require_dataset(datasets[key])
        for key in FULL_DIAGNOSTIC_DATASET_KEYS
    }


def main() -> int:
    """Load the configured ERA5 products and print a summary."""
    args = parse_args()
    datasets = load_era5_inputs(args)

    print("Loaded ERA5 inputs:")
    for name, ds in datasets.items():
        describe_dataset(name, ds)

    min_duration = 1
    hw_products = events.build_hw_event_ids(
        datasets["tas"]["tas"],  # type: ignore[index]
        datasets["hw_threshold"]["threshold"],  # type: ignore[index]
        datasets["hw_threshold"]["climatology"],  # type: ignore[index]
        region=args.region,
        min_duration=min_duration,
    )
    lwa_products = events.build_lwa_event_ids(
        datasets["lwa"]["LWA"],  # type: ignore[index]
        datasets["lwa_threshold"]["LWA"],  # type: ignore[index]
        region=args.region,
        variable="LWA",
        years=args.analysis_years,
        min_duration=min_duration,
    )
    lwa_a_products = events.build_lwa_event_ids(
        datasets["lwa"]["LWA_a"],  # type: ignore[index]
        datasets["lwa_threshold"]["LWA_a"],  # type: ignore[index]
        region=args.region,
        variable="LWA_a",
        years=args.analysis_years,
        min_duration=min_duration,
    )

    lwa_c_products = events.build_lwa_event_ids(
        datasets["lwa"]["LWA_c"],  # type: ignore[index]
        datasets["lwa_threshold"]["LWA_c"],  # type: ignore[index]
        region=args.region,
        variable="LWA_c",
        years=args.analysis_years,
        min_duration=min_duration,
    )

    print("Preprocessed regional inputs:")
    describe_dataarray("tas_region", hw_products["tas_region"])
    describe_dataarray("lwa_region", lwa_products["lwa_region"])
    describe_dataarray("lwa_a_region", lwa_a_products["lwa_a_region"])
    describe_dataarray("lwa_c_region", lwa_c_products["lwa_c_region"])

    print("Daily event-ID products:")
    describe_event_ids(
        "hw_event_id",
        hw_products["hw_exceedance_mask"],
        hw_products["hw_event_id"],
    )
    describe_event_ids(
        "lwa_event_id",
        lwa_products["lwa_exceedance_mask"],
        lwa_products["lwa_event_id"],
    )
    describe_event_ids(
        "lwa_a_event_id",
        lwa_a_products["lwa_a_exceedance_mask"],
        lwa_a_products["lwa_a_event_id"],
    )
    describe_event_ids(
        "lwa_c_event_id",
        lwa_c_products["lwa_c_exceedance_mask"],
        lwa_c_products["lwa_c_event_id"],
    )

    heat_budget_dataset = require_dataset(datasets["heat_budget"]) #ensuring correct type
    analysis_ds = harmonize.build_regional_analysis_dataset(
        heat_budget=heat_budget_dataset,
        hw_event_products=hw_products,
        lwa_event_products=[lwa_products, lwa_a_products, lwa_c_products],
        full_diagnostics=(
            full_diagnostic_datasets(datasets)
            if args.add_full_diagnostics
            else None
        ),
        region=args.region,
        attrs={
            "region": args.region,
            "quantile": str(args.quantile),
            "threshold_variable": args.threshold_variable,
            "heat_budget_bottom_boundary": args.bottom_boundary,
            "heat_budget_top_boundary": args.top_boundary,
            "start_year_ehb": args.start_year_ehb,
            "end_year_ehb": args.end_year_ehb,
            "heat_budget_root": str(args.heat_budget_root),
            "start_year": args.start_year,
            "end_year": args.end_year,
            "zg_level": args.zg_level,
            "min_duration": min_duration,
            "add_full_diagnostics": args.add_full_diagnostics,
        },
    )
    describe_analysis_dataset(analysis_ds)

    analysis_ds = append_event_summary_table(
        analysis_ds,
        threshold_variable=args.threshold_variable,
    )

    saved_path = analysis_io.save_harmonized_timeseries(analysis_ds, args.output_path)
    print(f"Saved harmonized Stage-1 regional dataset: {saved_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
