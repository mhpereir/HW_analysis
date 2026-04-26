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

from src import data_io, events, harmonize


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the ERA5 input loader."""
    parser = argparse.ArgumentParser(
        description="Load the currently configured ERA5 inputs for HW_analysis."
    )
    parser.add_argument(
        "--region",
        default="pnw_bartusek",
        help="Region key used by threshold products.",
    )
    parser.add_argument(
        "--quantile",
        default="95",
        help="Threshold quantile token, for example 95 or 97p5.",
    )
    parser.add_argument(
        "--zg-level",
        type=int,
        default=500,
        help="Pressure level used for the LWA products.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        help="Optional list of years to restrict tas and heat-budget loading.",
    )
    parser.add_argument(
        "--threshold-variable",
        default="tas",
        help="Variable used for thresholding.",
        choices=["tas", "lwa", "lwa_a", "lwa_c"],
    )
    return parser.parse_args()


def load_era5_inputs(args: argparse.Namespace) -> dict[str, object]:
    """Open all currently supported ERA5 inputs."""
    return {
        "tas": data_io.open_era5_tas(years=args.years),
        "lwa": data_io.open_era5_lwa(zg_level=args.zg_level),
        "lwa_threshold": data_io.open_era5_lwa_threshold(
            region=args.region,
            quantile=args.quantile,
            zg_level=args.zg_level,
        ),
        "hw_threshold": data_io.open_era5_hw_threshold(
            region=args.region,
            quantile=args.quantile,
            method="evolving",
        ),
        "heat_budget": data_io.open_era5_heat_budget(years=args.years),
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

def require_dataset(value: Any) -> xr.Dataset:
    if not isinstance(value, xr.Dataset):
        raise TypeError(f"Expected xr.Dataset, got {type(value).__name__}")
    return value

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
        region=args.region,
        min_duration=min_duration,
    )
    lwa_a_products = events.build_lwa_event_ids(
        datasets["lwa"]["LWA_a"],  # type: ignore[index]
        datasets["lwa_threshold"]["LWA_a"],  # type: ignore[index]
        region=args.region,
        variable="LWA_a",
        years=args.years,
        min_duration=min_duration,
    )

    lwa_c_products = events.build_lwa_event_ids(
        datasets["lwa"]["LWA_c"],  # type: ignore[index]
        datasets["lwa_threshold"]["LWA_c"],  # type: ignore[index]
        region=args.region,
        variable="LWA_c",
        years=args.years,
        min_duration=min_duration,
    )

    print("Preprocessed regional inputs:")
    describe_dataarray("tas_region", hw_products["tas_region"])
    describe_dataarray("lwa_a_region", lwa_a_products["lwa_a_region"])
    describe_dataarray("lwa_c_region", lwa_c_products["lwa_c_region"])

    print("Daily event-ID products:")
    describe_event_ids(
        "hw_event_id",
        hw_products["hw_exceedance_mask"],
        hw_products["hw_event_id"],
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
        lwa_event_products=[lwa_a_products, lwa_c_products],
        attrs={
            "region": args.region,
            "quantile": str(args.quantile),
            "zg_level": args.zg_level,
            "min_duration": min_duration,
        },
    )
    describe_analysis_dataset(analysis_ds)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
