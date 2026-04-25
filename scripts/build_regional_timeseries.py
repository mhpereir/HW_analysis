"""Load the current ERA5 inputs for the regional time-series workflow.

This script is intentionally thin: it opens the currently supported ERA5
products through ``src.data_io`` and reports what was loaded. It does not yet
harmonize timesteps, average regions, or write analysis-ready outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import data_io, preprocess


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

    dims_str = ", ".join(f"{dim}={size}" for dim, size in ds.sizes.items()) # type: ignore
    vars_str = ", ".join(ds.data_vars) # type: ignore
    print(f"{name}:")
    print(f"  dims: {dims_str}")
    print(f"  vars: {vars_str}")


def main() -> int:
    """Load the configured ERA5 products and print a summary."""
    args = parse_args()
    datasets = load_era5_inputs(args)

    print("Loaded ERA5 inputs:")
    for name, ds in datasets.items():
        describe_dataset(name, ds)

    tas_region = preprocess.compute_region_mean(datasets["tas"]["tas"], args.region) # type: ignore[index]
    print("Preprocessed regional inputs:")
    print(f"tas_region:")
    print(f"  dims: {', '.join(f'{dim}={size}' for dim, size in tas_region.sizes.items())}")
    print(f"  name: {tas_region.name}")
    print(f"  region: {tas_region.attrs.get('region')}")

    datasets["tas_region"] = tas_region

    

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
