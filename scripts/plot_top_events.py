"""Plot top-event diagnostics from the saved Stage-1 harmonized dataset.

This script consumes the harmonized regional time-series product written by
``build_regional_timeseries.py``. Event ranking and plotting will be layered on
top of this Stage-2 entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io


def parse_args() -> argparse.Namespace:
    """Parse command-line options for top-event plotting."""
    parser = argparse.ArgumentParser(
        description="Plot top-event diagnostics from a harmonized Stage-1 dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
        help="Path to the saved harmonized Stage-1 regional dataset.",
    )
    return parser.parse_args()


def open_harmonized_dataset(
    path: str | Path = analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
    *,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open the harmonized Stage-1 dataset used by top-event plots."""
    return analysis_io.open_harmonized_timeseries(path, chunks=chunks)


def load_plot_inputs(args: argparse.Namespace) -> xr.Dataset:
    """Open the saved harmonized dataset requested by CLI args."""
    return open_harmonized_dataset(args.input_path)


def describe_harmonized_dataset(ds: xr.Dataset) -> None:
    """Print a compact summary of the loaded harmonized dataset."""
    dims_str = ", ".join(f"{dim}={size}" for dim, size in ds.sizes.items())
    vars_str = ", ".join(ds.data_vars) # type: ignore
    print("Loaded harmonized Stage-1 regional dataset:")
    print(f"  dims: {dims_str}")
    print(f"  vars: {vars_str}")


def main() -> int:
    """Open the harmonized dataset and report what was loaded."""
    args = parse_args()
    ds = load_plot_inputs(args)
    try:
        describe_harmonized_dataset(ds)
    finally:
        ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
