"""Plot antecedent-temperature diagnostics from a prepared Stage-2 event table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from src import plot_style
from src.temperature_diagnostics import plot_antecedent_temperature


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.output_path.exists():
        raise FileExistsError(f"Output already exists: {args.output_path}")
    with xr.open_dataset(
        args.input_path, engine="h5netcdf", decode_timedelta=True
    ) as features:
        fig = plot_antecedent_temperature(features)
        try:
            args.output_path.parent.mkdir(parents=True, exist_ok=True)
            plot_style.save_figure(fig, args.output_path)
        finally:
            plt.close(fig)
    print(f"Wrote {args.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
