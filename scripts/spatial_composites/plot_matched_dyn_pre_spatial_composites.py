"""Plot matched positive/negative I_dyn_pre daily T2m and Z500 composites."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.spatial_composites import (  # noqa: E402
    build_matched_dyn_pre_spatial_composites as matched_builder,
    plot_dyn_net_spatial_composites as spatial_plotter,
)


DEFAULT_INPUT_PATH = matched_builder.DEFAULT_OUTPUT_PATH
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results/spatial_composites"
    / (
        "matched_dyn_pre_daily_t2m_z500_composites_pnw_bartusek_"
        "tas_q90_1940_2024_peak_anomaly_0p20.png"
    )
)


def parse_args() -> argparse.Namespace:
    """Parse matched-product and figure arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot matched positive/negative I_dyn_pre daily T2m/Z500 "
            "spatial composites."
        )
    )
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument(
        "--matching-specification",
        default=matched_builder.DEFAULT_MATCHING_SPECIFICATION,
        help="Required matching specification recorded in the input product.",
    )
    parser.add_argument(
        "--plot-lags",
        type=int,
        nargs="+",
        default=spatial_plotter.DEFAULT_PLOT_LAGS,
    )
    parser.add_argument("--temperature-limit", type=float, default=None)
    parser.add_argument("--height-contour-interval", type=float, default=None)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate paths and delegate numeric display checks."""
    spatial_plotter.validate_args(args)
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Matched composite does not exist: {input_path}.")
    if output_path.suffix.lower() != ".png":
        raise ValueError("--output-path must use the .png suffix.")
    if output_path.exists():
        raise FileExistsError(f"Matched spatial plot already exists: {output_path}.")


def main() -> int:
    """Validate the matched product and write its separate map figure."""
    args = parse_args()
    validate_args(args)
    input_path = args.input_path.expanduser().resolve()
    with xr.open_dataset(
        input_path,
        engine="h5netcdf",
        decode_timedelta=True,
    ) as ds:
        loaded = ds.load()
    figure = spatial_plotter.plot_matched_spatial_composites(
        loaded,
        plot_lags=args.plot_lags,
        temperature_limit=args.temperature_limit,
        height_contour_interval=args.height_contour_interval,
        expected_specification=args.matching_specification,
    )
    try:
        written = spatial_plotter.write_figure(figure, args.output_path)
    finally:
        plt.close(figure)
    print(f"Wrote matched I_dyn_pre spatial composite figure: {written}")
    print(f"Matching specification: {loaded.attrs['matching_specification']}")
    print(f"Matched pairs: {int(loaded.attrs['matching_pair_count'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
