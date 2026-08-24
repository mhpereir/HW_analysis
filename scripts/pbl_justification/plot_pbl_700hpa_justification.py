"""Plot the standalone PBL/700 hPa justification product."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt

from src import pbl_justification, pbl_justification_plotting


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the two-panel PBL/700 hPa justification figure."
    )
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument(
        "--map-margin-degrees",
        type=float,
        default=pbl_justification_plotting.MAP_MARGIN_DEGREES,
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_path = args.output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing figure: {output_path}")
    product = pbl_justification.open_product(input_path)
    try:
        figure = pbl_justification_plotting.plot_product(
            product.load(),
            map_margin_degrees=args.map_margin_degrees,
        )
    finally:
        product.close()
    try:
        written = pbl_justification_plotting.write_figure(figure, output_path)
    finally:
        plt.close(figure)
    print(f"Wrote PBL/700 hPa justification figure: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
