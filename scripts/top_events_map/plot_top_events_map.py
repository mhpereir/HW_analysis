"""Render single-panel T2m/Z500 maps from a prepared top_events_map product."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, config, top_events_map_plotting


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--outline-regions",
        nargs="+",
        choices=sorted(config.REGIONS),
        help="Default: both PNW regions for a PNW event; otherwise the event region.",
    )
    parser.add_argument(
        "--temperature-limit", type=float, help="Symmetric T2m anomaly limit [K]."
    )
    parser.add_argument(
        "--height-contour-interval",
        type=float,
        default=top_events_map_plotting.DEFAULT_HEIGHT_CONTOUR_INTERVAL,
        help="Z500 anomaly contour spacing [m] (default: 50).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    with analysis_io.open_top_event_maps(args.input_path) as product:
        paths = top_events_map_plotting.write_top_event_maps(
            product.load(),
            args.output_dir,
            outline_regions=args.outline_regions,
            temperature_limit=args.temperature_limit,
            height_contour_interval=args.height_contour_interval,
        )
    for path in paths:
        print(f"Wrote top-event map: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
