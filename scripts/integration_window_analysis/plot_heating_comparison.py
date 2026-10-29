"""Plot raw and climatology-corrected target-event ranks from saved tables."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from scripts.integration_window_analysis.build_heating_comparison import sha256
from src import analysis_io, plot_style
from src.integration_window_plotting import plot_heating_ranks


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-paths", nargs="+", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    outputs = {
        suffix: args.output_stem.with_suffix(suffix)
        for suffix in (".png", ".pdf", ".json")
    }
    for path in outputs.values():
        if path.exists():
            raise FileExistsError(path)
    hashes = {str(path.resolve()): sha256(path) for path in args.input_paths}
    with ExitStack() as stack:
        tables = [
            stack.enter_context(analysis_io.open_integration_window_ranks(path))
            for path in args.input_paths
        ]
        fig = plot_heating_ranks(tables)
        args.output_stem.parent.mkdir(parents=True, exist_ok=True)
        try:
            for suffix in (".png", ".pdf"):
                plot_style.save_figure(fig, outputs[suffix])
        finally:
            plt.close(fig)
        with Image.open(outputs[".png"]) as picture:
            picture.load()
            if min(picture.size) < 1000 or all(
                low == high for low, high in picture.convert("RGB").getextrema()
            ):
                raise ValueError("Blank or undersized rank figure.")
        if any(
            sha256(path) != hashes[str(path.resolve())] for path in args.input_paths
        ):
            raise ValueError("A comparison table changed during plotting.")
        manifest = {
            "generating_commit": os.environ["EXPECTED_COMMIT"],
            "pbs_job_id": os.environ["PBS_JOBID"],
            "input_sha256": hashes,
            "output_sha256": {
                path.name: sha256(path)
                for suffix, path in outputs.items()
                if suffix != ".json"
            },
            "regions": [table.attrs["region"] for table in tables],
            "population_sizes": [
                table.attrs["common_population_size"] for table in tables
            ],
            "acceptance": "Numerical validation passed; scheduler, logs and visual inspection remain required.",
        }
        with outputs[".json"].open("x") as stream:
            json.dump(manifest, stream, indent=2, default=int)
    print(f"Saved {outputs['.png']} and {outputs['.pdf']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
