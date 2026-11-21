"""Audit JJA legend metadata and seasonal populations before temporal plotting."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_composite_timeseries_all as all_plot
from scripts import plot_top_events as top_plot
from scripts.validate_season_selection import artifact_path, full_jja_ids, sha256
from src import analysis_io, climatology, composites, plotting, season_selection


def audit_region(inputs: dict) -> dict:
    """Compare configured inputs and derived context against direct reductions."""
    hashes = {}
    for key in ("input", "climatology"):
        hashes[key] = sha256(artifact_path(inputs[key]))
        if hashes[key] != inputs[f"{key}_sha256"]:
            raise ValueError(f"Changed {key}: {inputs[key]}")
    variables = all_plot.EXTENDED_COMPOSITE_VARIABLES
    with (
        analysis_io.open_harmonized_timeseries(Path(inputs["input"])) as source,
        analysis_io.open_regional_hourly_climatology(
            Path(inputs["climatology"])
        ) as climate,
    ):
        selected = season_selection.select_event_population(source)
        np.testing.assert_array_equal(selected.event_id, full_jja_ids(source))
        ranked = top_plot.select_top_tas_events(selected, n=10)
        np.testing.assert_array_equal(
            ranked.tas_peak, np.sort(selected.tas_peak)[-10:][::-1]
        )
        anomaly = climatology.apply_regional_hourly_climatology(
            source, climate, variables=variables, include_jja_means=True
        )
        summer = climate.climatology_time.dt.month.isin([6, 7, 8])
        assert int(summer.sum()) == 2208
        means = {
            n: float(np.mean(climate[n].values[summer.values], dtype=np.float64))
            for n in variables
        }
        # Match actual month/day/hour tuples independently of the anomaly helper.
        index = climate.indexes["climatology_time"]
        lookup = {(t.month, t.day, t.hour): i for i, t in enumerate(index)}
        sample = np.unique(np.linspace(0, source.sizes["time"] - 1, 101, dtype=int))
        times = source.indexes["time"][sample]
        positions = [lookup[(t.month, t.day, t.hour)] for t in times]
        for name in variables:
            expected = source[name].values[sample] - climate[name].values[positions]
            np.testing.assert_allclose(
                anomaly[name].values[sample], expected, rtol=0, atol=0
            )
            assert anomaly[name].attrs["climatology_jja_mean"] == means[name]
            assert "climatology_jja_mean" not in source[name].attrs
        composite = composites.all_event_peak_aligned_composite(
            anomaly,
            event_table=selected,
            variables=variables,
            pre_days=7,
            post_days=7,
            event_percentiles=(0.25, 0.5, 0.75),
        )
        composite.attrs.update(anomaly.attrs)
        smoothed = plotting.smooth_composite_for_display(
            composite,
            variables=all_plot.EXTENDED_SMOOTHED_VARIABLES,
            smoothing_window=24,
        )
        for dataset in (composite, smoothed):
            for name in variables:
                assert dataset[name].attrs["climatology_jja_mean"] == means[name]
        # A small independent reference window avoids duplicating a large stack.
        for name in variables:
            peaks = selected.peak_time.values
            expected = np.nanmean(anomaly[name].sel(time=peaks).values)
            np.testing.assert_allclose(composite[name].sel(lag_hour=0), expected)
        filenames = {}
        for representation, tag in (("absolute", None), ("clim_anom", "clim_anom")):
            filenames[representation] = [
                filename(event, filename_tag=tag)
                for i in range(ranked.sizes["event"])
                for event in [ranked.isel(event=i)]
                for filename in (
                    top_plot._event_figure_filename,
                    top_plot._smoothed_event_figure_filename,
                )
            ]
        return {
            "input_hashes": hashes,
            "time_samples": source.sizes["time"],
            "saved_events": source.sizes["event"],
            "full_jja_events": selected.sizes["event"],
            "selected_event_ids": selected.event_id.values.tolist(),
            "ranked_event_ids": ranked.event_id.values.tolist(),
            "ranked_peak_times": ranked.peak_time.values.astype(str).tolist(),
            "expected_top_files": filenames,
            "jja_calendar_hours": 2208,
            "jja_means_native_units": means,
            "legend_labels": {
                n: plotting._temporal_variable_label(composite, n) for n in variables
            },
            "timestamp_samples_checked": len(sample),
            "composite_metadata_and_peak_mean_checked": True,
            "smoothing_metadata_checked": True,
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest_path = artifact_path(str(args.manifest))
    output = artifact_path(str(args.output_dir))
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(manifest_path.read_text())
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "python": sys.executable,
        "manifest_sha256": sha256(manifest_path),
        "regions": {},
    }
    for region, inputs in manifest["regions"].items():
        record = audit_region(inputs)
        report["regions"][region] = record
        (output / f"{region}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"[audit] {region}: {record['full_jja_events']} full-JJA events; JJA metadata passed",
            flush=True,
        )
    report["passed"] = True
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
