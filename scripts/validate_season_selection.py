"""Audit seasonal defaults against immutable campaign inputs on a PBS worker.

Rendering is suppressed while inspecting the real all/top CLI reductions.
Production PNGs are generated separately by their normal tracked schedulers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_composite_timeseries_all as all_plot
from scripts import plot_composite_timeseries_all_clim_anom as anomaly_all
from scripts import plot_top_events as top_plot
from scripts import plot_top_events_clim_anom as anomaly_top
from scripts.event_features import build_stage2_baseline_features as baseline_builder
from scripts.event_features import build_stage2_event_features as event_builder
from src import analysis_io, plotting, season_selection
from src.artifact_paths import artifact_root


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def artifact_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or not path.resolve().is_relative_to(artifact_root()):
        raise ValueError(
            f"Expected a prepared artifact under {artifact_root()}: {path}"
        )
    if path.resolve() == artifact_root():
        raise ValueError("An artifact path cannot be the artifact root itself.")
    return path


def full_jja_ids(source: xr.Dataset) -> np.ndarray:
    """Independent whole-interval check, without calling the shared selector."""
    start, end = source.start_time.values, source.end_time.values
    year = start.astype("datetime64[Y]").astype("datetime64[M]")
    june = year + np.timedelta64(5, "M")
    september = year + np.timedelta64(8, "M")
    keep = (start >= june) & (end < september) & (end >= start)
    return source.event_id.values[keep]


def write_arrays(dataset: xr.Dataset, path: Path) -> None:
    numeric = dataset.copy(deep=False)
    numeric.attrs = {}
    for name in numeric.variables:
        numeric[name].attrs = {}
    numeric.to_netcdf(path, engine="h5netcdf")


def audit_reference(
    region: str, inputs: dict, output: Path, anomaly: bool, *, top_n: int = 10
) -> dict:
    representation = "clim_anom" if anomaly else "absolute"
    destination = output / region / representation
    destination.mkdir(parents=True)
    all_module, top_module = (
        (anomaly_all, anomaly_top) if anomaly else (all_plot, top_plot)
    )
    all_references, top_references, top_ids = [], [], []

    def capture_all(composite, path, **kwargs):
        all_references.append(composite.load())
        return []

    def capture_top(source, event, *, reference_composite, **kwargs):
        top_references.append(reference_composite.load())
        top_ids.append(int(event.event_id))
        return plt.figure()

    flags = [
        "--region",
        region,
        "--bottom-boundary",
        "surface",
        "--top-boundary",
        "700",
        "--threshold-variable",
        "tas",
        "--quantile",
        "90",
        "--start-year",
        "1940",
        "--end-year",
        "2024",
        "--input-path",
        inputs["input"],
        "--window-days",
        "7",
        "--smoothing-window",
        "24",
        "--layout",
        "paper",
        "--plot-extended-variables",
    ]
    if anomaly:
        flags += ["--climatology-path", inputs["climatology"]]
    with patch.object(plotting, "write_composite_timeseries_outputs", capture_all):
        with patch.object(
            sys,
            "argv",
            ["all", *flags, "--output-path", str(destination / "capture.png")],
        ):
            assert all_module.main() == 0
    with patch.object(top_plot, "plot_one_top_event", capture_top):
        with patch.object(top_plot.plot_style, "save_figure", lambda *a, **kw: None):
            with patch.object(
                sys,
                "argv",
                [
                    "top",
                    *flags,
                    "--output-dir",
                    str(destination / "capture"),
                    "--top-n",
                    str(top_n),
                ],
            ):
                assert top_module.main() == 0
    assert len(all_references) == 1 and len(top_references) == 2 * top_n
    raw = all_references[0]
    smooth = plotting.smooth_composite_for_display(
        raw, variables=all_plot.EXTENDED_SMOOTHED_VARIABLES, smoothing_window=24
    )
    for index, reference in enumerate(top_references):
        expected = smooth if index % 2 else raw
        assert set(reference.data_vars) == set(expected.data_vars)
        for name in expected.data_vars:
            xr.testing.assert_equal(reference[name], expected[name])
    assert top_ids[::2] == top_ids[1::2]
    write_arrays(raw, destination / "reference.nc")
    write_arrays(smooth, destination / "reference_smoothed.nc")
    return {
        "ranked_event_ids": top_ids[::2],
        "reference_arrays": len(raw.data_vars),
        "lag_hours": raw.sizes["lag_hour"],
        "raw_max_difference": 0.0,
        "smoothed_max_difference": 0.0,
        "all_ranked_references_equal": True,
    }


def audit_stage2(source: xr.Dataset, output: Path) -> dict:
    """Exercise 21-day integrals without changing the configured default window."""
    with patch.dict(
        event_builder.fixed.config.WINDOWS,
        {
            "heat_budget_pre": (-504, 0),
            "lwa_pre_peak": (-504, 0),
        },
    ):
        events = event_builder.build_event_features(source)
        baseline = baseline_builder.build_baseline_features(source)
        full = event_builder.build_event_features(source, require_full_event=True)
        all_seasons = event_builder.build_event_features(source, all_seasons=True)
    season_selection.validate_stage2_season(events)
    season_selection.validate_stage2_season(baseline)
    season_selection.validate_stage2_season(full, require_full_event=True)
    season_selection.validate_stage2_season(all_seasons, all_seasons=True)
    for incompatible in (full, all_seasons):
        try:
            season_selection.validate_stage2_season(incompatible)
        except ValueError:
            pass
        else:
            raise AssertionError(
                "Default Stage-2 consumer accepted incompatible membership"
            )
    np.testing.assert_array_equal(full.event_id.values, full_jja_ids(source))
    expected = source.event_id.values[np.isin(source.peak_time.dt.month, [6, 7, 8])]
    np.testing.assert_array_equal(events.event_id.values, expected)
    assert np.isin(events.event_id, all_seasons.event_id).all()
    assert np.isin(all_seasons.event_id, source.event_id).all()
    assert not np.isin(all_seasons.peak_time.dt.month, [6, 7, 8]).all()
    report = {}
    times = source.time.values
    for label, table, anchor, dimension in (
        ("events", events, "peak_time", "event"),
        ("baseline", baseline, "reference_time", "baseline_day"),
    ):
        selected = (table[anchor].dt.month == 6) & (table[anchor].dt.day <= 21)
        indices = np.flatnonzero(selected.values)
        assert indices.size > 0
        largest_error = 0.0
        values = source.diabatic.values
        for index in indices:
            endpoint = table[anchor].values[index]
            first, last = np.searchsorted(
                times, [endpoint - np.timedelta64(21, "D"), endpoint]
            )
            assert times[first] == endpoint - np.timedelta64(21, "D")
            assert times[last] == endpoint and last - first + 1 == 505
            assert int(table.n_samples_heat_budget_pre.values[index]) == 505
            expected_integral = np.nansum(values[first : last + 1])
            actual = float(table.I_diabatic_pre.values[index])
            np.testing.assert_allclose(actual, expected_integral, rtol=1e-10, atol=1e-8)
            largest_error = max(largest_error, abs(actual - expected_integral))
        table.to_netcdf(output / f"stage2_{label}_21day.nc", engine="h5netcdf")
        report[label] = {
            "rows": table.sizes[dimension],
            "early_june_anchors_checked": int(indices.size),
            "inclusive_samples": 505,
            "max_direct_sum_difference": largest_error,
        }
    report["full_event_rows"] = full.sizes["event"]
    report["all_season_event_rows"] = all_seasons.sizes["event"]
    report["incompatible_default_consumers_rejected"] = True
    return report


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
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256(manifest_path),
        "rendering": "suppressed; inspect real CLI selection, reductions and smoothing",
        "regions": {},
    }
    for region, inputs in manifest["regions"].items():
        for key in ("input", "climatology"):
            actual_hash = sha256(artifact_path(inputs[key]))
            assert actual_hash == inputs[f"{key}_sha256"], f"Changed {region} {key}"
        with analysis_io.open_harmonized_timeseries(Path(inputs["input"])) as source:
            independent_ids = full_jja_ids(source)
            selected = season_selection.select_event_population(source)
            np.testing.assert_array_equal(selected.event_id.values, independent_ids)
            metrics = selected.tas_peak.values
            ranked = top_plot.select_top_tas_events(selected, n=10)
            np.testing.assert_array_equal(
                ranked.tas_peak.values, np.sort(metrics)[-10:][::-1]
            )
            record = {
                "saved_events": source.sizes["event"],
                "jja_full_events": int(independent_ids.size),
                "selected_event_ids": independent_ids.tolist(),
                "previous_ranked_ids": top_plot.select_top_tas_events(
                    source, n=10
                ).event_id.values.tolist(),
                "ranked_peak_times": ranked.peak_time.values.astype(str).tolist(),
                "stage1_contract_version": source.attrs.get("stage1_contract_version"),
                "time_samples": source.sizes["time"],
                "input_hashes_verified": True,
            }
            if region == "pnw_bartusek":
                record["stage2_21day"] = audit_stage2(source, output)
            expected_ids = ranked.event_id.values.tolist()
        for anomaly in (False, True):
            key = "clim_anom" if anomaly else "absolute"
            record[key] = audit_reference(region, inputs, output, anomaly)
            assert record[key]["ranked_event_ids"] == expected_ids
        report["regions"][region] = record
        (output / f"{region}.json").write_text(json.dumps(record, indent=2) + "\n")
        print(
            f"[audit] {region}: {record['saved_events']} saved, {record['jja_full_events']} full JJA; references identical"
        )
    report["passed"] = True
    (output / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
