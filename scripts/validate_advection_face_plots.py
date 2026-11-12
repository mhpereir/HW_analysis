"""Audit face-plot CLI reductions against independent source-array calculations.

Run on a PBS worker with immutable production inputs. Rendering is captured here;
the five normal schedulers render the same configuration separately.
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

import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import plot_advection_direction_exploration as absolute
from scripts import plot_advection_direction_exploration_clim_anom as anomaly
from scripts import plot_advection_direction_exploration_matched_clim_anom as matched
from scripts import plot_advection_direction_exploration_top_events as top_absolute
from scripts import (
    plot_advection_direction_exploration_top_events_clim_anom as top_anomaly,
)
from scripts.validate_season_selection import (
    artifact_path,
    full_jja_ids,
    sha256,
    write_arrays,
)
from src import advection_direction
from src import advection_direction_plotting as renderer


def capture_cli(module, flags, destination, *, top=False, match=False):
    """Exercise the actual CLI selection and preparation without rendering."""
    captured = []
    prepared = []

    def capture(*args, **kwargs):
        captured.extend(value.load() for value in args if isinstance(value, xr.Dataset))
        return []

    build = matched.build_matched_composites

    def capture_match(*args, **kwargs):
        result = build(*args, **kwargs)
        prepared.append(result)
        return result

    writer = (
        "write_top_event_advection_direction_exploration_outputs"
        if top
        else "write_matched_advection_direction_exploration_outputs"
        if match
        else "write_advection_direction_exploration_outputs"
    )
    output_flag = "--output-dir" if top else "--output-path"
    with (
        patch.object(renderer, writer, capture),
        patch.object(matched, "build_matched_composites", capture_match),
        patch.object(
            sys, "argv", [module.__file__, *flags, output_flag, str(destination)]
        ),
    ):
        assert module.main() == 0
    return captured, prepared


def source_windows(source, climate, ids, variables, *, anomalies):
    """Index actual timestamps and calendar fields, without production helpers."""
    rows = pd.Index(source.event_id.values).get_indexer(ids)
    assert np.all(rows >= 0)
    dates = source.peak_time.values[rows, None] + np.arange(-168, 169) * np.timedelta64(
        1, "h"
    )
    indices = pd.DatetimeIndex(source.time.values).get_indexer(dates.ravel())
    assert np.all(indices >= 0), "Missing source hours in a plotted event window"
    clock = pd.DatetimeIndex(dates.ravel())
    keys = clock.month * 10000 + clock.day * 100 + clock.hour
    available = (
        climate.month.values.astype(np.int32) * 10000
        + climate.day.values * 100
        + climate.hour_utc.values
    )
    baseline_indices = pd.Index(available).get_indexer(keys)
    assert np.all(baseline_indices >= 0)
    result = {}
    for name in variables:
        values = source[name].values[indices].astype(float)
        if anomalies:
            values = values - climate[name].values[baseline_indices]
        result[name] = values.reshape(dates.shape)
    return result


def verify_arrays(actual, expected, destination):
    """Check reductions, grouped identities and a manually indexed rolling mean."""
    errors = {}
    np.testing.assert_array_equal(actual.lag_hour, np.arange(-168, 169))
    smooth = renderer._smooth_advection_composite_for_display(
        actual, smoothing_window=24
    )
    for name, values in expected.items():
        np.testing.assert_allclose(actual[name], values, rtol=1e-10, atol=1e-12)
        errors[name] = float(np.nanmax(np.abs(actual[name].values - values)))
        rolling = np.full_like(values, np.nan, dtype=float)
        for index in range(12, values.shape[-1] - 11):
            rolling[..., index] = np.mean(values[..., index - 12 : index + 12], axis=-1)
        np.testing.assert_allclose(smooth[name], rolling, rtol=1e-10, atol=1e-12)
    for dataset in (actual, smooth):
        grouped = advection_direction.grouped_advection_components(dataset)
        zonal = dataset.advection_west + dataset.advection_east
        meridional = dataset.advection_south + dataset.advection_north
        vertical = dataset.advection_top + dataset.get("advection_bottom", 0)
        for name, values in {
            "zonal": zonal,
            "meridional": meridional,
            "horizontal": zonal + meridional,
            "vertical": vertical,
            "face_total": zonal + meridional + vertical,
        }.items():
            np.testing.assert_allclose(
                grouped[f"advection_{name}"], values, rtol=1e-10, atol=1e-12
            )
        np.testing.assert_allclose(
            grouped.advection_face_total, dataset.advection, rtol=1e-10, atol=1e-12
        )
    write_arrays(actual, destination.with_suffix(".nc"))
    write_arrays(smooth, destination.with_name(destination.name + "_smoothed.nc"))
    return errors


def audit_region(region, inputs, output):
    output.mkdir()
    paths = {
        key: artifact_path(inputs[key])
        for key in ("input", "climatology", "event_features")
    }
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
        str(paths["input"]),
        "--window-days",
        "7",
        "--smoothing-window",
        "24",
    ]
    climate_flags = [*flags, "--climatology-path", str(paths["climatology"])]
    with (
        xr.open_dataset(
            paths["input"], engine="h5netcdf", decode_timedelta=True
        ) as source,
        xr.open_dataset(
            paths["climatology"], engine="h5netcdf", decode_timedelta=True
        ) as climate,
    ):
        assert (
            source.attrs["pipeline_stage"] == "stage_1_harmonized_regional_timeseries"
        )
        ids = full_jja_ids(source)
        assert len(ids) == inputs["expected_full_jja_events"]
        rows = pd.Index(source.event_id.values).get_indexer(ids)
        order = np.argsort(-source.tas_peak.values[rows], kind="stable")[:10]
        top_ids = ids[order]
        np.testing.assert_array_equal(top_ids, inputs["expected_top_ids"])
        variables = [
            "advection",
            *[
                f"advection_{face}"
                for face in ("west", "east", "south", "north", "top", "bottom")
                if f"advection_{face}" in source
            ],
        ]
        for name in variables:
            assert source[name].attrs["units"] == "K hr-1"
        checks = {}
        for representation, all_module, top_module, argv in (
            ("absolute", absolute, top_absolute, flags),
            ("clim_anom", anomaly, top_anomaly, climate_flags),
        ):
            window = source_windows(
                source, climate, ids, variables, anomalies=representation == "clim_anom"
            )
            expected = {
                name: np.nanmean(values, axis=0) for name, values in window.items()
            }
            (all_reference,), _ = capture_cli(
                all_module, argv, output / f"capture_{representation}.png"
            )
            (top_reference, events), _ = capture_cli(
                top_module, argv, output / f"capture_top_{representation}", top=True
            )
            assert (
                all_reference.attrs["n_events"]
                == top_reference.attrs["n_events"]
                == len(ids)
            )
            np.testing.assert_array_equal(events.event_id, top_ids)
            for name in variables:
                np.testing.assert_array_equal(all_reference[name], top_reference[name])
            checks[representation] = verify_arrays(
                all_reference, expected, output / representation
            )
            checks[f"top_{representation}"] = verify_arrays(
                events,
                {name: values[order] for name, values in window.items()},
                output / f"top_{representation}",
            )
        (negative, positive), (prepared,) = capture_cli(
            matched,
            [
                *climate_flags,
                "--event-features-path",
                str(paths["event_features"]),
                "--matching-specification",
                "peak_anomaly_0p20",
            ],
            output / "capture_matched.png",
            match=True,
        )
        pair_ids = {}
        with xr.open_dataset(
            paths["event_features"], engine="h5netcdf", decode_timedelta=True
        ) as features:
            eligible = full_jja_ids(features)
            eligible_rows = pd.Index(features.event_id.values).get_indexer(eligible)
            metric = features.I_dyn_pre.values[eligible_rows]
            covariate = features.tas_anom_peak.values[eligible_rows]
            finite = np.isfinite(metric) & np.isfinite(covariate) & (metric != 0)
            left = covariate[finite & (metric < 0)]
            right = covariate[finite & (metric > 0)]
            pooled_sd = np.sqrt(
                (
                    (len(left) - 1) * np.var(left, ddof=1)
                    + (len(right) - 1) * np.var(right, ddof=1)
                )
                / (len(left) + len(right) - 2)
            )
            pair_values = []
            for sign, dataset in (("negative", negative), ("positive", positive)):
                matched_ids = getattr(prepared.match, f"{sign}_event_ids")
                assert set(matched_ids) <= set(eligible) & set(ids)
                assert (
                    len(set(matched_ids))
                    == len(matched_ids)
                    == prepared.match.pair_count
                )
                matched_rows = pd.Index(features.event_id.values).get_indexer(
                    matched_ids
                )
                values = features.I_dyn_pre.values[matched_rows]
                assert np.all(values < 0) if sign == "negative" else np.all(values > 0)
                pair_values.append(features.tas_anom_peak.values[matched_rows])
                window = source_windows(
                    source, climate, matched_ids, variables, anomalies=True
                )
                checks[f"matched_{sign}"] = verify_arrays(
                    dataset,
                    {
                        name: np.nanmean(values, axis=0)
                        for name, values in window.items()
                    },
                    output / f"matched_{sign}",
                )
                pair_ids[sign] = matched_ids.tolist()
            assert np.all(
                np.abs(pair_values[0] - pair_values[1]) <= 0.2 * pooled_sd + 1e-12
            )
        return {
            "input_paths": {k: str(v) for k, v in paths.items()},
            "input_sha256": {k: sha256(v) for k, v in paths.items()},
            "stage1_attrs": dict(source.attrs),
            "climatology_attrs": dict(climate.attrs),
            "full_jja_event_ids": ids.tolist(),
            "top_event_ids": top_ids.tolist(),
            "top_peak_dates": [
                str(value) for value in source.peak_time.values[rows[order]]
            ],
            "matched_pairs": prepared.match.pair_count,
            "matched_event_ids": pair_ids,
            "max_absolute_errors": checks,
            "checks_passed": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(artifact_path(str(args.manifest)).read_text())
    output = artifact_path(str(args.output_dir))
    output.mkdir(parents=True, exist_ok=False)
    report = {
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip(),
        "job_id": os.environ.get("PBS_JOBID"),
        "python": sys.executable,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "regions": {},
    }
    for region, inputs in manifest["regions"].items():
        report["regions"][region] = audit_region(region, inputs, output / region)
    report["checks_passed"] = True
    (output / "validation.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n"
    )


if __name__ == "__main__":
    main()
