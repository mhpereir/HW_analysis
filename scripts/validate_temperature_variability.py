"""Independently check saved temperature statistics and native ERA5 cache samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import REGIONS


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def direct_statistics(values: np.ndarray) -> dict:
    """Use SciPy regression/ranks and direct sample moments, without the plotter."""
    x, y = values.T
    result = {"n": len(x)}
    for name, data in (("tas_region", x), ("T_mean", y)):
        constant = len(data) > 0 and np.ptp(data) == 0
        result.update(
            {
                f"{name}_mean_K": float(np.mean(data)) if len(data) else None,
                f"{name}_sd_K": (0.0 if constant else float(np.std(data, ddof=1)))
                if len(data) > 1
                else None,
                f"{name}_variance_K2": (
                    0.0 if constant else float(np.var(data, ddof=1))
                )
                if len(data) > 1
                else None,
                f"{name}_iqr_K": float(stats.iqr(data)) if len(data) else None,
            }
        )
    if len(x) < 2:
        result.update(
            dict.fromkeys(
                (
                    "pearson_r",
                    "spearman_rho",
                    "slope_K_per_K",
                    "intercept_K",
                    "r_squared",
                    "sd_ratio_T_mean_over_tas_region",
                    "variance_ratio_T_mean_over_tas_region",
                )
            )
        )
        return result
    variable_x, variable_y = np.ptp(x) > 0, np.ptp(y) > 0
    fit = stats.linregress(x, y) if variable_x else None
    result.update(
        pearson_r=float(stats.pearsonr(x, y).statistic)
        if variable_x and variable_y
        else None,
        spearman_rho=float(stats.spearmanr(x, y).statistic)
        if variable_x and variable_y
        else None,
        slope_K_per_K=float(fit.slope) if fit else None,
        intercept_K=float(fit.intercept) if fit else None,
        r_squared=float(fit.rvalue**2) if variable_x and variable_y else None,
        sd_ratio_T_mean_over_tas_region=float(np.std(y) / np.std(x))
        if variable_x
        else None,
        variance_ratio_T_mean_over_tas_region=float(np.var(y) / np.var(x))
        if variable_x
        else None,
    )
    return result


def check_values(actual: dict, expected: dict) -> None:
    for name, value in expected.items():
        if value is None or isinstance(value, str):
            assert actual[name] == value, name
        else:
            np.testing.assert_allclose(
                actual[name], value, rtol=1e-10, atol=1e-10, err_msg=name
            )


def validate_statistics(
    stage1: xr.Dataset, hourly: xr.Dataset, report: dict, daily_saved: pd.DataFrame
) -> dict:
    """Reconstruct complete UTC days with NumPy, then check every saved row."""
    selection = report["selection"]
    times = pd.DatetimeIndex(stage1.time.values).as_unit("ns")
    assert times.is_unique and times.is_monotonic_increasing and not times.hasnans
    selected = np.isin(times.month, selection["months_utc"])
    if selection["start_year"] is not None:
        selected &= times.year >= selection["start_year"]
    if selection["end_year"] is not None:
        selected &= times.year <= selection["end_year"]
    times = times[selected]
    positions = pd.DatetimeIndex(hourly.time.values).get_indexer(times)
    assert np.all(positions >= 0), "Hourly companion does not cover selected hours."
    values = np.column_stack(
        (
            stage1.tas_region.values[selected],
            stage1.T_mean.values[selected],
            hourly.tas_region.values[positions],
        )
    ).astype(float)
    dates, inverse, counts = np.unique(
        times.values.astype("datetime64[D]"), return_inverse=True, return_counts=True
    )
    boundaries = np.concatenate(([0], np.cumsum(counts)))
    hours = times.hour.to_numpy()
    complete = np.array(
        [
            stop - start == 24
            and np.array_equal(hours[start:stop], np.arange(24))
            and np.isfinite(values[start:stop]).all()
            for start, stop in pairwise(boundaries)
        ]
    )
    kept = complete[inverse]
    values, times = values[kept], times[kept]
    blocks = values.reshape(-1, 24, 3)
    assert np.all(np.ptp(blocks[:, :, 0], axis=1) <= 1e-6)
    daily = blocks.mean(axis=1)
    np.testing.assert_array_equal(
        pd.DatetimeIndex(daily_saved.index).as_unit("ns").values,
        dates[complete].astype("datetime64[ns]"),
    )
    np.testing.assert_allclose(
        daily_saved[["tas_region", "T_mean", "tas_region_hourly"]],
        daily,
        rtol=0,
        atol=1e-10,
    )
    check_values(
        report["coverage"],
        {
            "stage1_hours": stage1.sizes["time"],
            "selected_hours": int(selected.sum()),
            "selected_days_present": len(dates),
            "incomplete_or_nonfinite_days_dropped": int((~complete).sum()),
            "retained_days": len(daily),
            "retained_hours": len(values),
            "first_hour_utc": times.min().isoformat(),
            "last_hour_utc": times.max().isoformat(),
        },
    )
    arrays = {
        "hourly": (values[:, [2, 1]], times.year),
        "daily": (daily[:, [0, 1]], pd.DatetimeIndex(dates[complete]).year),
        "daily_from_hourly": (
            daily[:, [2, 1]],
            pd.DatetimeIndex(dates[complete]).year,
        ),
    }
    saved = {(r["timescale"], r["period"]): r for r in report["statistics"]}
    assert len(saved) == len(report["statistics"]), "Duplicated statistical rows."
    checked = set()
    for scale, (data, years) in arrays.items():
        for period in ["all", *map(str, np.unique(years))]:
            subset = data if period == "all" else data[years == int(period)]
            key = (scale, period)
            check_values(saved[key], direct_statistics(subset))
            checked.add(key)
    assert set(saved) == checked, "Unexpected statistical rows."
    delta = daily[:, 2] - daily[:, 0]
    check_values(
        report["daily_source_agreement"],
        {
            "n_days": len(daily),
            "mean_difference_K": float(delta.mean()),
            "rmse_K": float(np.sqrt(np.mean(delta**2))),
            "max_absolute_difference_K": float(np.max(np.abs(delta))),
        },
    )
    return {"statistical_rows_checked": len(checked), "daily_pairs_checked": len(daily)}


def validate_cache(hourly: xr.Dataset, manifest: dict) -> dict:
    """Check all hours and annual source identities, with independent raw samples."""
    first, last = manifest["start_year"], manifest["end_year"]
    expected = pd.date_range(
        f"{first}-01-01", f"{last + 1}-01-01", freq="h", inclusive="left"
    )
    np.testing.assert_array_equal(
        hourly.time.values.astype("datetime64[ns]"), expected.as_unit("ns").values
    )
    assert set(hourly.data_vars) == {"tas_region"}
    assert hourly.tas_region.attrs["units"] == "K"
    assert hourly.tas_region.attrs["native_time_resolution"] == "hourly"
    assert hourly.attrs["region"] == manifest["region"]
    assert np.isfinite(hourly.tas_region.values).all()
    sources = manifest["sources"]
    assert [s["year"] for s in sources] == list(range(first, last + 1))
    for source in sources:
        stat = Path(source["path"]).stat()
        assert stat.st_size == source["size_bytes"]
        assert stat.st_mtime_ns == source["mtime_ns"]
    lat, lon = REGIONS[manifest["region"]]
    samples = []
    for source in (
        sources[i] for i in sorted({0, len(sources) // 2, len(sources) - 1})
    ):
        year = source["year"]
        times = pd.to_datetime(
            [
                f"{year}-01-01T00",
                f"{year}-02-28T23",
                f"{year}-06-21T15",
                f"{year}-12-31T23",
            ]
        )
        with xr.open_dataset(source["path"], engine="h5netcdf", cache=False) as raw:
            latitude = raw.latitude.values
            longitude = (raw.longitude.values + 180) % 360 - 180
            ii = np.flatnonzero((latitude >= lat.start) & (latitude <= lat.stop))
            jj = np.flatnonzero((longitude >= lon.start) & (longitude <= lon.stop))
            assert [len(ii), len(jj)] == source["regional_shape"]
            cells = raw.t2m.isel(latitude=ii, longitude=jj).sel(valid_time=times)
            cells = cells.transpose(
                "valid_time", "latitude", "longitude"
            ).values.astype(float)
            assert np.isfinite(cells).all()
            weights = np.cos(np.deg2rad(latitude[ii]))
            direct = np.einsum("tij,i->t", cells, weights) / (weights.sum() * len(jj))
        actual = hourly.tas_region.sel(time=times).values
        np.testing.assert_allclose(actual, direct, rtol=0, atol=1e-10)
        stat = Path(source["path"]).stat()
        assert (
            stat.st_size == source["size_bytes"]
            and stat.st_mtime_ns == source["mtime_ns"]
        )
        samples.append(
            {
                "year": year,
                "times_utc": [str(t) for t in times],
                "raw_means_K": direct.tolist(),
                "cached_means_K": actual.tolist(),
                "maximum_absolute_error_K": float(np.max(np.abs(actual - direct))),
            }
        )
    return {
        "hours": len(expected),
        "source_years": len(sources),
        "raw_samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-manifest", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    report = json.loads((output / "statistics.json").read_text())
    manifest = json.loads(args.cache_manifest.read_text())
    inputs = report["inputs"]
    paths = {name: Path(info["path"]) for name, info in inputs.items()}
    hashes = {
        str(path): digest(path) for path in [*paths.values(), args.cache_manifest]
    }
    for name, path in paths.items():
        assert hashes[str(path)] == inputs[name]["sha256"]
    assert paths["hourly_t2m"].resolve() == Path(manifest["output"]["path"]).resolve()
    assert hashes[str(paths["hourly_t2m"])] == manifest["output"]["sha256"]
    assert report["execution"]["dirty"] is False
    assert report["execution"]["commit"] == os.environ["EXPECTED_COMMIT"]
    for name, expected in report["output_sha256"].items():
        assert digest(output / name) == expected
    daily = pd.read_csv(output / "daily_pairs.csv", index_col=0, parse_dates=True)
    table = pd.read_csv(output / "statistics.csv", dtype={"period": str})
    pd.testing.assert_frame_equal(
        table,
        pd.DataFrame(report["statistics"]),
        check_dtype=False,
        rtol=1e-10,
        atol=1e-10,
    )
    with (
        xr.open_dataset(
            paths["stage1"], engine="h5netcdf", decode_timedelta=True
        ) as stage1,
        xr.open_dataset(paths["hourly_t2m"], engine="h5netcdf") as hourly,
    ):
        assert stage1.attrs["region"] == hourly.attrs["region"]
        cache = validate_cache(hourly, manifest)
        numerical = validate_statistics(stage1, hourly, report, daily)
    for path, expected in hashes.items():
        assert digest(Path(path)) == expected, f"Input changed: {path}"
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "commit": os.environ["EXPECTED_COMMIT"],
        "pbs_job_id": os.environ["PBS_JOBID"],
        "cache": cache,
        "statistics": numerical,
        "input_sha256": hashes,
        "numerical_validation": "passed",
        "remaining_acceptance": "PBS accounting, logs, provenance and figure review",
    }
    with (output / "independent_validation.json").open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
