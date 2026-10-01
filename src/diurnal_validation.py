"""Independent array checks for the saved GMT diurnal diagnostic."""

from __future__ import annotations

import hashlib

import numpy as np
import xarray as xr

from .diurnal import DIURNAL_VARIABLES, HW_CLASS_LABELS


def validate_gmt_diurnal(source: xr.Dataset, saved: xr.Dataset) -> dict:
    """Check every hourly statistic against direct NumPy source reductions."""
    np.testing.assert_array_equal(saved.local_hour, np.arange(24))
    np.testing.assert_array_equal(saved.hw_class, HW_CLASS_LABELS)
    np.testing.assert_array_equal(saved["quantile"], [0.25, 0.5, 0.75])
    if int(saved.attrs["local_utc_offset_hours"]) != 0:
        raise ValueError("Production validation requires GMT offset zero.")
    if saved.attrs["season_months"] != "6 7 8":
        raise ValueError("Production validation requires June-August.")
    metadata = (
        "region",
        "threshold_variable",
        "quantile",
        "start_year",
        "end_year",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
    )
    for key in metadata:
        if saved.attrs.get(key) != source.attrs.get(key):
            raise ValueError(f"Source metadata mismatch: {key}")
    times = source.time.values.astype("datetime64[ns]")
    dates = times.astype("datetime64[D]")
    months = times.astype("datetime64[M]").astype(np.int64) % 12 + 1
    years = times.astype("datetime64[Y]").astype(np.int64) + 1970
    hours = (times - dates).astype("timedelta64[h]").astype(np.int64)
    season = (months >= 6) & (months <= 8)
    event_ids = np.nan_to_num(source.hw_event_id.values, nan=0).astype(np.int64)
    groups = (season & (event_ids > 0), season & (event_ids == 0))
    report = {
        "time_sha256": hashlib.sha256(times.astype("<i8").tobytes()).hexdigest(),
        "hw_event_id_sha256": hashlib.sha256(
            event_ids.astype("<i8").tobytes()
        ).hexdigest(),
        "source_time_samples": len(times),
        "jja_year_hour_counts": {
            str(y): int(np.count_nonzero(season & (years == y)))
            for y in np.unique(years)
        },
        "class_hour_counts": {},
        "maximum_absolute_errors": {},
        "metadata": {k: source.attrs.get(k) for k in metadata},
    }
    for index, (label, group) in enumerate(zip(HW_CLASS_LABELS, groups, strict=True)):
        count = int(np.count_nonzero(group))
        day_count = int(np.unique(dates[group]).size)
        np.testing.assert_equal(saved.class_sample_count.values[index], count)
        np.testing.assert_equal(saved.class_day_count.values[index], day_count)
        prefix = "hw" if index == 0 else "non_hw"
        np.testing.assert_equal(saved.attrs[f"n_{prefix}_samples"], count)
        np.testing.assert_equal(saved.attrs[f"n_{prefix}_days"], day_count)
        masks = [group & (hours == hour) for hour in range(24)]
        counts = np.array([np.count_nonzero(m) for m in masks])
        np.testing.assert_array_equal(
            saved.class_hour_sample_count.values[index], counts
        )
        report["class_hour_counts"][label] = counts.tolist()
        for name in DIURNAL_VARIABLES:
            values = np.asarray(source[name].values, dtype=float)
            expected_mean = np.full(24, np.nan)
            expected_quantiles = np.full((3, 24), np.nan)
            nonmissing = np.zeros(24, dtype=int)
            for hour, mask in enumerate(masks):
                sample = values[mask]
                sample = sample[~np.isnan(sample)]
                if not np.isfinite(sample).all():
                    raise ValueError(f"Nonfinite production values: {name}")
                nonmissing[hour] = sample.size
                if sample.size:
                    expected_mean[hour] = np.mean(sample)
                    expected_quantiles[:, hour] = np.quantile(sample, [0.25, 0.5, 0.75])
            mean = saved[name].values[index]
            quartiles = saved[f"sample_percentile_{name}"].values[index]
            np.testing.assert_allclose(
                mean,
                expected_mean,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
                err_msg=name,
            )
            np.testing.assert_allclose(
                quartiles,
                expected_quantiles,
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
                err_msg=name,
            )
            np.testing.assert_array_equal(
                saved[f"sample_count_{name}"].values[index], nonmissing
            )
            if (
                "units" in source[name].attrs
                and saved[name].attrs.get("units") != source[name].attrs["units"]
            ):
                raise ValueError(f"Units mismatch: {name}")
            report["maximum_absolute_errors"][f"{label}/{name}"] = {
                "mean": float(np.nanmax(np.abs(mean - expected_mean))),
                "quartiles": float(np.nanmax(np.abs(quartiles - expected_quantiles))),
            }
    residual = saved.dTdt - saved.advection - saved.adiabatic - saved.diabatic
    if not np.isfinite(residual).all():
        raise ValueError("Composite heat-budget closure is nonfinite.")
    np.testing.assert_allclose(residual, 0, rtol=0, atol=1e-10)
    report["maximum_closure_error_K_per_hour"] = float(np.abs(residual).max())
    report["status"] = "numerical_checks_passed"
    return report
