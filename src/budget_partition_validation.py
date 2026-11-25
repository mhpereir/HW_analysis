"""Independent numerical checks for an additive boundary-flux campaign."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


def validate_products(source, extended, climate, events, baseline, references):
    """Check source preservation, sample statistics and every absolute integral.

    Deliberately avoid the regrouping, climatology and window-reducer functions.
    Accepted references establish unchanged event and baseline populations.
    """
    for name in source.variables:
        xr.testing.assert_identical(extended[name], source[name])
    lateral = sum(
        source[f"advection_{face}"].values
        for face in ("west", "east", "south", "north")
    )
    vertical = source.advection_top.values.copy()
    if "advection_bottom" in source:
        vertical += source.advection_bottom.values
    expected = {
        "adv_horizontal": lateral,
        "vertical_contribution": source.adiabatic.values + vertical,
    }
    maxima = {}
    for name, values in expected.items():
        np.testing.assert_allclose(extended[name], values, rtol=1e-10, atol=1e-12)
        maxima[name] = float(np.max(np.abs(extended[name].values - values)))
    np.testing.assert_allclose(
        extended.adv_horizontal + extended.vertical_contribution,
        source.advection + source.adiabatic,
        rtol=1e-10,
        atol=1e-12,
    )
    times = pd.DatetimeIndex(source.time.values)
    keys = times.month * 10000 + times.day * 100 + times.hour
    unique, counts = np.unique(keys, return_counts=True)
    years = int(source.attrs["end_year"]) - int(source.attrs["start_year"]) + 1
    if not np.all(counts == years):
        raise ValueError(
            "Pilot requires one observation per source year and calendar hour."
        )
    ct = pd.DatetimeIndex(climate.climatology_time.values)
    np.testing.assert_array_equal(ct.month * 10000 + ct.day * 100 + ct.hour, unique)
    order = np.argsort(keys, kind="stable")
    statistics = {}
    for name, values in expected.items():
        samples = values[order].reshape(len(unique), years)
        if not np.isfinite(samples).all():
            raise ValueError(f"Nonfinite {name} samples.")
        for suffix, direct in (
            ("", samples.mean(axis=1)),
            ("_std", samples.std(axis=1, ddof=1)),
        ):
            actual = climate[f"{name}{suffix}"].values
            np.testing.assert_allclose(actual, direct, rtol=1e-10, atol=1e-12)
            statistics[f"{name}{suffix}"] = float(np.max(np.abs(actual - direct)))
        np.testing.assert_array_equal(climate[f"{name}_count"].values, counts)
    report = {
        "original_stage1_variables": "identical",
        "hourly_regrouping_maximum_difference": maxima,
        "climatology_maximum_difference": statistics,
        "calendar_hours": len(unique),
        "source_years": years,
    }
    for kind, table, anchor, dim in (
        ("event", events, "peak_time", "event"),
        ("baseline", baseline, "reference_time", "baseline_day"),
    ):
        reference = references[kind]
        for name in reference.variables:
            actual, previous = table[name], reference[name]
            if actual.dims != previous.dims:
                raise ValueError(f"Changed reference dimensions for {name}.")
            if np.issubdtype(previous.dtype, np.floating):
                np.testing.assert_allclose(
                    actual,
                    previous,
                    rtol=1e-12,
                    atol=1e-10,
                    equal_nan=True,
                    err_msg=name,
                )
            else:
                np.testing.assert_array_equal(actual, previous, err_msg=name)
        target_times = table[anchor].values[:, None] + np.arange(-96, 1).astype(
            "timedelta64[h]"
        )
        indices = np.searchsorted(source.time.values, target_times)
        np.testing.assert_array_equal(source.time.values[indices], target_times)
        differences = {}
        for name in ("dTdt", "advection", "adiabatic", "diabatic", *expected):
            direct = extended[name].values[indices].sum(axis=1)
            actual = table[f"I_{name}_pre"].values
            np.testing.assert_allclose(actual, direct, rtol=1e-12, atol=1e-10)
            differences[name] = float(np.max(np.abs(actual - direct)))
        np.testing.assert_array_equal(table.n_samples_heat_budget_pre.values, 97)
        report[kind] = {
            "rows": int(table.sizes[dim]),
            "retained_reference_fields": "unchanged",
            "direct_integral_maximum_difference_K": differences,
        }
    report["baseline"]["clean_rows"] = int((baseline.event_adjacent == 0).sum())
    return report
