"""Independent direct-source acceptance checks for Stage-2 temperature products.

This validator deliberately does not call the production window reducer or
anomaly-application helper. Keep its direct finite-sample reductions independent.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def _calendar_codes(time):
    return np.asarray(time.dt.month * 10000 + time.dt.day * 100 + time.dt.hour)


def _direct_sources(source, climate):
    source_keys = _calendar_codes(source.time)
    climate_keys = _calendar_codes(climate.climatology_time)
    lookup = {
        int(key): float(value)
        for key, value in zip(climate_keys, climate.T_mean.values, strict=True)
    }
    atmospheric = np.asarray(source.T_mean.values) - np.array(
        [lookup[int(key)] for key in source_keys]
    )
    tas = (
        source.tas_anom
        if "tas_anom" in source
        else source.tas_region - source.tas_climatology
    )
    return {"tas_anom": np.asarray(tas.values), "T_mean_anom": atmospheric}


def validate_temperature_product(
    source: xr.Dataset, climate: xr.Dataset, product: xr.Dataset
) -> dict:
    """Check every row against direct hourly means, exact points and old sums."""
    event = product.attrs.get("pipeline_stage") == "stage_2_event_features"
    anchor_name, row_dim = (
        ("peak_time", "event") if event else ("reference_time", "baseline_day")
    )
    anchors = np.asarray(product[anchor_name].values, dtype="datetime64[ns]")
    times = np.asarray(source.time.values, dtype="datetime64[ns]")
    sources = _direct_sources(source, climate)
    start, end = map(
        int, product.attrs["antecedent_temperature_window_hours"].split(",")
    )
    budget_start, budget_end = map(
        int, product.attrs["heat_budget_pre_window_hours"].split(",")
    )
    if end != budget_start or budget_end != 0 or end >= 0 or start >= end:
        raise ValueError("Temperature and budget windows are inconsistent.")
    if (
        product.attrs["antecedent_temperature_endpoint_inclusion"]
        != "left_closed_right_open"
    ):
        raise ValueError("Antecedent window must be half-open.")
    expected = {
        name: []
        for name in (
            "tas_anom_antecedent_mean",
            "tas_anom_at_budget_start",
            "T_mean_anom_antecedent_mean",
            "T_mean_anom_at_budget_start",
            "tas_anom_at_anchor",
            "n_samples_antecedent_temperature",
            "n_samples_budget_start",
            "n_samples_anchor",
        )
    }
    for name in tuple(expected):
        if not name.startswith("n_samples_"):
            expected[f"n_finite_{name}"] = []
    integral_sources = {
        "I_dTdt_pre": "dTdt",
        "I_advection_pre": "advection",
        "I_adiabatic_pre": "adiabatic",
        "I_diabatic_pre": "diabatic",
        ("I_lwa_a_pre_peak" if event else "I_lwa_a_pre_reference"): "lwa_a_region",
        ("I_lwa_c_pre_peak" if event else "I_lwa_c_pre_reference"): "lwa_c_region",
    }
    for name in (
        *integral_sources,
        "T_anom_mean_ant",
        "n_samples_heat_budget_pre",
        "n_samples_antecedent_state",
    ):
        expected[name] = []
    for anchor in anchors:
        mean_start, initial = anchor + np.array([start, end]).astype("timedelta64[h]")
        use = (times >= mean_start) & (times < initial)
        required = np.arange(mean_start, initial, np.timedelta64(1, "h"))
        complete = np.array_equal(times[use], required)
        expected["n_samples_antecedent_temperature"].append(int(use.sum()))
        for name, values in sources.items():
            feature = f"{name}_antecedent_mean"
            selected = values[use]
            finite = int(np.isfinite(selected).sum())
            expected[f"n_finite_{feature}"].append(finite)
            expected[feature].append(
                float(np.mean(selected))
                if complete and finite == required.size
                else np.nan
            )
        for suffix, timestamp in (("budget_start", initial), ("anchor", anchor)):
            point = times == timestamp
            expected[f"n_samples_{suffix}"].append(int(point.sum()))
            for name, values in sources.items():
                feature = f"{name}_at_{suffix}"
                if feature not in expected:
                    continue
                valid = point.any() and np.isfinite(values[point]).all()
                expected[feature].append(float(values[point][0]) if valid else np.nan)
                expected[f"n_finite_{feature}"].append(int(valid))
        for name, source_name in integral_sources.items():
            first, last = map(int, product[name].attrs["window_lag_hours"].split(","))
            selected = np.asarray(source[source_name].values)[
                (times >= anchor + np.timedelta64(first, "h"))
                & (times <= anchor + np.timedelta64(last, "h"))
            ]
            finite = selected[np.isfinite(selected)]
            expected[name].append(
                float(np.sum(finite, dtype=np.float64)) if finite.size else np.nan
            )
        legacy = sources["tas_anom"][
            (times >= anchor - np.timedelta64(168, "h"))
            & (times <= anchor - np.timedelta64(24, "h"))
        ]
        expected["T_anom_mean_ant"].append(
            float(np.mean(legacy[np.isfinite(legacy)]))
            if np.isfinite(legacy).any()
            else np.nan
        )
        expected["n_samples_antecedent_state"].append(legacy.size)
        expected["n_samples_heat_budget_pre"].append(
            int(
                (
                    (times >= anchor + np.timedelta64(budget_start, "h"))
                    & (times <= anchor)
                ).sum()
            )
        )
    for name, values in expected.items():
        if product[name].dims != (row_dim,):
            raise ValueError(f"Wrong dimension for {name}")
        np.testing.assert_allclose(
            product[name].values,
            values,
            rtol=1e-12,
            atol=1e-10,
            equal_nan=True,
            err_msg=f"Direct-source mismatch for {name}",
        )
    np.testing.assert_array_equal(
        product.I_dyn_pre, product.I_advection_pre + product.I_adiabatic_pre
    )
    # Stage-1 diabatic is the closure residual. It is not a separate direct Q.
    closure = product.I_dTdt_pre - product.I_dyn_pre - product.I_diabatic_pre
    np.testing.assert_allclose(closure, 0, rtol=0, atol=1e-9)
    names = [
        name for name in expected if name.startswith(("tas_anom_", "T_mean_anom_"))
    ]
    finite = np.logical_and.reduce(
        [np.isfinite(product[name].values) for name in names]
    )
    report = {
        "rows": int(anchors.size),
        "complete_temperature_rows": int(finite.sum()),
        "validated_features": list(expected),
        "antecedent_hours": [start, end],
        "budget_hours": [budget_start, budget_end],
        "max_abs_budget_closure_error_K": float(np.max(np.abs(closure.values))),
    }
    if event:
        difference = np.asarray(product.tas_anom_peak - product.tas_anom_at_anchor)
        valid = np.isfinite(difference)
        # Counts use an explicit reporting tolerance, separate from numerical validation.
        differing = valid & (np.abs(difference) > 1e-8)
        report["peak_vs_anchor_anomaly"] = {
            "finite_pairs": int(valid.sum()),
            "different_over_1e-8_K": int(differing.sum()),
            "max_abs_difference_K": float(np.max(np.abs(difference[valid]))),
            "mean_abs_difference_K": float(np.mean(np.abs(difference[valid]))),
        }
        idx = int(product.tas_anom_peak.argmax(dim=row_dim))
        report["highest_severity_event"] = {
            "event_id": int(product.event_id.values[idx]),
            "peak_time": str(product.peak_time.values[idx]),
            "tas_anom_peak": float(product.tas_anom_peak.values[idx]),
            **{
                name: float(product[name].values[idx])
                for name in (*names, "I_dTdt_pre", "I_dyn_pre", "I_diabatic_pre")
            },
        }
    else:
        report["clean_rows"] = int((product.event_adjacent == 0).sum())
        # Independently check adjacency over the union, including the new interval.
        event_values = np.asarray(source[str(product.attrs["event_id_source"])].values)
        windows = {
            da.attrs["window_name"]: (
                da.attrs["window_lag_hours"],
                da.attrs["window_endpoint_inclusion"],
            )
            for da in product.data_vars.values()
            if "window_name" in da.attrs
        }
        adjacent = []
        for anchor in anchors:
            union = np.zeros(times.size, dtype=bool)
            for lag_string, inclusion in windows.values():
                left, right = (
                    anchor + np.timedelta64(int(lag), "h")
                    for lag in lag_string.split(",")
                )
                union |= (times >= left) & (
                    (times < right)
                    if inclusion == "left_closed_right_open"
                    else (times <= right)
                )
            adjacent.append(bool(np.any(event_values[union] != 0)))
        np.testing.assert_array_equal(product.event_adjacent.values, adjacent)
    return report
