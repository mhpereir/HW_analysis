"""Independent direct anomaly reductions and reference-population checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.stage2_component_anomalies import (
    RAW_TO_ANOMALY,
    TENDENCIES,
    validate_component_anomalies,
)


def validate_anomalies_against_sources(
    source: xr.Dataset,
    climate: xr.Dataset,
    events: xr.Dataset,
    baseline: xr.Dataset,
    reference_events: xr.Dataset,
    reference_baseline: xr.Dataset,
) -> dict:
    """Check every saved anomaly without recalculating any raw integral.

    This path does not use the builder, WindowReducer or the shared calendar-key
    matcher. It directly selects complete source windows and independently maps
    their calendar month/day/hour to the accepted climatology.
    """
    times = pd.DatetimeIndex(source.time.values)
    ct = pd.DatetimeIndex(climate.climatology_time.values)
    climate_keys = pd.Index(ct.month * 10000 + ct.day * 100 + ct.hour)
    source_keys = times.month * 10000 + times.day * 100 + times.hour
    climate_positions = climate_keys.get_indexer(source_keys)
    source_values = {name: source[name].values for name in TENDENCIES}
    climate_values = {name: climate[name].values for name in TENDENCIES}
    report = {}
    for kind, table, reference, anchor_name, dim in (
        ("event", events, reference_events, "peak_time", "event"),
        ("baseline", baseline, reference_baseline, "reference_time", "baseline_day"),
    ):
        validate_component_anomalies(table)
        xr.testing.assert_identical(table[dim], reference[dim])
        for name in reference.data_vars:
            if name not in RAW_TO_ANOMALY:
                xr.testing.assert_identical(table[name], reference[name])
        hours = -int(reference.attrs["heat_budget_pre_window_hours"].split(",")[0])
        maxima = {name: 0.0 for name in TENDENCIES}
        for row, anchor in enumerate(reference[anchor_name].values):
            expected = anchor + np.arange(-hours, 1).astype("timedelta64[h]")
            positions = times.get_indexer(expected)
            if np.any(positions < 0) or np.any(climate_positions[positions] < 0):
                raise ValueError(
                    "Independent validation found missing source/climatology hours."
                )
            for name in TENDENCIES:
                samples = (
                    source_values[name][positions]
                    - climate_values[name][climate_positions[positions]]
                )
                if not np.isfinite(samples).all():
                    raise ValueError(
                        "Independent validation found nonfinite anomaly samples."
                    )
                direct = np.sum(samples)
                actual = table[f"I_{name}_anom_pre"].values[row]
                np.testing.assert_allclose(
                    actual,
                    direct,
                    rtol=1e-12,
                    atol=1e-10,
                    err_msg=f"{kind} {name} direct anomaly integral at {anchor}",
                )
                maxima[name] = max(maxima[name], float(abs(actual - direct)))
        residual = (
            table.I_dTdt_anom_pre - table.I_dyn_anom_pre - table.I_diabatic_anom_pre
        )
        report[kind] = {
            "rows": int(table.sizes[dim]),
            "reference_population_and_copied_features": "identical",
            "maximum_direct_anomaly_difference_K": maxima,
            "maximum_closure_residual_K": float(abs(residual).max()),
        }
    report["baseline"]["clean_rows"] = int((baseline.event_adjacent == 0).sum())
    report["baseline"]["adjacent_rows"] = int((baseline.event_adjacent != 0).sum())
    report["integration_hours"] = hours
    report["expected_hourly_samples"] = hours + 1
    report["raw_integrals_recalculated"] = False
    return report
