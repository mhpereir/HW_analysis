"""Independent source-slice and calendar-endpoint checks for heating ranks."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import pandas as pd
import xarray as xr


def validate_against_sources(
    table: xr.Dataset,
    source: xr.Dataset,
    climate: xr.Dataset,
    references: Mapping[int, xr.Dataset],
) -> dict:
    """Check saved results without using WindowReducer or the builder's ranking.

    Climatological endpoints are also checked against means of the source years,
    independently of the stored climatology and its calendar-matching helper.
    """
    time_index = pd.DatetimeIndex(source.time.values)
    tendencies = np.asarray(source.dTdt.values, dtype=float)
    temperature = np.asarray(source.T_mean.values, dtype=float)
    climate_lookup = {
        (time.month, time.day, time.hour): (float(mean), int(count))
        for time, mean, count in zip(
            pd.DatetimeIndex(climate.climatology_time.values),
            climate.T_mean.values,
            climate.T_mean_count.values,
            strict=True,
        )
    }
    used_keys = set()
    expected_raw = np.full(table.I_raw.shape, np.nan)
    expected_corrected = np.full(table.I_raw.shape, np.nan)
    expected_eligible = np.zeros(table.I_raw.shape, dtype=bool)
    expected_years = int(
        climate.attrs["climatology_end_year"]
        - climate.attrs["climatology_start_year"]
        + 1
    )
    for row, peak in enumerate(pd.DatetimeIndex(table.peak_time.values)):
        for column, day in enumerate(table.integration_days.values):
            start = peak - pd.Timedelta(days=int(day))
            grid = pd.date_range(start, peak, freq="h")
            positions = time_index.get_indexer(grid)
            found = positions >= 0
            values = tendencies[positions[found]]
            np.testing.assert_equal(
                table.sample_count.values[row, column], int(found.sum())
            )
            np.testing.assert_equal(
                table.finite_sample_count.values[row, column],
                int(np.isfinite(values).sum()),
            )
            endpoints = []
            for endpoint in (start, peak):
                key = (endpoint.month, endpoint.day, endpoint.hour)
                mean, count = climate_lookup.get(key, (np.nan, 0))
                endpoints.append((mean, count))
                if key in climate_lookup:
                    used_keys.add(key)
            good = (
                found.all()
                and np.isfinite(values).all()
                and all(
                    np.isfinite(mean) and count == expected_years
                    for mean, count in endpoints
                )
            )
            expected_eligible[row, column] = good
            np.testing.assert_allclose(
                table.climatology_T_start.values[row, column],
                endpoints[0][0],
                rtol=0,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                table.climatology_T_peak.values[row],
                endpoints[1][0],
                rtol=0,
                atol=1e-12,
            )
            if good:
                expected_raw[row, column] = math.fsum(values)
                expected_corrected[row, column] = math.fsum(values) - (
                    endpoints[1][0] - endpoints[0][0]
                )
    np.testing.assert_array_equal(table.eligible.values, expected_eligible)
    common = expected_eligible.all(axis=1)
    np.testing.assert_array_equal(table.common_cohort.values, common)
    for kind, expected in (("raw", expected_raw), ("corrected", expected_corrected)):
        np.testing.assert_allclose(
            table[f"I_{kind}"].values, expected, rtol=1e-11, atol=1e-11, err_msg=kind
        )
        for column in range(table.sizes["integration_days"]):
            for cohort in ("common", "available"):
                mask = common if cohort == "common" else expected_eligible[:, column]
                # Independently count competitors, keeping the documented exact
                # ties in the stored NumPy sums rather than fsum rounding.
                stored = table[f"I_{kind}"].values[:, column]
                population = stored[mask]
                ranks = (
                    1 + np.sum(population[None, :] > stored[:, None], axis=1)
                ).astype(float)
                ties = np.sum(population[None, :] == stored[:, None], axis=1)
                ranks[~mask] = np.nan
                ties[~mask] = 0
                np.testing.assert_equal(
                    table[f"rank_{kind}_{cohort}"].values[:, column], ranks
                )
                np.testing.assert_equal(
                    table[f"ties_{kind}_{cohort}"].values[:, column], ties
                )
    source_keys = time_index.month * 10000 + time_index.day * 100 + time_index.hour
    max_climate_error = 0.0
    for month, day, hour in sorted(used_keys):
        mask = source_keys == month * 10000 + day * 100 + hour
        years = time_index.year[mask]
        if np.unique(years).size != len(years):
            raise ValueError(
                "Source has duplicate observations for a year/calendar-hour key."
            )
        values = temperature[mask]
        finite = values[np.isfinite(values)]
        stored_mean, stored_count = climate_lookup[(month, day, hour)]
        np.testing.assert_equal(stored_count, finite.size)
        direct_mean = math.fsum(finite) / finite.size
        np.testing.assert_allclose(stored_mean, direct_mean, rtol=0, atol=1e-10)
        max_climate_error = max(max_climate_error, abs(stored_mean - direct_mean))
    reference_days = []
    for day, reference in references.items():
        if day not in table.integration_days.values:
            continue
        np.testing.assert_array_equal(reference.event_id.values, table.event_id.values)
        np.testing.assert_array_equal(
            reference.peak_time.values, table.peak_time.values
        )
        if reference.attrs["heat_budget_pre_window_hours"] != f"{-day * 24},0":
            raise ValueError("Reference product has the wrong integration window.")
        column = int(np.flatnonzero(table.integration_days.values == day)[0])
        eligible = expected_eligible[:, column]
        np.testing.assert_allclose(
            table.I_raw.values[eligible, column],
            reference.I_dTdt_pre.values[eligible],
            rtol=1e-11,
            atol=1e-11,
        )
        reference_days.append(int(day))
    return {
        "state": "passed",
        "candidate_population_size": table.sizes["event"],
        "common_population_size": int(common.sum()),
        "excluded_event_ids": table.event_id.values[~common].astype(int).tolist(),
        "integration_days": table.integration_days.values.astype(int).tolist(),
        "source_slice_checks": int(np.prod(table.I_raw.shape)),
        "source_climatology_keys_checked": len(used_keys),
        "max_climatology_mean_error_K": max_climate_error,
        "accepted_stage2_reference_days": sorted(reference_days),
        "remaining_acceptance": "PBS terminal status, clean logs, hashes and original-resolution figure inspection",
    }
