"""Independent checks for core JJA Stage-2 window-sensitivity products."""

from __future__ import annotations

import numpy as np
import xarray as xr


def _months(times: np.ndarray) -> np.ndarray:
    return times.astype("datetime64[M]").astype(int) % 12 + 1


def _bounds(times, anchors, start, end):
    starts = anchors + np.timedelta64(start, "h")
    ends = anchors + np.timedelta64(end, "h")
    return np.searchsorted(times, starts), np.searchsorted(times, ends, side="right")


def _check_equal(actual, expected, label):
    np.testing.assert_array_equal(actual, expected, err_msg=label)


def validate_core_pair(
    source: xr.Dataset,
    events: xr.Dataset,
    baseline: xr.Dataset,
    integration_hours: int,
) -> dict:
    """Check populations, coverage, metadata and every direct source reduction.

    This validator deliberately does not use the builders or WindowReducer.
    It covers the core, JJA full-event campaign, not extended-variable products.
    """
    hours = integration_hours
    if (
        isinstance(hours, bool)
        or not isinstance(hours, (int, np.integer))
        or hours <= 0
    ):
        raise ValueError("integration_hours must be a positive integer.")
    times = source.time.values.astype("datetime64[ns]")
    if np.isnat(times).any() or np.any(np.diff(times) <= np.timedelta64(0, "h")):
        raise ValueError("Stage-1 time must be finite and strictly increasing.")
    earliest_lag = -max(hours, 168)
    selected_source = source.attrs["event_id_source"]
    event_flags = np.asarray(source[selected_source].values)
    if not np.isfinite(event_flags).all():
        raise ValueError("Stage-1 selected event IDs contain missing values.")

    peaks = source.peak_time.values.astype("datetime64[ns]")
    starts = source.start_time.values.astype("datetime64[D]")
    ends = source.end_time.values.astype("datetime64[D]")
    if np.isnat(peaks).any() or np.isnat(starts).any() or np.isnat(ends).any():
        raise ValueError("Stage-1 event times contain missing values.")
    seasonal = np.asarray(
        [
            end >= start
            and np.isin(
                _months(np.arange(start, end + np.timedelta64(1, "D"))), [6, 7, 8]
            ).all()
            for start, end in zip(starts, ends, strict=True)
        ]
    )
    covered = (peaks + np.timedelta64(earliest_lag, "h") >= times[0]) & (
        peaks <= times[-1]
    )
    retained = np.flatnonzero(seasonal & covered)
    _check_equal(
        events.event.values, source.event.values[retained], "event coordinates"
    )
    for name, field in source.data_vars.items():
        if name in events and field.dims == ("event",):
            _check_equal(events[name].values, field.values[retained], name)
    _check_equal(
        events.attrs["dropped_boundary_events"],
        int((seasonal & ~covered).sum()),
        "event boundaries",
    )

    _, first, day_counts = np.unique(
        times.astype("datetime64[D]"), return_index=True, return_counts=True
    )
    for index, count in zip(first, day_counts, strict=True):
        if not np.all(event_flags[index : index + count] == event_flags[index]):
            raise ValueError("Selected event IDs change within a reference day.")
    references = times[first]
    selected = (event_flags[first] == 0) & np.isin(_months(references), [6, 7, 8])
    selected_references = references[selected]
    covered_days = (
        selected_references + np.timedelta64(earliest_lag, "h") >= times[0]
    ) & (selected_references <= times[-1])
    expected_references = selected_references[covered_days]
    _check_equal(
        baseline.reference_time.values, expected_references, "baseline population"
    )
    _check_equal(
        baseline.attrs["dropped_boundary_days"],
        int((~covered_days).sum()),
        "baseline boundaries",
    )
    left, right = _bounds(times, expected_references, earliest_lag, 0)
    adjacent = np.asarray(
        [np.any(event_flags[a:b] != 0) for a, b in zip(left, right, strict=True)]
    )
    _check_equal(baseline.event_adjacent.values, adjacent, "baseline adjacency")
    _check_equal(baseline.attrs["n_clean_days"], int((~adjacent).sum()), "clean count")
    _check_equal(
        baseline.attrs["n_event_adjacent_days"], int(adjacent.sum()), "adjacent count"
    )
    _check_equal(
        baseline.attrs["event_adjacency_window_hours"],
        f"{earliest_lag},0",
        "adjacency metadata",
    )
    _check_equal(baseline.attrs["event_id_source"], selected_source, "selected source")

    series = {
        name: np.asarray(source[name].values, dtype=float)
        for name in (
            "dTdt",
            "advection",
            "adiabatic",
            "diabatic",
            "lwa_a_region",
            "lwa_c_region",
        )
    }
    series["tas_anom"] = np.asarray(
        source["tas_anom"].values
        if "tas_anom" in source
        else source.tas_region.values - source.tas_climatology.values,
        dtype=float,
    )
    report = {}
    for kind, table, anchor_name, dimension, lwa_window in (
        ("event", events, "peak_time", "event", "lwa_pre_peak"),
        ("baseline", baseline, "reference_time", "baseline_day", "lwa_pre_reference"),
    ):
        _check_equal(
            table.attrs["pipeline_stage"], f"stage_2_{kind}_features", "product marker"
        )
        _check_equal(table.attrs["season_months"], "6,7,8", "season metadata")
        _check_equal(table.attrs["extended_variables_used"], 0, "core features")
        anchors = table[anchor_name].values.astype("datetime64[ns]")
        if anchors.size == 0:
            raise ValueError(f"Empty {kind} population.")
        windows = {
            "heat_budget_pre": (-hours, 0),
            lwa_window: (-hours, 0),
            "antecedent_state": (-168, -24),
        }
        for window, (start, end) in windows.items():
            lags = f"{start},{end}"
            _check_equal(table.attrs[f"{window}_window_hours"], lags, window)
            count_name = f"n_samples_{window}"
            _check_equal(
                table[count_name].values,
                np.full(anchors.size, end - start + 1),
                count_name,
            )
            _check_equal(
                table[count_name].attrs["window_lag_hours"], lags, "count metadata"
            )
            left, right = _bounds(times, anchors, start, end)
            for anchor, a, b in zip(anchors, left, right, strict=True):
                expected_times = anchor + np.arange(start, end + 1).astype(
                    "timedelta64[h]"
                )
                _check_equal(
                    times[a:b],
                    expected_times,
                    f"{kind} {window} complete hourly coverage",
                )

        features = {
            "I_dTdt_pre": ("dTdt", "heat_budget_pre"),
            "I_advection_pre": ("advection", "heat_budget_pre"),
            "I_adiabatic_pre": ("adiabatic", "heat_budget_pre"),
            "I_diabatic_pre": ("diabatic", "heat_budget_pre"),
            f"I_lwa_a_{lwa_window.removeprefix('lwa_')}": ("lwa_a_region", lwa_window),
            f"I_lwa_c_{lwa_window.removeprefix('lwa_')}": ("lwa_c_region", lwa_window),
            "T_anom_mean_ant": ("tas_anom", "antecedent_state"),
        }
        for name, (variable, window) in features.items():
            start, end = windows[window]
            left, right = _bounds(times, anchors, start, end)
            direct = []
            for a, b in zip(left, right, strict=True):
                values = series[variable][a:b]
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite required samples: {kind} {variable}.")
                direct.append(
                    float(np.mean(values) if variable == "tas_anom" else np.sum(values))
                )
            _check_equal(table[name].values, direct, f"{kind} {name} direct reductions")
            _check_equal(table[name].dims, (dimension,), "feature dimensions")
            _check_equal(
                table[name].attrs["window_lag_hours"], f"{start},{end}", "feature lags"
            )
            _check_equal(
                table[name].attrs["window_endpoint_inclusion"], "inclusive", "endpoints"
            )
            if variable in ("dTdt", "advection", "adiabatic", "diabatic"):
                _check_equal(table[name].attrs["units"], "K", "budget units")
        _check_equal(
            table.I_dyn_pre.values,
            table.I_adiabatic_pre.values + table.I_advection_pre.values,
            "dynamical identity",
        )
        _check_equal(
            table.I_dyn_pre.attrs["window_lag_hours"],
            f"{-hours},0",
            "dynamical metadata",
        )
        residual = (
            table.I_dTdt_pre.values
            - table.I_dyn_pre.values
            - table.I_diabatic_pre.values
        )
        np.testing.assert_allclose(
            residual, 0.0, atol=1e-8, rtol=0, err_msg="heat-budget closure"
        )
        report[kind] = {
            "rows": int(anchors.size),
            "maximum_closure_residual_K": float(np.abs(residual).max()),
        }

    report["baseline"]["clean_rows"] = int((~adjacent).sum())
    report["baseline"]["adjacent_rows"] = int(adjacent.sum())
    report["integration_hours"] = int(hours)
    report["expected_hourly_samples"] = int(hours + 1)
    # Identify the June 2021 event by its date, not a region-specific numeric ID.
    peaks = events.peak_time.values.astype("datetime64[D]")
    june_2021 = np.flatnonzero(
        (peaks >= np.datetime64("2021-06-20")) & (peaks <= np.datetime64("2021-07-02"))
    )
    report["june_2021_events"] = [
        {
            "event_id": int(events.event_id.values[i]),
            "peak_time": str(events.peak_time.values[i]),
            "integrated_heating_K": float(events.I_dTdt_pre.values[i]),
            "descending_heating_rank": int(
                1 + (events.I_dTdt_pre.values > events.I_dTdt_pre.values[i]).sum()
            ),
            "event_population": int(events.sizes["event"]),
        }
        for i in june_2021
    ]
    return report
