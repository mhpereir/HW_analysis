"""Heating-rank sensitivity with an atmospheric climatology endpoint correction."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from scripts.event_features.fixed_window_features import WindowReducer
from src import climatology, selectors

PIPELINE_STAGE = "integration_window_heating_ranks"
CORRECTION_METHOD = "T_mean_climatology_at_peak_minus_at_window_start"
RANK_METHOD = "descending_competition_exact_ties"
HOURLY_UNITS = {"K hr-1", "K h-1", "K hour-1", "K/hr", "K/hour"}


def descending_ranks(
    values: np.ndarray, eligible: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return competition ranks and exact tie counts, missing outside the cohort."""
    ranks = np.full(values.shape, np.nan)
    ties = np.zeros(values.shape, dtype=np.int64)
    indices = np.flatnonzero(eligible)
    selected = np.asarray(values, dtype=float)[indices]
    if not np.isfinite(selected).all():
        raise ValueError("Ranked values must be finite.")
    if indices.size:
        series = pd.Series(selected)
        ranks[indices] = series.rank(ascending=False, method="min").to_numpy()
        ties[indices] = series.map(series.value_counts()).to_numpy(dtype=np.int64)
    return ranks, ties


def _validate_inputs(
    source: xr.Dataset, events: xr.Dataset, climate: xr.Dataset
) -> None:
    climatology.validate_regional_hourly_climatology(
        climate, required_variables=("T_mean",)
    )
    climatology.validate_climatology_source_compatibility(source, climate)
    for name in (
        "region",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "stage1_contract_version",
    ):
        if name not in source.attrs:
            raise ValueError(f"Stage-1 metadata missing {name}.")
    for source_key, climate_key in (
        ("start_year", "climatology_start_year"),
        ("end_year", "climatology_end_year"),
    ):
        if source.attrs.get(source_key) != climate.attrs.get(climate_key):
            raise ValueError("Source and climatology reference years must agree.")
    if climate.attrs.get("climatology_event_exclusion") != "none":
        raise ValueError("An all-observation climatology is required.")
    if climate.attrs.get("climatology_method") != "arithmetic mean over source years":
        raise ValueError("The fixed arithmetic-mean climatology is required.")
    if (
        source.dTdt.dims != ("time",)
        or source.dTdt.attrs.get("units") not in HOURLY_UNITS
    ):
        raise ValueError(
            "Stage-1 dTdt must be hourly, one-dimensional and in K per hour."
        )
    if (
        source.T_mean.attrs.get("units") != "K"
        or climate.T_mean.attrs.get("units") != "K"
    ):
        raise ValueError("Source and climatological T_mean must use K.")
    required_attrs = {
        "pipeline_stage": "stage_2_event_features",
        "heat_budget_pre_window_hours": "-96,0",
        "window_endpoint_inclusion": "inclusive",
        "integral_method": "hourly_sum_assuming_1h_spacing",
        "season_months": "6,7,8",
        "require_full_event": 1,
    }
    for key, expected in required_attrs.items():
        if events.attrs.get(key) != expected:
            raise ValueError(
                f"Accepted 4-day event product requires {key}={expected!r}."
            )
    summary = source[
        [name for name, field in source.data_vars.items() if field.dims == ("event",)]
    ]
    expected = selectors.select_events_by_season(
        summary, [6, 7, 8], require_full_event=True
    )
    for name in ("event", "event_id", "peak_time", "start_time", "end_time"):
        np.testing.assert_array_equal(
            events[name].values,
            expected[name].values,
            err_msg=f"Event population: {name}",
        )
    if not events.sizes.get("event") or not pd.Index(events.event_id.values).is_unique:
        raise ValueError("A nonempty population with unique event IDs is required.")


def _match_temperature(
    climate: xr.Dataset, times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    keys = climatology.calendar_hour_keys(xr.DataArray(times, dims="time"))
    positions = pd.DatetimeIndex(climate.climatology_time.values).get_indexer(
        keys.values
    )
    values = np.full(times.shape, np.nan)
    counts = np.zeros(times.shape, dtype=np.int64)
    found = positions >= 0
    values[found] = climate.T_mean.values[positions[found]]
    counts[found] = climate.T_mean_count.values[positions[found]]
    return values, counts


def build_heating_comparison(
    source: xr.Dataset,
    events: xr.Dataset,
    climate: xr.Dataset,
    *,
    integration_days: Sequence[int] = tuple(range(4, 32)),
    target_peak: str = "2021-06-29T00:00:00",
    target_event_id: int | None = None,
) -> xr.Dataset:
    """Build all-event heating and rank tables without changing any inputs."""
    days = tuple(integration_days)
    if not days or any(
        isinstance(day, (bool, np.bool_))
        or not isinstance(day, (int, np.integer))
        or day < 1
        for day in days
    ):
        raise ValueError("integration_days must contain positive integers.")
    if tuple(sorted(set(days))) != days:
        raise ValueError("integration_days must be unique and increasing.")
    _validate_inputs(source, events, climate)
    windows = {str(day): (-24 * int(day), 0) for day in days}
    reducer = WindowReducer(source, windows=windows)
    times = reducer.time_values
    if np.any(times != times.astype("datetime64[h]").astype("datetime64[ns]")):
        raise ValueError("Source timestamps must fall on exact UTC hours.")
    peaks = events.peak_time.values.astype("datetime64[ns]")
    if np.isnat(peaks).any() or np.any(
        peaks != peaks.astype("datetime64[h]").astype("datetime64[ns]")
    ):
        raise ValueError("Event peaks must be finite exact UTC hours.")
    target = np.flatnonzero(peaks == np.datetime64(target_peak, "ns"))
    if target.size != 1:
        raise ValueError("target_peak must identify exactly one candidate event.")
    target_index = int(target[0])
    actual_id = int(events.event_id.values[target_index])
    if target_event_id is not None and target_event_id != actual_id:
        raise ValueError("Target event ID does not match its peak timestamp.")
    tendencies = np.asarray(source.dTdt.values, dtype=float)
    shape = (peaks.size, len(days))
    raw = np.full(shape, np.nan)
    sample_count = np.zeros(shape, dtype=np.int64)
    finite_count = sample_count.copy()
    reasons = np.full(shape, "", dtype=object)
    starts = peaks[:, None] - np.asarray(days)[None, :] * np.timedelta64(1, "D")
    climate_start = np.full(shape, np.nan)
    climate_start_count = sample_count.copy()
    climate_peak, climate_peak_count = _match_temperature(climate, peaks)
    expected_years = int(
        climate.attrs["climatology_end_year"]
        - climate.attrs["climatology_start_year"]
        + 1
    )
    for column, day in enumerate(days):
        reduced = reducer.sums("dTdt", peaks, str(day))
        climate_start[:, column], climate_start_count[:, column] = _match_temperature(
            climate, starts[:, column]
        )
        for row, peak in enumerate(peaks):
            left = int(np.searchsorted(times, starts[row, column], side="left"))
            right = int(np.searchsorted(times, peak, side="right"))
            segment = times[left:right]
            window_values = tendencies[left:right]
            sample_count[row, column] = segment.size
            finite_count[row, column] = np.isfinite(window_values).sum()
            problems = []
            if (
                segment.size != 24 * day + 1
                or not segment.size
                or segment[0] != starts[row, column]
                or segment[-1] != peak
                or np.any(np.diff(segment) != np.timedelta64(1, "h"))
            ):
                problems.append("incomplete_hourly_window")
            if not np.isfinite(window_values).all():
                problems.append("nonfinite_dTdt")
            if not np.isfinite([climate_start[row, column], climate_peak[row]]).all():
                problems.append("missing_climatology_endpoint")
            if (
                climate_start_count[row, column] != expected_years
                or climate_peak_count[row] != expected_years
            ):
                problems.append("incomplete_climatology_year_count")
            reasons[row, column] = ";".join(problems)
            if not problems:
                raw[row, column] = reduced[row]
    eligible = reasons == ""
    common = eligible.all(axis=1)
    if not common[target_index]:
        raise ValueError(
            "Target event lacks complete coverage across the requested windows."
        )
    correction = climate_peak[:, None] - climate_start
    corrected = raw - correction
    if 4 in days:
        mask = eligible[:, days.index(4)]
        np.testing.assert_allclose(
            raw[mask, days.index(4)],
            events.I_dTdt_pre.values[mask],
            rtol=1e-11,
            atol=1e-11,
            err_msg="Accepted 4-day raw integral",
        )
    out = xr.Dataset(
        coords={
            "event": events.event.values,
            "integration_days": np.asarray(days, dtype=np.int64),
        }
    )
    for name in ("event_id", "peak_time", "start_time", "end_time"):
        out[name] = ("event", events[name].values, dict(events[name].attrs))
    out["window_start"] = (("event", "integration_days"), starts)
    for name, values in {
        "I_raw": raw,
        "climatological_temperature_change": correction,
        "I_corrected": corrected,
        "climatology_T_start": climate_start,
    }.items():
        out[name] = (("event", "integration_days"), values, {"units": "K"})
    out["climatology_T_peak"] = ("event", climate_peak, {"units": "K"})
    out["climatology_T_peak_count"] = ("event", climate_peak_count)
    out["common_cohort"] = ("event", common.astype(np.int8))
    out["available_population_size"] = ("integration_days", eligible.sum(axis=0))
    for name, values in {
        "eligible": eligible.astype(np.int8),
        "exclusion_reason": reasons.astype(str),
        "sample_count": sample_count,
        "finite_sample_count": finite_count,
        "climatology_T_start_count": climate_start_count,
    }.items():
        out[name] = (("event", "integration_days"), values)
    for kind, values in (("raw", raw), ("corrected", corrected)):
        for cohort in ("common", "available"):
            ranks = np.full(shape, np.nan)
            ties = np.zeros(shape, dtype=np.int64)
            for column in range(len(days)):
                mask = common if cohort == "common" else eligible[:, column]
                ranks[:, column], ties[:, column] = descending_ranks(
                    values[:, column], mask
                )
            out[f"rank_{kind}_{cohort}"] = (("event", "integration_days"), ranks)
            out[f"ties_{kind}_{cohort}"] = (("event", "integration_days"), ties)
    out.attrs.update(
        {
            "pipeline_stage": PIPELINE_STAGE,
            "product_contract_version": 1,
            "correction_method": CORRECTION_METHOD,
            "rank_method": RANK_METHOD,
            "primary_cohort": "complete_intersection_across_all_requested_windows",
            "integral_method": "hourly_sum_assuming_1h_spacing",
            "window_endpoint_inclusion": "inclusive",
            "candidate_population_size": peaks.size,
            "common_population_size": int(common.sum()),
            "target_event_id": actual_id,
            "target_peak_time": str(peaks[target_index]),
            "climatology_start_year": int(climate.attrs["climatology_start_year"]),
            "climatology_end_year": int(climate.attrs["climatology_end_year"]),
            "climatology_event_exclusion": "none",
        }
    )
    for name in (
        "region",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "threshold_variable",
        "quantile",
    ):
        out.attrs[name] = source.attrs[name]
    validate_heating_comparison(out)
    return out


def validate_heating_comparison(table: xr.Dataset) -> None:
    """Validate the saved handoff before I/O and plotting."""
    for key, value in (
        ("pipeline_stage", PIPELINE_STAGE),
        ("correction_method", CORRECTION_METHOD),
        ("rank_method", RANK_METHOD),
    ):
        if table.attrs.get(key) != value:
            raise ValueError(f"Invalid heating comparison {key}.")
    days = table.integration_days.values
    if (
        not days.size
        or np.any(days < 1)
        or np.any(np.diff(days) <= 0)
        or not np.issubdtype(days.dtype, np.integer)
    ):
        raise ValueError(
            "Integration days must be positive, unique, increasing integers."
        )
    eligible = table.eligible.values.astype(bool)
    common = table.common_cohort.values.astype(bool)
    np.testing.assert_array_equal(
        table.window_start.values,
        table.peak_time.values[:, None] - days[None, :] * np.timedelta64(1, "D"),
    )
    for name in (
        "I_raw",
        "I_corrected",
        "climatological_temperature_change",
        "climatology_T_peak",
        "climatology_T_start",
    ):
        if table[name].attrs.get("units") != "K":
            raise ValueError(f"{name} must have units K.")
    expected_years = (
        table.attrs["climatology_end_year"] - table.attrs["climatology_start_year"] + 1
    )
    if np.any(
        table.climatology_T_start_count.values[eligible] != expected_years
    ) or np.any(
        table.climatology_T_peak_count.values[eligible.any(axis=1)] != expected_years
    ):
        raise ValueError(
            "Eligible climatology endpoints must contain all reference years."
        )
    np.testing.assert_array_equal(common, eligible.all(axis=1))
    np.testing.assert_array_equal(
        table.available_population_size.values, eligible.sum(axis=0)
    )
    if (
        int(common.sum()) != table.attrs["common_population_size"]
        or table.sizes["event"] != table.attrs["candidate_population_size"]
    ):
        raise ValueError("Cohort size metadata is inconsistent.")
    target = np.flatnonzero(table.event_id.values == table.attrs["target_event_id"])
    if (
        target.size != 1
        or not common[target[0]]
        or table.peak_time.values[target[0]]
        != np.datetime64(table.attrs["target_peak_time"])
    ):
        raise ValueError("Invalid target identity or coverage.")
    np.testing.assert_array_equal(table.exclusion_reason.values == "", eligible)
    expected_counts = np.broadcast_to(24 * days + 1, eligible.shape)
    for name in ("sample_count", "finite_sample_count"):
        np.testing.assert_array_equal(
            table[name].values[eligible], expected_counts[eligible]
        )
    np.testing.assert_allclose(
        table.climatological_temperature_change.values,
        table.climatology_T_peak.values[:, None] - table.climatology_T_start.values,
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        table.I_corrected.values,
        table.I_raw.values - table.climatological_temperature_change.values,
        rtol=0,
        atol=1e-12,
    )
    for kind in ("raw", "corrected"):
        values = table[f"I_{kind}"].values
        if (
            not np.isfinite(values[eligible]).all()
            or not np.isnan(values[~eligible]).all()
        ):
            raise ValueError("Heating values do not match eligibility.")
        for column in range(days.size):
            for cohort in ("common", "available"):
                mask = common if cohort == "common" else eligible[:, column]
                ranks, ties = descending_ranks(values[:, column], mask)
                np.testing.assert_equal(
                    table[f"rank_{kind}_{cohort}"].values[:, column], ranks
                )
                np.testing.assert_equal(
                    table[f"ties_{kind}_{cohort}"].values[:, column], ties
                )
