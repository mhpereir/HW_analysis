"""Scientific regressions for independent per-run integration windows."""

import numpy as np
import pytest
import xarray as xr

from scripts.event_features import event_feature_config as config
from scripts.event_features.build_stage2_baseline_features import (
    build_baseline_features,
)
from scripts.event_features.build_stage2_event_features import build_event_features
from src.stage2_validation import validate_core_pair


@pytest.fixture
def source():
    times = np.arange("2021-05-01", "2021-09-02", dtype="datetime64[h]")
    peaks = np.asarray(
        ["2021-06-01", "2021-06-29", "2021-07-21"], dtype="datetime64[ns]"
    )
    flags = np.zeros(times.size, dtype=int)
    for event_id, peak in enumerate(peaks, 1):
        flags[(times >= peak) & (times < peak + np.timedelta64(3, "D"))] = event_id
    signal = np.sin(np.arange(times.size) / 20) / 24
    ds = xr.Dataset(
        {
            "dTdt": ("time", signal),
            "advection": ("time", signal * 2),
            "adiabatic": ("time", signal * 3),
            "diabatic": ("time", signal * -4),
            "lwa_a_region": ("time", 1e6 + np.arange(times.size) * 0.01),
            "lwa_c_region": ("time", 1e5 + np.arange(times.size) * 0.03),
            "tas_region": ("time", 290 + signal),
            "tas_climatology": ("time", np.full(times.size, 285.0)),
            "hw_event_id": ("time", flags),
            "event_id": ("event", [1, 2, 3]),
            "start_time": ("event", peaks),
            "end_time": ("event", peaks + np.timedelta64(2, "D")),
            "peak_time": ("event", peaks),
            "duration": ("event", [3, 3, 3]),
        },
        coords={"time": times, "event": [0, 1, 2]},
        attrs={"event_id_source": "hw_event_id"},
    )
    for name in config.EVENT_SUMMARY_FEATURES:
        if name not in ds:
            ds[name] = ("event", [1.0, 2.0, 3.0])
    for name in ("dTdt", "advection", "adiabatic", "diabatic"):
        ds[name].attrs["units"] = "K hr-1"
    return ds


def pair(source, hours):
    common = {"season_months": [6, 7, 8], "integration_hours": hours}
    return (
        build_event_features(source, require_full_event=True, **common),
        build_baseline_features(source, **common),
    )


@pytest.mark.parametrize("hours", [48, 96, 168, 336, 504])
def test_independent_reductions_populations_and_roundtrip(source, hours, tmp_path):
    events, baseline = pair(source, hours)
    path = tmp_path / "events.nc"
    events.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as saved:
        report = validate_core_pair(source, saved, baseline, hours)
    assert report["event"]["rows"] == 3
    assert report["expected_hourly_samples"] == hours + 1
    assert report["baseline"]["clean_rows"] > 0
    assert report["june_2021_events"][0]["event_id"] == 2
    np.testing.assert_array_equal(events.n_samples_antecedent_state, [145] * 3)


def test_override_does_not_leak_and_earlier_events_change_adjacency(source):
    original = source.copy(deep=True)
    short_events, short_base = pair(source, 96)
    long_events, long_base = pair(source, 504)
    repeated_events, repeated_base = pair(source, None)
    xr.testing.assert_identical(short_events, repeated_events)
    xr.testing.assert_identical(short_base, repeated_base)
    xr.testing.assert_identical(source, original)
    assert config.WINDOWS["heat_budget_pre"] == (-96, 0)
    assert config.WINDOWS["lwa_pre_peak"] == (-96, 0)
    np.testing.assert_array_equal(short_base.reference_time, long_base.reference_time)
    np.testing.assert_array_equal(
        short_events.T_anom_mean_ant, long_events.T_anom_mean_ant
    )
    assert long_base.attrs["n_clean_days"] < short_base.attrs["n_clean_days"]
    assert (
        long_base.event_adjacent.where(
            long_base.reference_time == np.datetime64("2021-07-15"), drop=True
        ).item()
        == 1
    )
    assert (
        short_base.event_adjacent.where(
            short_base.reference_time == np.datetime64("2021-07-15"), drop=True
        ).item()
        == 0
    )


def test_resolved_windows_control_boundary_selection(source):
    clipped = source.sel(time=slice("2021-05-20", None))
    short_events, _ = pair(clipped, 96)
    long_events, long_base = pair(clipped, 504)
    np.testing.assert_array_equal(short_events.event_id, [1, 2, 3])
    np.testing.assert_array_equal(long_events.event_id, [2, 3])
    assert long_events.attrs["dropped_boundary_events"] == 1
    assert long_base.attrs["dropped_boundary_days"] > 0
    validate_core_pair(clipped, long_events, long_base, 504)


@pytest.mark.parametrize("hours", [0, -1, 1.5, True, "168"])
def test_invalid_integration_spans_fail(source, hours):
    with pytest.raises(ValueError, match="positive integer"):
        pair(source, hours)


def test_validator_rejects_hidden_hourly_gaps(source):
    bad = source.drop_sel(time=np.datetime64("2021-06-20T12", "ns"))
    events, baseline = pair(bad, 504)
    with pytest.raises(AssertionError):
        validate_core_pair(bad, events, baseline, 504)


def test_validator_rejects_finite_sum_over_nonfinite_source(source):
    source.dTdt.loc[{"time": np.datetime64("2021-06-20T12", "ns")}] = np.nan
    events, baseline = pair(source, 504)
    with pytest.raises(ValueError, match="Nonfinite required samples"):
        validate_core_pair(source, events, baseline, 504)


@pytest.mark.parametrize(
    "corruption", ["value", "lag", "adjacency", "source_population"]
)
def test_validator_detects_product_corruption(source, corruption):
    events, baseline = pair(source, 504)
    if corruption == "value":
        events.I_lwa_a_pre_peak.values[0] += 1
    elif corruption == "lag":
        events.I_dyn_pre.attrs["window_lag_hours"] = "-96,0"
    elif corruption == "adjacency":
        baseline.event_adjacent.values[0] = 1 - baseline.event_adjacent.values[0]
    else:
        events.event_id.values[0] = 999
    with pytest.raises(AssertionError):
        validate_core_pair(source, events, baseline, 504)
