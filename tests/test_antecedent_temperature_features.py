"""Strict antecedent-window semantics shared by both Stage-2 builders."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from HW_analysis.tests.stage2_fixtures import climatology_for
from HW_analysis.tests.test_build_stage2_event_features import _make_feature_dataset

from scripts.event_features import build_stage2_baseline_features as baseline
from scripts.event_features import build_stage2_event_features as event
from scripts.event_features import event_feature_config as config
from scripts.event_features import fixed_window_features as fixed


@pytest.fixture
def sources():
    ds = _make_feature_dataset()
    hours = np.arange(ds.sizes["time"], dtype=float)
    ds["T_mean"] = ("time", 280.0 + hours / 100)
    ds["tas_anom"] = ("time", hours / 50)
    ds["hw_event_id"] = ("time", np.zeros(hours.size, dtype=int))
    ds.attrs["event_id_source"] = "hw_event_id"
    climate = climatology_for(ds)
    climate["T_mean"][:] = 280.0
    return ds, climate


@pytest.mark.parametrize("span", [96, 48])
def test_runtime_lags_half_open_counts_parity_and_unchanged_legacy(
    sources, monkeypatch, span
):
    ds, climate = sources
    before, climate_before = ds.copy(deep=True), climate.copy(deep=True)
    monkeypatch.setitem(config.WINDOWS, "heat_budget_pre", (-span, 0))
    out = event.build_event_features(ds, climatology=climate, all_seasons=True)
    other = baseline.build_baseline_features(ds, climatology=climate, season_months=[6])
    anchor = ds.peak_time.values[0]
    initial = anchor - np.timedelta64(span, "h")
    start = initial - np.timedelta64(72, "h")
    use = (ds.time.values >= start) & (ds.time.values < initial)
    assert use.sum() == 72
    at_start = ds.sel(time=initial)
    assert out.tas_anom_antecedent_mean.item() == ds.tas_anom.values[use].mean()
    assert (
        out.T_mean_anom_antecedent_mean.item() == (ds.T_mean.values[use] - 280).mean()
    )
    assert out.tas_anom_at_budget_start.item() == at_start.tas_anom.item()
    assert out.T_mean_anom_at_budget_start.item() == at_start.T_mean.item() - 280
    assert out.tas_anom_at_anchor.item() == ds.tas_anom.sel(time=anchor).item()
    assert out.tas_anom_at_anchor.item() != out.tas_anom_peak.item()
    assert out.n_samples_antecedent_temperature.item() == 72
    assert out.n_samples_heat_budget_pre.item() == span + 1
    assert out.I_dTdt_pre.item() == span + 1
    assert out.I_dyn_pre.item() == 5 * (span + 1)
    legacy = ds.sel(
        time=slice(anchor - np.timedelta64(168, "h"), anchor - np.timedelta64(24, "h"))
    )
    assert out.T_anom_mean_ant.item() == legacy.tas_anom.mean().item()
    assert out.n_samples_antecedent_state.item() == 145
    assert out.attrs["antecedent_temperature_window_hours"] == f"{-span - 72},{-span}"
    assert out.attrs["budget_start_lag_hours"] == -span
    row = other.where(other.reference_time == anchor, drop=True)
    for name in fixed.TEMPERATURE_FEATURES:
        assert row[name].item() == out[name].item()
        assert out[name].attrs["units"] == "K"
    xr.testing.assert_identical(ds, before)
    xr.testing.assert_identical(climate, climate_before)


@pytest.mark.parametrize("location", ["mean", "start", "anchor"])
@pytest.mark.parametrize("kind", ["missing", "nan", "inf"])
def test_missing_and_nonfinite_temperature_samples_are_not_interpolated(
    sources, location, kind
):
    ds, climate = sources
    lag = {"mean": -140, "start": -96, "anchor": 0}[location]
    target = ds.peak_time.values[0] + np.timedelta64(lag, "h")
    if kind == "missing":
        ds = ds.sel(time=ds.time != target)
    else:
        for variable in ("T_mean", "tas_anom"):
            ds[variable].loc[{"time": target}] = np.nan if kind == "nan" else np.inf
    out = event.build_event_features(ds, climatology=climate, all_seasons=True)
    suffix = {
        "mean": "antecedent_mean",
        "start": "at_budget_start",
        "anchor": "at_anchor",
    }[location]
    for source in (
        ("tas_anom", "T_mean_anom") if location != "anchor" else ("tas_anom",)
    ):
        name = f"{source}_{suffix}"
        assert np.isnan(out[name]).all()
        assert out[f"n_finite_{name}"].item() == (71 if location == "mean" else 0)
    count = {
        "mean": "antecedent_temperature",
        "start": "budget_start",
        "anchor": "anchor",
    }[location]
    expected = 72 if location == "mean" else 1
    assert out[f"n_samples_{count}"].item() == expected - (kind == "missing")
    if location == "start":
        assert np.isfinite(out.tas_anom_antecedent_mean).all()


def test_full_count_with_irregular_timestamps_still_rejects_mean(sources):
    ds, _ = sources
    index = 24 * 15
    times = ds.time.values.copy()
    times[index] += np.timedelta64(30, "m")
    ds = ds.assign_coords(time=times)
    result, count, finite = fixed.WindowReducer(ds).strict_temperature_values(
        "tas_anom",
        ds.peak_time.values,
        fixed.ANTECEDENT_WINDOW,
    )
    assert count.item() == finite.item() == 72
    assert np.isnan(result).all()


@pytest.mark.parametrize(
    "attr",
    [
        "region",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "source_stage1_contract_version",
    ],
)
def test_rejects_mismatched_climatology(sources, attr):
    ds, climate = sources
    climate.attrs[attr] = "wrong"
    with pytest.raises(ValueError, match="does not match|contract"):
        event.build_event_features(ds, climatology=climate, all_seasons=True)


def test_rejects_missing_climatology_calendar_key(sources):
    ds, climate = sources
    with pytest.raises(ValueError, match="missing calendar-hour keys"):
        event.build_event_features(
            ds,
            climatology=climate.isel(climatology_time=slice(1, None)),
            all_seasons=True,
        )


@pytest.mark.parametrize("span", [48, 96])
def test_round_trip_values_counts_and_metadata(sources, tmp_path, monkeypatch, span):
    ds, climate = sources
    monkeypatch.setitem(config.WINDOWS, "heat_budget_pre", (-span, 0))
    for builder, writer in (
        (event.build_event_features, event.write_feature_outputs),
        (baseline.build_baseline_features, baseline.write_feature_outputs),
    ):
        out = builder(
            ds, climatology=climate, all_seasons=True, climatology_path="climate.nc"
        )
        path, csv = tmp_path / "features.nc", tmp_path / "features.csv"
        writer(out, path, csv_output_path=csv)
        with xr.open_dataset(path, engine="h5netcdf") as reopened:
            xr.testing.assert_identical(out, reopened)
        frame = pd.read_csv(csv)
        for name in fixed.TEMPERATURE_FEATURES:
            np.testing.assert_allclose(frame[name], out[name])
            assert f"n_finite_{name}" in frame
        assert out.attrs["temperature_climatology_path"] == "climate.nc"


def test_longer_budget_expands_boundary_and_baseline_adjacency_union(
    sources, monkeypatch
):
    ds, climate = sources
    monkeypatch.setitem(config.WINDOWS, "heat_budget_pre", (-192, 0))
    anchor = ds.peak_time.values[0]
    old_event = anchor - np.timedelta64(250, "h")
    day = ds.time.values.astype("datetime64[D]") == old_event.astype("datetime64[D]")
    ds["hw_event_id"].values[day] = 1
    out = baseline.build_baseline_features(ds, climatology=climate, season_months=[6])
    assert out.where(out.reference_time == anchor, drop=True).event_adjacent.item() == 1
    assert out.attrs["event_adjacency_window_hours"] == "-264,0"
    assert out.attrs["antecedent_temperature_window_hours"] == "-264,-192"
    spec = fixed.active_feature_spec(
        ds, use_extended_variables=False, allow_missing_extended=False
    )
    first = ds.time.values[0]
    anchors = first + np.array([200, 264]).astype("timedelta64[h]")
    np.testing.assert_array_equal(
        fixed.complete_anchor_mask(
            ds.time.values, anchors, fixed.active_window_names(spec)
        ),
        [False, True],
    )


def test_june_reference_uses_may_antecedent_samples(sources):
    ds, climate = sources
    ds.hw_event_id.loc[{"time": slice("2000-05-26", "2000-05-26T23")}] = 1
    out = baseline.build_baseline_features(ds, climatology=climate, season_months=[6])
    row = out.where(out.reference_time == np.datetime64("2000-06-01"), drop=True)
    expected = ds.tas_anom.sel(time=slice("2000-05-25", "2000-05-27T23")).mean()
    assert row.tas_anom_antecedent_mean.item() == expected.item()
    assert row.event_adjacent.item() == 1


@pytest.mark.parametrize("duration", [0, -1, 1.5, True])
def test_invalid_antecedent_duration_fails(monkeypatch, duration):
    monkeypatch.setattr(config, "ANTECEDENT_TEMPERATURE_DURATION_HOURS", duration)
    with pytest.raises(ValueError):
        fixed.temperature_window_lags()


def test_explicit_climatology_is_required(sources):
    ds, _ = sources
    with pytest.raises(TypeError, match="climatology"):
        event.build_event_features(ds, all_seasons=True)
    with pytest.raises(TypeError, match="climatology"):
        baseline.build_baseline_features(ds, all_seasons=True)


def test_independent_validator_accepts_products_and_detects_corruption(sources):
    from src.stage2_temperature_validation import validate_temperature_product

    ds, climate = sources
    ds["diabatic"] = ds.dTdt - ds.advection - ds.adiabatic
    ds.diabatic.attrs["units"] = "K hr-1"
    for builder in (event.build_event_features, baseline.build_baseline_features):
        product = builder(ds, climatology=climate, all_seasons=True)
        report = validate_temperature_product(ds, climate, product)
        assert report["complete_temperature_rows"] == report["rows"]
        assert report["max_abs_budget_closure_error_K"] == 0
        product["tas_anom_antecedent_mean"].values[0] += 1
        with pytest.raises(AssertionError, match="Direct-source mismatch"):
            validate_temperature_product(ds, climate, product)
