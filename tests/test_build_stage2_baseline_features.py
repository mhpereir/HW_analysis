import argparse

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import build_stage2_baseline_features as build_baseline


def test_parse_args_requires_explicit_baseline_universe(monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_stage2_baseline_features.py"])

    with pytest.raises(SystemExit) as excinfo:
        build_baseline.parse_args()

    assert excinfo.value.code == 2


def test_parse_args_has_no_require_full_event_option(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["build_stage2_baseline_features.py", "--all-seasons", "--require-full-event"],
    )

    with pytest.raises(SystemExit) as excinfo:
        build_baseline.parse_args()

    assert excinfo.value.code == 2


def test_validate_args_rejects_allow_missing_without_extended(tmp_path):
    args = argparse.Namespace(
        output_path=tmp_path / "baseline.nc",
        csv_output_path=None,
        overwrite=False,
        season_months=[6],
        use_extended_variables=False,
        allow_missing_extended=True,
    )

    with pytest.raises(ValueError, match="--allow-missing-extended"):
        build_baseline.validate_args(args)


def test_build_baseline_uses_selected_source_and_marks_event_adjacency():
    ds = _make_baseline_dataset(event_id_source="lwa_a_event_id")

    out = build_baseline.build_baseline_features(ds, season_months=[6])

    assert out.sizes["baseline_day"] == 19
    reference_days = out["reference_time"].values.astype("datetime64[D]")
    assert np.datetime64("2000-06-05") in reference_days
    assert np.datetime64("2000-06-10") not in reference_days
    assert out.attrs["event_id_source"] == "lwa_a_event_id"
    assert out.attrs["n_event_adjacent_days"] == 7
    assert out.attrs["n_clean_days"] == 12
    np.testing.assert_array_equal(
        out["event_adjacent"]
        .where(
            (out["reference_time"] >= np.datetime64("2000-06-11"))
            & (out["reference_time"] <= np.datetime64("2000-06-17")),
            drop=True,
        )
        .values,
        np.ones(7, dtype=np.int8),
    )


def test_build_baseline_uses_tas_source_when_selected():
    out = build_baseline.build_baseline_features(
        _make_baseline_dataset(event_id_source="hw_event_id"),
        season_months=[6],
    )

    reference_days = out["reference_time"].values.astype("datetime64[D]")
    assert np.datetime64("2000-06-05") not in reference_days
    assert np.datetime64("2000-06-10") in reference_days
    assert out.attrs["event_id_source"] == "hw_event_id"


def test_build_baseline_writes_reference_features_without_event_peak_analogues():
    out = build_baseline.build_baseline_features(
        _make_baseline_dataset(),
        season_months=[6],
    )

    assert out["n_samples_heat_budget_pre"].min().item() == 97
    assert out["n_samples_lwa_pre_reference"].min().item() == 97
    assert out["n_samples_antecedent_state"].min().item() == 145
    assert out["I_dTdt_pre"].min().item() == 97.0
    assert out["I_lwa_a_pre_reference"].min().item() == 485.0
    assert out["I_lwa_c_pre_reference"].min().item() == 582.0
    assert out["I_lwa_a_pre_reference"].attrs["window_name"] == "lwa_pre_reference"
    assert out["n_samples_lwa_pre_reference"].attrs["window_name"] == "lwa_pre_reference"
    assert out["T_anom_mean_ant"].min().item() == 10.0
    assert out.attrs["pipeline_stage"] == "stage_2_baseline_features"
    assert out.attrs["feature_method"] == "fixed_windows_relative_to_reference_time"
    assert out.attrs["dropped_boundary_days"] == 0

    event_only = {
        "event_id",
        "start_time",
        "end_time",
        "duration",
        "peak_time",
        "peak_value",
        "tas_peak",
        "tas_anom_peak",
        "tas_excess_peak",
        "tas_excess_integral",
        "lwa_a_peak",
        "lwa_c_peak",
    }
    assert event_only.isdisjoint(out.data_vars)
    assert not any(name.endswith("_peak") for name in out.data_vars)


def test_build_baseline_applies_season_and_drops_boundary_days():
    out = build_baseline.build_baseline_features(
        _make_baseline_dataset(),
        all_seasons=True,
    )

    assert out.attrs["n_calendar_days"] == 32
    assert out.attrs["n_non_event_days"] == 31
    assert out.attrs["n_selected_before_boundary"] == 31
    assert out.attrs["dropped_boundary_days"] == 7
    assert out.sizes["baseline_day"] == 24
    assert out["reference_time"].values.min() == np.datetime64("2000-05-27T00:00:00")


def test_build_baseline_adds_extended_features():
    out = build_baseline.build_baseline_features(
        _make_baseline_dataset(add_extended=True),
        season_months=[6],
        use_extended_variables=True,
    )

    assert out["I_nslr_pre"].min().item() == 679.0
    assert out["cloud_cover_mean_ant"].min().item() == 0.5
    assert out["pbl_p_mean_ant"].min().item() == 90000.0
    assert "soil_moisture_change" in out


def test_build_baseline_requires_event_id_source_metadata():
    ds = _make_baseline_dataset()
    del ds.attrs["event_id_source"]

    with pytest.raises(ValueError, match="event_id_source metadata"):
        build_baseline.build_baseline_features(ds, season_months=[6])


def test_build_baseline_requires_event_id_source_variable():
    ds = _make_baseline_dataset()
    ds.attrs["event_id_source"] = "missing_event_id"

    with pytest.raises(ValueError, match="not present"):
        build_baseline.build_baseline_features(ds, season_months=[6])


def test_build_baseline_requires_fixed_window_source_variables():
    ds = _make_baseline_dataset().drop_vars("dTdt")

    with pytest.raises(ValueError, match="required time-indexed variables: dTdt"):
        build_baseline.build_baseline_features(ds, season_months=[6])


def test_build_baseline_rejects_missing_event_ids():
    ds = _make_baseline_dataset()
    ds["hw_event_id"][0] = np.nan

    with pytest.raises(ValueError, match="contains missing values"):
        build_baseline.build_baseline_features(ds, season_months=[6])


def test_build_baseline_rejects_inconsistent_within_day_event_ids():
    ds = _make_baseline_dataset()
    idx = np.flatnonzero(ds["time"].values == np.datetime64("2000-06-05T12"))[0]
    ds["hw_event_id"][idx] = 0

    with pytest.raises(ValueError, match="inconsistent within"):
        build_baseline.build_baseline_features(ds, season_months=[6])


def _make_baseline_dataset(
    *,
    event_id_source: str = "hw_event_id",
    add_extended: bool = False,
) -> xr.Dataset:
    time = np.arange(
        np.datetime64("2000-05-20T00", "h"),
        np.datetime64("2000-06-21T00", "h"),
        np.timedelta64(1, "h"),
    )
    days = time.astype("datetime64[D]")
    hw_event_id = np.where(days == np.datetime64("2000-06-05"), 8, 0).astype(float)
    lwa_a_event_id = np.where(days == np.datetime64("2000-06-10"), 4, 0).astype(float)
    ds = xr.Dataset(
        data_vars={
            "dTdt": ("time", np.full(time.size, 1.0)),
            "advection": ("time", np.full(time.size, 2.0)),
            "adiabatic": ("time", np.full(time.size, 3.0)),
            "diabatic": ("time", np.full(time.size, 4.0)),
            "tas_region": ("time", np.full(time.size, 300.0)),
            "tas_climatology": ("time", np.full(time.size, 290.0)),
            "lwa_a_region": ("time", np.full(time.size, 5.0)),
            "lwa_c_region": ("time", np.full(time.size, 6.0)),
            "hw_event_id": ("time", hw_event_id),
            "lwa_a_event_id": ("time", lwa_a_event_id),
        },
        coords={"time": time},
        attrs={"event_id_source": event_id_source},
    )
    for name in ("dTdt", "advection", "adiabatic", "diabatic"):
        ds[name].attrs["units"] = "K hr-1"

    if add_extended:
        ds["soil_moisture"] = ("time", np.linspace(0.0, 1.0, time.size))
        ds["cloud_cover"] = ("time", np.full(time.size, 0.5))
        ds["pbl_p_mean"] = ("time", np.full(time.size, 90000.0))
        ds["nslr_heating_rate_approx"] = ("time", np.full(time.size, 7.0))
        ds["nssr_heating_rate_approx"] = ("time", np.full(time.size, 8.0))
        ds["sshf_heating_rate_approx"] = ("time", np.full(time.size, 9.0))
        ds["slhf_heating_rate_approx"] = ("time", np.full(time.size, 10.0))
        ds["surface_energy_heating_rate_approx"] = ("time", np.full(time.size, 11.0))
        for name in (
            "nslr_heating_rate_approx",
            "nssr_heating_rate_approx",
            "sshf_heating_rate_approx",
            "slhf_heating_rate_approx",
            "surface_energy_heating_rate_approx",
        ):
            ds[name].attrs["units"] = "K hr-1"
    return ds
