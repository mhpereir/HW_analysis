import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import build_stage2_event_features as build_event_features
from HW_analysis.scripts.event_features import event_feature_config as feature_config


def test_config_uses_expected_default_windows():
    assert feature_config.WINDOWS["heat_budget_pre"] == (-96, 0)
    assert feature_config.WINDOWS["lwa_pre_peak"] == (-96, 0)
    assert feature_config.WINDOWS["antecedent_state"] == (-168, -24)


def test_parse_args_requires_explicit_event_universe(monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_stage2_event_features.py"])

    with pytest.raises(SystemExit) as excinfo:
        build_event_features.parse_args()

    assert excinfo.value.code == 2


def test_parse_args_accepts_all_seasons(monkeypatch, tmp_path):
    input_path = tmp_path / "stage1.nc"
    output_path = tmp_path / "features.nc"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage2_event_features.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--all-seasons",
        ],
    )

    args = build_event_features.parse_args()

    assert args.input_path == input_path
    assert args.output_path == output_path
    assert args.all_seasons
    assert args.season_months is None
    assert args.csv_output_path is None


def test_validate_args_rejects_require_full_event_with_all_seasons(tmp_path):
    args = argparse.Namespace(
        output_path=tmp_path / "features.nc",
        csv_output_path=None,
        overwrite=False,
        season_months=None,
        require_full_event=True,
        use_extended_variables=False,
        allow_missing_extended=False,
    )

    with pytest.raises(ValueError, match="--require-full-event"):
        build_event_features.validate_args(args)


def test_validate_args_rejects_existing_output_without_overwrite(tmp_path):
    output_path = tmp_path / "features.nc"
    output_path.write_text("exists")
    args = argparse.Namespace(
        output_path=output_path,
        csv_output_path=None,
        overwrite=False,
        season_months=[6, 7, 8],
        require_full_event=False,
        use_extended_variables=False,
        allow_missing_extended=False,
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_event_features.validate_args(args)


def test_build_default_features_uses_inclusive_windows_and_derived_tas_anom():
    ds = _make_feature_dataset()

    out = build_event_features.build_event_features(ds, all_seasons=True)

    assert out.sizes["event"] == 1
    np.testing.assert_array_equal(out["event_id"].values, [1])
    assert out["n_samples_heat_budget_pre"].item() == 97
    assert out["n_samples_lwa_pre_peak"].item() == 97
    assert out["n_samples_antecedent_state"].item() == 145
    assert out["I_dTdt_pre"].item() == 97.0
    assert out["I_advection_pre"].item() == 194.0
    assert out["I_adiabatic_pre"].item() == 291.0
    assert out["I_diabatic_pre"].item() == 388.0
    assert out["I_lwa_a_pre_peak"].item() == 485.0
    assert out["I_lwa_c_pre_peak"].item() == 582.0
    assert out["T_anom_mean_ant"].item() == 10.0
    assert out["days_from_solstice"].item() == -11.0
    assert out.attrs["all_seasons"] == 1
    assert out.attrs["dropped_boundary_events"] == 0
    assert out["I_dTdt_pre"].attrs["window_endpoint_inclusion"] == "inclusive"


def test_build_features_drops_boundary_events_when_required_window_is_outside_data():
    ds = _make_feature_dataset(include_boundary_event=True)

    out = build_event_features.build_event_features(ds, all_seasons=True)

    np.testing.assert_array_equal(out["event_id"].values, [1])
    assert out.attrs["dropped_boundary_events"] == 1


def test_build_features_applies_season_selection_and_full_event_requirement():
    ds = _make_feature_dataset(include_cross_month_event=True)

    out = build_event_features.build_event_features(
        ds,
        season_months=[6],
        require_full_event=True,
    )

    np.testing.assert_array_equal(out["event_id"].values, [1])
    assert out.attrs["season_months"] == "6"
    assert out.attrs["require_full_event"] == 1


def test_build_extended_features_adds_optional_diagnostics():
    ds = _make_feature_dataset(add_extended=True)

    out = build_event_features.build_event_features(
        ds,
        all_seasons=True,
        use_extended_variables=True,
    )

    assert out["I_nslr_pre"].item() == 679.0
    assert out["I_nssr_pre"].item() == 776.0
    assert out["I_sshf_pre"].item() == 873.0
    assert out["I_slhf_pre"].item() == 970.0
    assert out["I_surface_energy_pre"].item() == 1067.0
    assert out["soil_moisture_mean_ant"].item() == pytest.approx(154.0 / 145.0)
    assert out["cloud_cover_mean_ant"].item() == 0.5
    assert out["pbl_p_mean_ant"].item() == 90000.0
    assert out["soil_moisture_change"].item() == 9.0
    assert out["I_sshf_pre"].attrs["sign_convention"] == "native Stage-1/source signs retained"


def test_build_extended_features_raises_for_missing_extended_variables():
    ds = _make_feature_dataset()

    with pytest.raises(ValueError, match="missing required extended variables"):
        build_event_features.build_event_features(
            ds,
            all_seasons=True,
            use_extended_variables=True,
        )


def test_build_extended_features_can_skip_missing_extended_variables():
    ds = _make_feature_dataset()

    with pytest.warns(RuntimeWarning, match="Skipping missing extended variables"):
        out = build_event_features.build_event_features(
            ds,
            all_seasons=True,
            use_extended_variables=True,
            allow_missing_extended=True,
        )

    assert "I_nslr_pre" not in out
    assert "I_dTdt_pre" in out


def test_write_feature_outputs_writes_netcdf_and_optional_csv(tmp_path):
    features = xr.Dataset(
        data_vars={"event_id": ("event", np.array([1])), "I_dTdt_pre": ("event", np.array([1.0]))},
        coords={"event": np.array([0])},
    )
    output_path = tmp_path / "features.nc"
    csv_path = tmp_path / "features.csv"

    written = build_event_features.write_feature_outputs(
        features,
        output_path,
        csv_output_path=csv_path,
    )

    assert written == [output_path.resolve(), csv_path.resolve()]
    assert output_path.exists()
    assert csv_path.exists()


def test_main_orchestrates_open_build_and_write(monkeypatch, tmp_path):
    opened = _make_feature_dataset()
    built = xr.Dataset(coords={"event": np.array([0])})
    captured = {}

    def fake_open(path: str | Path):
        captured["input_path"] = path
        return opened

    def fake_build(ds, **kwargs):
        captured["build_ds"] = ds
        captured["build_kwargs"] = kwargs
        return built

    def fake_write(features, output_path, *, csv_output_path=None):
        captured["features"] = features
        captured["output_path"] = output_path
        captured["csv_output_path"] = csv_output_path
        return [Path(output_path)]

    monkeypatch.setattr("sys.argv", ["build_stage2_event_features.py", "--all-seasons"])
    monkeypatch.setattr(build_event_features.analysis_io, "open_harmonized_timeseries", fake_open)
    monkeypatch.setattr(build_event_features, "build_event_features", fake_build)
    monkeypatch.setattr(build_event_features, "write_feature_outputs", fake_write)

    result = build_event_features.main()

    assert result == 0
    assert captured["input_path"] == feature_config.DEFAULT_INPUT_PATH
    assert captured["build_ds"] is opened
    assert captured["build_kwargs"] == {
        "use_extended_variables": False,
        "allow_missing_extended": False,
        "season_months": None,
        "all_seasons": True,
        "require_full_event": False,
        "input_path": feature_config.DEFAULT_INPUT_PATH,
    }
    assert captured["features"] is built
    assert captured["output_path"] == feature_config.DEFAULT_OUTPUT_PATH
    assert captured["csv_output_path"] is None


def _make_feature_dataset(
    *,
    add_extended: bool = False,
    include_boundary_event: bool = False,
    include_cross_month_event: bool = False,
) -> xr.Dataset:
    time = np.arange(
        np.datetime64("2000-05-20T00", "h"),
        np.datetime64("2000-06-20T01", "h"),
        np.timedelta64(1, "h"),
    )
    event = [0]
    event_ids = [1]
    start_times = [np.datetime64("2000-06-09T00", "ns")]
    end_times = [np.datetime64("2000-06-11T00", "ns")]
    peak_times = [np.datetime64("2000-06-10T00", "ns")]

    if include_boundary_event:
        event.append(1)
        event_ids.append(2)
        start_times.append(np.datetime64("2000-05-21T00", "ns"))
        end_times.append(np.datetime64("2000-05-23T00", "ns"))
        peak_times.append(np.datetime64("2000-05-22T00", "ns"))

    if include_cross_month_event:
        event.append(len(event))
        event_ids.append(3)
        start_times.append(np.datetime64("2000-05-31T00", "ns"))
        end_times.append(np.datetime64("2000-06-02T00", "ns"))
        peak_times.append(np.datetime64("2000-06-01T00", "ns"))

    n_event = len(event)
    ds = xr.Dataset(
        data_vars={
            "T_mean": ("time", np.full(time.size, 280.0)),
            "volume": ("time", np.ones(time.size)),
            "dTdt": ("time", np.full(time.size, 1.0)),
            "advection": ("time", np.full(time.size, 2.0)),
            "adiabatic": ("time", np.full(time.size, 3.0)),
            "diabatic": ("time", np.full(time.size, 4.0)),
            "tas_region": ("time", np.full(time.size, 300.0)),
            "tas_climatology": ("time", np.full(time.size, 290.0)),
            "hw_threshold": ("time", np.full(time.size, 295.0)),
            "lwa_a_region": ("time", np.full(time.size, 5.0)),
            "lwa_c_region": ("time", np.full(time.size, 6.0)),
            "event_id": ("event", np.asarray(event_ids, dtype=np.int64)),
            "start_time": ("event", np.asarray(start_times, dtype="datetime64[ns]")),
            "end_time": ("event", np.asarray(end_times, dtype="datetime64[ns]")),
            "duration": ("event", np.full(n_event, 3, dtype=np.int64)),
            "peak_time": ("event", np.asarray(peak_times, dtype="datetime64[ns]")),
            "tas_peak": ("event", np.full(n_event, 305.0)),
            "tas_anom_peak": ("event", np.full(n_event, 15.0)),
            "tas_excess_peak": ("event", np.full(n_event, 10.0)),
            "tas_excess_integral": ("event", np.full(n_event, 20.0)),
            "lwa_a_peak": ("event", np.full(n_event, 12.0)),
            "lwa_c_peak": ("event", np.full(n_event, 13.0)),
        },
        coords={"time": time, "event": np.asarray(event, dtype=np.int64)},
    )
    for name in ("dTdt", "advection", "adiabatic", "diabatic"):
        ds[name].attrs["units"] = "K hr-1"

    if add_extended:
        final_start = np.datetime64("2000-06-09T00", "h")
        soil = np.where(time >= final_start, 10.0, 1.0)
        ds["soil_moisture"] = ("time", soil)
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
