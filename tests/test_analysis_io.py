import numpy as np
import pytest
import xarray as xr

from src import analysis_io


def test_default_harmonized_timeseries_path_constant_uses_stage1_filename():
    assert analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH.name == (
        "harmonized_regional_timeseries_pnw_bartusek_tas_q90_1940_2024.nc"
    )


def test_default_harmonized_timeseries_path_includes_run_tokens():
    path = analysis_io.default_harmonized_timeseries_path(
        region="pnw_bartusek",
        threshold_variable="lwa",
        quantile="q97p5",
        start_year=1940,
        end_year=2024,
    )

    assert path.parent == analysis_io.DEFAULT_STAGE1_OUTPUT_DIR
    assert path.name == (
        "harmonized_regional_timeseries_pnw_bartusek_lwa_q97p5_1940_2024.nc"
    )


def test_save_harmonized_timeseries_creates_parent_and_writes_readable_file(tmp_path):
    ds = _make_harmonized_dataset()
    path = tmp_path / "nested" / "stage1.nc"

    saved_path = analysis_io.save_harmonized_timeseries(ds, path)

    assert saved_path == path.resolve()
    assert saved_path.exists()

    with xr.open_dataset(saved_path, engine="h5netcdf") as reopened:
        assert reopened.attrs["pipeline_stage"] == analysis_io.EXPECTED_PIPELINE_STAGE
        assert set(analysis_io.REQUIRED_HARMONIZED_VARIABLES) <= set(reopened.data_vars)
        assert reopened["hw_flag"].dtype == np.int8
        assert reopened["hw_flag"].attrs["projected_from_daily"] == 1


def test_open_harmonized_timeseries_validates_and_returns_dataset(tmp_path):
    path = analysis_io.save_harmonized_timeseries(
        _make_harmonized_dataset(),
        tmp_path / "stage1.nc",
    )

    out = analysis_io.open_harmonized_timeseries(path)

    try:
        assert out.attrs["pipeline_stage"] == analysis_io.EXPECTED_PIPELINE_STAGE
        assert out.attrs["time_axis"] == "time"
        assert out.sizes["time"] == 2
        np.testing.assert_allclose(out["T_mean"].values, [1.0, 2.0])
    finally:
        out.close()


def test_save_harmonized_timeseries_rejects_non_stage1_dataset(tmp_path):
    ds = _make_harmonized_dataset()
    ds.attrs["pipeline_stage"] = "not_stage_1"

    with pytest.raises(ValueError, match="pipeline_stage"):
        analysis_io.save_harmonized_timeseries(ds, tmp_path / "stage1.nc")


def test_open_harmonized_timeseries_rejects_missing_required_variables(tmp_path):
    ds = _make_harmonized_dataset().drop_vars("hw_event_id")
    ds["hw_flag"].attrs["projected_from_daily"] = 1
    path = tmp_path / "invalid.nc"
    ds.to_netcdf(path, engine="h5netcdf")

    with pytest.raises(ValueError, match="missing required variables"):
        opened = analysis_io.open_harmonized_timeseries(path)
        opened.close()


def _make_harmonized_dataset() -> xr.Dataset:
    time = np.array(["2000-05-01T00:00", "2000-05-01T01:00"], dtype="datetime64[m]")
    values = np.array([1.0, 2.0])
    flags = np.array([False, True])
    event_ids = np.array([0, 1], dtype=np.int64)

    return xr.Dataset(
        data_vars={
            "T_mean": ("time", values),
            "volume": ("time", values),
            "dTdt": ("time", values),
            "advection": ("time", values),
            "adiabatic": ("time", values),
            "diabatic": ("time", values),
            "tas_region": ("time", values),
            "tas_climatology": ("time", values),
            "hw_threshold": ("time", values),
            "hw_flag": ("time", flags, {"projected_from_daily": True}),
            "hw_event_id": ("time", event_ids),
            "lwa_region": ("time", values),
            "lwa_threshold": ("time", values),
            "lwa_flag": ("time", flags),
            "lwa_event_id": ("time", event_ids),
            "lwa_a_region": ("time", values),
            "lwa_a_threshold": ("time", values),
            "lwa_a_flag": ("time", flags),
            "lwa_a_event_id": ("time", event_ids),
            "lwa_c_region": ("time", values),
            "lwa_c_threshold": ("time", values),
            "lwa_c_flag": ("time", flags),
            "lwa_c_event_id": ("time", event_ids),
        },
        coords={"time": time},
        attrs={
            "pipeline_stage": analysis_io.EXPECTED_PIPELINE_STAGE,
            "analysis_time_resolution": "hourly",
            "time_axis": "time",
        },
    )
