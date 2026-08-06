from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.src import analysis_io


def test_default_harmonized_timeseries_path_constant_uses_stage1_filename():
    assert analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH.name == (
        "harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_2024.nc"
    )


def test_default_harmonized_timeseries_path_includes_run_tokens():
    path = analysis_io.default_harmonized_timeseries_path(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary=700,
        threshold_variable="lwa",
        quantile="q97p5",
        start_year=1940,
        end_year=2024,
    )

    assert path.parent == analysis_io.DEFAULT_STAGE1_OUTPUT_DIR
    assert path.name == (
        "harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_lwa_q97p5_1940_2024.nc"
    )


def test_default_harmonized_timeseries_path_omits_boundaries_when_not_given():
    path = analysis_io.default_harmonized_timeseries_path(
        region="pnw_bartusek",
        threshold_variable="lwa",
        quantile="q97p5",
        start_year=1940,
        end_year=2024,
    )

    assert path.name == (
        "harmonized_regional_timeseries_pnw_bartusek_lwa_q97p5_1940_2024.nc"
    )


def test_default_regional_hourly_climatology_path_includes_run_tokens():
    path = analysis_io.default_regional_hourly_climatology_path(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary=700,
        start_year=1940,
        end_year=2024,
    )

    assert path.parent == analysis_io.DEFAULT_STAGE1_CLIMATOLOGY_OUTPUT_DIR
    assert path.name == (
        "regional_hourly_climatology_"
        "pnw_bartusek_surface_700hPa_1940_2024.nc"
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


def test_open_harmonized_timeseries_enables_stable_timedelta_decoding(
    monkeypatch,
    tmp_path,
):
    captured = {}
    ds = _make_harmonized_dataset()

    def fake_open_dataset(path: Path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return ds

    monkeypatch.setattr(xr, "open_dataset", fake_open_dataset)

    out = analysis_io.open_harmonized_timeseries(
        tmp_path / "stage1.nc",
        chunks={"time": 12},
    )

    assert out is ds
    assert captured["path"] == (tmp_path / "stage1.nc").resolve()
    assert captured["kwargs"] == {
        "engine": "h5netcdf",
        "decode_timedelta": True,
        "chunks": {"time": 12},
    }


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


def test_stage1_contract_version_2_requires_face_variables(tmp_path):
    ds = _make_harmonized_dataset()
    ds.attrs["stage1_contract_version"] = 2

    with pytest.raises(ValueError, match="contract version 2"):
        analysis_io.save_harmonized_timeseries(ds, tmp_path / "stage1.nc")


def test_stage1_contract_version_2_with_face_variables_is_valid(tmp_path):
    ds = _make_harmonized_dataset()
    ds.attrs["stage1_contract_version"] = 2
    for name in analysis_io.REQUIRED_STAGE1_V2_VARIABLES:
        ds[name] = ("time", [0.1, 0.2])

    path = analysis_io.save_harmonized_timeseries(ds, tmp_path / "stage1.nc")

    with analysis_io.open_harmonized_timeseries(path) as reopened:
        assert reopened.attrs["stage1_contract_version"] == 2
        assert analysis_io.REQUIRED_STAGE1_V2_VARIABLES <= set(reopened.data_vars)


def test_save_and_open_regional_hourly_climatology(tmp_path):
    climate_time = np.array(
        ["2000-05-01T00:00", "2000-05-01T01:00"],
        dtype="datetime64[m]",
    )
    ds = xr.Dataset(
        {"T_mean": ("climatology_time", [1.0, 2.0])},
        coords={"climatology_time": climate_time},
        attrs={"pipeline_stage": analysis_io.EXPECTED_CLIMATOLOGY_PIPELINE_STAGE},
    )

    path = analysis_io.save_regional_hourly_climatology(
        ds,
        tmp_path / "climatology.nc",
    )

    with analysis_io.open_regional_hourly_climatology(path) as reopened:
        np.testing.assert_allclose(reopened["T_mean"], [1.0, 2.0])


def test_atomic_netcdf_writer_cleans_partial_file_after_failure(
    monkeypatch,
    tmp_path,
):
    ds = xr.Dataset({"value": ("time", [1.0])})
    output = tmp_path / "product.nc"

    def fail_write(self, path, **kwargs):
        Path(path).write_bytes(b"partial")
        raise RuntimeError("simulated write failure")

    monkeypatch.setattr(xr.Dataset, "to_netcdf", fail_write)

    with pytest.raises(RuntimeError, match="simulated write failure"):
        analysis_io._write_netcdf_atomically(ds, output)

    assert not output.exists()
    assert list(tmp_path.iterdir()) == []


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
