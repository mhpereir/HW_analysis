import numpy as np
import pytest
import xarray as xr

from scripts import build_regional_timeseries
from src import analysis_io


def test_parse_args_requires_start_and_end_year(monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_regional_timeseries.py"])

    with pytest.raises(SystemExit) as excinfo:
        build_regional_timeseries.parse_args()

    assert excinfo.value.code == 2


def test_parse_args_rejects_descending_year_range(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "2024",
            "--end-year",
            "1940",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        build_regional_timeseries.parse_args()

    assert excinfo.value.code == 2


def test_parse_args_builds_inclusive_analysis_years(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "1940",
            "--end-year",
            "1942",
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.start_year == 1940
    assert args.end_year == 1942
    assert args.analysis_years == [1940, 1941, 1942]
    assert args.output_path == analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH


def test_parse_args_accepts_custom_output_path(monkeypatch, tmp_path):
    output_path = tmp_path / "custom_stage1.nc"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "1940",
            "--end-year",
            "1940",
            "--output-path",
            str(output_path),
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.output_path == output_path


def test_append_hw_event_summary_table_adds_event_dimension():
    ds = _make_harmonized_timeseries_for_events()

    out = build_regional_timeseries.append_hw_event_summary_table(ds)

    assert out.sizes["time"] == 4
    assert out.sizes["event"] == 2
    assert {"event_id", "start_time", "end_time", "duration", "tas_anom_peak"} <= set(out)
    np.testing.assert_array_equal(out["event_id"].values, [1, 2])
    np.testing.assert_array_equal(out["duration"].values, [1, 1])
    np.testing.assert_allclose(out["tas_anom_peak"].values, [5.0, 5.0])


def _make_harmonized_timeseries_for_events() -> xr.Dataset:
    time = np.array(
        [
            "2000-05-01T00:00",
            "2000-05-01T12:00",
            "2000-05-02T00:00",
            "2000-05-03T00:00",
        ],
        dtype="datetime64[m]",
    )
    return xr.Dataset(
        data_vars={
            "hw_event_id": ("time", np.array([1, 1, 0, 2], dtype=np.int64)),
            "tas_region": ("time", np.array([300.0, 301.0, 280.0, 305.0])),
            "tas_climatology": ("time", np.array([295.0, 295.0, 275.0, 300.0])),
            "hw_threshold": ("time", np.array([298.0, 298.0, 278.0, 303.0])),
            "lwa_a_region": ("time", np.array([1.0, 2.0, 0.5, 3.0])),
            "lwa_c_region": ("time", np.array([4.0, 5.0, 0.5, 6.0])),
        },
        coords={"time": time},
    )
