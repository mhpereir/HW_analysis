import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts import plot_composite_timeseries_all as plot_composite_timeseries
from src import analysis_io


def test_parse_args_uses_default_input_path(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plot_composite_timeseries_all.py"])

    args = plot_composite_timeseries.parse_args()

    assert args.input_path == analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH
    assert args.season_months is None
    assert not args.require_full_event
    assert not args.plot_extended_variables


def test_parse_args_accepts_season_months(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["plot_composite_timeseries_all.py", "--season-months", "6", "7", "8"],
    )

    args = plot_composite_timeseries.parse_args()

    assert args.season_months == [6, 7, 8]


def test_parse_args_accepts_plot_extended_variables(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        ["plot_composite_timeseries_all.py", "--plot-extended-variables"],
    )

    args = plot_composite_timeseries.parse_args()

    assert args.plot_extended_variables


def test_validate_args_rejects_negative_window_days():
    args = argparse.Namespace(
        window_days=-1,
        smoothing_window=24,
        season_months=None,
        require_full_event=False,
    )

    try:
        plot_composite_timeseries.validate_args(args)
    except ValueError as exc:
        assert "--window-days" in str(exc)
    else:
        raise AssertionError("Expected negative window_days to raise ValueError.")


def test_validate_args_rejects_invalid_season_months():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        season_months=[6, 13],
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="--season-months"):
        plot_composite_timeseries.validate_args(args)


def test_smoothed_output_path_uses_default_name_for_default_output():
    path = plot_composite_timeseries._smoothed_output_path(
        plot_composite_timeseries.DEFAULT_OUTPUT_PATH
    )

    assert path.name == "hw_all_events_composite_smoothed.png"


def test_main_orchestrates_dataset_composite_and_plotting(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "stage1.nc"
    output_path = tmp_path / "composite.png"
    opened = xr.Dataset()
    composite = xr.Dataset()
    captured = {}

    def fake_open(path):
        captured["input_path"] = path
        return opened

    def fake_composite(ds, **kwargs):
        captured["composite_ds"] = ds
        captured["composite_kwargs"] = kwargs
        return composite

    def fake_write(ds, output, **kwargs):
        captured["plot_ds"] = ds
        captured["output_path"] = output
        captured["plot_kwargs"] = kwargs
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_all.py",
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--window-days",
        "3",
        "--smoothing-window",
        "6",
    ])
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", fake_open)
    monkeypatch.setattr(
        plot_composite_timeseries.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )
    monkeypatch.setattr(
        plot_composite_timeseries.plotting,
        "write_composite_timeseries_outputs",
        fake_write,
    )

    result = plot_composite_timeseries.main()

    assert result == 0
    assert captured["input_path"] == input_path
    assert captured["composite_ds"] is opened
    assert captured["composite_kwargs"] == {
        "variables": plot_composite_timeseries.COMPOSITE_VARIABLES,
        "pre_days": 3,
        "post_days": 3,
        "event_percentiles": (0.25, 0.5, 0.75),
    }
    assert captured["plot_ds"] is composite
    assert captured["output_path"] == output_path
    assert captured["plot_kwargs"] == {
        "smoothed_output_path": output_path.with_name("composite_smoothed.png"),
        "smoothing_window": 6,
        "smoothed_variables": plot_composite_timeseries.SMOOTHED_VARIABLES,
        "plot_extended_variables": False,
    }
    assert "Wrote HW all-event composite figures:" in capsys.readouterr().out


def test_main_uses_extended_variables_when_requested(monkeypatch, tmp_path):
    input_path = tmp_path / "stage1.nc"
    output_path = tmp_path / "composite.png"
    opened = xr.Dataset()
    composite = xr.Dataset()
    captured = {}

    def fake_open(path):
        return opened

    def fake_composite(ds, **kwargs):
        captured["composite_kwargs"] = kwargs
        return composite

    def fake_write(ds, output, **kwargs):
        captured["plot_kwargs"] = kwargs
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_all.py",
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--plot-extended-variables",
    ])
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", fake_open)
    monkeypatch.setattr(
        plot_composite_timeseries.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )
    monkeypatch.setattr(
        plot_composite_timeseries.plotting,
        "write_composite_timeseries_outputs",
        fake_write,
    )

    result = plot_composite_timeseries.main()

    assert result == 0
    assert captured["composite_kwargs"]["variables"] == (
        plot_composite_timeseries.EXTENDED_COMPOSITE_VARIABLES
    )
    assert captured["plot_kwargs"]["smoothed_variables"] == (
        plot_composite_timeseries.EXTENDED_SMOOTHED_VARIABLES
    )
    assert captured["plot_kwargs"]["plot_extended_variables"]


def test_main_filters_event_table_before_composite_when_season_requested(monkeypatch, tmp_path):
    input_path = tmp_path / "stage1.nc"
    output_path = tmp_path / "composite.png"
    opened = _make_harmonized_dataset()
    composite = xr.Dataset()
    captured = {}

    def fake_open(path):
        captured["input_path"] = path
        return opened

    def fake_composite(ds, **kwargs):
        captured["composite_ds"] = ds
        captured["composite_kwargs"] = kwargs
        return composite

    def fake_write(ds, output, **kwargs):
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_all.py",
        "--input-path",
        str(input_path),
        "--output-path",
        str(output_path),
        "--season-months",
        "6",
        "--require-full-event",
    ])
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", fake_open)
    monkeypatch.setattr(
        plot_composite_timeseries.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )
    monkeypatch.setattr(
        plot_composite_timeseries.plotting,
        "write_composite_timeseries_outputs",
        fake_write,
    )

    result = plot_composite_timeseries.main()

    assert result == 0
    assert captured["composite_ds"] is opened
    event_table = captured["composite_kwargs"]["event_table"]
    np.testing.assert_array_equal(event_table["event_id"].values, [1])


def _make_harmonized_dataset() -> xr.Dataset:
    time = np.array(["2000-06-01T00:00", "2000-06-01T01:00"], dtype="datetime64[h]")
    event = np.arange(2)
    return xr.Dataset(
        data_vars={
            "T_mean": ("time", np.array([280.0, 281.0])),
            "event_id": ("event", np.array([1, 2], dtype=np.int64)),
            "start_time": (
                "event",
                np.array(["2000-06-01", "2000-05-31"], dtype="datetime64[ns]"),
            ),
            "end_time": (
                "event",
                np.array(["2000-06-02", "2000-06-02"], dtype="datetime64[ns]"),
            ),
            "peak_time": (
                "event",
                np.array(["2000-06-01", "2000-06-01"], dtype="datetime64[ns]"),
            ),
        },
        coords={"time": time, "event": event},
    )
