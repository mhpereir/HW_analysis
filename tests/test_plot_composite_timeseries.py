import argparse
from pathlib import Path

import xarray as xr

from scripts import plot_composite_timeseries
from src import analysis_io


def test_parse_args_uses_default_input_path(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plot_composite_timeseries.py"])

    args = plot_composite_timeseries.parse_args()

    assert args.input_path == analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH


def test_validate_args_rejects_negative_window_days():
    args = argparse.Namespace(window_days=-1, smoothing_window=24)

    try:
        plot_composite_timeseries.validate_args(args)
    except ValueError as exc:
        assert "--window-days" in str(exc)
    else:
        raise AssertionError("Expected negative window_days to raise ValueError.")


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

    def fake_open(path: str | Path):
        captured["input_path"] = path
        return opened

    def fake_composite(ds: xr.Dataset, **kwargs):
        captured["composite_ds"] = ds
        captured["composite_kwargs"] = kwargs
        return composite

    def fake_write(ds: xr.Dataset, output: Path, **kwargs):
        captured["plot_ds"] = ds
        captured["output_path"] = output
        captured["plot_kwargs"] = kwargs
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries.py",
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
        "event_percentiles": (0.05, 0.5, 0.95),
    }
    assert captured["plot_ds"] is composite
    assert captured["output_path"] == output_path
    assert captured["plot_kwargs"] == {
        "smoothed_output_path": output_path.with_name("composite_smoothed.png"),
        "smoothing_window": 6,
        "smoothed_variables": plot_composite_timeseries.SMOOTHED_VARIABLES,
    }
    assert "Wrote HW all-event composite figures:" in capsys.readouterr().out
