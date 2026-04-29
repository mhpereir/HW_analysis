from pathlib import Path

import xarray as xr

from scripts import plot_top_events
from src import analysis_io


def test_parse_args_uses_default_input_path(monkeypatch):
    monkeypatch.setattr("sys.argv", ["plot_top_events.py"])

    args = plot_top_events.parse_args()

    assert args.input_path == analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH


def test_parse_args_accepts_custom_input_path(monkeypatch, tmp_path):
    input_path = tmp_path / "stage1.nc"
    monkeypatch.setattr(
        "sys.argv",
        ["plot_top_events.py", "--input-path", str(input_path)],
    )

    args = plot_top_events.parse_args()

    assert args.input_path == input_path


def test_open_harmonized_dataset_delegates_to_analysis_io(monkeypatch, tmp_path):
    captured = {}
    expected = xr.Dataset()

    def fake_open(path: str | Path, *, chunks=None):
        captured["path"] = path
        captured["chunks"] = chunks
        return expected

    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", fake_open)

    out = plot_top_events.open_harmonized_dataset(
        tmp_path / "stage1.nc",
        chunks={"time": 128},
    )

    assert out is expected
    assert captured["path"] == tmp_path / "stage1.nc"
    assert captured["chunks"] == {"time": 128}
