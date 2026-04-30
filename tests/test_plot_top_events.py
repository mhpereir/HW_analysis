from pathlib import Path

import numpy as np
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


def test_select_top_tas_events_returns_ranked_top_events():
    ds = _make_plot_dataset()

    selected = plot_top_events.select_top_tas_events(ds, n=2)

    np.testing.assert_array_equal(selected["event_id"].values, [2, 3])
    np.testing.assert_array_equal(selected["selection_rank"].values, [1, 2])
    np.testing.assert_allclose(selected["tas_peak"].values, [305.0, 301.0])


def test_write_top_event_plots_writes_one_figure_per_event(tmp_path):
    ds = _make_plot_dataset()
    selected = plot_top_events.select_top_tas_events(ds, n=2)

    written = plot_top_events.write_top_event_plots(
        ds,
        selected,
        output_dir=tmp_path,
        window_days=1,
    )

    assert len(written) == 2
    assert all(path.exists() for path in written)
    assert written[0].name.startswith("top_event_rank_01_event_0002_")


def _make_plot_dataset() -> xr.Dataset:
    time = np.array(
        [
            "2000-05-01T00:00",
            "2000-05-01T12:00",
            "2000-05-02T00:00",
            "2000-05-03T00:00",
            "2000-05-04T00:00",
        ],
        dtype="datetime64[m]",
    )
    event = np.arange(3)
    return xr.Dataset(
        data_vars={
            "T_mean": ("time", np.array([280.0, 281.0, 282.0, 283.0, 284.0])),
            "volume": ("time", np.array([10.0, 11.0, 12.0, 13.0, 14.0])),
            "dTdt": ("time", np.array([0.1, 0.2, -0.1, 0.0, 0.3])),
            "advection": ("time", np.array([1.0, 2.0, 3.0, 4.0, 5.0])),
            "adiabatic": ("time", np.array([0.5, 0.4, 0.3, 0.2, 0.1])),
            "diabatic": ("time", np.array([-1.0, -0.5, 0.0, 0.5, 1.0])),
            "lwa_a_region": ("time", np.array([2.0, 3.0, 4.0, 5.0, 6.0])),
            "lwa_c_region": ("time", np.array([6.0, 5.0, 4.0, 3.0, 2.0])),
            "hw_event_id": ("time", np.array([1, 1, 2, 0, 3], dtype=np.int64)),
            "event_id": ("event", np.array([1, 2, 3], dtype=np.int64)),
            "start_time": (
                "event",
                np.array(["2000-05-01T00:00", "2000-05-02T00:00", "2000-05-04T00:00"], dtype="datetime64[m]"),
            ),
            "end_time": (
                "event",
                np.array(["2000-05-01T12:00", "2000-05-02T00:00", "2000-05-04T00:00"], dtype="datetime64[m]"),
            ),
            "peak_time": (
                "event",
                np.array(["2000-05-01T12:00", "2000-05-02T00:00", "2000-05-04T00:00"], dtype="datetime64[m]"),
            ),
            "tas_peak": ("event", np.array([300.0, 305.0, 301.0])),
        },
        coords={"time": time, "event": event},
    )
