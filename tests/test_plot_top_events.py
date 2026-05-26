from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from scripts import plot_top_events
from src import analysis_io


RUN_ARGS = [
    "--region", "pnw_hotz",
    "--bottom-boundary", "surface",
    "--top-boundary", "700",
    "--threshold-variable", "tas",
    "--quantile", "90",
    "--start-year", "1940",
    "--end-year", "2024",
]


def _argv(*extra: str) -> list[str]:
    return ["plot_top_events.py", *RUN_ARGS, *extra]


def test_parse_args_builds_default_paths(monkeypatch):
    monkeypatch.setattr("sys.argv", _argv())

    args = plot_top_events.parse_args()

    assert args.input_path == analysis_io.default_harmonized_timeseries_path(
        region="pnw_hotz",
        bottom_boundary="surface",
        top_boundary="700hPa",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=2024,
    )
    assert args.output_dir == (
        plot_top_events.REPO_ROOT
        / "results"
        / "plots_top_events"
        / "region_pnw_hotz"
        / "boundary_surface_700hPa"
        / "time_range_1940_2024"
    )
    assert args.smoothing_window == plot_top_events.DEFAULT_SMOOTHING_WINDOW
    assert not args.plot_extended_variables


def test_parse_args_accepts_custom_input_path(monkeypatch, tmp_path):
    input_path = tmp_path / "stage1.nc"
    monkeypatch.setattr(
        "sys.argv",
        _argv("--input-path", str(input_path)),
    )

    args = plot_top_events.parse_args()

    assert args.input_path == input_path


def test_parse_args_accepts_plot_extended_variables(monkeypatch):
    monkeypatch.setattr("sys.argv", _argv("--plot-extended-variables"))

    args = plot_top_events.parse_args()

    assert args.plot_extended_variables


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


def test_write_top_event_plots_writes_raw_and_smoothed_figures_per_event(tmp_path):
    ds = _make_plot_dataset()
    selected = plot_top_events.select_top_tas_events(ds, n=2)

    written = plot_top_events.write_top_event_plots(
        ds,
        selected,
        output_dir=tmp_path,
        window_days=1,
    )

    assert len(written) == 4
    assert all(path.exists() for path in written)
    assert written[0].name.startswith("top_event_rank_01_event_0002_")
    assert written[1].name.startswith("top_event_rank_01_event_0002_")
    assert written[1].name.endswith("_smoothed.png")


def test_write_top_event_plots_computes_one_reference_composite(monkeypatch, tmp_path):
    ds = _make_plot_dataset()
    selected = plot_top_events.select_top_tas_events(ds, n=2)
    reference_composite = xr.Dataset()
    smoothed_reference_composite = xr.Dataset()
    captured = {
        "composite_calls": 0,
        "plot_references": [],
        "smooth_calls": [],
    }

    def fake_composite(source, **kwargs):
        captured["composite_calls"] += 1
        captured["composite_source"] = source
        captured["composite_kwargs"] = kwargs
        return reference_composite

    def fake_smooth(source, **kwargs):
        captured["smooth_calls"].append((source, kwargs))
        if source is reference_composite:
            return smoothed_reference_composite
        smoothed = source.copy(deep=False)
        smoothed.attrs["smoothing_window"] = kwargs["smoothing_window"]
        return smoothed

    def fake_plot(
        event_window,
        event,
        *,
        reference_composite=None,
        plot_extended_variables=False,
    ):
        captured["plot_references"].append(reference_composite)
        captured.setdefault("plot_extended_variables", []).append(plot_extended_variables)
        fig = plt.figure()
        fig.add_subplot(111)
        return fig

    monkeypatch.setattr(
        plot_top_events.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )
    monkeypatch.setattr(
        plot_top_events.plotting,
        "plot_top_event_timeseries",
        fake_plot,
    )
    monkeypatch.setattr(
        plot_top_events.plotting,
        "smooth_composite_for_display",
        fake_smooth,
    )

    written = plot_top_events.write_top_event_plots(
        ds,
        selected,
        output_dir=tmp_path,
        window_days=1,
        smoothing_window=6,
    )

    assert len(written) == 4
    assert captured["composite_calls"] == 1
    assert captured["composite_source"] is ds
    assert captured["composite_kwargs"] == {
        "variables": plot_top_events.TOP_EVENT_VARIABLES,
        "pre_days": 1,
        "post_days": 1,
        "event_percentiles": plot_top_events.REFERENCE_EVENT_PERCENTILES,
    }
    assert captured["plot_references"] == [
        reference_composite,
        smoothed_reference_composite,
        reference_composite,
        smoothed_reference_composite,
    ]
    assert captured["plot_extended_variables"] == [False, False, False, False]
    assert len(captured["smooth_calls"]) == 3
    first_smooth_source, first_smooth_kwargs = captured["smooth_calls"][0]
    assert first_smooth_source is reference_composite
    assert first_smooth_kwargs == {
        "variables": plot_top_events.SMOOTHED_TOP_EVENT_VARIABLES,
        "smoothing_window": 6,
    }
    assert all(
        kwargs == {
            "variables": plot_top_events.SMOOTHED_TOP_EVENT_VARIABLES,
            "smoothing_window": 6,
            "lag_dim": "time",
        }
        for _, kwargs in captured["smooth_calls"][1:]
    )
    assert all(path.exists() for path in written)


def test_write_top_event_plots_uses_extended_variables_when_requested(monkeypatch, tmp_path):
    ds = _make_plot_dataset()
    selected = plot_top_events.select_top_tas_events(ds, n=1)
    reference_composite = xr.Dataset()
    smoothed_reference_composite = xr.Dataset()
    captured = {"plot_extended_variables": []}

    def fake_composite(source, **kwargs):
        captured["composite_kwargs"] = kwargs
        return reference_composite

    def fake_smooth(source, **kwargs):
        captured.setdefault("smooth_kwargs", []).append(kwargs)
        if source is reference_composite:
            return smoothed_reference_composite
        return source

    def fake_plot(
        event_window,
        event,
        *,
        reference_composite=None,
        plot_extended_variables=False,
    ):
        captured["plot_extended_variables"].append(plot_extended_variables)
        fig = plt.figure()
        fig.add_subplot(111)
        return fig

    monkeypatch.setattr(
        plot_top_events.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )
    monkeypatch.setattr(
        plot_top_events.plotting,
        "plot_top_event_timeseries",
        fake_plot,
    )
    monkeypatch.setattr(
        plot_top_events.plotting,
        "smooth_composite_for_display",
        fake_smooth,
    )

    written = plot_top_events.write_top_event_plots(
        ds,
        selected,
        output_dir=tmp_path,
        window_days=1,
        plot_extended_variables=True,
    )

    assert len(written) == 2
    assert captured["composite_kwargs"]["variables"] == (
        plot_top_events.EXTENDED_TOP_EVENT_VARIABLES
    )
    assert captured["smooth_kwargs"][0]["variables"] == (
        plot_top_events.EXTENDED_SMOOTHED_TOP_EVENT_VARIABLES
    )
    assert captured["plot_extended_variables"] == [True, True]


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
