import numpy as np
import xarray as xr

from scripts import plot_composite_timeseries


def test_build_all_hw_event_stack_uses_all_events():
    ds = _make_composite_dataset()

    stacked = plot_composite_timeseries.build_all_hw_event_stack(ds, window_days=0)

    assert stacked.sizes == {"event": 2, "lag_hour": 1}
    np.testing.assert_array_equal(stacked["event"].values, [1, 2])
    np.testing.assert_allclose(stacked["T_mean"].values[:, 0], [281.0, 283.0])
    assert stacked.attrs["n_events"] == 2


def test_build_event_mean_composite_averages_over_events():
    stacked = plot_composite_timeseries.build_all_hw_event_stack(
        _make_composite_dataset(),
        window_days=0,
    )

    composite = plot_composite_timeseries.build_event_mean_composite(stacked)

    assert composite.sizes == {"lag_hour": 1}
    np.testing.assert_allclose(composite["T_mean"].values, [282.0])
    np.testing.assert_allclose(composite["volume"].values, [12.0])
    assert composite.attrs["composite_reduction"] == "mean over all HW events"


def test_write_composite_plot_writes_png(tmp_path):
    stacked = plot_composite_timeseries.build_all_hw_event_stack(
        _make_composite_dataset(),
        window_days=0,
    )
    composite = plot_composite_timeseries.build_event_mean_composite(stacked)

    path = plot_composite_timeseries.write_composite_plot(
        composite,
        tmp_path / "composite.png",
    )

    assert path.exists()
    assert path.name == "composite.png"


def test_smooth_composite_smooths_only_first_three_panel_variables():
    composite = plot_composite_timeseries.build_event_mean_composite(
        plot_composite_timeseries.build_all_hw_event_stack(
            _make_composite_dataset(),
            window_days=1,
        )
    )

    smoothed = plot_composite_timeseries.smooth_composite(composite, smoothing_window=3)

    assert smoothed.attrs["smoothing_window"] == 3
    assert not np.array_equal(smoothed["T_mean"].values, composite["T_mean"].values)
    assert not np.array_equal(smoothed["advection"].values, composite["advection"].values)
    np.testing.assert_allclose(smoothed["lwa_a_region"].values, composite["lwa_a_region"].values)
    np.testing.assert_allclose(smoothed["lwa_c_region"].values, composite["lwa_c_region"].values)


def test_write_composite_outputs_writes_raw_and_smoothed_pngs(tmp_path):
    composite = plot_composite_timeseries.build_event_mean_composite(
        plot_composite_timeseries.build_all_hw_event_stack(
            _make_composite_dataset(),
            window_days=0,
        )
    )

    written = plot_composite_timeseries.write_composite_outputs(
        composite,
        tmp_path / "hw_all_events_composite.png",
        smoothing_window=2,
    )

    assert [path.name for path in written] == [
        "hw_all_events_composite.png",
        "hw_all_events_composite_smoothed.png",
    ]
    assert all(path.exists() for path in written)


def _make_composite_dataset() -> xr.Dataset:
    time = np.array(
        [
            "2000-05-01T00:00",
            "2000-05-01T01:00",
            "2000-05-01T02:00",
            "2000-05-01T03:00",
        ],
        dtype="datetime64[h]",
    )
    event = np.arange(2)
    return xr.Dataset(
        data_vars={
            "T_mean": ("time", np.array([280.0, 281.0, 282.0, 283.0])),
            "volume": ("time", np.array([10.0, 11.0, 12.0, 13.0])),
            "dTdt": ("time", np.array([0.0, 0.1, 0.2, 0.3])),
            "advection": ("time", np.array([1.0, 2.0, 3.0, 4.0])),
            "adiabatic": ("time", np.array([0.5, 0.4, 0.3, 0.2])),
            "diabatic": ("time", np.array([-1.0, -0.5, 0.0, 0.5])),
            "lwa_a_region": ("time", np.array([2.0, 3.0, 4.0, 5.0])),
            "lwa_c_region": ("time", np.array([5.0, 4.0, 3.0, 2.0])),
            "event_id": ("event", np.array([1, 2], dtype=np.int64)),
            "peak_time": (
                "event",
                np.array(["2000-05-01T01:00", "2000-05-01T03:00"], dtype="datetime64[h]"),
            ),
        },
        coords={"time": time, "event": event},
    )
