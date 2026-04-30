import numpy as np
import xarray as xr

from src import plotting


def test_smooth_composite_for_display_smooths_only_requested_variables():
    composite = _make_composite()

    smoothed = plotting.smooth_composite_for_display(
        composite,
        variables=("T_mean", "advection"),
        smoothing_window=3,
    )

    assert smoothed.attrs["smoothing_window"] == 3
    assert smoothed.attrs["smoothing_applied_to"] == "T_mean, advection"
    assert not np.array_equal(smoothed["T_mean"].values, composite["T_mean"].values)
    assert not np.array_equal(
        smoothed["advection"].values,
        composite["advection"].values,
    )
    np.testing.assert_allclose(smoothed["volume"].values, composite["volume"].values)
    np.testing.assert_allclose(
        smoothed["lwa_a_region"].values,
        composite["lwa_a_region"].values,
    )


def test_write_composite_timeseries_plot_writes_png(tmp_path):
    path = plotting.write_composite_timeseries_plot(
        _make_composite(),
        tmp_path / "composite.png",
    )

    assert path.exists()
    assert path.name == "composite.png"


def test_write_composite_timeseries_outputs_writes_raw_and_smoothed_pngs(tmp_path):
    written = plotting.write_composite_timeseries_outputs(
        _make_composite(),
        tmp_path / "hw_all_events_composite.png",
        smoothed_output_path=tmp_path / "hw_all_events_composite_smoothed.png",
        smoothing_window=2,
        smoothed_variables=("T_mean", "volume", "dTdt"),
    )

    assert [path.name for path in written] == [
        "hw_all_events_composite.png",
        "hw_all_events_composite_smoothed.png",
    ]
    assert all(path.exists() for path in written)


def _make_composite() -> xr.Dataset:
    lag_hour = np.arange(-2, 3)
    return xr.Dataset(
        data_vars={
            "T_mean": ("lag_hour", np.array([280.0, 281.0, 284.0, 283.0, 282.0])),
            "volume": ("lag_hour", np.array([10.0, 11.0, 14.0, 13.0, 12.0])),
            "dTdt": ("lag_hour", np.array([0.0, 0.1, 0.4, 0.3, 0.2])),
            "advection": ("lag_hour", np.array([1.0, 2.0, 5.0, 4.0, 3.0])),
            "adiabatic": ("lag_hour", np.array([0.5, 0.4, 0.1, 0.2, 0.3])),
            "diabatic": ("lag_hour", np.array([-1.0, -0.5, 1.0, 0.5, 0.0])),
            "lwa_a_region": ("lag_hour", np.array([2.0, 3.0, 6.0, 5.0, 4.0])),
            "lwa_c_region": ("lag_hour", np.array([6.0, 5.0, 2.0, 3.0, 4.0])),
        },
        coords={"lag_hour": lag_hour},
        attrs={"n_events": 2, "pre_days": 2, "post_days": 2},
    )
