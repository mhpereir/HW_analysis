import numpy as np
import matplotlib.pyplot as plt
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
    assert smoothed.attrs["smoothing_applied_to"] == (
        "T_mean, event_percentile_T_mean, advection, event_percentile_advection"
    )
    assert not np.array_equal(smoothed["T_mean"].values, composite["T_mean"].values)
    assert not np.array_equal(
        smoothed["event_percentile_T_mean"].values,
        composite["event_percentile_T_mean"].values,
    )
    assert not np.array_equal(
        smoothed["advection"].values,
        composite["advection"].values,
    )
    np.testing.assert_allclose(smoothed["volume"].values, composite["volume"].values)
    np.testing.assert_allclose(
        smoothed["event_percentile_volume"].values,
        composite["event_percentile_volume"].values,
    )
    np.testing.assert_allclose(
        smoothed["lwa_a_region"].values,
        composite["lwa_a_region"].values,
    )


def test_plot_composite_timeseries_adds_percentile_envelopes():
    fig = plotting.plot_composite_timeseries(_make_composite())
    try:
        assert sum(len(ax.collections) for ax in fig.axes) == 8
    finally:
        plt.close(fig)


def test_plot_composite_timeseries_adds_iqr_legend_to_first_panel():
    fig = plotting.plot_composite_timeseries(_make_composite())
    try:
        assert fig.axes[4].get_legend().get_texts()[0].get_text() == "IQR"
    finally:
        plt.close(fig)


def test_temperature_volume_axis_labels_match_tick_colors():
    fig = plotting.plot_composite_timeseries(_make_composite())
    try:
        assert fig.axes[0].yaxis.label.get_color() == "tab:red"
        assert fig.axes[4].yaxis.label.get_color() == "tab:blue"
    finally:
        plt.close(fig)


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


def test_plot_split_composite_timeseries_draws_mean_lines_for_each_bin():
    fig = plotting.plot_split_composite_timeseries(_make_split_composite())
    try:
        # dTdt panel: 2 bins * (mean + 2 IQR bounds), plus zero line and peak marker.
        assert len(fig.axes[1].lines) == 8
    finally:
        plt.close(fig)


def test_plot_split_composite_timeseries_draws_iqr_as_lines_not_fills():
    fig = plotting.plot_split_composite_timeseries(_make_split_composite())
    try:
        assert sum(len(ax.collections) for ax in fig.axes) == 0
        assert _legend_label_count(fig, "IQR bounds") == 1
        assert fig.axes[4].get_legend().get_texts()[-1].get_text() == "IQR bounds"
    finally:
        plt.close(fig)


def test_write_split_composite_timeseries_outputs_writes_raw_and_smoothed_pngs(tmp_path):
    written = plotting.write_split_composite_timeseries_outputs(
        _make_split_composite(),
        tmp_path / "hw_split_composite.png",
        smoothed_output_path=tmp_path / "hw_split_composite_smoothed.png",
        smoothing_window=2,
        smoothed_variables=("T_mean", "volume", "dTdt"),
    )

    assert [path.name for path in written] == [
        "hw_split_composite.png",
        "hw_split_composite_smoothed.png",
    ]
    assert all(path.exists() for path in written)


def _make_composite() -> xr.Dataset:
    lag_hour = np.arange(-2, 3)
    variables = {
        "T_mean": np.array([280.0, 281.0, 284.0, 283.0, 282.0]),
        "volume": np.array([10.0, 11.0, 14.0, 13.0, 12.0]),
        "dTdt": np.array([0.0, 0.1, 0.4, 0.3, 0.2]),
        "advection": np.array([1.0, 2.0, 5.0, 4.0, 3.0]),
        "adiabatic": np.array([0.5, 0.4, 0.1, 0.2, 0.3]),
        "diabatic": np.array([-1.0, -0.5, 1.0, 0.5, 0.0]),
        "lwa_a_region": np.array([2.0, 3.0, 6.0, 5.0, 4.0]),
        "lwa_c_region": np.array([6.0, 5.0, 2.0, 3.0, 4.0]),
    }
    data_vars = {
        name: ("lag_hour", values)
        for name, values in variables.items()
    }
    data_vars.update(
        {
            f"event_percentile_{name}": (
                ("quantile", "lag_hour"),
                np.vstack([values - 1.0, values, values + 1.0]),
            )
            for name, values in variables.items()
        }
    )
    return xr.Dataset(
        data_vars=data_vars,
        coords={"lag_hour": lag_hour, "quantile": [0.25, 0.5, 0.75]},
        attrs={"n_events": 2, "pre_days": 2, "post_days": 2},
    )


def _legend_label_count(fig, label: str) -> int:
    """Return the number of legend entries matching one label."""
    count = 0
    for ax in fig.axes:
        legend = ax.get_legend()
        if legend is None:
            continue
        count += sum(text.get_text() == label for text in legend.get_texts())
    return count


def _make_split_composite() -> xr.Dataset:
    base = _make_composite()
    bins = []
    for offset in (0.0, 2.0):
        ds = base.copy(deep=True)
        for name in ds.data_vars:
            ds[name] = ds[name] + offset
        bins.append(ds)
    out = xr.concat(
        bins,
        dim=xr.IndexVariable("split_bin", ["q0-0.5 (n=2)", "q0.5-1 (n=2)"]),
    )
    out = out.assign_coords(split_n_events=("split_bin", np.array([2, 2])))
    out.attrs.update(
        {
            "n_events": 4,
            "pre_days": 2,
            "post_days": 2,
            "split_variable": "duration",
        }
    )
    return out
