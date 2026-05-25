import numpy as np
import matplotlib.dates as mdates
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


def test_plot_composite_timeseries_uses_variable_colors_for_multi_variable_panels():
    fig = plotting.plot_composite_timeseries(_make_composite())
    try:
        tendency_colors = _line_colors_by_label(fig.axes[2])
        assert tendency_colors["advection"] == plotting.VARIABLE_COLORS["advection"]
        assert tendency_colors["adiabatic"] == plotting.VARIABLE_COLORS["adiabatic"]
        assert tendency_colors["diabatic"] == plotting.VARIABLE_COLORS["diabatic"]

        lwa_colors = _line_colors_by_label(fig.axes[3])
        assert lwa_colors["lwa_a_region"] == plotting.VARIABLE_COLORS["lwa_a_region"]
        assert lwa_colors["lwa_c_region"] == plotting.VARIABLE_COLORS["lwa_c_region"]
    finally:
        plt.close(fig)


def test_plot_composite_timeseries_expands_tendency_axis_range():
    composite = _make_composite()
    fig = plotting.plot_composite_timeseries(composite)
    expected_fig, expected_ax = plt.subplots()
    try:
        for name in ("advection", "adiabatic", "diabatic"):
            expected_ax.plot(composite["lag_hour"].values, composite[name].values)
            lower = composite[f"event_percentile_{name}"].sel(quantile=0.25)
            upper = composite[f"event_percentile_{name}"].sel(quantile=0.75)
            expected_ax.fill_between(
                composite["lag_hour"].values,
                lower.values,
                upper.values,
            )
        expected_ax.axhline(0)
        expected_lower, expected_upper = expected_ax.get_ylim()
        lower, upper = fig.axes[2].get_ylim()

        np.testing.assert_allclose(
            upper - lower,
            1.5 * (expected_upper - expected_lower),
        )
    finally:
        plt.close(fig)
        plt.close(expected_fig)


def test_plot_composite_timeseries_extended_layout_uses_optional_panels():
    composite = _make_composite()

    fig = plotting.plot_composite_timeseries(
        composite,
        plot_extended_variables=True,
    )
    try:
        assert len(fig.axes) == 12

        assert set(_line_colors_by_label(fig.axes[4])) == {"advection"}
        assert set(_line_colors_by_label(fig.axes[6])) == {"adiabatic"}
        assert set(_line_colors_by_label(fig.axes[8])) == {"diabatic"}

        pbl_axis = fig.axes[3]
        assert pbl_axis.yaxis_inverted()
        assert pbl_axis.get_ylabel() == "PBL top pressure [hPa]"
        np.testing.assert_allclose(
            pbl_axis.lines[0].get_ydata(),
            composite["pbl_p_mean"].values * 0.01,
        )

        assert set(_line_colors_by_label(fig.axes[5])) == {
            "nslr_heating_rate_approx",
            "nssr_heating_rate_approx",
        }
        assert set(_line_colors_by_label(fig.axes[7])) == {
            "sshf_heating_rate_approx",
            "slhf_heating_rate_approx",
        }
        assert fig.axes[9].get_ylabel() == "soil moisture [m3 m-3]"
        assert fig.axes[11].get_ylabel() == "cloud cover fraction"
    finally:
        plt.close(fig)


def test_plot_composite_timeseries_extended_layout_requires_optional_variables():
    composite = _make_composite().drop_vars("cloud_cover")

    try:
        plotting.plot_composite_timeseries(composite, plot_extended_variables=True)
    except ValueError as exc:
        assert "Extended plot requires missing variables in composite: cloud_cover" in str(exc)
    else:
        raise AssertionError("Expected missing extended variable to raise ValueError.")


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


def test_plot_split_composite_timeseries_labels_dtdt_panel_variable():
    fig = plotting.plot_split_composite_timeseries(_make_split_composite())
    try:
        legend_labels = [text.get_text() for text in fig.axes[1].get_legend().get_texts()]

        assert "dTdt" in legend_labels
    finally:
        plt.close(fig)


def test_plot_split_composite_timeseries_expands_tendency_axis_range():
    composite = _make_split_composite()
    fig = plotting.plot_split_composite_timeseries(composite)
    expected_fig, expected_ax = plt.subplots()
    try:
        for split_index in range(composite.sizes["split_bin"]):
            subset = composite.isel(split_bin=split_index)
            for name in ("advection", "adiabatic", "diabatic"):
                expected_ax.plot(composite["lag_hour"].values, subset[name].values)
                lower = subset[f"event_percentile_{name}"].sel(quantile=0.25)
                upper = subset[f"event_percentile_{name}"].sel(quantile=0.75)
                expected_ax.plot(composite["lag_hour"].values, lower.values)
                expected_ax.plot(composite["lag_hour"].values, upper.values)
        expected_ax.axhline(0)
        expected_lower, expected_upper = expected_ax.get_ylim()
        lower, upper = fig.axes[2].get_ylim()

        np.testing.assert_allclose(
            upper - lower,
            1.5 * (expected_upper - expected_lower),
        )
    finally:
        plt.close(fig)
        plt.close(expected_fig)


def test_plot_split_composite_timeseries_extended_layout_uses_optional_panels():
    composite = _make_split_composite()

    fig = plotting.plot_split_composite_timeseries(
        composite,
        plot_extended_variables=True,
    )
    try:
        assert len(fig.axes) == 12

        assert plotting.VARIABLE_COLORS["advection"] in _non_marker_line_colors(fig.axes[4])
        assert plotting.VARIABLE_COLORS["adiabatic"] in _non_marker_line_colors(fig.axes[6])
        assert plotting.VARIABLE_COLORS["diabatic"] in _non_marker_line_colors(fig.axes[8])

        pbl_axis = fig.axes[3]
        assert pbl_axis.yaxis_inverted()
        assert pbl_axis.get_ylabel() == "PBL top pressure [hPa]"
        np.testing.assert_allclose(
            pbl_axis.lines[0].get_ydata(),
            composite.isel(split_bin=0)["pbl_p_mean"].values * 0.01,
        )

        assert plotting.VARIABLE_COLORS["nslr_heating_rate_approx"] in (
            _non_marker_line_colors(fig.axes[5])
        )
        assert plotting.VARIABLE_COLORS["nssr_heating_rate_approx"] in (
            _non_marker_line_colors(fig.axes[5])
        )
        assert plotting.VARIABLE_COLORS["sshf_heating_rate_approx"] in (
            _non_marker_line_colors(fig.axes[7])
        )
        assert plotting.VARIABLE_COLORS["slhf_heating_rate_approx"] in (
            _non_marker_line_colors(fig.axes[7])
        )
        assert fig.axes[9].get_ylabel() == "soil moisture [m3 m-3]"
        assert fig.axes[11].get_ylabel() == "cloud cover fraction"
    finally:
        plt.close(fig)


def test_plot_split_composite_timeseries_extended_labels_single_variable_panels():
    fig = plotting.plot_split_composite_timeseries(
        _make_split_composite(),
        plot_extended_variables=True,
    )
    try:
        for axis_index, label in zip(
            (2, 4, 6, 8),
            ("dTdt", "advection", "adiabatic", "diabatic"),
            strict=True,
        ):
            legend_labels = [
                text.get_text()
                for text in fig.axes[axis_index].get_legend().get_texts()
            ]
            assert label in legend_labels
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


def test_plot_top_event_timeseries_draws_event_and_reference_lines():
    fig = plotting.plot_top_event_timeseries(
        _make_top_event_window(),
        _make_top_event(),
        reference_composite=_make_composite(),
    )
    try:
        dtdt_styles = {
            line.get_label(): line.get_linestyle()
            for line in fig.axes[1].lines
        }

        assert dtdt_styles["dTdt"] == "--"
        assert dtdt_styles["_all_event_average"] == "-"
    finally:
        plt.close(fig)


def test_plot_top_event_timeseries_draws_reference_iqr():
    fig = plotting.plot_top_event_timeseries(
        _make_top_event_window(),
        _make_top_event(),
        reference_composite=_make_composite(),
    )
    try:
        assert sum(len(ax.collections) for ax in fig.axes) == 8
        legend_labels = [text.get_text() for text in fig.axes[4].get_legend().get_texts()]

        assert legend_labels == ["all-event average", "IQR"]
        assert _legend_label_count(fig, "all-event average") == 1
        assert _legend_label_count(fig, "IQR") == 1
    finally:
        plt.close(fig)


def test_plot_top_event_timeseries_aligns_reference_lag_zero_to_peak():
    event = _make_top_event()
    fig = plotting.plot_top_event_timeseries(
        _make_top_event_window(),
        event,
        reference_composite=_make_composite(),
    )
    try:
        reference_line = next(
            line
            for line in fig.axes[1].lines
            if line.get_label() == "_all_event_average"
        )
        xdata = np.asarray(reference_line.get_xdata(), dtype="datetime64[ns]")
        peak_time = np.asarray(event["peak_time"].values).astype("datetime64[ns]")[()]
        zero_lag_index = int(np.flatnonzero(_make_composite()["lag_hour"].values == 0)[0])

        assert xdata[zero_lag_index] == peak_time
    finally:
        plt.close(fig)


def test_plot_top_event_timeseries_marks_event_bounds_with_lines_not_span():
    fig = plotting.plot_top_event_timeseries(_make_top_event_window(), _make_top_event())
    try:
        assert sum(len(ax.collections) for ax in fig.axes) == 0
        for ax in fig.axes[:4]:
            marker_styles = [line.get_linestyle() for line in ax.lines[-3:]]

            assert marker_styles == [":", ":", "--"]
    finally:
        plt.close(fig)


def test_plot_top_event_timeseries_expands_tendency_axis_range():
    event_window = _make_top_event_window()
    fig = plotting.plot_top_event_timeseries(event_window, _make_top_event())
    expected_fig, expected_ax = plt.subplots()
    try:
        for name in ("advection", "adiabatic", "diabatic"):
            expected_ax.plot(event_window["time"].values, event_window[name].values)
        expected_ax.axhline(0)
        expected_lower, expected_upper = expected_ax.get_ylim()
        lower, upper = fig.axes[2].get_ylim()

        np.testing.assert_allclose(
            upper - lower,
            1.5 * (expected_upper - expected_lower),
        )
    finally:
        plt.close(fig)
        plt.close(expected_fig)


def test_plot_top_event_timeseries_uses_concise_datetime_formatter():
    fig = plotting.plot_top_event_timeseries(_make_top_event_window(), _make_top_event())
    try:
        assert isinstance(fig.axes[-1].xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    finally:
        plt.close(fig)


def test_plot_top_event_timeseries_extended_layout_uses_optional_panels():
    event_window = _make_top_event_window()
    composite = _make_composite()

    fig = plotting.plot_top_event_timeseries(
        event_window,
        _make_top_event(),
        reference_composite=composite,
        plot_extended_variables=True,
    )
    try:
        assert len(fig.axes) == 12
        assert set(_line_colors_by_label(fig.axes[4])) == {"advection"}
        assert set(_line_colors_by_label(fig.axes[6])) == {"adiabatic"}
        assert set(_line_colors_by_label(fig.axes[8])) == {"diabatic"}

        pbl_axis = fig.axes[3]
        assert pbl_axis.yaxis_inverted()
        np.testing.assert_allclose(
            pbl_axis.lines[0].get_ydata(),
            event_window["pbl_p_mean"].values * 0.01,
        )
        assert set(_line_colors_by_label(fig.axes[5])) == {
            "nslr_heating_rate_approx",
            "nssr_heating_rate_approx",
        }
        assert set(_line_colors_by_label(fig.axes[7])) == {
            "sshf_heating_rate_approx",
            "slhf_heating_rate_approx",
        }
        assert fig.axes[9].get_ylabel() == "soil moisture [m3 m-3]"
        assert fig.axes[11].get_ylabel() == "cloud cover fraction"
    finally:
        plt.close(fig)


def test_plot_top_event_timeseries_extended_uses_concise_datetime_formatter():
    fig = plotting.plot_top_event_timeseries(
        _make_top_event_window(),
        _make_top_event(),
        reference_composite=_make_composite(),
        plot_extended_variables=True,
    )
    try:
        assert isinstance(fig.axes[8].xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
        assert isinstance(fig.axes[9].xaxis.get_major_formatter(), mdates.ConciseDateFormatter)
    finally:
        plt.close(fig)


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
        "pbl_p_mean": np.array([90000.0, 88000.0, 85000.0, 87000.0, 89000.0]),
        "pbl_p_p05": np.array([82000.0, 80000.0, 78000.0, 79000.0, 81000.0]),
        "pbl_p_p95": np.array([95000.0, 94000.0, 92000.0, 93000.0, 94000.0]),
        "nslr_heating_rate_approx": np.array([-0.2, -0.1, -0.3, -0.4, -0.2]),
        "nssr_heating_rate_approx": np.array([0.0, 0.2, 0.6, 0.5, 0.1]),
        "sshf_heating_rate_approx": np.array([0.1, 0.2, 0.4, 0.3, 0.2]),
        "slhf_heating_rate_approx": np.array([-0.1, -0.2, -0.4, -0.3, -0.2]),
        "soil_moisture": np.array([0.2, 0.19, 0.18, 0.17, 0.16]),
        "cloud_cover": np.array([0.7, 0.6, 0.4, 0.3, 0.5]),
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


def _make_top_event_window() -> xr.Dataset:
    time = np.array(
        [
            "2000-05-01T22:00",
            "2000-05-01T23:00",
            "2000-05-02T00:00",
            "2000-05-02T01:00",
            "2000-05-02T02:00",
        ],
        dtype="datetime64[h]",
    )
    variables = {
        "T_mean": np.array([280.0, 281.0, 284.0, 283.0, 282.0]),
        "volume": np.array([10.0, 11.0, 14.0, 13.0, 12.0]),
        "dTdt": np.array([0.0, 0.1, 0.4, 0.3, 0.2]),
        "advection": np.array([1.0, 2.0, 5.0, 4.0, 3.0]),
        "adiabatic": np.array([0.5, 0.4, 0.1, 0.2, 0.3]),
        "diabatic": np.array([-1.0, -0.5, 1.0, 0.5, 0.0]),
        "lwa_a_region": np.array([2.0, 3.0, 6.0, 5.0, 4.0]),
        "lwa_c_region": np.array([6.0, 5.0, 2.0, 3.0, 4.0]),
        "pbl_p_mean": np.array([90000.0, 88000.0, 85000.0, 87000.0, 89000.0]),
        "pbl_p_p05": np.array([82000.0, 80000.0, 78000.0, 79000.0, 81000.0]),
        "pbl_p_p95": np.array([95000.0, 94000.0, 92000.0, 93000.0, 94000.0]),
        "nslr_heating_rate_approx": np.array([-0.2, -0.1, -0.3, -0.4, -0.2]),
        "nssr_heating_rate_approx": np.array([0.0, 0.2, 0.6, 0.5, 0.1]),
        "sshf_heating_rate_approx": np.array([0.1, 0.2, 0.4, 0.3, 0.2]),
        "slhf_heating_rate_approx": np.array([-0.1, -0.2, -0.4, -0.3, -0.2]),
        "soil_moisture": np.array([0.2, 0.19, 0.18, 0.17, 0.16]),
        "cloud_cover": np.array([0.7, 0.6, 0.4, 0.3, 0.5]),
    }
    return xr.Dataset(
        data_vars={name: ("time", values) for name, values in variables.items()},
        coords={"time": time},
    )


def _make_top_event() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "event_id": 2,
            "selection_rank": 1,
            "start_time": np.datetime64("2000-05-01T23:00"),
            "end_time": np.datetime64("2000-05-02T01:00"),
            "peak_time": np.datetime64("2000-05-02T00:00"),
            "tas_peak": 305.0,
        }
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


def _line_colors_by_label(ax) -> dict[str, str]:
    """Return plotted line colors keyed by their visible legend labels."""
    return {
        line.get_label(): line.get_color()
        for line in ax.lines
        if not line.get_label().startswith("_")
    }


def _non_marker_line_colors(ax) -> set[str]:
    return {line.get_color() for line in ax.lines if line.get_color() != "0.2"}


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
