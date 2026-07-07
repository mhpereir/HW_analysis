from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import plot_adiabatic_diabatic_advection as plot_diag


def test_plot_tendency_scatter_creates_three_data_axes_and_colorbar():
    features = _make_feature_table()

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        assert len(fig.axes) == 4
        assert [ax.get_title() for ax in fig.axes[:3]] == [
            "Diabatic vs Adiabatic",
            "Advection vs Adiabatic",
            "Sqrt LWA a Peak",
        ]
        assert fig.axes[-1].get_ylabel() == "Peak TAS Anomaly (K)"
    finally:
        plot_diag.plt.close(fig)


def test_validate_feature_variables_only_requires_plotted_and_color_variables():
    features = _make_feature_table()

    plot_diag.validate_feature_variables(
        features,
        color_variable=plot_diag.COLOR_VARIABLE,
    )

    missing_color = features.drop_vars(plot_diag.COLOR_VARIABLE)
    with pytest.raises(ValueError, match=plot_diag.COLOR_VARIABLE):
        plot_diag.validate_feature_variables(
            missing_color,
            color_variable=plot_diag.COLOR_VARIABLE,
        )

    missing_lwa_source = features.drop_vars("lwa_a_peak")
    with pytest.raises(ValueError, match="lwa_a_peak"):
        plot_diag.validate_feature_variables(
            missing_lwa_source,
            color_variable=plot_diag.COLOR_VARIABLE,
        )


def test_each_data_panel_draws_one_to_one_line():
    features = _make_feature_table()

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        for ax in fig.axes[:3]:
            one_to_one_lines = [
                line for line in ax.lines if line.get_gid() == "one_to_one"
            ]
            assert len(one_to_one_lines) == 1
            np.testing.assert_allclose(
                one_to_one_lines[0].get_xdata(),
                one_to_one_lines[0].get_ydata(),
            )
    finally:
        plot_diag.plt.close(fig)


def test_advection_panel_draws_one_to_negative_one_line():
    features = _make_feature_table()

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        diabatic_axis, advection_axis = fig.axes[:2]
        assert not [
            line
            for line in diabatic_axis.lines
            if line.get_gid() == "one_to_negative_one"
        ]

        negative_one_lines = [
            line
            for line in advection_axis.lines
            if line.get_gid() == "one_to_negative_one"
        ]
        assert len(negative_one_lines) == 1
        np.testing.assert_allclose(
            negative_one_lines[0].get_xdata(),
            -negative_one_lines[0].get_ydata(),
        )
    finally:
        plot_diag.plt.close(fig)


def test_shared_x_axis_uses_finite_x_data_extent_only():
    features = _make_feature_table()
    features["I_diabatic_pre"] = ("event", np.array([10.0, 12.0, 14.0, 16.0, 18.0, 20.0]))
    features["I_advection_pre"] = (
        "event",
        np.array([-20.0, -18.0, -16.0, -14.0, -12.0, -10.0]),
    )

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        expected_xlim = (
            float(features["I_adiabatic_pre"].min()),
            float(features["I_adiabatic_pre"].max()),
        )
        for ax in fig.axes[:3]:
            np.testing.assert_allclose(ax.get_xlim(), expected_xlim)
    finally:
        plot_diag.plt.close(fig)


def test_scatter_colors_use_peak_temperature_anomaly():
    features = _make_feature_table()

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        for ax in fig.axes[:3]:
            scatter = ax.collections[0]
            np.testing.assert_allclose(
                np.asarray(scatter.get_array()),
                features["tas_anom_peak"].values,
            )
    finally:
        plot_diag.plt.close(fig)


def test_scatter_colors_use_configured_colormap():
    features = _make_feature_table()

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        for ax in fig.axes[:3]:
            scatter = ax.collections[0]
            assert scatter.cmap.name == plot_diag.COLOR_MAP
    finally:
        plot_diag.plt.close(fig)


def test_feature_values_square_roots_integrated_lwa_exposure():
    features = xr.Dataset(
        data_vars={
            "I_lwa_a_pre_peak": ("event", np.array([0.0, 4.0, 9.0, -1.0])),
        },
        coords={"event": np.arange(4)},
    )

    out = plot_diag.feature_values(features, "sqrt_I_lwa_a_pre_peak")

    np.testing.assert_allclose(out[:3], np.array([0.0, 2.0, 3.0]))
    assert np.isnan(out[3])


def test_feature_values_square_roots_peak_lwa():
    features = xr.Dataset(
        data_vars={
            "lwa_a_peak": ("event", np.array([0.0, 16.0, 25.0, -1.0])),
        },
        coords={"event": np.arange(4)},
    )

    out = plot_diag.feature_values(features, "sqrt_lwa_a_peak")

    np.testing.assert_allclose(out[:3], np.array([0.0, 4.0, 5.0]))
    assert np.isnan(out[3])


def test_main_writes_one_raw_output(monkeypatch, tmp_path):
    input_path = tmp_path / "features.nc"
    output_path = tmp_path / "adiabatic_advection_vs_adiabatic_scatter.png"
    written = []

    def fake_open(path):
        assert path == input_path
        return _make_feature_table()

    def fake_write(
        features,
        path,
        *,
        color_variable,
        point_size,
        alpha,
    ):
        written.append(
            (
                Path(path).name,
                color_variable,
                point_size,
                alpha,
            )
        )
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_adiabatic_diabatic_advection.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--point-size",
            "10",
            "--alpha",
            "0.6",
        ],
    )
    monkeypatch.setattr(plot_diag, "open_event_features", fake_open)
    monkeypatch.setattr(plot_diag, "write_tendency_scatter_plot", fake_write)

    assert plot_diag.main() == 0
    assert written == [
        (
            "adiabatic_advection_vs_adiabatic_scatter.png",
            "tas_anom_peak",
            10.0,
            0.6,
        ),
    ]


def _make_feature_table() -> xr.Dataset:
    event = np.arange(6)
    return xr.Dataset(
        data_vars={
            "I_dTdt_pre": ("event", np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])),
            "I_adiabatic_pre": ("event", np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0])),
            "I_diabatic_pre": ("event", np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0])),
            "I_advection_pre": ("event", np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 4.0])),
            "I_lwa_a_pre_peak": ("event", np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])),
            "lwa_a_peak": ("event", np.array([4.0, 9.0, 16.0, 25.0, 36.0, 49.0])),
            "tas_anom_peak": ("event", np.array([2.0, 3.0, 4.0, 4.5, 5.0, 6.0])),
        },
        coords={"event": event},
    )
