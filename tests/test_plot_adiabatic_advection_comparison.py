from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import plot_adiabatic_advection_comparison as plot_diag


def test_net_dynamical_contribution_sums_adiabatic_and_advection():
    x_values = np.array([1.0, 2.0, 4.0, -3.0])
    y_values = np.array([-1.0, 0.0, 2.0, 1.0])

    net_dynamical = plot_diag.net_dynamical_contribution(
        x_values,
        y_values,
    )

    np.testing.assert_allclose(
        net_dynamical,
        x_values + y_values,
    )
    assert net_dynamical[0] == 0.0
    assert net_dynamical[1] > 0.0
    assert net_dynamical[3] < 0.0


def test_panels_use_expected_x_and_y_values():
    features = _make_feature_table()

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        top_offsets = np.asarray(fig.axes[0].collections[0].get_offsets())
        middle_offsets = np.asarray(fig.axes[1].collections[0].get_offsets())
        temperature_offsets = np.asarray(fig.axes[2].collections[0].get_offsets())
        diabatic_offsets = np.asarray(fig.axes[3].collections[0].get_offsets())
        expected_x = features["I_adiabatic_pre"].values
        expected_net_dynamical = (
            features["I_adiabatic_pre"].values
            + features["I_advection_pre"].values
        )

        np.testing.assert_allclose(top_offsets[:, 0], expected_x)
        np.testing.assert_allclose(middle_offsets[:, 0], expected_x)
        np.testing.assert_allclose(temperature_offsets[:, 0], expected_net_dynamical)
        np.testing.assert_allclose(diabatic_offsets[:, 0], expected_net_dynamical)
        np.testing.assert_allclose(
            top_offsets[:, 1],
            features["I_advection_pre"].values,
        )
        np.testing.assert_allclose(middle_offsets[:, 1], expected_net_dynamical)
        np.testing.assert_allclose(
            temperature_offsets[:, 1],
            features["I_dTdt_pre"].values,
        )
        np.testing.assert_allclose(
            diabatic_offsets[:, 1],
            features["I_diabatic_pre"].values,
        )
    finally:
        plot_diag.plt.close(fig)


def test_panels_share_color_normalization_centered_on_configured_value(monkeypatch):
    monkeypatch.setattr(plot_diag, "USE_CENTERED_COLOR_NORMALIZATION", True)
    monkeypatch.setattr(plot_diag, "COLOR_NORMALIZATION_CENTER", 4.0)

    fig = plot_diag.plot_tendency_scatter(_make_feature_table())
    try:
        norms = [ax.collections[0].norm for ax in fig.axes[:4]]
        assert all(norm is norms[0] for norm in norms)
        assert norms[0](4.0) == pytest.approx(0.5)
    finally:
        plot_diag.plt.close(fig)


def test_centered_color_normalization_can_be_disabled(monkeypatch):
    monkeypatch.setattr(plot_diag, "USE_CENTERED_COLOR_NORMALIZATION", False)

    norm = plot_diag.color_norm_for_values(np.array([2.0, 4.0, 6.0]))

    assert norm is None


def test_validate_feature_variables_requires_only_plotted_and_color_variables():
    features = _make_feature_table()

    plot_diag.validate_feature_variables(
        features,
        color_variable=plot_diag.COLOR_VARIABLE,
    )

    for variable in (
        plot_diag.X_VARIABLE,
        plot_diag.ADVECTION_VARIABLE,
        plot_diag.TEMPERATURE_CHANGE_VARIABLE,
        plot_diag.DIABATIC_VARIABLE,
        plot_diag.COLOR_VARIABLE,
    ):
        with pytest.raises(ValueError, match=variable):
            plot_diag.validate_feature_variables(
                features.drop_vars(variable),
                color_variable=plot_diag.COLOR_VARIABLE,
            )

    plot_diag.validate_feature_variables(
        features.drop_vars(plot_diag.COLOR_VARIABLE),
        color_variable=None,
    )


def test_nonfinite_required_values_are_excluded_from_relevant_panels():
    features = _make_feature_table()
    features["I_adiabatic_pre"] = (
        "event",
        np.array([1.0, np.nan, 4.0, 5.0, 7.0, 8.0]),
    )
    features["I_advection_pre"] = (
        "event",
        np.array([-1.0, 0.0, np.inf, 2.0, 3.0, 4.0]),
    )
    features["tas_anom_peak"] = (
        "event",
        np.array([2.0, 3.0, 4.0, np.nan, 5.0, 6.0]),
    )
    features["I_dTdt_pre"] = (
        "event",
        np.array([-3.0, -1.0, 0.0, 1.0, np.nan, 5.0]),
    )
    features["I_diabatic_pre"] = (
        "event",
        np.array([1.0, 2.0, 3.0, 4.0, 5.0, np.nan]),
    )
    expected_adiabatic_x = np.array([1.0, 7.0, 8.0])
    expected_temperature_x = np.array([0.0, 12.0])
    expected_diabatic_x = np.array([0.0, 10.0])

    fig = plot_diag.plot_tendency_scatter(features)
    try:
        for ax in fig.axes[:2]:
            offsets = np.asarray(ax.collections[0].get_offsets())
            np.testing.assert_allclose(offsets[:, 0], expected_adiabatic_x)
        temperature_offsets = np.asarray(fig.axes[2].collections[0].get_offsets())
        np.testing.assert_allclose(temperature_offsets[:, 0], expected_temperature_x)
        diabatic_offsets = np.asarray(fig.axes[3].collections[0].get_offsets())
        np.testing.assert_allclose(diabatic_offsets[:, 0], expected_diabatic_x)
    finally:
        plot_diag.plt.close(fig)


def test_main_writes_configured_output(monkeypatch, tmp_path):
    input_path = tmp_path / "features.nc"
    output_path = tmp_path / "adiabatic_advection_distance_comparison.png"
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
        written.append((Path(path), color_variable, point_size, alpha))
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_adiabatic_advection_comparison.py",
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
            output_path,
            "tas_anom_peak",
            10.0,
            0.6,
        ),
    ]


def _make_feature_table() -> xr.Dataset:
    event = np.arange(6)
    return xr.Dataset(
        data_vars={
            "I_adiabatic_pre": ("event", np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0])),
            "I_advection_pre": ("event", np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 4.0])),
            "I_dTdt_pre": ("event", np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0])),
            "I_diabatic_pre": ("event", np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0])),
            "tas_anom_peak": ("event", np.array([2.0, 3.0, 4.0, 4.5, 5.0, 6.0])),
        },
        coords={"event": event},
    )
