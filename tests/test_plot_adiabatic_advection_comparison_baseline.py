from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from HW_analysis.scripts.event_features import (
    plot_adiabatic_advection_comparison_baseline as plot_diag,
)


def test_presentation_layout_retains_first_and_fourth_panels_in_one_column():
    baseline = _make_baseline_table()
    events = _make_event_table()

    fig = plot_diag.plot_tendency_scatter(
        baseline,
        events,
        layout=plot_diag.PRESENTATION_LAYOUT,
    )
    try:
        assert len(fig.axes) == 3
        plot_axes = fig.axes[:2]
        assert [ax.get_title() for ax in plot_axes] == [
            "Advection vs Adiabatic Heating",
            r"Diabatic Heating vs $I_{dyn,net}$",
        ]
        assert [ax.get_xlabel() for ax in plot_axes] == [
            plot_diag.variable_label(plot_diag.X_VARIABLE),
            plot_diag.NET_DYNAMICAL_LABEL,
        ]
        np.testing.assert_allclose(fig.get_size_inches(), np.array([7.5, 9.0]))
        assert plot_axes[0].get_legend() is not None
        assert plot_axes[1].get_legend() is None
        assert fig.axes[-1].get_ylabel() == "Peak TAS Anomaly (K)"
        _assert_offsets(
            plot_axes[0].collections[0],
            np.array([1.0, 4.0, 5.0, 8.0]),
            np.array([-2.0, 0.0, 2.0, 4.0]),
        )
        _assert_offsets(
            plot_axes[0].collections[1],
            np.array([-3.0, 6.0, 10.0]),
            np.array([3.0, -6.0, -9.0]),
        )
        _assert_offsets(
            plot_axes[1].collections[0],
            np.array([-1.0, 4.0, 7.0, 12.0]),
            np.array([2.0, 5.0, 6.0, 9.0]),
        )
        _assert_offsets(
            plot_axes[1].collections[1],
            np.array([0.0, 0.0, 1.0]),
            np.array([1.0, 7.0, 11.0]),
        )
    finally:
        plot_diag.plt.close(fig)


def test_layout_defaults_use_distinct_output_paths_and_reject_unknown_layouts():
    assert plot_diag.default_output_path(plot_diag.FULL_LAYOUT) == (
        plot_diag.DEFAULT_OUTPUT_PATH
    )
    assert plot_diag.default_output_path(plot_diag.PRESENTATION_LAYOUT) == (
        plot_diag.DEFAULT_PRESENTATION_OUTPUT_PATH
    )
    assert plot_diag.DEFAULT_PRESENTATION_OUTPUT_PATH != plot_diag.DEFAULT_OUTPUT_PATH
    with pytest.raises(ValueError, match="layout must be one of"):
        plot_diag.plot_tendency_scatter(
            _make_baseline_table(),
            _make_event_table(),
            layout="unknown",
        )


def test_plot_creates_four_panels_with_baseline_and_colored_event_layers():
    fig = plot_diag.plot_tendency_scatter(
        _make_baseline_table(),
        _make_event_table(),
    )
    try:
        np.testing.assert_allclose(fig.get_size_inches(), np.array([12.0, 6.6]))
        assert len(fig.axes) == 5
        plot_axes = fig.axes[:4]
        assert [ax.get_title() for ax in plot_axes] == [
            "Advection vs Adiabatic Heating",
            "Net Dynamical Contribution",
            r"Integrated dT/dt vs $I_{dyn,net}$",
            r"Diabatic Heating vs $I_{dyn,net}$",
        ]
        for ax in plot_axes:
            assert len(ax.collections) == 2
            baseline_scatter, event_scatter = ax.collections
            assert baseline_scatter.get_alpha() == 0.2
            assert event_scatter.get_alpha() == 0.9
            assert baseline_scatter.get_zorder() < event_scatter.get_zorder()
            assert baseline_scatter.get_label() == "Clean baseline days"
            assert event_scatter.get_label() == "Events"
            assert ax.texts[0].get_text() == "baseline n = 4\nevents n = 3"
        assert fig.axes[0].get_legend() is not None
        assert [text.get_text() for text in fig.axes[0].get_legend().get_texts()] == [
            "Clean baseline days",
            "Events",
        ]
        assert fig.axes[0].get_legend()._loc == 1
        assert fig.axes[1].get_legend() is None
        assert fig.axes[-1].get_ylabel() == "Peak TAS Anomaly (K)"
        event_norms = [ax.collections[1].norm for ax in plot_axes]
        assert all(norm is event_norms[0] for norm in event_norms)
        for ax in plot_axes:
            np.testing.assert_allclose(
                np.asarray(ax.collections[1].get_array()),
                np.array([2.0, 4.0, 6.0]),
            )
    finally:
        plot_diag.plt.close(fig)


def test_panels_use_expected_baseline_and_event_values():
    baseline = _make_baseline_table()
    events = _make_event_table()

    fig = plot_diag.plot_tendency_scatter(baseline, events)
    try:
        baseline_x = np.array([1.0, 4.0, 5.0, 8.0])
        baseline_advection = np.array([-2.0, 0.0, 2.0, 4.0])
        baseline_net = np.array([-1.0, 4.0, 7.0, 12.0])
        baseline_temperature_change = np.array([-3.0, 0.0, 1.0, 5.0])
        baseline_diabatic = np.array([2.0, 5.0, 6.0, 9.0])

        event_x = np.array([-3.0, 6.0, 10.0])
        event_advection = np.array([3.0, -6.0, -9.0])
        event_net = np.array([0.0, 0.0, 1.0])
        event_temperature_change = np.array([-1.0, 2.0, 4.0])
        event_diabatic = np.array([1.0, 7.0, 11.0])

        _assert_offsets(fig.axes[0].collections[0], baseline_x, baseline_advection)
        _assert_offsets(fig.axes[0].collections[1], event_x, event_advection)

        _assert_offsets(fig.axes[1].collections[0], baseline_x, baseline_net)
        _assert_offsets(fig.axes[1].collections[1], event_x, event_net)

        _assert_offsets(
            fig.axes[2].collections[0], baseline_net, baseline_temperature_change
        )
        _assert_offsets(fig.axes[2].collections[1], event_net, event_temperature_change)

        _assert_offsets(fig.axes[3].collections[0], baseline_net, baseline_diabatic)
        _assert_offsets(fig.axes[3].collections[1], event_net, event_diabatic)
    finally:
        plot_diag.plt.close(fig)


def test_tendency_values_consume_stored_i_dyn_pre_without_reconstructing_it():
    features = _make_event_table()
    features["I_dyn_pre"] = ("event", np.array([10.0, 20.0, 30.0]))

    values = plot_diag.tendency_values(features)

    np.testing.assert_allclose(values["net_dynamical"], [10.0, 20.0, 30.0])


def test_nonfinite_values_are_filtered_per_layer_and_limits_use_plotted_points():
    baseline = _make_baseline_table()
    events = _make_event_table()
    baseline["I_advection_pre"] = (
        "baseline_day",
        np.array([-2.0, -1.0, np.nan, 2.0, 3.0, 4.0]),
    )
    events["I_adiabatic_pre"] = ("event", np.array([-20.0, 6.0, 10.0]))
    events["I_advection_pre"] = ("event", np.array([5.0, np.nan, -9.0]))
    events["I_dTdt_pre"] = ("event", np.array([0.0, 2.0, np.nan]))
    events["I_diabatic_pre"] = ("event", np.array([50.0, 7.0, np.nan]))
    events["tas_anom_peak"] = ("event", np.array([2.0, np.nan, 6.0]))

    fig = plot_diag.plot_tendency_scatter(baseline, events)
    try:
        _assert_offsets(
            fig.axes[0].collections[0],
            np.array([1.0, 5.0, 8.0]),
            np.array([-2.0, 2.0, 4.0]),
        )
        _assert_offsets(
            fig.axes[0].collections[1],
            np.array([-20.0, 10.0]),
            np.array([5.0, -9.0]),
        )
        assert fig.axes[0].texts[0].get_text() == "baseline n = 3\nevents n = 2"

        _assert_offsets(
            fig.axes[2].collections[1],
            np.array([0.0]),
            np.array([0.0]),
        )
        _assert_offsets(
            fig.axes[3].collections[1],
            np.array([0.0]),
            np.array([50.0]),
        )
        for ax in fig.axes[:2]:
            np.testing.assert_allclose(ax.get_xlim(), np.array([-21.5, 11.5]))
        for ax in fig.axes[2:4]:
            np.testing.assert_allclose(ax.get_xlim(), np.array([-1.65, 12.65]))
        np.testing.assert_allclose(fig.axes[0].get_ylim(), np.array([-9.7, 5.7]))
        np.testing.assert_allclose(fig.axes[1].get_ylim(), np.array([-1.65, 12.65]))
        np.testing.assert_allclose(fig.axes[2].get_ylim(), np.array([-3.4, 5.4]))
        np.testing.assert_allclose(fig.axes[3].get_ylim(), np.array([-2.5, 52.5]))
        for ax, expected_colors in zip(
            fig.axes[:4],
            (
                np.array([2.0, 6.0]),
                np.array([2.0, 6.0]),
                np.array([2.0]),
                np.array([2.0]),
            ),
        ):
            np.testing.assert_allclose(
                np.asarray(ax.collections[1].get_array()),
                expected_colors,
            )
        for ax in fig.axes[:4]:
            assert ax.get_xlim()[0] < 0.0 < ax.get_xlim()[1]
            assert ax.get_ylim()[0] < 0.0 < ax.get_ylim()[1]
    finally:
        plot_diag.plt.close(fig)


@pytest.mark.parametrize(
    ("values", "required_values", "expected"),
    [
        (np.array([np.nan, -2.0, 8.0]), (), (-2.5, 8.5)),
        (np.array([4.0, 4.0]), (), (3.8, 4.2)),
        (np.array([0.0]), (), (-0.05, 0.05)),
        (np.array([2.0, 8.0]), (0.0,), (-0.4, 8.4)),
        (np.array([-8.0, -2.0]), (0.0,), (-8.4, 0.4)),
    ],
)
def test_padded_data_limits(values, required_values, expected):
    np.testing.assert_allclose(
        plot_diag.plot_style.padded_data_limits(
            values,
            required_values=required_values,
        ),
        expected,
    )


def test_finite_range_color_norm_handles_constant_and_nonfinite_values():
    norm = plot_diag.plot_style.finite_range_color_norm(np.array([4.0, np.nan, 4.0]))

    assert norm is not None
    assert norm.vmin < 4.0 < norm.vmax
    assert plot_diag.plot_style.finite_range_color_norm(np.array([np.nan])) is None


def test_negative_one_reference_line_uses_combined_plotted_populations():
    fig = plot_diag.plot_tendency_scatter(
        _make_baseline_table(),
        _make_event_table(),
    )
    try:
        negative_one = [
            line
            for line in fig.axes[0].lines
            if line.get_gid() == "one_to_negative_one"
        ]
        assert len(negative_one) == 1
        np.testing.assert_allclose(negative_one[0].get_xdata(), np.array([-4.0, 10.0]))
        np.testing.assert_allclose(negative_one[0].get_ydata(), np.array([4.0, -10.0]))
        assert all(
            not any(line.get_gid() == "one_to_negative_one" for line in ax.lines)
            for ax in fig.axes[1:]
        )
    finally:
        plot_diag.plt.close(fig)


def test_validate_feature_variables_checks_both_tables():
    baseline = _make_baseline_table()
    events = _make_event_table()

    plot_diag.validate_feature_variables(baseline, events)

    with pytest.raises(ValueError, match="Baseline-day.*event_adjacent"):
        plot_diag.validate_feature_variables(
            baseline.drop_vars("event_adjacent"),
            events,
        )
    with pytest.raises(ValueError, match="Event-feature.*I_dTdt_pre"):
        plot_diag.validate_feature_variables(
            baseline,
            events.drop_vars("I_dTdt_pre"),
        )
    with pytest.raises(ValueError, match="Event-feature.*tas_anom_peak"):
        plot_diag.validate_feature_variables(
            baseline,
            events.drop_vars("tas_anom_peak"),
        )
    with pytest.raises(ValueError, match="Baseline-day.*I_dyn_pre"):
        plot_diag.validate_feature_variables(
            baseline.drop_vars("I_dyn_pre"),
            events,
        )


def test_plot_rejects_no_clean_baseline_rows_or_empty_event_table():
    baseline = _make_baseline_table()
    baseline["event_adjacent"] = (
        "baseline_day",
        np.ones(baseline.sizes["baseline_day"], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="no clean rows"):
        plot_diag.plot_tendency_scatter(baseline, _make_event_table())

    with pytest.raises(ValueError, match="no event rows"):
        plot_diag.plot_tendency_scatter(
            _make_baseline_table(),
            _make_event_table().isel(event=slice(0, 0)),
        )

    events = _make_event_table()
    events["tas_anom_peak"] = ("event", np.full(3, np.nan))
    with pytest.raises(ValueError, match="contains no finite values"):
        plot_diag.plot_tendency_scatter(_make_baseline_table(), events)


def test_write_tendency_scatter_plot_writes_png(tmp_path):
    output_path = tmp_path / "diagnostics" / "combined.png"

    written = plot_diag.write_tendency_scatter_plot(
        _make_baseline_table(),
        _make_event_table(),
        output_path,
    )

    assert written == output_path.resolve()
    assert written.is_file()
    assert written.stat().st_size > 0


def test_main_forwards_arguments_and_closes_both_datasets(monkeypatch, tmp_path):
    baseline_path = tmp_path / "baseline.nc"
    event_path = tmp_path / "events.nc"
    output_path = tmp_path / "combined.png"
    closed = []
    written = []

    class Closable:
        def __init__(self, label):
            self.label = label

        def close(self):
            closed.append(self.label)

    baseline = Closable("baseline")
    events = Closable("events")

    def fake_open_baseline(path):
        assert path == baseline_path
        return baseline

    def fake_open_events(path):
        assert path == event_path
        return events

    def fake_write(
        baseline_features,
        event_features,
        path,
        *,
        layout,
        color_variable,
        point_size,
        alpha,
        event_point_size,
        event_alpha,
    ):
        assert baseline_features is baseline
        assert event_features is events
        written.append(
            (
                Path(path),
                layout,
                color_variable,
                point_size,
                alpha,
                event_point_size,
                event_alpha,
            )
        )
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_adiabatic_advection_comparison_baseline.py",
            "--input-path",
            str(baseline_path),
            "--event-input-path",
            str(event_path),
            "--output-path",
            str(output_path),
            "--color-variable",
            "tas_anom_peak",
            "--layout",
            "presentation",
            "--point-size",
            "10",
            "--alpha",
            "0.1",
            "--event-point-size",
            "50",
            "--event-alpha",
            "0.8",
        ],
    )
    monkeypatch.setattr(plot_diag, "open_baseline_features", fake_open_baseline)
    monkeypatch.setattr(plot_diag, "open_event_features", fake_open_events)
    monkeypatch.setattr(plot_diag, "write_tendency_scatter_plot", fake_write)

    assert plot_diag.main() == 0
    assert written == [
        (
            output_path,
            "presentation",
            "tas_anom_peak",
            10.0,
            0.1,
            50.0,
            0.8,
        )
    ]
    assert closed == ["events", "baseline"]


def _assert_offsets(collection, x_values, y_values):
    offsets = np.asarray(collection.get_offsets(), dtype=float)
    np.testing.assert_allclose(offsets[:, 0], x_values)
    np.testing.assert_allclose(offsets[:, 1], y_values)


def _make_baseline_table() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "event_adjacent": (
                "baseline_day",
                np.array([0, 1, 0, 0, 1, 0], dtype=np.int8),
            ),
            "I_adiabatic_pre": (
                "baseline_day",
                np.array([1.0, 2.0, 4.0, 5.0, 7.0, 8.0]),
            ),
            "I_advection_pre": (
                "baseline_day",
                np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 4.0]),
            ),
            "I_dyn_pre": (
                "baseline_day",
                np.array([-1.0, 1.0, 4.0, 7.0, 10.0, 12.0]),
            ),
            "I_dTdt_pre": (
                "baseline_day",
                np.array([-3.0, -1.0, 0.0, 1.0, 3.0, 5.0]),
            ),
            "I_diabatic_pre": (
                "baseline_day",
                np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0]),
            ),
        },
        coords={"baseline_day": np.arange(6)},
    )


def _make_event_table() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "I_adiabatic_pre": ("event", np.array([-3.0, 6.0, 10.0])),
            "I_advection_pre": ("event", np.array([3.0, -6.0, -9.0])),
            "I_dyn_pre": ("event", np.array([0.0, 0.0, 1.0])),
            "I_dTdt_pre": ("event", np.array([-1.0, 2.0, 4.0])),
            "I_diabatic_pre": ("event", np.array([1.0, 7.0, 11.0])),
            "tas_anom_peak": ("event", np.array([2.0, 4.0, 6.0])),
        },
        coords={"event": np.arange(3)},
    )
