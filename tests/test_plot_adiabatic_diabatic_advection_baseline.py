from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.event_features import (
    plot_adiabatic_diabatic_advection_baseline as plot_diag,
)


def test_plot_creates_three_axes_with_background_and_foreground_collections():
    fig = plot_diag.plot_tendency_scatter(
        _make_baseline_table(),
        _make_event_table(),
    )
    try:
        assert len(fig.axes) == 3
        assert [ax.get_title() for ax in fig.axes] == [
            "Diabatic vs Adiabatic",
            "Advection vs Adiabatic",
            "Sqrt LWA a Exposure",
        ]
        for ax in fig.axes:
            assert len(ax.collections) == 2
            baseline_scatter, event_scatter = ax.collections
            assert baseline_scatter.get_alpha() == 0.2
            assert event_scatter.get_alpha() == 0.9
            assert baseline_scatter.get_zorder() < event_scatter.get_zorder()
            assert baseline_scatter.get_label() == "Clean baseline days"
            assert event_scatter.get_label() == "Events"
        assert fig.axes[0].get_legend() is not None
        assert [
            text.get_text() for text in fig.axes[0].get_legend().get_texts()
        ] == ["Clean baseline days", "Events"]
    finally:
        plot_diag.plt.close(fig)


def test_plot_filters_baseline_only_and_uses_matching_lwa_exposures():
    fig = plot_diag.plot_tendency_scatter(
        _make_baseline_table(),
        _make_event_table(),
    )
    try:
        expected_baseline_x = np.array([1.0, 4.0, 5.0, 8.0])
        expected_event_x = np.array([-3.0, 6.0, 10.0])
        expected_baseline_y = (
            np.array([2.0, 5.0, 6.0, 9.0]),
            np.array([-2.0, 0.0, 2.0, 4.0]),
            np.array([1.0, 3.0, 4.0, 6.0]),
        )
        expected_event_y = (
            np.array([1.0, 7.0, 11.0]),
            np.array([3.0, -6.0, -9.0]),
            np.array([2.0, 5.0, 7.0]),
        )

        for ax, baseline_y, event_y in zip(
            fig.axes,
            expected_baseline_y,
            expected_event_y,
        ):
            baseline_offsets = np.asarray(ax.collections[0].get_offsets())
            event_offsets = np.asarray(ax.collections[1].get_offsets())
            np.testing.assert_allclose(baseline_offsets[:, 0], expected_baseline_x)
            np.testing.assert_allclose(baseline_offsets[:, 1], baseline_y)
            np.testing.assert_allclose(event_offsets[:, 0], expected_event_x)
            np.testing.assert_allclose(event_offsets[:, 1], event_y)
            assert ax.texts[0].get_text() == "baseline n = 4\nevents n = 3"
    finally:
        plot_diag.plt.close(fig)


def test_feature_values_square_roots_both_integrated_lwa_sources():
    baseline = xr.Dataset(
        {"I_lwa_a_pre_reference": ("baseline_day", np.array([0.0, 4.0, -1.0]))}
    )
    events = xr.Dataset({"I_lwa_a_pre_peak": ("event", np.array([9.0, 16.0, -1.0]))})

    baseline_values = plot_diag.feature_values(
        baseline,
        "sqrt_I_lwa_a_pre_reference",
    )
    event_values = plot_diag.feature_values(events, "sqrt_I_lwa_a_pre_peak")

    np.testing.assert_allclose(baseline_values[:2], np.array([0.0, 2.0]))
    np.testing.assert_allclose(event_values[:2], np.array([3.0, 4.0]))
    assert np.isnan(baseline_values[2])
    assert np.isnan(event_values[2])


def test_nonfinite_values_are_filtered_independently_and_limits_use_both_tables():
    baseline = _make_baseline_table()
    events = _make_event_table()
    baseline["I_diabatic_pre"] = (
        "baseline_day",
        np.array([2.0, 3.0, np.nan, 6.0, 8.0, 9.0]),
    )
    events["I_advection_pre"] = ("event", np.array([3.0, np.nan, -9.0]))

    fig = plot_diag.plot_tendency_scatter(baseline, events)
    try:
        diabatic_baseline = np.asarray(fig.axes[0].collections[0].get_offsets())
        diabatic_events = np.asarray(fig.axes[0].collections[1].get_offsets())
        advection_baseline = np.asarray(fig.axes[1].collections[0].get_offsets())
        advection_events = np.asarray(fig.axes[1].collections[1].get_offsets())
        np.testing.assert_allclose(diabatic_baseline[:, 0], np.array([1.0, 5.0, 8.0]))
        np.testing.assert_allclose(diabatic_events[:, 0], np.array([-3.0, 6.0, 10.0]))
        np.testing.assert_allclose(
            advection_baseline[:, 0],
            np.array([1.0, 4.0, 5.0, 8.0]),
        )
        np.testing.assert_allclose(advection_events[:, 0], np.array([-3.0, 10.0]))
        for ax in fig.axes:
            np.testing.assert_allclose(ax.get_xlim(), np.array([-3.0, 10.0]))
    finally:
        plot_diag.plt.close(fig)


def test_reference_lines_use_combined_populations():
    baseline = _make_baseline_table()
    events = _make_event_table()
    events["I_diabatic_pre"] = ("event", np.array([1.0, 7.0, 25.0]))

    fig = plot_diag.plot_tendency_scatter(baseline, events)
    try:
        one_to_one = [
            line for line in fig.axes[0].lines if line.get_gid() == "one_to_one"
        ]
        assert len(one_to_one) == 1
        assert max(one_to_one[0].get_xdata()) == 25.0
        np.testing.assert_allclose(
            one_to_one[0].get_xdata(),
            one_to_one[0].get_ydata(),
        )

        negative_one = [
            line
            for line in fig.axes[1].lines
            if line.get_gid() == "one_to_negative_one"
        ]
        assert len(negative_one) == 1
        np.testing.assert_allclose(
            negative_one[0].get_xdata(),
            -negative_one[0].get_ydata(),
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
    with pytest.raises(ValueError, match="Baseline-day.*I_lwa_a_pre_reference"):
        plot_diag.validate_feature_variables(
            baseline.drop_vars("I_lwa_a_pre_reference"),
            events,
        )
    with pytest.raises(ValueError, match="Event-feature.*I_lwa_a_pre_peak"):
        plot_diag.validate_feature_variables(
            baseline,
            events.drop_vars("I_lwa_a_pre_peak"),
        )


def test_plot_rejects_no_clean_baseline_rows_or_empty_event_table():
    baseline = _make_baseline_table()
    baseline["event_adjacent"] = (
        "baseline_day",
        np.ones(baseline.sizes["baseline_day"], dtype=np.int8),
    )
    with pytest.raises(ValueError, match="no clean rows"):
        plot_diag.plot_tendency_scatter(baseline, _make_event_table())

    empty_events = _make_event_table().isel(event=slice(0, 0))
    with pytest.raises(ValueError, match="no event rows"):
        plot_diag.plot_tendency_scatter(_make_baseline_table(), empty_events)


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
        point_size,
        alpha,
        event_point_size,
        event_alpha,
    ):
        assert baseline_features is baseline
        assert event_features is events
        written.append(
            (Path(path), point_size, alpha, event_point_size, event_alpha)
        )
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_adiabatic_diabatic_advection_baseline.py",
            "--input-path",
            str(baseline_path),
            "--event-input-path",
            str(event_path),
            "--output-path",
            str(output_path),
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
    assert written == [(output_path, 10.0, 0.1, 50.0, 0.8)]
    assert closed == ["events", "baseline"]


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
            "I_diabatic_pre": (
                "baseline_day",
                np.array([2.0, 3.0, 5.0, 6.0, 8.0, 9.0]),
            ),
            "I_advection_pre": (
                "baseline_day",
                np.array([-2.0, -1.0, 0.0, 2.0, 3.0, 4.0]),
            ),
            "I_lwa_a_pre_reference": (
                "baseline_day",
                np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0]),
            ),
        },
        coords={"baseline_day": np.arange(6)},
    )


def _make_event_table() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "I_adiabatic_pre": ("event", np.array([-3.0, 6.0, 10.0])),
            "I_diabatic_pre": ("event", np.array([1.0, 7.0, 11.0])),
            "I_advection_pre": ("event", np.array([3.0, -6.0, -9.0])),
            "I_lwa_a_pre_peak": ("event", np.array([4.0, 25.0, 49.0])),
        },
        coords={"event": np.arange(3)},
    )
