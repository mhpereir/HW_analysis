"""Regression checks for shared presentation-plot styling."""

import numpy as np
import pytest
from HW_analysis.src import plot_style


def test_day_lag_axis_keeps_fractional_major_ticks_for_short_windows():
    fig, ax = plot_style.plt.subplots()
    try:
        ax.plot([-1 / 24, 0, 1 / 24], [1, 2, 1])
        plot_style.format_day_lag_axis(ax.xaxis)
        plot_style.format_numeric_axes(fig)
        fig.canvas.draw()
        ticks = ax.get_xticks()
        visible = ticks[(ticks >= ax.get_xlim()[0]) & (ticks <= ax.get_xlim()[1])]
        assert len(visible) >= 3
        assert 0 in visible
        assert np.any(visible != np.round(visible))
        minor_ticks = ax.get_xticks(minor=True)
        np.testing.assert_array_equal(minor_ticks, np.round(minor_ticks))
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        assert len(set(labels)) == len(labels)
    finally:
        plot_style.plt.close(fig)


@pytest.mark.parametrize("max_intervals", [0, -1, 2.5, True])
def test_major_tick_limit_rejects_invalid_counts(max_intervals):
    fig, ax = plot_style.plt.subplots()
    try:
        with pytest.raises(ValueError, match="positive integer"):
            plot_style.limit_major_ticks(ax.xaxis, max_intervals=max_intervals)
    finally:
        plot_style.plt.close(fig)


def test_major_tick_limit_preserves_shared_decimal_formatter():
    fig, ax = plot_style.plt.subplots()
    try:
        ax.set_xlim(-27.8, 11.8)
        plot_style.limit_major_ticks(ax.xaxis, max_intervals=4)
        plot_style.format_numeric_axes(fig)
        fig.canvas.draw()
        ticks = ax.get_xticks()
        visible = ticks[(ticks >= ax.get_xlim()[0]) & (ticks <= ax.get_xlim()[1])]
        assert 2 <= visible.size <= 5
        assert isinstance(
            ax.xaxis.get_major_formatter(), plot_style.FixedDecimalScaleFormatter
        )
        assert ax.xaxis.get_major_formatter()(-20.0) == "-20.00"
    finally:
        plot_style.plt.close(fig)


def test_event_legend_handle_preserves_marker_size_and_opacity():
    handle = plot_style.event_severity_legend_handle(point_size=24.0, alpha=0.7)
    assert handle.get_markersize() == pytest.approx(np.sqrt(24.0))
    assert handle.get_alpha() == 0.7
    assert handle.get_label() == "Events"


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("scale", [1.0, 1e5])
def test_sum_reference_guides_are_bounded_clipped_and_preserve_axes(invert, scale):
    fig, ax = plot_style.plt.subplots()
    try:
        xlim = np.array([-28.0, 10.0]) * scale
        ylim = np.array([-5.0, 20.0]) * scale
        ax.set_xlim(xlim[::-1] if invert else xlim)
        ax.set_ylim(ylim[::-1] if invert else ylim)
        before = (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_autoscalex_on(),
            ax.get_autoscaley_on(),
        )
        plot_style.add_sum_reference_lines(ax)
        assert (
            ax.get_xlim(),
            ax.get_ylim(),
            ax.get_autoscalex_on(),
            ax.get_autoscaley_on(),
        ) == before
        assert 1 <= len(ax.lines) <= 17
        for line in ax.lines:
            x, y = line.get_data()
            level = (x + y)[0]
            np.testing.assert_allclose(x + y, level)
            assert level / 5 == pytest.approx(round(level / 5))
            assert np.all((x >= xlim[0]) & (x <= xlim[1]))
            assert np.all((y >= ylim[0]) & (y <= ylim[1]))
            assert line.get_zorder() == 0
            assert line.get_linestyle() == ":"
            assert line.get_label() == f"reference {level:g} K"
        assert {text.get_text() for text in ax.texts} == {
            line.get_label().removeprefix("reference ") for line in ax.lines
        }
    finally:
        plot_style.plt.close(fig)


@pytest.mark.parametrize("spacing", [0, -5, np.nan, np.inf])
def test_sum_reference_guides_reject_invalid_spacing(spacing):
    fig, ax = plot_style.plt.subplots()
    try:
        with pytest.raises(ValueError, match="spacing"):
            plot_style.add_sum_reference_lines(ax, spacing=spacing)
        assert not ax.lines and not ax.texts
    finally:
        plot_style.plt.close(fig)


@pytest.mark.parametrize("max_lines", [0, 1, True, 2.5])
def test_sum_reference_guides_reject_invalid_line_limit(max_lines):
    fig, ax = plot_style.plt.subplots()
    try:
        with pytest.raises(ValueError, match="max_lines"):
            plot_style.add_sum_reference_lines(ax, max_lines=max_lines)
    finally:
        plot_style.plt.close(fig)


def test_sum_reference_guides_require_linear_axes():
    fig, ax = plot_style.plt.subplots()
    try:
        ax.set_xscale("log")
        with pytest.raises(ValueError, match="linear"):
            plot_style.add_sum_reference_lines(ax)
    finally:
        plot_style.plt.close(fig)
