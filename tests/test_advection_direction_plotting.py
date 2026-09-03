from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from HW_analysis.src import advection_direction, advection_direction_plotting


def test_plot_advection_direction_exploration_has_selected_two_panels():
    composite = _make_composite()

    fig = advection_direction_plotting.plot_advection_direction_exploration(composite)
    try:
        assert len(fig.axes) == 2
        assert "Signed face contributions" == fig.axes[0].get_title()
        assert "Grouped advective contributions" == fig.axes[1].get_title()
        assert all("Component ratios" not in ax.get_title() for ax in fig.axes)
        assert all("glyph" not in ax.get_title().lower() for ax in fig.axes)
        assert {text.get_text() for text in fig.axes[0].get_legend().get_texts()} == {
            "West",
            "East",
            "South",
            "North",
            "Top",
        }
        assert {text.get_text() for text in fig.axes[1].get_legend().get_texts()} == {
            "Zonal (west + east)",
            "Meridional (south + north)",
            "Horizontal",
            "Vertical",
            "All faces",
        }
        assert fig.axes[1].get_legend()._ncols == 5
        assert [ax.get_ylabel() for ax in fig.axes] == ["K hr-1", "K hr-1"]
        assert fig.axes[1].get_xlabel() == "Days relative to event peak"
        visible_ticks = fig.axes[1].get_xticks()
        lower, upper = fig.axes[1].get_xlim()
        visible_ticks = visible_ticks[
            (visible_ticks >= lower) & (visible_ticks <= upper)
        ]
        np.testing.assert_array_equal(visible_ticks, np.arange(-2, 3))
        assert all(float(tick).is_integer() for tick in visible_ticks)
        advection_direction_plotting.plot_style.format_numeric_axes(fig)
        formatter = fig.axes[1].xaxis.get_major_formatter()
        assert [formatter(tick) for tick in visible_ticks] == [
            "−2",
            "−1",
            "0",
            "1",
            "2",
        ]
    finally:
        plt.close(fig)


def test_add_upper_axis_headroom_expands_only_upper_limit():
    fig, ax = plt.subplots()
    try:
        ax.plot([0, 1], [-2.0, 3.0])
        lower, upper = ax.get_ylim()
        span = upper - lower

        advection_direction_plotting._add_upper_axis_headroom(ax)

        new_lower, new_upper = ax.get_ylim()
        assert new_lower == lower
        assert np.isclose(
            new_upper,
            upper + advection_direction_plotting.LEGEND_HEADROOM_FRACTION * span,
        )
    finally:
        plt.close(fig)


def test_climatological_anomaly_title_is_explicit():
    composite = _make_composite()
    composite.attrs["data_representation"] = "climatological_anomaly"

    fig = advection_direction_plotting.plot_advection_direction_exploration(composite)
    try:
        assert "advection climatological-anomaly composite" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_write_advection_direction_exploration_plot_writes_nonempty_png(tmp_path):
    output = tmp_path / "advection_face_contributions.png"

    written = advection_direction_plotting.write_advection_direction_exploration_plot(
        _make_composite(),
        output,
    )

    assert written == output.resolve()
    assert written.stat().st_size > 0


def test_dual_output_writer_preserves_raw_and_smooths_faces_before_groups(
    monkeypatch,
    tmp_path,
):
    composite = _make_composite()
    original = composite.copy(deep=True)
    captured = []

    def fake_write(ds, output_path):
        captured.append(ds)
        return Path(output_path).resolve()

    monkeypatch.setattr(
        advection_direction_plotting,
        "write_advection_direction_exploration_plot",
        fake_write,
    )
    output = tmp_path / "advection.png"
    smoothed_output = tmp_path / "advection_smoothed.png"

    written = (
        advection_direction_plotting.write_advection_direction_exploration_outputs(
            composite,
            output,
            smoothed_output_path=smoothed_output,
            smoothing_window=24,
        )
    )

    assert written == [output.resolve(), smoothed_output.resolve()]
    assert captured[0] is composite
    smoothed = captured[1]
    assert smoothed.attrs["smoothing_window"] == 24
    expected_west = (
        composite["advection_west"]
        .rolling(
            lag_hour=24,
            center=True,
            min_periods=24,
        )
        .mean()
    )
    xr.testing.assert_allclose(smoothed["advection_west"], expected_west)
    grouped = advection_direction.grouped_advection_components(smoothed)
    xr.testing.assert_allclose(
        grouped["advection_face_total"],
        sum(
            smoothed[f"advection_{face}"]
            for face in ("west", "east", "south", "north", "top")
        ),
    )
    xr.testing.assert_identical(composite, original)


def test_smoothed_advection_plot_title_identifies_running_mean():
    composite = _make_composite()
    composite.attrs["smoothing_window"] = 24

    fig = advection_direction_plotting.plot_advection_direction_exploration(composite)
    try:
        assert "24-hour running mean" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def test_write_advection_direction_outputs_writes_two_nonempty_pngs(tmp_path):
    output = tmp_path / "advection.png"
    smoothed_output = tmp_path / "advection_smoothed.png"

    written = (
        advection_direction_plotting.write_advection_direction_exploration_outputs(
            _make_composite(),
            output,
            smoothed_output_path=smoothed_output,
            smoothing_window=24,
        )
    )

    assert written == [output.resolve(), smoothed_output.resolve()]
    assert all(path.stat().st_size > 0 for path in written)


def test_matched_plot_uses_positive_solid_and_negative_dashed_lines():
    negative, positive = _make_matched_composites()

    fig = advection_direction_plotting.plot_matched_advection_direction_exploration(
        negative,
        positive,
    )
    try:
        assert len(fig.axes) == 2
        for ax in fig.axes:
            matched_lines = {
                line.get_gid(): line
                for line in ax.lines
                if str(line.get_gid()).startswith("matched_")
            }
            assert len(matched_lines) == 10
            assert all(
                line.get_linestyle() == "-"
                for gid, line in matched_lines.items()
                if gid.startswith("matched_positive_")
            )
            assert all(
                line.get_linestyle() == "--"
                for gid, line in matched_lines.items()
                if gid.startswith("matched_negative_")
            )
        assert [ax.get_ylabel() for ax in fig.axes] == [
            "Δ [K hr-1]",
            "Δ [K hr-1]",
        ]
        title = fig._suptitle.get_text()
        assert "n=3 pairs" in title
        assert "0.20 pooled SD" in title
        for ax in fig.axes:
            legend_labels = {text.get_text() for text in ax.get_legend().get_texts()}
            assert r"Positive $I_{\mathrm{dyn,pre}}$" in legend_labels
            assert r"Negative $I_{\mathrm{dyn,pre}}$" in legend_labels
    finally:
        plt.close(fig)


def test_matched_plot_rejects_different_event_counts():
    negative, positive = _make_matched_composites()
    positive.attrs["n_events"] = 2

    with np.testing.assert_raises_regex(ValueError, "same positive event count"):
        advection_direction_plotting.plot_matched_advection_direction_exploration(
            negative,
            positive,
        )


def test_write_matched_advection_plot_writes_nonempty_png(tmp_path):
    negative, positive = _make_matched_composites()
    output = tmp_path / "matched_advection.png"

    written = (
        advection_direction_plotting.write_matched_advection_direction_exploration_plot(
            negative,
            positive,
            output,
        )
    )

    assert written == output.resolve()
    assert written.stat().st_size > 0


def test_matched_dual_output_writer_smooths_populations_independently(
    monkeypatch,
    tmp_path,
):
    negative, positive = _make_matched_composites()
    negative_original = negative.copy(deep=True)
    positive_original = positive.copy(deep=True)
    captured = []

    def fake_write(negative_ds, positive_ds, output_path):
        captured.append((negative_ds, positive_ds))
        return Path(output_path).resolve()

    monkeypatch.setattr(
        advection_direction_plotting,
        "write_matched_advection_direction_exploration_plot",
        fake_write,
    )
    output = tmp_path / "matched.png"
    smoothed_output = tmp_path / "matched_smoothed.png"

    written = advection_direction_plotting.write_matched_advection_direction_exploration_outputs(
        negative,
        positive,
        output,
        smoothed_output_path=smoothed_output,
        smoothing_window=24,
    )

    assert written == [output.resolve(), smoothed_output.resolve()]
    assert captured[0][0] is negative
    assert captured[0][1] is positive
    negative_smoothed, positive_smoothed = captured[1]
    assert negative_smoothed.attrs["smoothing_window"] == 24
    assert positive_smoothed.attrs["smoothing_window"] == 24
    expected_negative = (
        negative["advection_west"]
        .rolling(
            lag_hour=24,
            center=True,
            min_periods=24,
        )
        .mean()
    )
    expected_positive = (
        positive["advection_west"]
        .rolling(
            lag_hour=24,
            center=True,
            min_periods=24,
        )
        .mean()
    )
    xr.testing.assert_allclose(
        negative_smoothed["advection_west"],
        expected_negative,
    )
    xr.testing.assert_allclose(
        positive_smoothed["advection_west"],
        expected_positive,
    )
    xr.testing.assert_identical(negative, negative_original)
    xr.testing.assert_identical(positive, positive_original)

    fig = advection_direction_plotting.plot_matched_advection_direction_exploration(
        negative_smoothed,
        positive_smoothed,
    )
    try:
        assert "24-hour running mean" in fig._suptitle.get_text()
    finally:
        plt.close(fig)


def _make_composite() -> xr.Dataset:
    lag = np.arange(-48, 49)
    phase = np.linspace(-np.pi, np.pi, lag.size)
    face_values = {
        "west": 0.04 + 0.02 * np.cos(phase),
        "east": -0.02 + 0.01 * np.sin(phase),
        "south": 0.01 + 0.01 * np.cos(phase),
        "north": -0.015 + 0.005 * np.sin(phase),
        "top": 0.005 * np.cos(phase),
    }
    data_vars = {
        advection_direction.stage1_face_variable(face): (
            "lag_hour",
            values,
        )
        for face, values in face_values.items()
    }
    total = sum(face_values.values())
    data_vars["advection"] = ("lag_hour", total)
    return xr.Dataset(
        data_vars,
        coords={"lag_hour": lag},
        attrs={
            "region": "pnw_bartusek",
            "n_events": 12,
            "pre_days": 2,
            "post_days": 2,
        },
    )


def _make_matched_composites() -> tuple[xr.Dataset, xr.Dataset]:
    negative = _make_composite()
    positive = _make_composite()
    for name in positive.data_vars:
        positive[name] = positive[name] + 0.01
    shared_attrs = {
        "data_representation": "climatological_anomaly",
        "matching_label": "Peak anomaly",
        "matching_variables": "tas_anom_peak",
        "matching_caliper_sd": 0.2,
        "n_events": 3,
    }
    negative.attrs.update(shared_attrs, matched_sign="negative")
    positive.attrs.update(shared_attrs, matched_sign="positive")
    return negative, positive
