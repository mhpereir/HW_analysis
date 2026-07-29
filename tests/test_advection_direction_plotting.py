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
        assert {
            text.get_text() for text in fig.axes[0].get_legend().get_texts()
        } == {"West", "East", "South", "North", "Top"}
        assert {
            text.get_text() for text in fig.axes[1].get_legend().get_texts()
        } == {
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
            upper
            + advection_direction_plotting.LEGEND_HEADROOM_FRACTION * span,
        )
    finally:
        plt.close(fig)


def test_write_advection_direction_exploration_plot_writes_nonempty_png(tmp_path):
    output = tmp_path / "advection_face_contributions.png"

    written = (
        advection_direction_plotting.write_advection_direction_exploration_plot(
            _make_composite(),
            output,
        )
    )

    assert written == output.resolve()
    assert written.stat().st_size > 0


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
