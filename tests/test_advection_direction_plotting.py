import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from HW_analysis.src import advection_direction, advection_direction_plotting


def test_plot_advection_direction_exploration_has_four_panels_and_daily_glyphs():
    composite = _make_composite()

    fig = advection_direction_plotting.plot_advection_direction_exploration(
        composite,
        ratio_epsilon=0.005,
    )
    try:
        assert len(fig.axes) == 4
        assert len(fig.axes[3].patches) == 4
        assert len(fig.axes[3].collections) == 4
        assert "Signed face contributions" == fig.axes[0].get_title()
        assert "Component ratios" in fig.axes[2].get_title()
        assert {
            line.get_label() for line in fig.axes[2].lines
        } >= {
            "Meridional / zonal",
            "Horizontal / vertical",
        }
        assert "not airflow direction" in fig.axes[3].get_title()
        assert {text.get_text() for text in fig.axes[3].texts} >= {
            "W",
            "E",
            "S",
            "N",
            "T",
        }
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
