import numpy as np
import xarray as xr
from HW_analysis.scripts.spatial_composites import (
    plot_dyn_net_spatial_composites as plotter,
)


def test_plot_spatial_composites_creates_sign_by_lag_geoaxes():
    fig = plotter.plot_spatial_composites(_composite_dataset())
    try:
        map_axes = fig.axes[:6]
        assert len(map_axes) == 6
        assert map_axes[0].get_title() == "$t-2$"
        assert map_axes[1].get_title() == "$t$ (peak)"
        assert map_axes[2].get_title() == "$t+2$"
        assert map_axes[3].get_title() == ""
        assert all(len(ax.collections) > 1 for ax in map_axes)
        assert len(fig.axes) == 7  # six maps plus shared colorbar
    finally:
        plotter.plt.close(fig)


def test_plot_spatial_composites_accepts_explicit_lags():
    fig = plotter.plot_spatial_composites(
        _composite_dataset(),
        plot_lags=(-3, -1, 0, 1, 3),
    )
    try:
        assert len(fig.axes) == 11  # ten maps plus shared colorbar
        assert [axis.get_title() for axis in fig.axes[:5]] == [
            "$t-3$",
            "$t-1$",
            "$t$ (peak)",
            "$t+1$",
            "$t+3$",
        ]
    finally:
        plotter.plt.close(fig)


def test_plot_spatial_composites_rejects_missing_lag():
    with np.testing.assert_raises_regex(
        ValueError,
        "absent from the composite",
    ):
        plotter.plot_spatial_composites(
            _composite_dataset(),
            plot_lags=(-2, 0, 4),
        )


def test_write_figure_creates_png(tmp_path):
    fig = plotter.plt.figure()
    output_path = tmp_path / "figures" / "composite.png"
    try:
        written = plotter.write_figure(fig, output_path)
    finally:
        plotter.plt.close(fig)

    assert written == output_path.resolve()
    assert written.stat().st_size > 0


def test_rounded_symmetric_limit_uses_requested_step():
    assert plotter.rounded_symmetric_limit(np.array([-1.1, 0.4]), step=0.5) == 1.5


def _composite_dataset() -> xr.Dataset:
    latitude = np.linspace(10.0, 80.0, 8)
    longitude = np.linspace(-170.0, -40.0, 14)
    lags = np.arange(-3, 4)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    temperature = 2.0 * np.sin(np.deg2rad(lon_grid + 110.0))
    height = 80.0 * np.cos(np.deg2rad(lat_grid - 45.0)) - 60.0
    lag_scale = (1.0 + 0.1 * lags)[:, None, None]
    return xr.Dataset(
        {
            "t2m_anomaly": (
                ("dyn_sign", "lag", "latitude", "longitude"),
                np.stack([lag_scale * temperature, -lag_scale * temperature]),
            ),
            "z500_anomaly": (
                ("dyn_sign", "lag", "latitude", "longitude"),
                np.stack([lag_scale * height, -lag_scale * height]),
            ),
            "event_count": ("dyn_sign", [141, 117]),
            "I_dyn_net_mean": ("dyn_sign", [2.0, -1.5]),
        },
        coords={
            "dyn_sign": ["positive", "negative"],
            "lag": lags,
            "latitude": latitude,
            "longitude": longitude,
        },
    )
