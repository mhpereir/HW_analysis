import numpy as np
import xarray as xr

from HW_analysis.scripts.spatial_composites import (
    plot_dyn_net_spatial_composites as plotter,
)


def test_plot_spatial_composites_creates_stacked_geoaxes():
    fig = plotter.plot_spatial_composites(_composite_dataset())
    try:
        map_axes = fig.axes[:2]
        assert len(map_axes) == 2
        assert map_axes[0].get_title().startswith("Positive")
        assert map_axes[1].get_title().startswith("Negative")
        assert all(len(ax.collections) > 1 for ax in map_axes)
        assert len(fig.axes) == 3  # two maps plus shared colorbar
    finally:
        plotter.plt.close(fig)


def test_write_figure_creates_png(tmp_path):
    fig = plotter.plot_spatial_composites(_composite_dataset())
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
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    temperature = 2.0 * np.sin(np.deg2rad(lon_grid + 110.0))
    height = 80.0 * np.cos(np.deg2rad(lat_grid - 45.0)) - 60.0
    return xr.Dataset(
        {
            "t2m_anomaly": (
                ("dyn_sign", "latitude", "longitude"),
                np.stack([temperature, -temperature]),
            ),
            "z500_anomaly": (
                ("dyn_sign", "latitude", "longitude"),
                np.stack([height, -height]),
            ),
            "event_count": ("dyn_sign", [141, 117]),
            "I_dyn_net_mean": ("dyn_sign", [2.0, -1.5]),
        },
        coords={
            "dyn_sign": ["positive", "negative"],
            "latitude": latitude,
            "longitude": longitude,
        },
    )
