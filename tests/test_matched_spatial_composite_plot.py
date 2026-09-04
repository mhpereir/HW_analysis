import numpy as np
import pytest
import xarray as xr
from HW_analysis.scripts.spatial_composites import (
    plot_dyn_net_spatial_composites as plotter,
)


def test_plot_matched_spatial_composites_labels_pair_design():
    figure = plotter.plot_matched_spatial_composites(
        _matched_composite_dataset(),
        expected_specification="peak_anomaly_0p20",
    )
    try:
        assert len(figure.axes) == 7
        assert figure._suptitle is not None
        title = figure._suptitle.get_text()
        assert "Matched Heatwave Spatial Composite" in title
        assert "Peak anomaly, 0.20 pooled SD (n = 4 pairs)" in title
        row_labels = [text.get_text() for text in figure.axes[0].texts]
        row_labels += [text.get_text() for text in figure.axes[3].texts]
        assert any("Positive $I_{dyn,pre}$" in text for text in row_labels)
        assert any("Negative $I_{dyn,pre}$" in text for text in row_labels)
        assert sum("n = 4" in text for text in row_labels) == 2
    finally:
        plotter.plt.close(figure)


def test_validate_matched_composite_rejects_unequal_counts():
    dataset = _matched_composite_dataset()
    dataset["event_count"] = ("dyn_sign", [4, 3])

    with pytest.raises(ValueError, match="sign counts"):
        plotter.validate_matched_composite(dataset)


def test_validate_matched_composite_rejects_wrong_specification():
    with pytest.raises(ValueError, match="Expected matching specification"):
        plotter.validate_matched_composite(
            _matched_composite_dataset(),
            expected_specification="integrated_warming_antecedent_0p20",
        )


def test_validate_matched_composite_rejects_broken_pair_audit():
    dataset = _matched_composite_dataset()
    dataset["matched_pair_id"] = ("event", np.arange(8))

    with pytest.raises(ValueError, match="exactly twice"):
        plotter.validate_matched_composite(dataset)


def _matched_composite_dataset() -> xr.Dataset:
    latitude = np.linspace(10.0, 80.0, 8)
    longitude = np.linspace(-170.0, -40.0, 14)
    lags = np.arange(-3, 4)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    temperature = 2.0 * np.sin(np.deg2rad(lon_grid + 110.0))
    height = 80.0 * np.cos(np.deg2rad(lat_grid - 45.0)) - 60.0
    lag_scale = (1.0 + 0.1 * lags)[:, None, None]
    pair_distances = np.array([0.01, 0.02, 0.03, 0.04])
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
            "event_count": ("dyn_sign", [4, 4]),
            "I_dyn_net_mean": ("dyn_sign", [2.0, -1.5]),
            "I_dyn_pre_mean": ("dyn_sign", [2.0, -1.5]),
            "event_id": ("event", np.arange(101, 109)),
            "I_dyn_pre": ("event", [1.0] * 4 + [-1.0] * 4),
            "event_dyn_sign": (
                "event",
                ["positive"] * 4 + ["negative"] * 4,
            ),
            "matched_pair_id": ("event", [0, 1, 2, 3, 0, 1, 2, 3]),
            "matched_pair_distance": (
                "event",
                np.tile(pair_distances, 2),
            ),
        },
        coords={
            "dyn_sign": ["positive", "negative"],
            "lag": lags,
            "latitude": latitude,
            "longitude": longitude,
            "event": np.arange(8),
        },
        attrs={
            "pipeline_stage": "daily_matched_idyn_spatial_composites",
            "matching_group_variable": "I_dyn_pre",
            "matching_specification": "peak_anomaly_0p20",
            "matching_label": "Peak anomaly",
            "matching_variables": "tas_anom_peak",
            "matching_caliper_sd": 0.2,
            "matching_pair_count": 4,
            "matching_source_negative_count": 4,
            "matching_source_positive_count": 5,
            "matching_source_zero_count": 0,
            "matching_unmatched_negative_count": 0,
            "matching_unmatched_positive_count": 1,
            "matching_settings_path": "/settings.json",
            "matching_settings_sha256": "settings-sha",
            "event_features_path": "/features.nc",
            "event_features_sha256": "features-sha",
        },
    )
