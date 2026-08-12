from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from HW_analysis.scripts.Idyn_matching_exploration import matching_settings
from HW_analysis.scripts.spatial_composites import (
    build_matched_dyn_pre_spatial_composites as matched_builder,
)


def test_prepare_matched_events_uses_selector_and_records_pair_audit(monkeypatch):
    features = _event_features()
    settings = matching_settings.load_matching_settings()
    original = matched_builder.selectors.match_events_by_metric_sign
    calls = []

    def recording_selector(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        matched_builder.selectors,
        "match_events_by_metric_sign",
        recording_selector,
    )
    selected = matched_builder.prepare_matched_events(
        features,
        settings=settings,
        specification_id="peak_anomaly_0p20",
    )

    assert len(calls) == 1
    assert calls[0][0][1] == "I_dyn_pre"
    assert calls[0][1]["match_variables"] == ("tas_anom_peak",)
    assert calls[0][1]["caliper_sd"] == 0.2
    assert selected.match.pair_count == 4
    np.testing.assert_array_equal(
        selected.events["event_dyn_sign"],
        ["positive"] * 4 + ["negative"] * 4,
    )
    np.testing.assert_array_equal(
        selected.events["matched_pair_id"],
        [0, 1, 2, 3, 0, 1, 2, 3],
    )
    np.testing.assert_array_equal(selected.events["event"], np.arange(8))
    np.testing.assert_allclose(
        selected.events["matched_pair_distance"][:4],
        selected.events["matched_pair_distance"][4:],
    )
    assert selected.events.attrs["matching_specification"] == (
        "peak_anomaly_0p20"
    )
    assert selected.events.attrs["matching_group_variable"] == "I_dyn_pre"
    assert selected.events.attrs["matching_source_negative_count"] == 4
    assert selected.events.attrs["matching_source_positive_count"] == 5
    assert selected.events.attrs["matching_unmatched_positive_count"] == 1


def test_build_matched_spatial_composites_writes_separate_contract(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    climatology_path = tmp_path / "climatology.nc"
    _write_spatial_file(
        daily_dir / "ERA5_daily_t2m_z500_2000.nc",
        year=2000,
        t2m_offset=2.0,
        z_height_offset=50.0,
    )
    _write_spatial_file(climatology_path, year=2004)

    output, selected = matched_builder.build_matched_spatial_composites(
        _event_features(),
        settings=matching_settings.load_matching_settings(),
        specification_id="peak_anomaly_0p20",
        daily_dir=daily_dir,
        climatology_path=climatology_path,
        lat_bounds=(10.0, 80.0),
        lon_bounds=(-170.0, -40.0),
        daily_lags=(-1, 0, 1),
        event_features_path=tmp_path / "features.nc",
        event_features_sha256="abc123",
    )

    assert selected.match.pair_count == 4
    assert output.attrs["pipeline_stage"] == (
        "daily_matched_idyn_spatial_composites"
    )
    assert output.attrs["matching_specification"] == "peak_anomaly_0p20"
    assert output.attrs["matching_caliper_sd"] == 0.2
    assert output.attrs["matching_pair_count"] == 4
    assert output.attrs["matching_source_negative_count"] == 4
    assert output.attrs["matching_source_positive_count"] == 5
    assert output.attrs["matching_unmatched_negative_count"] == 0
    assert output.attrs["matching_unmatched_positive_count"] == 1
    assert output.attrs["event_features_sha256"] == "abc123"
    np.testing.assert_array_equal(output["event_count"], [4, 4])
    np.testing.assert_array_equal(output["lag"], [-1, 0, 1])
    np.testing.assert_allclose(output["I_dyn_pre_mean"], [1.0, -1.0])
    np.testing.assert_allclose(
        output["I_dyn_net_mean"],
        output["I_dyn_pre_mean"],
    )
    np.testing.assert_allclose(output["t2m_anomaly"], 2.0, atol=1e-10)
    np.testing.assert_allclose(output["z500_anomaly"], 50.0, atol=1e-10)
    assert output.sizes["event"] == 8
    assert "matched_pair_id" in output
    assert "matched_pair_distance" in output


def test_open_event_features_requires_stage2_marker(tmp_path):
    path = tmp_path / "features.nc"
    xr.Dataset({"event_id": ("event", [1])}).to_netcdf(
        path,
        engine="h5netcdf",
    )

    with pytest.raises(ValueError, match="pipeline_stage='stage_2_event_features'"):
        matched_builder.open_event_features(path)


def test_matched_builder_reads_canonical_idyn_without_reconstructing_components():
    source = Path(matched_builder.__file__).read_text()

    assert "settings.group_variable" in source
    assert "I_adiabatic_pre" not in source
    assert "I_advection_pre" not in source


def _event_features() -> xr.Dataset:
    event_ids = np.array([14, 12, 11, 13, 25, 23, 21, 24, 22])
    peak_lookup = {
        event_id: np.datetime64("2000-06-10T00:00", "ns")
        + np.timedelta64(index * 4, "D")
        for index, event_id in enumerate(sorted(event_ids))
    }
    negative_anomaly = {11: 2.5, 12: 3.0, 13: 3.5, 14: 4.0}
    positive_anomaly = {21: 2.51, 22: 3.01, 23: 3.51, 24: 4.01, 25: 8.0}
    return xr.Dataset(
        {
            "event_id": ("event", event_ids),
            "peak_time": (
                "event",
                np.array([peak_lookup[event_id] for event_id in event_ids]),
            ),
            "I_dyn_pre": (
                "event",
                np.where(event_ids < 20, -1.0, 1.0),
            ),
            "tas_anom_peak": (
                "event",
                np.array(
                    [
                        negative_anomaly.get(
                            event_id,
                            positive_anomaly.get(event_id),
                        )
                        for event_id in event_ids
                    ]
                ),
            ),
        },
        coords={"event": np.arange(event_ids.size)},
        attrs={"pipeline_stage": "stage_2_event_features"},
    )


def _write_spatial_file(
    path: Path,
    *,
    year: int,
    t2m_offset: float = 0.0,
    z_height_offset: float = 0.0,
) -> None:
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    latitude = np.array([90.0, 80.0, 20.0, 10.0, 0.0])
    longitude = np.array([0.0, 190.0, 200.0, 320.0, 330.0])
    doy = time.dayofyear.to_numpy(dtype=float)
    base_t2m = 250.0 + doy[:, None, None]
    base_height = 5000.0 + doy[:, None, None]
    shape = (time.size, latitude.size, longitude.size)
    t2m = np.broadcast_to(base_t2m + t2m_offset, shape).copy()
    z = np.broadcast_to(
        (
            base_height
            + z_height_offset
        )
        * matched_builder.spatial_builder.GEOPOTENTIAL_TO_HEIGHT_M_S2,
        shape,
    ).copy()
    xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), t2m),
            "z": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                z[:, None, :, :],
            ),
        },
        coords={
            "valid_time": time,
            "pressure_level": [500.0],
            "latitude": latitude,
            "longitude": longitude,
        },
    ).to_netcdf(path, engine="h5netcdf")
