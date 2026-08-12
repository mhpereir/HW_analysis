from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts import (
    plot_advection_direction_exploration_matched_clim_anom as matched_script,
)
from HW_analysis.scripts.Idyn_matching_exploration import matching_settings


def test_build_matched_composites_uses_selector_and_stage1_event_alignment(
    monkeypatch,
):
    stage1, anomaly_source, features = _make_inputs()
    settings = matching_settings.load_matching_settings()
    original = matched_script.selectors.match_events_by_metric_sign
    calls = []

    def recording_selector(*args, **kwargs):
        calls.append((args, kwargs))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        matched_script.selectors,
        "match_events_by_metric_sign",
        recording_selector,
    )

    prepared = matched_script.build_matched_composites(
        stage1,
        anomaly_source,
        features,
        settings=settings,
        specification_id="peak_anomaly_0p20",
        window_days=1,
    )

    assert len(calls) == 1
    assert calls[0][0][1] == "I_dyn_pre"
    assert calls[0][1]["match_variables"] == ("tas_anom_peak",)
    assert calls[0][1]["caliper_sd"] == 0.2
    assert prepared.match.pair_count == 4
    assert prepared.negative.attrs["matched_sign"] == "negative"
    assert prepared.positive.attrs["matched_sign"] == "positive"
    assert prepared.negative.attrs["n_events"] == 4
    assert prepared.positive.attrs["n_events"] == 4
    assert prepared.negative.attrs["matching_specification"] == (
        "peak_anomaly_0p20"
    )
    assert prepared.negative.attrs["matching_label"] == "Peak anomaly"


def test_build_matched_composites_rejects_peak_time_mismatch():
    stage1, anomaly_source, features = _make_inputs()
    features["peak_time"] = features["peak_time"] + np.timedelta64(1, "h")

    with pytest.raises(ValueError, match="peak times do not align"):
        matched_script.build_matched_composites(
            stage1,
            anomaly_source,
            features,
            settings=matching_settings.load_matching_settings(),
            specification_id="peak_anomaly_0p20",
            window_days=1,
        )


def test_open_event_features_requires_stage2_contract_marker(tmp_path):
    path = tmp_path / "not-stage2.nc"
    xr.Dataset({"event_id": ("event", [1])}).to_netcdf(
        path,
        engine="h5netcdf",
    )

    with pytest.raises(ValueError, match="pipeline_stage='stage_2_event_features'"):
        matched_script.open_event_features(path)


def test_matched_script_reads_canonical_idyn_without_reconstructing_components():
    source = Path(matched_script.__file__).read_text()

    assert "settings.group_variable" in source
    assert "I_adiabatic_pre" not in source
    assert "I_advection_pre" not in source


def _make_inputs() -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]:
    feature_ids = np.array([14, 12, 11, 13, 25, 23, 21, 24, 22])
    stage1_ids = np.array([25, 11, 22, 14, 21, 13, 24, 12, 23])
    peak_lookup = {
        event_id: np.datetime64("2000-06-03T00:00", "ns")
        + np.timedelta64(index, "D")
        for index, event_id in enumerate(sorted(feature_ids))
    }
    negative_anomaly = {11: 2.5, 12: 3.0, 13: 3.5, 14: 4.0}
    positive_anomaly = {21: 2.51, 22: 3.01, 23: 3.51, 24: 4.01, 25: 8.0}
    anomaly = np.array(
        [
            negative_anomaly.get(event_id, positive_anomaly.get(event_id))
            for event_id in feature_ids
        ]
    )
    features = xr.Dataset(
        data_vars={
            "event_id": ("event", feature_ids),
            "peak_time": (
                "event",
                np.array([peak_lookup[event_id] for event_id in feature_ids]),
            ),
            "I_dyn_pre": (
                "event",
                np.where(feature_ids < 20, -1.0, 1.0),
            ),
            "tas_anom_peak": ("event", anomaly),
        },
        coords={"event": np.arange(feature_ids.size)},
        attrs={"pipeline_stage": "stage_2_event_features"},
    )

    time = np.arange(
        np.datetime64("2000-06-01T00:00", "h"),
        np.datetime64("2000-06-15T01:00", "h"),
        np.timedelta64(1, "h"),
    ).astype("datetime64[ns]")
    phase = np.linspace(-np.pi, np.pi, time.size)
    face_values = {
        "advection_west": 0.04 + 0.02 * np.cos(phase),
        "advection_east": -0.02 + 0.01 * np.sin(phase),
        "advection_south": 0.01 + 0.01 * np.cos(phase),
        "advection_north": -0.015 + 0.005 * np.sin(phase),
        "advection_top": 0.005 * np.cos(phase),
    }
    total = sum(face_values.values())
    stage1 = xr.Dataset(
        data_vars={
            "event_id": ("event", stage1_ids),
            "peak_time": (
                "event",
                np.array([peak_lookup[event_id] for event_id in stage1_ids]),
            ),
            "advection": ("time", total),
            **{
                name: ("time", values)
                for name, values in face_values.items()
            },
        },
        coords={
            "event": np.arange(stage1_ids.size),
            "time": time,
        },
        attrs={"region": "pnw_bartusek"},
    )
    anomaly_source = stage1.copy(deep=True)
    anomaly_source.attrs["data_representation"] = "climatological_anomaly"
    return stage1, anomaly_source, features
