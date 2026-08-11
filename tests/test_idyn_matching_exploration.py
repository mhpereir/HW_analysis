from pathlib import Path

import numpy as np
import xarray as xr

from HW_analysis.scripts.Idyn_matching_exploration import (
    explore_idyn_matching as exploration,
)
from HW_analysis.scripts.Idyn_matching_exploration import matching_settings


def test_prepare_exploration_builds_deterministic_complete_primary_match():
    features = make_event_features()

    settings = matching_settings.load_matching_settings()
    result = exploration.prepare_exploration(features, settings=settings)

    assert result.negative_indices.size == 4
    assert result.positive_indices.size == 5
    assert result.primary_match.pair_count == 4
    assert set(result.event_ids[result.primary_match.negative_indices]) == {
        11,
        12,
        13,
        14,
    }
    assert len(set(result.primary_match.positive_indices)) == 4
    assert result.balance["tas_anom_peak"]["smd_after"] < 0.05


def test_prepare_exploration_calls_reusable_selector(monkeypatch):
    settings = matching_settings.load_matching_settings()
    selector = exploration.selectors.match_events_by_metric_sign
    calls = []

    def recording_selector(*args, **kwargs):
        calls.append((args, kwargs))
        return selector(*args, **kwargs)

    monkeypatch.setattr(
        exploration.selectors,
        "match_events_by_metric_sign",
        recording_selector,
    )

    exploration.prepare_exploration(make_event_features(), settings=settings)

    expected_calls = len(settings.specifications) + (
        len(settings.frontier_families) * len(settings.frontier_calipers_sd)
    )
    assert len(calls) == expected_calls


def test_proposed_match_and_tradeoff_specs_are_prepared():
    settings = matching_settings.load_matching_settings()
    result = exploration.prepare_exploration(
        make_event_features(),
        settings=settings,
    )

    assert set(result.frontier_matches) == set(settings.frontier_families)
    assert set(result.frontier_matches["integrated_warming_antecedent"]) == set(
        settings.frontier_calipers_sd
    )
    score = exploration.balance_score(
        result,
        result.specification_matches["integrated_warming_antecedent_0p20"],
    )
    assert score["matched_pairs"] > 0
    assert 0 <= score["variables_improved"] <= len(settings.balance_variables)


def test_main_writes_four_nonempty_figures(monkeypatch, tmp_path):
    input_path = tmp_path / "features.nc"
    output_dir = tmp_path / "plots"
    make_event_features().to_netcdf(input_path, engine="h5netcdf")
    monkeypatch.setattr(
        "sys.argv",
        [
            "explore_idyn_matching.py",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert exploration.main() == 0

    for path in exploration.output_paths(output_dir).values():
        assert path.is_file()
        assert path.stat().st_size > 0


def make_event_features() -> xr.Dataset:
    event_ids = np.array([14, 12, 11, 13, 25, 23, 21, 24, 22])
    negative_anomaly = {
        11: 2.50,
        12: 3.00,
        13: 3.50,
        14: 4.00,
    }
    positive_anomaly = {
        21: 2.51,
        22: 3.01,
        23: 3.51,
        24: 4.01,
        25: 6.00,
    }
    anomaly = np.array(
        [
            negative_anomaly.get(event_id, positive_anomaly.get(event_id))
            for event_id in event_ids
        ]
    )
    is_negative = event_ids < 20
    i_dyn = np.where(is_negative, -1.0, 1.0)
    duration_days = np.where(event_ids % 2 == 0, 3, 2)

    return xr.Dataset(
        data_vars={
            "event_id": ("event", event_ids),
            "I_advection_pre": ("event", i_dyn * 0.4),
            "I_adiabatic_pre": ("event", i_dyn * 0.6),
            "I_dTdt_pre": ("event", anomaly - 1.0),
            "tas_anom_peak": ("event", anomaly),
            "tas_peak": ("event", 288.0 + anomaly),
            "tas_excess_peak": ("event", anomaly - 2.0),
            "tas_excess_integral": ("event", (anomaly - 2.0) * duration_days),
            "duration": (
                "event",
                duration_days.astype("timedelta64[D]").astype("timedelta64[ns]"),
            ),
            "days_from_solstice": ("event", (event_ids % 10).astype(float)),
            "T_anom_mean_ant": ("event", anomaly * 0.2),
        },
        coords={"event": np.arange(event_ids.size)},
    )
