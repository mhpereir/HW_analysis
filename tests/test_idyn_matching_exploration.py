from pathlib import Path

import numpy as np
import xarray as xr

from HW_analysis.scripts.Idyn_matching_exploration import (
    explore_idyn_matching as exploration,
)


def test_prepare_exploration_builds_deterministic_complete_primary_match():
    features = make_event_features()

    result = exploration.prepare_exploration(features, caliper=0.2)

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


def test_optimal_match_uses_caliper_and_leaves_reference_event_unmatched():
    event_ids = np.array([1, 2, 3, 4])
    i_dyn = np.array([-1.0, -1.0, 1.0, 1.0])
    values = {"tas_anom_peak": np.array([0.0, 10.0, 0.01, 1.0])}

    result = exploration.optimal_sign_match(
        event_ids,
        i_dyn,
        values,
        match_variables=("tas_anom_peak",),
        caliper=0.05,
    )

    assert result.pair_count == 1
    np.testing.assert_array_equal(result.negative_indices, [0])
    np.testing.assert_array_equal(result.positive_indices, [2])


def test_main_writes_three_nonempty_figures(monkeypatch, tmp_path):
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
            "tas_anom_peak": ("event", anomaly),
            "tas_peak": ("event", 288.0 + anomaly),
            "tas_excess_peak": ("event", anomaly - 2.0),
            "tas_excess_integral": ("event", (anomaly - 2.0) * duration_days),
            "duration": (
                "event",
                duration_days.astype("timedelta64[D]").astype("timedelta64[ns]"),
            ),
            "days_from_solstice": ("event", np.arange(-4, 5, dtype=float)),
            "T_anom_mean_ant": ("event", anomaly * 0.2),
        },
        coords={"event": np.arange(event_ids.size)},
    )
