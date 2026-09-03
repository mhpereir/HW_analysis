import numpy as np
import pytest
import xarray as xr
from HW_analysis.src import selectors


def test_select_events_by_season_uses_peak_month_with_drop_true():
    event_table = _make_event_table()

    out = selectors.select_events_by_season(event_table, [6, 7, 8])

    np.testing.assert_array_equal(out["event_id"].values, [1, 2, 3])
    assert out.sizes["event"] == 3
    assert out.attrs["selection_type"] == "season"
    assert out.attrs["selection_months"] == "6,7,8"
    assert out.attrs["selection_time_name"] == "peak_time"
    assert out.attrs["selection_require_full_event"] == 0
    assert out.attrs["n_selected_events"] == 3


def test_select_events_by_season_drop_false_masks_only_event_variables():
    event_table = _make_event_table()

    out = selectors.select_events_by_season(event_table, [6], drop=False)

    assert out.sizes["event"] == event_table.sizes["event"]
    np.testing.assert_allclose(out["event_id"].values[:2], [1.0, 2.0])
    assert np.isnan(out["event_id"].values[2])
    assert np.isnat(out["peak_time"].values[2])
    np.testing.assert_allclose(out["T_mean"].values, event_table["T_mean"].values)
    assert out["T_mean"].dims == ("time",)


def test_select_events_by_season_full_interval_rejects_events_outside_season():
    event_table = _make_event_table()

    out = selectors.select_events_by_season(
        event_table,
        [6, 7, 8],
        require_full_event=True,
    )

    np.testing.assert_array_equal(out["event_id"].values, [1])
    assert out.attrs["selection_require_full_event"] == 1
    assert out.attrs["n_selected_events"] == 1


def test_select_events_by_season_full_interval_supports_djf_wraparound():
    event_table = _make_event_table()

    out = selectors.select_events_by_season(
        event_table,
        [12, 1, 2],
        require_full_event=True,
    )

    np.testing.assert_array_equal(out["event_id"].values, [4])


def test_select_events_by_season_rejects_invalid_months():
    event_table = _make_event_table()

    with pytest.raises(ValueError, match="between 1 and 12"):
        selectors.select_events_by_season(event_table, [0, 13])


def test_select_events_by_season_rejects_missing_time_variable():
    event_table = _make_event_table().drop_vars("peak_time")

    with pytest.raises(ValueError, match="missing time variable 'peak_time'"):
        selectors.select_events_by_season(event_table, [6, 7, 8])


def test_select_events_by_id_preserves_requested_order_and_other_dimensions():
    event_table = _make_event_table()

    out = selectors.select_events_by_id(event_table, np.array([3, 1]))

    np.testing.assert_array_equal(out["event_id"].values, [3, 1])
    np.testing.assert_array_equal(
        out["T_mean"].values,
        event_table["T_mean"].values,
    )
    assert out.attrs["selection_type"] == "event_id"
    assert out.attrs["selection_event_ids"] == "3,1"
    assert out.attrs["n_selected_events"] == 2


def test_select_events_by_id_rejects_missing_or_duplicate_ids():
    event_table = _make_event_table()

    with pytest.raises(ValueError, match="missing requested event IDs: 99"):
        selectors.select_events_by_id(event_table, [1, 99])
    with pytest.raises(ValueError, match="must be unique"):
        selectors.select_events_by_id(event_table, [1, 1])


def test_select_event_quantile_bin_supports_numeric_duration():
    event_table = _make_metric_event_table()

    out = selectors.select_event_quantile_bin(
        event_table,
        "duration",
        qmin=0.0,
        qmax=0.5,
        inclusive="left",
    )

    np.testing.assert_array_equal(out["event_id"].values, [1, 2])
    assert out.attrs["selection_lower_value"] == 1.0
    assert out.attrs["selection_upper_value"] == 2.5


def test_select_event_quantile_bin_supports_timedelta_duration_in_days():
    event_table = _make_metric_event_table(
        duration=np.array([1, 2, 3, 4], dtype="timedelta64[D]"),
    )

    out = selectors.select_event_quantile_bin(
        event_table,
        "duration",
        qmin=0.5,
        qmax=1.0,
        inclusive="both",
    )

    np.testing.assert_array_equal(out["event_id"].values, [3, 4])
    assert out.attrs["selection_lower_value"] == 2.5
    assert out.attrs["selection_upper_value"] == 4.0
    assert out.attrs["selection_metric_units"] == "days"


def test_select_events_by_metric_supports_timedelta_duration_day_bounds():
    event_table = _make_metric_event_table(
        duration=np.array([1, 2, 3, 4], dtype="timedelta64[D]"),
    )

    out = selectors.select_events_by_metric(
        event_table,
        "duration",
        min_value=2.0,
        max_value=3.0,
    )

    np.testing.assert_array_equal(out["event_id"].values, [2, 3])
    assert out.attrs["selection_metric_units"] == "days"


def test_select_event_quantile_bin_rejects_datetime_metric():
    event_table = _make_metric_event_table()

    with pytest.raises(TypeError, match="datetime64"):
        selectors.select_event_quantile_bin(
            event_table,
            "peak_time",
            qmin=0.0,
            qmax=0.5,
        )


def test_match_events_by_metric_sign_reproduces_single_variable_caliper():
    event_table = _make_matching_event_table()

    result = selectors.match_events_by_metric_sign(
        event_table,
        "I_dyn",
        match_variables=("severity",),
        caliper_sd=0.1,
    )

    assert result.pair_count == 3
    np.testing.assert_array_equal(result.negative_event_ids, [11, 12, 13])
    np.testing.assert_array_equal(result.positive_event_ids, [21, 22, 23])
    assert len(set(result.positive_indices)) == result.pair_count
    assert result.calipers_sd == {"severity": 0.1}
    assert result.method == selectors.SIGN_MATCH_METHOD


def test_match_events_by_metric_sign_applies_caliper_to_every_variable():
    event_table = _make_matching_event_table()

    result = selectors.match_events_by_metric_sign(
        event_table,
        event_table["I_dyn"],
        match_variables=("severity", "timing"),
        caliper_sd={"severity": 0.1, "timing": 0.1},
    )

    assert result.pair_count == 2
    np.testing.assert_array_equal(result.negative_event_ids, [11, 12])
    np.testing.assert_array_equal(result.positive_event_ids, [21, 22])


def test_match_events_by_metric_sign_maximizes_pair_count_before_distance():
    event_table = xr.Dataset(
        data_vars={
            "event_id": ("event", [11, 12, 21, 22]),
            "I_dyn": ("event", [-1.0, -1.0, 1.0, 1.0]),
            "severity": ("event", [0.0, 1.0, 0.4, -0.4]),
        },
        coords={"event": np.arange(4)},
    )

    result = selectors.match_events_by_metric_sign(
        event_table,
        "I_dyn",
        match_variables=("severity",),
        caliper_sd=1.0,
    )

    assert result.pair_count == 2
    np.testing.assert_array_equal(result.negative_event_ids, [11, 12])
    np.testing.assert_array_equal(result.positive_event_ids, [22, 21])


def test_match_events_by_metric_sign_is_invariant_to_event_row_order():
    event_table = _make_matching_event_table()
    shuffled = event_table.isel(event=[4, 2, 6, 0, 3, 1, 5])

    expected = selectors.match_events_by_metric_sign(
        event_table,
        "I_dyn",
        match_variables=("severity", "timing"),
        caliper_sd=0.1,
    )
    actual = selectors.match_events_by_metric_sign(
        shuffled,
        "I_dyn",
        match_variables=("severity", "timing"),
        caliper_sd=0.1,
    )

    np.testing.assert_array_equal(
        actual.negative_event_ids,
        expected.negative_event_ids,
    )
    np.testing.assert_array_equal(
        actual.positive_event_ids,
        expected.positive_event_ids,
    )
    np.testing.assert_allclose(actual.distances, expected.distances)


def test_match_events_by_metric_sign_converts_timedelta_matching_variable():
    event_table = _make_matching_event_table()
    event_table["duration"] = (
        "event",
        np.array([2, 3, 4, 2, 3, 4, 9], dtype="timedelta64[D]"),
    )

    result = selectors.match_events_by_metric_sign(
        event_table,
        "I_dyn",
        match_variables=("duration",),
        caliper_sd=0.1,
    )

    assert result.pair_count == 3
    np.testing.assert_array_equal(result.negative_event_ids, [11, 12, 13])
    np.testing.assert_array_equal(result.positive_event_ids, [21, 22, 23])


def test_match_events_by_metric_sign_rejects_incomplete_caliper_mapping():
    event_table = _make_matching_event_table()

    with pytest.raises(ValueError, match="exactly match"):
        selectors.match_events_by_metric_sign(
            event_table,
            "I_dyn",
            match_variables=("severity", "timing"),
            caliper_sd={"severity": 0.1},
        )


def _make_event_table() -> xr.Dataset:
    event = np.arange(5)
    time = np.array(
        ["2000-06-01T00:00", "2000-06-01T01:00", "2000-06-01T02:00"],
        dtype="datetime64[h]",
    )
    return xr.Dataset(
        data_vars={
            "event_id": ("event", np.array([1, 2, 3, 4, 5], dtype=np.int64)),
            "start_time": (
                "event",
                np.array(
                    [
                        "2000-06-15T00:00",
                        "2000-05-31T00:00",
                        "2000-08-31T00:00",
                        "2000-12-31T00:00",
                        "NaT",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            "end_time": (
                "event",
                np.array(
                    [
                        "2000-06-17T00:00",
                        "2000-06-02T00:00",
                        "2000-09-01T00:00",
                        "2001-01-02T00:00",
                        "NaT",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            "peak_time": (
                "event",
                np.array(
                    [
                        "2000-06-16T00:00",
                        "2000-06-01T00:00",
                        "2000-08-31T00:00",
                        "2001-01-01T00:00",
                        "NaT",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
            "tas_peak": ("event", np.array([31.0, 32.0, 33.0, 34.0, 35.0])),
            "T_mean": ("time", np.array([280.0, 281.0, 282.0])),
        },
        coords={"event": event, "time": time},
    )


def _make_metric_event_table(
    *,
    duration: np.ndarray | None = None,
) -> xr.Dataset:
    if duration is None:
        duration = np.array([1.0, 2.0, 3.0, 4.0])
    event = np.arange(4)
    return xr.Dataset(
        data_vars={
            "event_id": ("event", np.array([1, 2, 3, 4], dtype=np.int64)),
            "duration": ("event", duration),
            "peak_time": (
                "event",
                np.array(
                    [
                        "2000-06-01T00:00",
                        "2000-06-02T00:00",
                        "2000-06-03T00:00",
                        "2000-06-04T00:00",
                    ],
                    dtype="datetime64[ns]",
                ),
            ),
        },
        coords={"event": event},
    )


def _make_matching_event_table() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "event_id": (
                "event",
                np.array([11, 12, 13, 21, 22, 23, 24], dtype=np.int64),
            ),
            "I_dyn": ("event", np.array([-1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0])),
            "severity": (
                "event",
                np.array([1.0, 2.0, 3.0, 1.01, 2.01, 3.01, 9.0]),
            ),
            "timing": (
                "event",
                np.array([0.0, 0.0, 10.0, 0.01, 0.02, 0.0, 50.0]),
            ),
        },
        coords={"event": np.arange(7)},
    )
