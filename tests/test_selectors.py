import numpy as np
import pytest
import xarray as xr

from src import selectors


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
