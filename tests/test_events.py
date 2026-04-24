import numpy as np
import xarray as xr
import pytest

from src import events


def test_mask_to_event_ids_labels_contiguous_true_runs():
    mask = xr.DataArray(
        [False, True, True, False, True],
        dims=("time",),
        coords={"time": np.arange(5)},
    )

    event_ids = events.mask_to_event_ids(mask)

    np.testing.assert_array_equal(event_ids.values, [0, 1, 1, 0, 2])


def test_mask_to_event_ids_filters_short_events_and_renumbers():
    mask = xr.DataArray(
        [True, False, True, True, False, True],
        dims=("time",),
        coords={"time": np.arange(6)},
    )

    event_ids = events.mask_to_event_ids(mask, min_duration=2)

    np.testing.assert_array_equal(event_ids.values, [0, 0, 1, 1, 0, 0])


def test_mask_to_event_ids_all_false_returns_zeros():
    mask = xr.DataArray(
        [False, False, False],
        dims=("time",),
        coords={"time": np.arange(3)},
    )

    event_ids = events.mask_to_event_ids(mask)

    np.testing.assert_array_equal(event_ids.values, [0, 0, 0])


def test_mask_to_event_ids_treats_missing_values_as_non_event_days():
    mask = xr.DataArray(
        [True, np.nan, True],
        dims=("time",),
        coords={"time": np.arange(3)},
    )

    event_ids = events.mask_to_event_ids(mask)

    np.testing.assert_array_equal(event_ids.values, [1, 0, 2])


def test_mask_to_event_ids_preserves_time_coordinate():
    times = np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]")
    mask = xr.DataArray([True, False], dims=("time",), coords={"time": times})

    event_ids = events.mask_to_event_ids(mask)

    np.testing.assert_array_equal(event_ids["time"].values, times)


def test_mask_to_event_ids_applies_custom_name():
    mask = xr.DataArray([True], dims=("time",), coords={"time": [0]})

    event_ids = events.mask_to_event_ids(mask, name="hw_event_id")

    assert event_ids.name == "hw_event_id"


def test_mask_to_event_ids_raises_for_missing_time_dim():
    mask = xr.DataArray([True], dims=("day",), coords={"day": [0]})

    with pytest.raises(ValueError, match="missing required time dimension"):
        events.mask_to_event_ids(mask)


def test_mask_to_event_ids_raises_for_non_1d_masks():
    mask = xr.DataArray(
        [[True, False], [False, True]],
        dims=("member", "time"),
        coords={"member": ["a", "b"], "time": [0, 1]},
    )

    with pytest.raises(ValueError, match="only 1D masks"):
        events.mask_to_event_ids(mask)


def test_mask_to_event_ids_raises_for_invalid_min_duration():
    mask = xr.DataArray([True], dims=("time",), coords={"time": [0]})

    with pytest.raises(ValueError, match="min_duration must be >= 1"):
        events.mask_to_event_ids(mask, min_duration=0)
