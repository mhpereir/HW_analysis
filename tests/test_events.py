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


def test_build_hw_event_ids_returns_regional_mask_and_event_ids():
    times = np.array(["2001-01-01", "2001-01-02", "2001-01-03"], dtype="datetime64[D]")
    tas = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[4.0, 4.0], [4.0, 4.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [42.0, 44.0], "lon": [-124.0, -122.0]},
        name="tas",
    )
    threshold = xr.DataArray(
        [[2.0, 2.0, 2.0]],
        dims=("year", "dayofyear"),
        coords={"year": [2001], "dayofyear": [1, 2, 3]},
        name="threshold",
    )
    climatology = xr.DataArray(
        [[0.5, 1.5, 2.5]],
        dims=("year", "dayofyear"),
        coords={"year": [2001], "dayofyear": [1, 2, 3]},
        name="climatology",
    )

    out = events.build_hw_event_ids(
        tas,
        threshold,
        climatology,
        region="pnw_bartusek",
        min_duration=2,
    )

    assert set(out) == {
        "tas_region",
        "tas_climatology",
        "hw_threshold",
        "hw_exceedance_mask",
        "hw_event_id",
    }
    np.testing.assert_array_equal(out["hw_exceedance_mask"].values, [False, True, True])
    np.testing.assert_array_equal(out["hw_event_id"].values, [0, 1, 1])
    np.testing.assert_allclose(out["tas_climatology"].values, [0.5, 1.5, 2.5])
    assert out["tas_region"].attrs["region"] == "pnw_bartusek"


def test_build_lwa_a_event_ids_filters_years_and_uses_lwa_threshold():
    times = np.array(
        ["2001-01-01", "2001-01-02", "2002-01-01", "2002-01-02"],
        dtype="datetime64[D]",
    )
    lwa_a = xr.DataArray(
        np.array(
            [
                [[5.0, 5.0], [5.0, 5.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[5.0, 5.0], [5.0, 5.0]],
                [[6.0, 6.0], [6.0, 6.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [42.0, 44.0], "lon": [-124.0, -122.0]},
        name="LWA_a",
    )
    threshold = xr.DataArray(
        [4.0, 4.0],
        dims=("dayofyear",),
        coords={"dayofyear": [1, 2]},
        name="LWA_a",
    )

    out = events.build_lwa_event_ids(
        lwa_a,
        threshold,
        region="pnw_bartusek",
        variable="LWA_a",
        years=[2002],
        min_duration=1,
    )

    assert set(out) == {
        "lwa_a_region",
        "lwa_a_threshold",
        "lwa_a_exceedance_mask",
        "lwa_a_event_id",
    }
    np.testing.assert_array_equal(
        out["lwa_a_region"]["time"].values,
        np.array(["2002-01-01", "2002-01-02"], dtype="datetime64[D]"),
    )
    np.testing.assert_array_equal(out["lwa_a_exceedance_mask"].values, [True, True])
    np.testing.assert_array_equal(out["lwa_a_event_id"].values, [1, 1])


def test_build_lwa_event_ids_supports_lwa_c_variant():
    times = np.array(["2001-01-01", "2001-01-02"], dtype="datetime64[D]")
    lwa_c = xr.DataArray(
        np.array(
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ]
        ),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [42.0, 44.0], "lon": [-124.0, -122.0]},
        name="LWA_c",
    )
    threshold = xr.DataArray(
        [3.0, 3.0],
        dims=("dayofyear",),
        coords={"dayofyear": [1, 2]},
        name="LWA_c",
    )

    out = events.build_lwa_event_ids(
        lwa_c,
        threshold,
        region="pnw_bartusek",
        variable="LWA_c",
    )

    assert set(out) == {
        "lwa_c_region",
        "lwa_c_threshold",
        "lwa_c_exceedance_mask",
        "lwa_c_event_id",
    }
    np.testing.assert_array_equal(out["lwa_c_exceedance_mask"].values, [False, True])
    np.testing.assert_array_equal(out["lwa_c_event_id"].values, [0, 1])


def test_build_lwa_event_ids_rejects_unknown_lwa_variant():
    times = np.array(["2001-01-01"], dtype="datetime64[D]")
    lwa = xr.DataArray(
        np.ones((1, 2, 2)),
        dims=("time", "lat", "lon"),
        coords={"time": times, "lat": [42.0, 44.0], "lon": [-124.0, -122.0]},
        name="other",
    )
    threshold = xr.DataArray(
        [0.0],
        dims=("dayofyear",),
        coords={"dayofyear": [1]},
    )

    with pytest.raises(ValueError, match="Unsupported LWA variable"):
        events.build_lwa_event_ids(
            lwa,
            threshold,
            region="pnw_bartusek",
            variable="other",
        )
