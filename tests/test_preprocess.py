import numpy as np
import pytest
import xarray as xr

from src import preprocess


def test_floor_daily_time_floors_dataarray_time_to_midnight():
    da = xr.DataArray(
        [1.0, 2.0],
        dims=("time",),
        coords={
            "time": np.array(
                ["2000-01-01T06:00", "2000-01-02T23:00"],
                dtype="datetime64[m]",
            )
        },
        name="value",
    )

    out = preprocess.floor_daily_time(da)

    np.testing.assert_array_equal(
        out["time"].values,
        np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
    )
    assert out.name == "value"


def test_floor_daily_time_works_on_dataset():
    ds = xr.Dataset(
        {"value": ("time", [1.0, 2.0])},
        coords={
            "time": np.array(
                ["2000-01-01T12:00", "2000-01-02T03:00"],
                dtype="datetime64[m]",
            )
        },
    )

    out = preprocess.floor_daily_time(ds)

    np.testing.assert_array_equal(
        out["time"].values,
        np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[ns]"),
    )
    assert "value" in out.data_vars


def test_floor_daily_time_raises_when_time_coordinate_missing():
    da = xr.DataArray([1.0], dims=("x",), coords={"x": [0]})

    with pytest.raises(ValueError, match="missing required time coordinate"):
        preprocess.floor_daily_time(da)


def test_threshold_to_time_projects_dayofyear_threshold_to_time_axis():
    threshold = xr.DataArray(
        [10.0, 20.0],
        dims=("dayofyear",),
        coords={"dayofyear": [1, 2]},
        name="threshold",
    )
    time = xr.DataArray(
        np.array(["2001-01-01", "2001-01-02", "2002-01-01"], dtype="datetime64[D]"),
        dims=("time",),
        name="time",
    )

    out = preprocess.threshold_to_time(threshold, time)

    assert out.dims == ("time",)
    np.testing.assert_allclose(out.values, [10.0, 20.0, 10.0])
    np.testing.assert_array_equal(out["time"].values, time.values.astype("datetime64[ns]"))
    assert out.name == "threshold"
    assert out.attrs["projected_to_time"] is True


def test_threshold_to_time_projects_year_dayofyear_threshold_to_time_axis():
    threshold = xr.DataArray(
        [[10.0, 20.0], [30.0, 40.0]],
        dims=("year", "dayofyear"),
        coords={"year": [2001, 2002], "dayofyear": [1, 2]},
        name="threshold",
    )
    time = xr.DataArray(
        np.array(["2001-01-02", "2002-01-01"], dtype="datetime64[D]"),
        dims=("time",),
    )

    out = preprocess.threshold_to_time(threshold, time)

    assert out.dims == ("time",)
    np.testing.assert_allclose(out.values, [20.0, 30.0])


def test_threshold_to_time_preserves_missing_threshold_dates_as_nan():
    threshold = xr.DataArray(
        [[10.0]],
        dims=("year", "dayofyear"),
        coords={"year": [2001], "dayofyear": [1]},
        name="threshold",
    )
    time = xr.DataArray(
        np.array(["2001-01-01", "2002-01-01"], dtype="datetime64[D]"),
        dims=("time",),
    )

    out = preprocess.threshold_to_time(threshold, time)

    assert out.values[0] == 10.0
    assert np.isnan(out.values[1])


def test_threshold_to_time_applies_custom_name():
    threshold = xr.DataArray(
        [10.0],
        dims=("dayofyear",),
        coords={"dayofyear": [1]},
        name="threshold",
    )
    time = xr.DataArray(np.array(["2001-01-01"], dtype="datetime64[D]"), dims=("time",))

    out = preprocess.threshold_to_time(threshold, time, name="hw_threshold_time")

    assert out.name == "hw_threshold_time"


def test_threshold_to_time_raises_for_unsupported_threshold_dims():
    threshold = xr.DataArray(
        [10.0],
        dims=("quantile",),
        coords={"quantile": [95]},
    )
    time = xr.DataArray(np.array(["2001-01-01"], dtype="datetime64[D]"), dims=("time",))

    with pytest.raises(ValueError, match="threshold must have dimensions"):
        preprocess.threshold_to_time(threshold, time)


def test_threshold_to_time_raises_for_non_1d_time():
    threshold = xr.DataArray([10.0], dims=("dayofyear",), coords={"dayofyear": [1]})
    time = xr.DataArray(
        np.array([["2001-01-01"]], dtype="datetime64[D]"),
        dims=("x", "y"),
    )

    with pytest.raises(ValueError, match="time must be a 1D"):
        preprocess.threshold_to_time(threshold, time)


def test_exceedance_mask_above_returns_boolean_time_mask():
    series = xr.DataArray(
        [11.0, 19.0, 30.0],
        dims=("time",),
        coords={"time": [0, 1, 2]},
        name="tas",
    )
    threshold = xr.DataArray(
        [10.0, 20.0, np.nan],
        dims=("time",),
        coords={"time": [0, 1, 2]},
        name="threshold",
    )

    out = preprocess.exceedance_mask(series, threshold, mode="above")

    assert out.dtype == bool
    np.testing.assert_array_equal(out.values, [True, False, False])
    assert out.name == "exceedance_mask"
    assert out.attrs["mode"] == "above"


def test_exceedance_mask_below_returns_boolean_time_mask():
    series = xr.DataArray([9.0, 21.0], dims=("time",), coords={"time": [0, 1]})
    threshold = xr.DataArray([10.0, 20.0], dims=("time",), coords={"time": [0, 1]})

    out = preprocess.exceedance_mask(series, threshold, mode="below", name="cold_mask")

    np.testing.assert_array_equal(out.values, [True, False])
    assert out.name == "cold_mask"


def test_exceedance_mask_raises_for_invalid_mode():
    series = xr.DataArray([1.0], dims=("time",), coords={"time": [0]})
    threshold = xr.DataArray([0.0], dims=("time",), coords={"time": [0]})

    with pytest.raises(ValueError, match="mode must be"):
        preprocess.exceedance_mask(series, threshold, mode="sideways")


def test_exceedance_mask_raises_for_mismatched_dims():
    series = xr.DataArray([1.0], dims=("time",), coords={"time": [0]})
    threshold = xr.DataArray([0.0], dims=("daily_time",), coords={"daily_time": [0]})

    with pytest.raises(ValueError, match="matching dimensions"):
        preprocess.exceedance_mask(series, threshold)


def _sample_field() -> xr.DataArray:
    values = np.arange(2 * 3 * 4, dtype=float).reshape(2, 3, 4)
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={
            "time": [0, 1],
            "lat": [39.0, 41.0, 59.0],
            "lon": [-131.0, -129.0, -111.0, -109.0],
        },
        name="tas",
    )


def test_compute_region_mean_reduces_lat_lon_to_time():
    out = preprocess.compute_region_mean(_sample_field(), "pnw_bartusek")

    assert out.dims == ("time",)
    assert out.sizes["time"] == 2


def test_compute_region_mean_matches_manual_weighted_mean():
    da = _sample_field()
    selected = da.sel(lat=slice(40, 60), lon=slice(-130.0, -110.0))
    weights = np.cos(np.deg2rad(selected["lat"]))
    expected = selected.weighted(weights).mean(dim=["lat", "lon"])

    out = preprocess.compute_region_mean(da, "pnw_bartusek")

    xr.testing.assert_allclose(out, expected)


def test_compute_region_mean_converts_0_360_longitudes_before_selection():
    da = xr.DataArray(
        np.array([[[1.0, 10.0, 20.0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [50.0], "lon": [229.0, 231.0, 249.0]},
        name="tas",
    )

    out = preprocess.compute_region_mean(da, "pnw_bartusek")

    np.testing.assert_allclose(out.values, [15.0])


def test_compute_region_mean_handles_descending_latitudes():
    da = xr.DataArray(
        np.array([[[1.0, 3.0], [10.0, 30.0]]]),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [59.0, 41.0], "lon": [-129.0, -111.0]},
        name="tas",
    )

    out = preprocess.compute_region_mean(da, "pnw_bartusek")

    weights = np.cos(np.deg2rad(da["lat"]))
    expected = da.weighted(weights).mean(dim=["lat", "lon"])
    xr.testing.assert_allclose(out, expected)


def test_compute_region_mean_preserves_name():
    out = preprocess.compute_region_mean(_sample_field(), "pnw_bartusek")

    assert out.name == "tas"


def test_compute_region_mean_adds_region_metadata():
    out = preprocess.compute_region_mean(_sample_field(), "pnw_bartusek")

    assert out.attrs["region"] == "pnw_bartusek"
    assert out.attrs["spatial_mean"] == "cosine-latitude weighted mean"
    assert out.attrs["lat_bounds"] == (40, 60)
    assert out.attrs["lon_bounds"] == (-130.0, -110.0)


def test_compute_region_mean_raises_for_unknown_region():
    with pytest.raises(ValueError, match="Unknown region"):
        preprocess.compute_region_mean(_sample_field(), "missing")


def test_compute_region_mean_raises_for_missing_lat_dim():
    da = _sample_field().rename({"lat": "latitude"})

    with pytest.raises(ValueError, match="missing latitude dimension"):
        preprocess.compute_region_mean(da, "pnw_bartusek")


def test_compute_region_mean_raises_for_missing_lon_dim():
    da = _sample_field().rename({"lon": "longitude"})

    with pytest.raises(ValueError, match="missing longitude dimension"):
        preprocess.compute_region_mean(da, "pnw_bartusek")


def test_compute_region_mean_raises_for_empty_region_selection():
    da = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [0.0], "lon": [0.0]},
        name="tas",
    )

    with pytest.raises(ValueError, match="selected no grid cells"):
        preprocess.compute_region_mean(da, "pnw_bartusek")


def test_compute_region_mean_rejects_dataset_inputs():
    ds = xr.Dataset({"tas": _sample_field()})

    with pytest.raises(TypeError, match="xarray.DataArray"):
        preprocess.compute_region_mean(ds, "pnw_bartusek") # type: ignore[arg-type]


def test_compute_region_weighted_quantiles_reduces_lat_lon():
    out = preprocess.compute_region_weighted_quantiles(
        _sample_field(),
        "pnw_bartusek",
        (0.0, 1.0),
    )

    assert out.dims == ("quantile", "time")
    np.testing.assert_allclose(out.sel(quantile=0.0), [5.0, 17.0])
    assert np.all(out.sel(quantile=1.0).values <= np.array([10.0, 22.0]))
    assert np.all(out.sel(quantile=1.0).values >= np.array([5.0, 17.0]))
    assert out.attrs["region"] == "pnw_bartusek"
    assert out.attrs["spatial_quantile"] == "cosine-latitude area-weighted quantile"


def test_compute_region_weighted_quantiles_rejects_invalid_quantile():
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        preprocess.compute_region_weighted_quantiles(_sample_field(), "pnw_bartusek", 1.5)


def test_compute_region_area_is_positive_for_selected_grid():
    area = preprocess.compute_region_area(_sample_field(), "pnw_bartusek")

    assert area > 0.0


def test_compute_region_area_requires_multiple_cells_per_axis():
    da = xr.DataArray(
        np.ones((1, 1, 1)),
        dims=("time", "lat", "lon"),
        coords={"time": [0], "lat": [50.0], "lon": [-120.0]},
    )

    with pytest.raises(ValueError, match="At least two coordinate values"):
        preprocess.compute_region_area(da, "pnw_bartusek")
