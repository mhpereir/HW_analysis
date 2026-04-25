import numpy as np
import pytest
import xarray as xr

from src import harmonize


def test_project_daily_to_hourly_replicates_daily_values_by_date():
    daily = xr.DataArray(
        [10.0, 20.0],
        dims=("time",),
        coords={"time": np.array(["2000-01-01", "2000-01-02"], dtype="datetime64[D]")},
        name="daily_value",
    )
    hourly_time = xr.DataArray(
        np.array(
            ["2000-01-01T00:00", "2000-01-01T12:00", "2000-01-02T06:00"],
            dtype="datetime64[m]",
        ),
        dims=("hourly_time",),
        name="hourly_time",
    )

    out = harmonize.project_daily_to_hourly(daily, hourly_time)

    np.testing.assert_allclose(out.values, [10.0, 10.0, 20.0])


def test_project_daily_to_hourly_preserves_missing_daily_dates_as_nan():
    daily = xr.DataArray(
        [10.0],
        dims=("time",),
        coords={"time": np.array(["2000-01-01"], dtype="datetime64[D]")},
        name="daily_value",
    )
    hourly_time = xr.DataArray(
        np.array(["2000-01-01T00:00", "2000-01-02T00:00"], dtype="datetime64[m]"),
        dims=("hourly_time",),
    )

    out = harmonize.project_daily_to_hourly(daily, hourly_time)

    assert out.values[0] == 10.0
    assert np.isnan(out.values[1])


def test_project_daily_to_hourly_preserves_original_hourly_timestamps():
    daily = xr.DataArray(
        [10.0],
        dims=("time",),
        coords={"time": np.array(["2000-01-01"], dtype="datetime64[D]")},
        name="daily_value",
    )
    hourly_values = np.array(["2000-01-01T03:00", "2000-01-01T18:00"], dtype="datetime64[m]")
    hourly_time = xr.DataArray(hourly_values, dims=("hourly_time",))

    out = harmonize.project_daily_to_hourly(daily, hourly_time)

    np.testing.assert_array_equal(out["hourly_time"].values, hourly_values.astype("datetime64[ns]"))


def test_project_daily_to_hourly_applies_custom_name_and_attrs():
    daily = xr.DataArray(
        [1],
        dims=("time",),
        coords={"time": np.array(["2000-01-01"], dtype="datetime64[D]")},
        name="event_id",
        attrs={"source": "test"},
    )
    hourly_time = xr.DataArray(
        np.array(["2000-01-01T00:00"], dtype="datetime64[m]"),
        dims=("hourly_time",),
    )

    out = harmonize.project_daily_to_hourly(daily, hourly_time, name="event_id_hourly")

    assert out.name == "event_id_hourly"
    assert out.attrs["source"] == "test"
    assert out.attrs["projected_from_daily"] is True
    assert out.attrs["daily_time_dim"] == "time"
    assert out.attrs["hourly_time_dim"] == "hourly_time"


def test_project_daily_to_hourly_raises_for_duplicate_daily_dates_after_flooring():
    daily = xr.DataArray(
        [1.0, 2.0],
        dims=("time",),
        coords={
            "time": np.array(
                ["2000-01-01T00:00", "2000-01-01T12:00"],
                dtype="datetime64[m]",
            )
        },
    )
    hourly_time = xr.DataArray(
        np.array(["2000-01-01T00:00"], dtype="datetime64[m]"),
        dims=("hourly_time",),
    )

    with pytest.raises(ValueError, match="duplicate dates"):
        harmonize.project_daily_to_hourly(daily, hourly_time)


def test_project_daily_to_hourly_raises_for_non_1d_daily_inputs():
    daily = xr.DataArray(
        [[1.0]],
        dims=("time", "lat"),
        coords={"time": np.array(["2000-01-01"], dtype="datetime64[D]"), "lat": [50.0]},
    )
    hourly_time = xr.DataArray(
        np.array(["2000-01-01T00:00"], dtype="datetime64[m]"),
        dims=("hourly_time",),
    )

    with pytest.raises(ValueError, match="only 1D daily inputs"):
        harmonize.project_daily_to_hourly(daily, hourly_time)


def test_project_daily_to_hourly_raises_for_missing_daily_time_coordinate():
    daily = xr.DataArray([1.0], dims=("day",), coords={"day": [0]})
    hourly_time = xr.DataArray(
        np.array(["2000-01-01T00:00"], dtype="datetime64[m]"),
        dims=("hourly_time",),
    )

    with pytest.raises(ValueError, match="missing required time coordinate"):
        harmonize.project_daily_to_hourly(daily, hourly_time)
