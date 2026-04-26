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


def test_project_daily_to_hourly_raises_for_missing_daily_dates():
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

    with pytest.raises(ValueError, match="missing dates required by the hourly target"):
        harmonize.project_daily_to_hourly(daily, hourly_time)


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


def test_build_regional_analysis_dataset_projects_daily_products_to_hourly_time():
    hourly_time = np.array(
        ["2000-01-01T00:00", "2000-01-01T12:00", "2000-01-02T00:00"],
        dtype="datetime64[m]",
    )
    heat_budget = _make_heat_budget(hourly_time)
    hw_products = {
        "tas_region": _daily_array([280.0, 285.0], name="tas"),
        "hw_threshold": _daily_array([282.0, 282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False, True], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0, 1], name="hw_event_id"),
    }
    lwa_a_products = {
        "lwa_a_region": _daily_array([1.0, 2.0], name="LWA_a"),
        "lwa_a_threshold": _daily_array([1.5, 1.5], name="lwa_a_threshold"),
        "lwa_a_exceedance_mask": _daily_array([False, True], name="lwa_a_exceedance_mask"),
        "lwa_a_event_id": _daily_array([0, 1], name="lwa_a_event_id"),
    }

    out = harmonize.build_regional_analysis_dataset(
        heat_budget=heat_budget,
        hw_event_products=hw_products,
        lwa_event_products=[lwa_a_products],
        attrs={"region": "pnw_bartusek"},
    )

    assert out.sizes == {"time": 3}
    assert out.attrs["pipeline_stage"] == "stage_1_harmonized_regional_timeseries"
    assert out.attrs["analysis_time_resolution"] == "hourly"
    assert out.attrs["region"] == "pnw_bartusek"
    assert {"T_mean", "volume", "dTdt", "advection", "adiabatic", "diabatic"} <= set(out)
    np.testing.assert_allclose(out["tas_region"].values, [280.0, 280.0, 285.0])
    np.testing.assert_array_equal(out["hw_event_id"].values, [0, 0, 1])
    np.testing.assert_array_equal(out["lwa_a_flag"].values, [False, False, True])
    assert np.issubdtype(out["hw_event_id"].dtype, np.integer)
    assert out["lwa_a_flag"].dtype == bool
    assert out["tas_region"].attrs["native_time_resolution"] == "daily"
    assert out["tas_region"].attrs["analysis_time_resolution"] == "hourly"
    assert out["T_mean"].attrs["native_time_resolution"] == "hourly"


def test_build_regional_analysis_dataset_raises_for_missing_daily_dates():
    hourly_time = np.array(
        ["2000-01-01T00:00", "2000-01-02T00:00"],
        dtype="datetime64[m]",
    )
    heat_budget = _make_heat_budget(hourly_time)
    hw_products = {
        "tas_region": _daily_array([280.0], name="tas"),
        "hw_threshold": _daily_array([282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0], name="hw_event_id"),
    }

    with pytest.raises(ValueError, match="missing dates required by the hourly target"):
        harmonize.build_regional_analysis_dataset(
            heat_budget=heat_budget,
            hw_event_products=hw_products,
        )


def test_build_regional_analysis_dataset_raises_for_missing_heat_budget_variable():
    hourly_time = np.array(["2000-01-01T00:00"], dtype="datetime64[m]")
    heat_budget = _make_heat_budget(hourly_time).drop_vars("advection_term")
    hw_products = {
        "tas_region": _daily_array([280.0], name="tas"),
        "hw_threshold": _daily_array([282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0], name="hw_event_id"),
    }

    with pytest.raises(ValueError, match="missing required variables"):
        harmonize.build_regional_analysis_dataset(
            heat_budget=heat_budget,
            hw_event_products=hw_products,
        )


def _daily_array(values, *, name: str) -> xr.DataArray:
    return xr.DataArray(
        values,
        dims=("time",),
        coords={
            "time": np.array(
                ["2000-01-01", "2000-01-02"][: len(values)],
                dtype="datetime64[D]",
            )
        },
        name=name,
    )


def _make_heat_budget(hourly_time: np.ndarray) -> xr.Dataset:
    coords = {"time": hourly_time}
    data = np.arange(hourly_time.size, dtype=float)
    return xr.Dataset(
        {
            "T_domain_avg": ("time", data + 1.0),
            "domain_volume": ("time", data + 2.0),
            "dT_dt": ("time", data + 3.0),
            "advection_term": ("time", data + 4.0),
            "adiabatic_term": ("time", data + 5.0),
            "diabatic_term": ("time", data + 6.0),
        },
        coords=coords,
    )
