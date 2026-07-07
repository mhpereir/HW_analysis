import numpy as np
import pytest
import xarray as xr

from HW_analysis.src import config, harmonize, preprocess


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
        "tas_climatology": _daily_array([279.0, 281.0], name="tas_climatology"),
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
    lwa_products = {
        "lwa_region": _daily_array([3.0, 4.0], name="LWA"),
        "lwa_threshold": _daily_array([3.5, 3.5], name="lwa_threshold"),
        "lwa_exceedance_mask": _daily_array([False, True], name="lwa_exceedance_mask"),
        "lwa_event_id": _daily_array([0, 2], name="lwa_event_id"),
    }

    out = harmonize.build_regional_analysis_dataset(
        heat_budget=heat_budget,
        hw_event_products=hw_products,
        lwa_event_products=[lwa_products, lwa_a_products],
        attrs={"region": "pnw_bartusek"},
    )

    assert out.sizes == {"time": 3}
    assert out.attrs["pipeline_stage"] == "stage_1_harmonized_regional_timeseries"
    assert out.attrs["analysis_time_resolution"] == "hourly"
    assert out.attrs["region"] == "pnw_bartusek"
    assert {"T_mean", "volume", "dTdt", "advection", "adiabatic", "diabatic"} <= set(out)
    np.testing.assert_allclose(out["T_mean"].values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(out["volume"].values, [2.0, 3.0, 4.0])
    np.testing.assert_allclose(out["dTdt"].values, [5400.0, 4800.0, 4500.0])
    np.testing.assert_allclose(out["advection"].values, [7200.0, 6000.0, 5400.0])
    np.testing.assert_allclose(out["adiabatic"].values, [9000.0, 7200.0, 6300.0])
    np.testing.assert_allclose(out["diabatic"].values, [10800.0, 8400.0, 7200.0])
    np.testing.assert_allclose(out["tas_region"].values, [280.0, 280.0, 285.0])
    np.testing.assert_allclose(out["tas_climatology"].values, [279.0, 279.0, 281.0])
    np.testing.assert_array_equal(out["hw_event_id"].values, [0, 0, 1])
    np.testing.assert_array_equal(out["lwa_event_id"].values, [0, 0, 2])
    np.testing.assert_array_equal(out["lwa_flag"].values, [0, 0, 1])
    np.testing.assert_array_equal(out["lwa_a_flag"].values, [0, 0, 1])
    assert np.issubdtype(out["hw_event_id"].dtype, np.integer)
    assert out["lwa_a_flag"].dtype == np.int8
    assert out["tas_region"].attrs["native_time_resolution"] == "daily"
    assert out["tas_region"].attrs["analysis_time_resolution"] == "hourly"
    assert out["T_mean"].attrs["native_time_resolution"] == "hourly"
    assert out["dTdt"].attrs["units"] == "K hr-1"
    assert out["dTdt"].attrs["normalized_by"] == "domain_volume"
    assert out["advection"].attrs["sign_convention"] == (
        "source sign retained; positive values indicate advection into the domain"
    )


def test_build_regional_analysis_dataset_raises_for_missing_daily_dates():
    hourly_time = np.array(
        ["2000-01-01T00:00", "2000-01-02T00:00"],
        dtype="datetime64[m]",
    )
    heat_budget = _make_heat_budget(hourly_time)
    hw_products = {
        "tas_region": _daily_array([280.0], name="tas"),
        "tas_climatology": _daily_array([279.0], name="tas_climatology"),
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
        "tas_climatology": _daily_array([279.0], name="tas_climatology"),
        "hw_threshold": _daily_array([282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0], name="hw_event_id"),
    }

    with pytest.raises(ValueError, match="missing required variables"):
        harmonize.build_regional_analysis_dataset(
            heat_budget=heat_budget,
            hw_event_products=hw_products,
        )


def test_build_regional_analysis_dataset_adds_optional_full_diagnostics():
    hourly_time = np.array(
        ["2000-01-01T00:00", "2000-01-01T01:00"],
        dtype="datetime64[m]",
    )
    heat_budget = _make_heat_budget(hourly_time)
    hw_products = {
        "tas_region": _daily_array([280.0], name="tas"),
        "tas_climatology": _daily_array([279.0], name="tas_climatology"),
        "hw_threshold": _daily_array([282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0], name="hw_event_id"),
    }
    full_diagnostics = _make_full_diagnostics(hourly_time)

    out = harmonize.build_regional_analysis_dataset(
        heat_budget=heat_budget,
        hw_event_products=hw_products,
        full_diagnostics=full_diagnostics,
        region="pnw_bartusek",
    )

    expected_names = {
        "nslr",
        "nssr",
        "slhf",
        "sshf",
        "soil_moisture",
        "cloud_cover",
        "pbl_p_mean",
        "pbl_p_p05",
        "pbl_p_p95",
        "nslr_heating_rate_approx",
        "nssr_heating_rate_approx",
        "slhf_heating_rate_approx",
        "sshf_heating_rate_approx",
        "surface_energy_heating_rate_approx",
    }
    assert expected_names <= set(out.data_vars)
    np.testing.assert_allclose(out["nssr"].values, [10.0, 20.0])
    np.testing.assert_allclose(out["soil_moisture"].values, [0.1, 0.2])
    np.testing.assert_allclose(out["cloud_cover"].values, [0.25, 0.5])
    assert out["pbl_p_p05"].dims == ("time",)
    assert out["nssr"].attrs["source_variable"] == "ssr"
    assert out["nssr"].attrs["alignment_method"] == "exact_time_selection"

    region_area = preprocess.compute_region_area(full_diagnostics["nssr"]["ssr"], "pnw_bartusek")
    expected_rate = (
        out["nssr"].values
        * region_area
        * config.G_M_S2
        / (config.CP_J_KG_K * out["volume"].values)
    )
    np.testing.assert_allclose(out["nssr_heating_rate_approx"].values, expected_rate)
    expected_total = (
        out["nslr_heating_rate_approx"]
        + out["nssr_heating_rate_approx"]
        + out["slhf_heating_rate_approx"]
        + out["sshf_heating_rate_approx"]
    )
    xr.testing.assert_allclose(out["surface_energy_heating_rate_approx"], expected_total)


def test_build_regional_analysis_dataset_requires_region_for_full_diagnostics():
    hourly_time = np.array(["2000-01-01T00:00"], dtype="datetime64[m]")
    heat_budget = _make_heat_budget(hourly_time)
    hw_products = {
        "tas_region": _daily_array([280.0], name="tas"),
        "tas_climatology": _daily_array([279.0], name="tas_climatology"),
        "hw_threshold": _daily_array([282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0], name="hw_event_id"),
    }

    with pytest.raises(ValueError, match="region is required"):
        harmonize.build_regional_analysis_dataset(
            heat_budget=heat_budget,
            hw_event_products=hw_products,
            full_diagnostics={},
        )


def test_build_regional_analysis_dataset_reports_missing_full_diagnostic_dataset():
    hourly_time = np.array(["2000-01-01T00:00"], dtype="datetime64[m]")
    heat_budget = _make_heat_budget(hourly_time)
    hw_products = {
        "tas_region": _daily_array([280.0], name="tas"),
        "tas_climatology": _daily_array([279.0], name="tas_climatology"),
        "hw_threshold": _daily_array([282.0], name="hw_threshold"),
        "hw_exceedance_mask": _daily_array([False], name="hw_exceedance_mask"),
        "hw_event_id": _daily_array([0], name="hw_event_id"),
    }

    with pytest.raises(ValueError, match="missing datasets"):
        harmonize.build_regional_analysis_dataset(
            heat_budget=heat_budget,
            hw_event_products=hw_products,
            full_diagnostics={},
            region="pnw_bartusek",
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


def _make_full_diagnostics(hourly_time: np.ndarray) -> dict[str, xr.Dataset]:
    lat = [59.0, 41.0]
    lon = [-129.0, -111.0]
    shape = (hourly_time.size, len(lat), len(lon))

    def gridded(source_name: str, values: list[float]) -> xr.Dataset:
        data = np.asarray(values, dtype=float).reshape(hourly_time.size, 1, 1)
        data = np.broadcast_to(data, shape)
        return xr.Dataset(
            {source_name: (("time", "lat", "lon"), data)},
            coords={"time": hourly_time, "lat": lat, "lon": lon},
        )

    return {
        "nslr": gridded("str", [-1.0, -2.0]),
        "nssr": gridded("ssr", [10.0, 20.0]),
        "slhf": gridded("slhf", [-3.0, -4.0]),
        "sshf": gridded("sshf", [-5.0, -6.0]),
        "soil_moisture": gridded("swvl1", [0.1, 0.2]),
        "pbl_p": gridded("pbl_p", [70000.0, 65000.0]),
        "cloud_cover": xr.Dataset(
            {"total_cloud_cover": ("time", [0.25, 0.5])},
            coords={"time": hourly_time},
        ),
    }
