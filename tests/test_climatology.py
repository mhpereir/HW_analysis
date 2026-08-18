import numpy as np
import pytest
import xarray as xr

from HW_analysis.src import climatology


def test_calendar_hour_keys_match_leap_and_nonleap_calendar_dates():
    time = xr.DataArray(
        np.array(
            ["1999-05-01T06:00", "2000-05-01T06:00", "2000-02-29T12:00"],
            dtype="datetime64[h]",
        ),
        dims="time",
    )

    keys = climatology.calendar_hour_keys(time)

    assert keys.values[0] == keys.values[1]
    assert str(keys.values[2]).startswith("2000-02-29T12:00")


def test_build_regional_hourly_climatology_computes_mean_std_and_count():
    source = _source_dataset(
        [
            "1999-05-01T00:00",
            "1999-05-01T01:00",
            "2000-05-01T00:00",
            "2000-05-01T01:00",
        ],
        values=[1.0, 2.0, 3.0, 6.0],
    )

    climate = climatology.build_regional_hourly_climatology(
        source,
        variables=("T_mean",),
    )

    np.testing.assert_allclose(climate["T_mean"], [2.0, 4.0])
    np.testing.assert_allclose(climate["T_mean_std"], [np.sqrt(2.0), np.sqrt(8.0)])
    np.testing.assert_array_equal(climate["T_mean_count"], [2, 2])
    np.testing.assert_array_equal(climate["month"], [5, 5])
    np.testing.assert_array_equal(climate["day"], [1, 1])
    np.testing.assert_array_equal(climate["hour_utc"], [0, 1])
    assert climate.attrs["climatology_start_year"] == 1999
    assert climate.attrs["climatology_end_year"] == 2000


def test_build_regional_hourly_climatology_skips_nonfinite_values_per_key():
    source = _source_dataset(
        [
            "1999-05-01T00:00",
            "1999-05-01T01:00",
            "2000-05-01T00:00",
            "2000-05-01T01:00",
            "2001-05-01T00:00",
            "2001-05-01T01:00",
        ],
        values=[1.0, 2.0, np.nan, 4.0, 5.0, np.inf],
    )

    climate = climatology.build_regional_hourly_climatology(
        source,
        variables=("T_mean",),
    )

    np.testing.assert_allclose(climate["T_mean"], [3.0, 3.0])
    np.testing.assert_allclose(climate["T_mean_std"], [np.sqrt(8.0), np.sqrt(2.0)])
    np.testing.assert_array_equal(climate["T_mean_count"], [2, 2])


def test_build_regional_hourly_climatology_rejects_duplicate_year_key():
    source = _source_dataset(
        ["1999-05-01T00:00", "1999-05-01T00:30"],
        values=[1.0, 2.0],
    )

    with pytest.raises(ValueError, match="more than one timestamp"):
        climatology.build_regional_hourly_climatology(
            source,
            variables=("T_mean",),
        )


def test_apply_regional_hourly_climatology_matches_before_event_stacking():
    source = _source_dataset(
        [
            "1999-05-01T00:00",
            "1999-05-01T01:00",
            "2000-05-01T00:00",
            "2000-05-01T01:00",
        ],
        values=[1.0, 2.0, 3.0, 6.0],
    )
    climate = climatology.build_regional_hourly_climatology(
        source,
        variables=("T_mean",),
    )

    anomalies = climatology.apply_regional_hourly_climatology(
        source,
        climate,
        variables=("T_mean",),
    )

    np.testing.assert_allclose(anomalies["T_mean"], [-1.0, -2.0, 1.0, 2.0])
    assert anomalies.attrs["data_representation"] == "climatological_anomaly"
    assert anomalies["T_mean"].attrs["climatology_baseline_variable"] == "T_mean"


def test_apply_regional_hourly_climatology_rejects_missing_key():
    source = _source_dataset(
        ["1999-05-01T00:00", "2000-05-01T00:00"],
        values=[1.0, 3.0],
    )
    climate = climatology.build_regional_hourly_climatology(
        source,
        variables=("T_mean",),
    )
    target = _source_dataset(["2001-05-01T01:00"], values=[4.0])

    with pytest.raises(ValueError, match="missing calendar-hour keys"):
        climatology.apply_regional_hourly_climatology(
            target,
            climate,
            variables=("T_mean",),
        )


def test_face_anomalies_reconstruct_total_advection_anomaly():
    source = _source_dataset(
        ["1999-05-01T00:00", "2000-05-01T00:00"],
        values=[1.0, 3.0],
    )
    faces = (
        "advection_west",
        "advection_east",
        "advection_south",
        "advection_north",
        "advection_top",
    )
    for index, name in enumerate(faces, start=1):
        source[name] = ("time", np.array([index, index + 2.0]))
    source["advection"] = sum(source[name] for name in faces)
    variables = ("advection", *faces)
    climate = climatology.build_regional_hourly_climatology(
        source,
        variables=variables,
    )

    anomalies = climatology.apply_regional_hourly_climatology(
        source,
        climate,
        variables=variables,
    )

    np.testing.assert_allclose(
        sum(anomalies[name] for name in faces),
        anomalies["advection"],
    )


def _source_dataset(times, *, values) -> xr.Dataset:
    time = np.asarray(times, dtype="datetime64[m]")
    return xr.Dataset(
        {"T_mean": ("time", np.asarray(values, dtype=float), {"units": "K"})},
        coords={"time": time},
        attrs={
            "stage1_contract_version": 2,
            "region": "pnw_bartusek",
            "heat_budget_bottom_boundary": "surface",
            "heat_budget_top_boundary": "700hPa",
        },
    )
