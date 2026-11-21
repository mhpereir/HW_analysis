import numpy as np
import pandas as pd
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
    source_values = source["T_mean"].values.copy()
    source_attrs = dict(source.attrs)

    anomalies = climatology.apply_regional_hourly_climatology(
        source,
        climate,
        variables=("T_mean",),
    )

    np.testing.assert_allclose(anomalies["T_mean"], [-1.0, -2.0, 1.0, 2.0])
    assert anomalies.attrs["data_representation"] == "climatological_anomaly"
    assert anomalies["T_mean"].attrs["climatology_baseline_variable"] == "T_mean"
    np.testing.assert_allclose(source["T_mean"], source_values)
    assert source.attrs == source_attrs


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


@pytest.mark.parametrize(
    ("climatology_attr", "mismatched_value"),
    [
        ("region", "alaska"),
        ("heat_budget_bottom_boundary", "850hPa"),
        ("heat_budget_top_boundary", "500hPa"),
        ("source_stage1_contract_version", 1),
    ],
)
def test_apply_regional_hourly_climatology_rejects_mismatched_source_metadata(
    climatology_attr,
    mismatched_value,
):
    source = _source_dataset(
        ["1999-05-01T00:00", "2000-05-01T00:00"],
        values=[1.0, 3.0],
    )
    climate = climatology.build_regional_hourly_climatology(
        source,
        variables=("T_mean",),
    )
    climate.attrs[climatology_attr] = mismatched_value

    with pytest.raises(ValueError, match="does not match Stage-1 source"):
        climatology.apply_regional_hourly_climatology(
            source,
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


def _summer_climatology() -> xr.Dataset:
    time = pd.date_range("2000-05-01", "2000-10-01", freq="h", inclusive="left")
    # The large non-JJA values must not affect the summer summary. June has
    # fewer days, so averaging monthly means would also give a wrong answer.
    values = np.where(time.month.isin((6, 7, 8)), time.month - 5 + time.hour, 1000.0)
    source = _source_dataset(time, values=values)
    return climatology.build_regional_hourly_climatology(source, variables=("T_mean",))


def test_jja_climatological_mean_weights_calendar_hours_not_months_or_counts():
    climate = _summer_climatology()
    climate["T_mean_count"][:] = np.arange(climate.sizes["climatology_time"]) + 1
    before = climate.copy(deep=True)

    means = climatology.jja_climatological_means(climate, variables=("T_mean",))

    assert means["T_mean"] == pytest.approx((30 + 2 * 31 + 3 * 31) / 92 + 11.5)
    xr.testing.assert_identical(climate, before)


def test_jja_climatological_mean_rejects_missing_summer_hour():
    climate = _summer_climatology().drop_sel(climatology_time="2000-07-01T12:00")

    with pytest.raises(ValueError, match="all 2,208"):
        climatology.jja_climatological_means(climate, variables=("T_mean",))


@pytest.mark.parametrize("value", [np.nan, np.inf])
def test_jja_climatological_mean_rejects_nonfinite_summer_values(value):
    climate = _summer_climatology()
    climate["T_mean"].loc[{"climatology_time": "2000-07-01T12:00"}] = value

    with pytest.raises(ValueError, match="must be finite"):
        climatology.jja_climatological_means(climate, variables=("T_mean",))


def test_apply_optional_jja_metadata_preserves_anomalies_and_inputs():
    climate = _summer_climatology()
    source = _source_dataset(["2000-06-15T00:00", "2000-08-31T23:00"], values=[10., 20.])
    original = source.copy(deep=True)
    baseline = climate.copy(deep=True)
    plain = climatology.apply_regional_hourly_climatology(source, climate, variables=("T_mean",))
    annotated = climatology.apply_regional_hourly_climatology(
        source, climate, variables=("T_mean",), include_jja_means=True
    )
    np.testing.assert_array_equal(plain.T_mean.values, annotated.T_mean.values)
    expected = climate.T_mean.where(climate.month.isin([6, 7, 8]), drop=True).values.mean()
    assert annotated.T_mean.attrs["climatology_jja_mean"] == expected
    assert "climatology_jja_mean" not in plain.T_mean.attrs
    xr.testing.assert_identical(source, original)
    xr.testing.assert_identical(climate, baseline)
