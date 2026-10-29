from argparse import Namespace

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from HW_analysis.scripts import build_stage1_harmonized_timeseries as raw_builder
from HW_analysis.src import harmonize, pressure_layer


def seasonal_time(year):
    return pd.date_range(f"{year}-05-01T01:00", f"{year}-10-31T22:00", freq="h")


def test_production_coverage_accepts_complete_years_and_off_season_gaps():
    # Leap years have the same May-October interval as non-leap years.
    times = seasonal_time(2000).append(seasonal_time(2001))
    source = xr.Dataset(coords={"time": times})
    harmonize.validate_production_hourly_coverage(source, years=[2000, 2001])
    assert source.sizes["time"] == 2 * 4414


@pytest.mark.parametrize("years", [[], [2001, 2000], [2001, 2001]])
def test_production_coverage_rejects_invalid_year_selection(years):
    source = xr.Dataset(coords={"time": seasonal_time(2001)})
    with pytest.raises(ValueError, match="nonempty, unique and increasing"):
        harmonize.validate_production_hourly_coverage(source, years=years)


@pytest.mark.parametrize(
    "source",
    [
        xr.Dataset(),
        xr.Dataset(coords={"time": [0, 1]}),
        xr.Dataset(coords={"time": ("sample", seasonal_time(2001))}),
    ],
)
def test_production_coverage_requires_a_datetime_time_axis(source):
    with pytest.raises(ValueError, match="requires"):
        harmonize.validate_production_hourly_coverage(source, years=[2001])


@pytest.mark.parametrize(
    "fault",
    [
        "first_hour",
        "last_hour",
        "interior_hour",
        "duplicate_hour",
        "unordered",
        "shifted",
        "extra_hour",
        "missing_year",
        "missing_date",
        "empty",
    ],
)
def test_raw_and_reference_builders_reject_the_same_incomplete_coverage(
    monkeypatch, fault
):
    time = seasonal_time(2001)
    years = [2001]
    if fault == "first_hour":
        time = time[1:]
    elif fault == "last_hour":
        time = time[:-1]
    elif fault == "interior_hour":
        time = time.delete(100)
    elif fault == "duplicate_hour":
        # Preserve the expected count so sample counts alone cannot pass this.
        time = time.delete(100).insert(100, time[99])
    elif fault == "unordered":
        order = np.arange(len(time))
        order[[100, 101]] = order[[101, 100]]
        time = time[order]
    elif fault == "shifted":
        time = time + pd.Timedelta(minutes=30)
    elif fault == "extra_hour":
        time = time.insert(0, pd.Timestamp("2001-05-01T00:00"))
    elif fault == "missing_year":
        years = [2000, 2001]
    elif fault == "missing_date":
        time = time.delete(100).insert(100, pd.NaT)
    elif fault == "empty":
        time = time[:0]
    source = xr.Dataset(coords={"time": time})

    # Exercise the real raw-source CLI preflight for surface and elevated layers.
    # With no daily inputs, an omitted coverage check would reach event creation
    # and fail differently, rather than rejecting the defective EHB time axis.
    for bottom, top in [("surface", "700hPa"), ("700hPa", "500hPa")]:
        args = Namespace(analysis_years=years, bottom_boundary=bottom, top_boundary=top)
        monkeypatch.setattr(raw_builder, "parse_args", lambda args=args: args)
        monkeypatch.setattr(
            raw_builder, "load_era5_inputs", lambda args: {"heat_budget": source}
        )
        with pytest.raises(ValueError, match="May 1 01:00 through October 31 22:00"):
            raw_builder.main()

    annual = (
        source.sel(time=source.time.dt.year == 2000)
        if fault == "missing_year"
        else source
    )
    with pytest.raises(ValueError, match="May 1 01:00 through October 31 22:00"):
        pressure_layer.validate_annual_budget(
            annual, year=years[0], bottom=700, top=500
        )

    reference = source.assign(
        {
            name: xr.zeros_like(source.time, dtype=float)
            for name in pressure_layer.REUSED_VARIABLES
        }
    )
    reference.attrs.update(
        stage1_contract_version=2,
        region="pnw_bartusek",
        threshold_variable="tas",
        quantile="90",
        analysis_time_resolution="hourly",
        add_full_diagnostics=1,
        cloud_cover_source_layout="global-hourly-grid",
    )
    with pytest.raises(ValueError, match="Reference complete-year coverage"):
        pressure_layer.select_reference(
            reference,
            years=years,
            region="pnw_bartusek",
            threshold_variable="tas",
            quantile="90",
        )
