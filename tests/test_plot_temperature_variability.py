"""Independent numerical and CLI checks for the temperature diagnostic."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from HW_analysis.scripts import plot_temperature_variability as plotter
from HW_analysis.src import analysis_io


def make_inputs(years=(2000,)):
    time = pd.DatetimeIndex(
        np.concatenate(
            [
                pd.date_range(f"{year}-06-01", periods=96, freq="h").values
                for year in years
            ]
        )
    )
    slow = np.tile(np.repeat([-2.0, -1.0, 1.0, 2.0], 24), len(years))
    diurnal = np.sin(2 * np.pi * time.hour.to_numpy() / 24)
    stage1 = xr.Dataset(
        {
            "T_mean": ("time", 280 + 2 * slow + 0.5 * diurnal),
            "tas_region": ("time", 290 + slow),
        },
        coords={"time": time},
        attrs={
            "region": "pnw_hotz",
            "heat_budget_bottom_boundary": "surface",
            "heat_budget_top_boundary": "700hPa",
        },
    )
    stage1.T_mean.attrs.update(units="K", native_time_resolution="hourly")
    stage1.tas_region.attrs.update(
        units="K",
        native_time_resolution="daily",
        lat_bounds=(49, 59),
        lon_bounds=(-125, -115),
    )
    hourly = xr.Dataset(
        {"tas_region": ("time", 290 + slow + 6 * diurnal)},
        coords={"time": time},
        attrs={"region": "pnw_hotz"},
    )
    hourly.tas_region.attrs.update(
        units="K",
        native_time_resolution="hourly",
        lat_bounds=(49, 59),
        lon_bounds=(-125, -115),
    )
    return stage1, hourly


def test_aggregation_can_reverse_variability_order_without_changing_sources():
    stage1, hourly = make_inputs((2000, 2001))
    originals = stage1.copy(deep=True), hourly.copy(deep=True)
    pairs, daily, coverage = plotter.prepare_pairs(stage1, hourly)
    np.testing.assert_allclose(daily.T_mean, np.tile([276, 278, 282, 284], 2))
    np.testing.assert_allclose(daily.tas_region, np.tile([288, 289, 291, 292], 2))
    stats = plotter.pair_statistics(pairs["daily"])
    assert stats["sd_ratio_T_mean_over_tas_region"] == pytest.approx(2)
    assert stats["variance_ratio_T_mean_over_tas_region"] == pytest.approx(4)
    assert stats["pearson_r"] == pytest.approx(1)
    assert stats["slope_K_per_K"] == pytest.approx(2)
    assert stats["intercept_K"] == pytest.approx(-300)
    assert stats["T_mean_sd_K"] == pytest.approx(
        np.std([276, 278, 282, 284] * 2, ddof=1)
    )
    hourly_stats = plotter.pair_statistics(pairs["hourly"])
    expected = np.std(stage1.T_mean.values, ddof=1) / np.std(
        hourly.tas_region.values, ddof=1
    )
    assert hourly_stats["sd_ratio_T_mean_over_tas_region"] == pytest.approx(expected)
    assert expected < 1
    assert coverage["retained_hours"] == 192
    assert (
        coverage["selected_days_present"] == 8
    )  # No artificial rows in the interannual gap.
    assert plotter.source_agreement(daily)["rmse_K"] == pytest.approx(0, abs=1e-12)
    for original, current in zip(originals, (stage1, hourly), strict=True):
        xr.testing.assert_identical(original, current)


@pytest.mark.parametrize(
    "field", ["T_mean", "tas_region", "hourly_surface", "missing_timestamp"]
)
def test_one_missing_hour_removes_entire_day_from_both_scales(field):
    stage1, hourly = make_inputs()
    if field == "missing_timestamp":
        hourly = hourly.isel(time=np.arange(1, 96))
    elif field == "hourly_surface":
        hourly["tas_region"].values[4] = np.inf
    else:
        stage1[field].values[4] = np.nan
    pairs, daily, coverage = plotter.prepare_pairs(stage1, hourly)
    assert len(daily) == 3
    assert len(pairs["hourly"]) == 72
    assert coverage["incomplete_or_nonfinite_days_dropped"] == 1
    assert daily.index[0] == pd.Timestamp("2000-06-02")
    assert pairs["hourly"].index.min() == daily.index.min()


def test_celsius_and_daily_source_disagreement_are_explicit():
    stage1, hourly = make_inputs()
    hourly.tas_region.values[:] -= 273.15 - 1.5
    hourly.tas_region.attrs["units"] = "degrees_Celsius"
    pairs, daily, _ = plotter.prepare_pairs(stage1, hourly)
    agreement = plotter.source_agreement(daily)
    assert agreement["mean_difference_K"] == pytest.approx(1.5)
    assert agreement["rmse_K"] == pytest.approx(1.5)
    assert agreement["max_absolute_difference_K"] == pytest.approx(1.5)
    np.testing.assert_allclose(
        pairs["daily_from_hourly"].tas_region - pairs["daily"].tas_region, 1.5
    )


@pytest.mark.parametrize(
    "change,match",
    [
        ("daily", "native_time_resolution"),
        ("projection", "Daily-projected"),
        ("region", "same region"),
        ("bounds", "lat_bounds"),
        ("units", "explicit K or Celsius"),
        ("duplicates", "unique"),
        ("unordered", "ordered"),
        ("half_hour", "exact UTC hours"),
        ("nonconstant_daily", "constant within"),
        ("dimensions", "exactly the dimension"),
    ],
)
def test_reject_incompatible_sources(change, match):
    stage1, hourly = make_inputs()
    if change == "daily":
        hourly.tas_region.attrs["native_time_resolution"] = "daily"
    elif change == "projection":
        hourly.tas_region.attrs["projection_method"] = "calendar_day_lookup"
    elif change == "region":
        hourly.attrs["region"] = "alaska"
    elif change == "bounds":
        hourly.tas_region.attrs["lat_bounds"] = "(40, 60)"
    elif change == "units":
        del hourly.tas_region.attrs["units"]
    elif change == "duplicates":
        hourly = hourly.isel(time=[0, *range(96)])
    elif change == "unordered":
        hourly = hourly.isel(time=slice(None, None, -1))
    elif change == "half_hour":
        hourly = hourly.assign_coords(time=hourly.time + np.timedelta64(30, "m"))
    elif change == "nonconstant_daily":
        stage1.tas_region.values[1] += 1
    elif change == "dimensions":
        hourly = hourly.expand_dims(latitude=[54])
    with pytest.raises(ValueError, match=match):
        plotter.prepare_pairs(stage1, hourly)


def test_month_and_year_selection_and_partial_boundary_days():
    stage1, hourly = make_inputs((2000, 2001))
    pairs, daily, _ = plotter.prepare_pairs(
        stage1, hourly, start_year=2001, end_year=2001
    )
    assert set(daily.index.year) == {2001}
    assert len(pairs["hourly"]) == 96
    with pytest.raises(ValueError, match="no Stage-1 samples"):
        plotter.prepare_pairs(stage1, hourly, months=[7])
    # A clipped calendar day cannot contribute a 23-hour daily average.
    _, daily, coverage = plotter.prepare_pairs(stage1.isel(time=slice(1, None)), hourly)
    assert len(daily) == 7
    assert coverage["incomplete_or_nonfinite_days_dropped"] == 1
    with pytest.raises(ValueError, match="At least two complete"):
        plotter.prepare_pairs(stage1.isel(time=slice(1, 48)), hourly)


def test_statistics_handle_degenerate_samples_and_spearman_ties():
    frame = pd.DataFrame({"tas_region": [1, 1, 2, 3], "T_mean": [5, 5, 3, 1]})
    stats = plotter.pair_statistics(frame)
    assert stats["pearson_r"] == pytest.approx(-1)
    assert stats["spearman_rho"] == pytest.approx(-1)
    assert stats["r_squared"] == pytest.approx(1)
    assert stats["slope_K_per_K"] == pytest.approx(-2)
    for data in (frame.iloc[:0], frame.iloc[:1], frame.assign(tas_region=1)):
        stats = plotter.pair_statistics(data)
        assert stats["pearson_r"] is None
        assert stats["sd_ratio_T_mean_over_tas_region"] is None
        json.dumps(stats, allow_nan=False)
    stats = plotter.pair_statistics(frame.assign(T_mean=1))
    assert stats["pearson_r"] is None
    assert stats["r_squared"] is None
    assert stats["slope_K_per_K"] == 0
    assert stats["sd_ratio_T_mean_over_tas_region"] == 0


def test_constant_noninteger_kelvin_is_not_a_spurious_correlation():
    # Summing repeated decimal Kelvin values can give a rounded mean unequal
    # to each sample, and a tiny nonzero variance in an unguarded calculation.
    frame = pd.DataFrame(
        {"tas_region": np.full(100, 283.15), "T_mean": np.full(100, 279.1)}
    )
    stats = plotter.pair_statistics(frame)
    assert stats["tas_region_sd_K"] == 0
    assert stats["T_mean_variance_K2"] == 0
    assert stats["pearson_r"] is None
    assert stats["spearman_rho"] is None
    assert stats["sd_ratio_T_mean_over_tas_region"] is None


def test_season_selection_at_utc_month_boundary():
    stage1, hourly = make_inputs()
    time = pd.date_range("2000-05-30", periods=96, freq="h")
    stage1 = stage1.assign_coords(time=time)
    hourly = hourly.assign_coords(time=time)
    pairs, daily, coverage = plotter.prepare_pairs(stage1, hourly)
    assert daily.index.tolist() == [
        pd.Timestamp("2000-06-01"),
        pd.Timestamp("2000-06-02"),
    ]
    assert coverage["selected_hours"] == 48
    assert len(pairs["hourly"]) == 48
    _, all_days, _ = plotter.prepare_pairs(stage1, hourly, months=list(range(1, 13)))
    assert len(all_days) == 4


def test_yearly_stats_and_scatter_use_all_pairs():
    stage1, hourly = make_inputs((2000, 2001))
    pairs, _, _ = plotter.prepare_pairs(stage1, hourly)
    rows = plotter.statistics_table(pairs)
    assert len(rows) == 9
    assert [row["n"] for row in rows if row["timescale"] == "daily"] == [8, 4, 4]
    fig = plotter.plot_comparison(pairs, rows, "Synthetic diagnostic")
    try:
        np.testing.assert_allclose(
            fig.axes[0].collections[0].get_offsets(), pairs["hourly"]
        )
        np.testing.assert_allclose(
            fig.axes[1].collections[0].get_offsets(), pairs["daily"]
        )
        np.testing.assert_allclose(fig.axes[3].lines[0].get_ydata(), [2, 2])
    finally:
        plt.close(fig)


@pytest.mark.parametrize("daily_only", [False, True])
def test_cli_exports_reproducible_outputs_and_refuses_overwrite(tmp_path, daily_only):
    stage1, hourly = make_inputs((2000, 2001))
    stage1.attrs["pipeline_stage"] = analysis_io.EXPECTED_PIPELINE_STAGE
    for name in analysis_io.REQUIRED_HARMONIZED_VARIABLES.difference(stage1.data_vars):
        stage1[name] = xr.zeros_like(stage1.T_mean)
    stage1_path, hourly_path = tmp_path / "stage1.nc", tmp_path / "hourly.nc"
    analysis_io.save_harmonized_timeseries(stage1, stage1_path)
    hourly.to_netcdf(hourly_path, engine="h5netcdf")
    before = plotter.sha256(stage1_path), plotter.sha256(hourly_path)
    output_dir = tmp_path / "new-run"
    args = ["--input-path", str(stage1_path), "--output-dir", str(output_dir)]
    args += ["--daily-only"] if daily_only else ["--hourly-t2m-path", str(hourly_path)]
    assert plotter.main(args) == output_dir
    report = json.loads((output_dir / "statistics.json").read_text())
    assert report["coverage"]["retained_days"] == 8
    assert report["hourly_available"] is not daily_only
    assert report["inputs"]["stage1"]["sha256"] == before[0]
    assert report["execution"]["script_sha256"]
    assert (
        (output_dir / "temperature_variability.png").read_bytes().startswith(b"\x89PNG")
    )
    assert (output_dir / "temperature_variability.pdf").read_bytes().startswith(b"%PDF")
    for name, digest in report["output_sha256"].items():
        assert plotter.sha256(output_dir / name) == digest
    table = pd.read_csv(output_dir / "statistics.csv")
    assert "tas_region_sd_K" in table
    assert "t2m_sd_K" not in table
    daily_columns = pd.read_csv(output_dir / "daily_pairs.csv", nrows=0).columns
    assert "tas_region" in daily_columns and "t2m" not in daily_columns
    row = table.loc[(table.timescale == "daily") & (table.period == "all")].iloc[0]
    assert row.sd_ratio_T_mean_over_tas_region == pytest.approx(2)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        plotter.main(args)
    assert before == (plotter.sha256(stage1_path), plotter.sha256(hourly_path))


def test_cli_requires_explicit_mode_and_season_options():
    base = ["--input-path", "stage1.nc", "--output-dir", "output"]
    with pytest.raises(SystemExit):
        plotter.parse_args(base)
    assert plotter.parse_args([*base, "--daily-only"]).season_months == [6, 7, 8]
    assert (
        plotter.parse_args(
            [*base, "--hourly-t2m-path", "hourly.nc"]
        ).hourly_t2m_variable
        == "tas_region"
    )
    args = plotter.parse_args([*base, "--daily-only", "--all-seasons"])
    assert args.season_months == list(range(1, 13))
    assert plotter.parse_args([*base, "--stage1-hourly-t2m"]).stage1_hourly_t2m
    with pytest.raises(SystemExit):
        plotter.parse_args([*base, "--daily-only", "--hourly-t2m-path", "hourly.nc"])
    with pytest.raises(SystemExit):
        plotter.parse_args(
            [*base, "--daily-only", "--start-year", "2001", "--end-year", "2000"]
        )


def test_embedded_stage1_hourly_field_produces_same_statistics(tmp_path):
    stage1, hourly = make_inputs((2000, 2001))
    stage1["tas_region_hourly"] = hourly.tas_region
    stage1.tas_region_hourly.attrs["region"] = stage1.attrs["region"]
    stage1.attrs["pipeline_stage"] = analysis_io.EXPECTED_PIPELINE_STAGE
    for name in analysis_io.REQUIRED_HARMONIZED_VARIABLES.difference(stage1.data_vars):
        stage1[name] = xr.zeros_like(stage1.T_mean)
    path = tmp_path / "stage1.nc"
    analysis_io.save_harmonized_timeseries(stage1, path)
    output = tmp_path / "figure"
    plotter.main(
        ["--input-path", str(path), "--stage1-hourly-t2m", "--output-dir", str(output)]
    )
    report = json.loads((output / "statistics.json").read_text())
    expected, _, _ = plotter.prepare_pairs(stage1, hourly)
    assert report["statistics"] == plotter.statistics_table(expected)
    assert "embedded Stage-1" in report["inputs"]["hourly_t2m"]["source"]


def test_external_variable_name_requires_an_explicit_override():
    stage1, hourly = make_inputs()
    expected, expected_daily, _ = plotter.prepare_pairs(stage1, hourly)
    external = hourly.rename(tas_region="t2m")
    with pytest.raises(ValueError, match="missing 'tas_region'"):
        plotter.prepare_pairs(stage1, external)
    pairs, daily, _ = plotter.prepare_pairs(stage1, external, hourly_variable="t2m")
    assert plotter.statistics_table(pairs) == plotter.statistics_table(expected)
    pd.testing.assert_frame_equal(daily, expected_daily)
