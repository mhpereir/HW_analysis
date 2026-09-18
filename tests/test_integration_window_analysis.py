"""Scientific and display regressions for the 4-31-day rank sweep."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src import analysis_io, climatology, plot_style
from src.integration_window_analysis import build_heating_comparison, descending_ranks
from src.integration_window_plotting import plot_heating_ranks
from src.integration_window_validation import validate_against_sources


@pytest.fixture
def inputs():
    times = np.concatenate(
        [
            pd.date_range(f"{year}-05-01T01:00", f"{year}-07-01", freq="h").values
            for year in (2019, 2020, 2021)
        ]
    )
    index = pd.DatetimeIndex(times)
    elapsed = np.concatenate(
        [
            (
                pd.date_range(f"{year}-05-01T01:00", f"{year}-07-01", freq="h")
                - pd.Timestamp(f"{year}-05-01")
            )
            / pd.Timedelta(days=1)
            for year in (2019, 2020, 2021)
        ]
    )
    peaks = np.array(
        ["2019-06-01", "2019-06-15", "2020-06-15", "2021-06-29"], dtype="datetime64[ns]"
    )
    source = xr.Dataset(
        {
            "dTdt": (
                "time",
                np.where(index.year == 2021, 0.105, 0.1),
                {"units": "K hr-1"},
            ),
            "T_mean": ("time", 280 + 0.01 * elapsed**2, {"units": "K"}),
            "event_id": ("event", [7, 9, 23, 42]),
            "peak_time": ("event", peaks),
            "start_time": ("event", peaks),
            "end_time": ("event", peaks + np.timedelta64(1, "D")),
        },
        coords={"time": times, "event": np.arange(4)},
        attrs={
            "region": "pnw_bartusek",
            "heat_budget_bottom_boundary": "surface",
            "heat_budget_top_boundary": "700hPa",
            "stage1_contract_version": 2,
            "start_year": 2019,
            "end_year": 2021,
            "threshold_variable": "tas",
            "quantile": "90",
        },
    )
    climate = climatology.build_regional_hourly_climatology(
        source, variables=("T_mean",)
    )
    events = source[["event_id", "peak_time", "start_time", "end_time"]].copy(deep=True)
    events["I_dTdt_pre"] = (
        "event",
        [
            source.dTdt.sel(time=slice(peak - np.timedelta64(4, "D"), peak))
            .sum()
            .item()
            for peak in peaks
        ],
    )
    events.attrs = {
        "pipeline_stage": "stage_2_event_features",
        "heat_budget_pre_window_hours": "-96,0",
        "window_endpoint_inclusion": "inclusive",
        "integral_method": "hourly_sum_assuming_1h_spacing",
        "season_months": "6,7,8",
        "require_full_event": 1,
    }
    return source, events, climate


def test_known_correction_ties_fixed_cohort_and_source_validation(inputs):
    source, events, climate = inputs
    before = [item.copy(deep=True) for item in inputs]
    table = build_heating_comparison(*inputs, target_event_id=42)
    target = table.sel(event=3, integration_days=4)
    assert target.I_raw.item() == pytest.approx(97 * 0.105)
    assert target.climatological_temperature_change.item() == pytest.approx(
        0.01 * (59**2 - 55**2)
    )
    assert target.rank_raw_common.item() == 1
    assert target.rank_corrected_common.item() == 3
    assert target.rank_corrected_available.item() == 4
    np.testing.assert_array_equal(table.common_cohort, [0, 1, 1, 1])
    assert table.attrs["common_population_size"] == 3
    assert table.sel(event=0, integration_days=31).sample_count.item() == 744
    assert (
        "incomplete_hourly_window"
        in table.sel(event=0, integration_days=31).exclusion_reason.item()
    )
    assert np.isnan(table.sel(event=0).rank_raw_common).all()
    assert table.sel(event=0, integration_days=30).eligible.item() == 1
    assert table.sel(event=1, integration_days=4).ties_corrected_common.item() == 2
    assert table.sel(event=2, integration_days=4).rank_corrected_common.item() == 1
    # June dates have identical climatology in leap and non-leap years.
    np.testing.assert_equal(
        table.sel(event=1).climatological_temperature_change.values,
        table.sel(event=2).climatological_temperature_change.values,
    )
    validation = validate_against_sources(table, source, climate, {4: events})
    assert validation["excluded_event_ids"] == [7]
    assert validation["accepted_stage2_reference_days"] == [4]
    for actual, original in zip(inputs, before, strict=True):
        xr.testing.assert_identical(actual, original)


def test_negative_seasonal_change_keeps_its_sign(inputs):
    source, events, _ = inputs
    source["T_mean"] = -source.T_mean
    source.T_mean.attrs["units"] = "K"
    climate = climatology.build_regional_hourly_climatology(
        source, variables=("T_mean",)
    )
    table = build_heating_comparison(source, events, climate)
    target = table.sel(event=3)
    assert (target.climatological_temperature_change < 0).all()
    assert (target.I_corrected > target.I_raw).all()


def test_independent_validation_detects_wrong_climatological_means(inputs):
    source, events, climate = inputs
    climate.T_mean.values[:] += 0.2
    table = build_heating_comparison(*inputs)
    with pytest.raises(AssertionError):
        validate_against_sources(table, source, climate, {4: events})


@pytest.mark.parametrize("problem", ["gap", "nan", "inf"])
def test_incomplete_interior_window_is_excluded_across_all_days(inputs, problem):
    source, events, climate = inputs
    stamp = np.datetime64("2020-06-05T12:00")
    if problem == "gap":
        source = source.drop_sel(time=stamp)
    else:
        source.dTdt.loc[{"time": stamp}] = np.nan if problem == "nan" else np.inf
    table = build_heating_comparison(source, events, climate)
    assert table.sel(event=2, integration_days=4).eligible.item() == 1
    assert table.sel(event=2, integration_days=14).eligible.item() == 0
    assert table.sel(event=2).common_cohort.item() == 0
    assert np.isnan(table.sel(event=2).rank_raw_common).all()


def test_missing_target_coverage_fails(inputs):
    source, events, climate = inputs
    source.dTdt.loc[{"time": np.datetime64("2021-06-10")}] = np.nan
    with pytest.raises(ValueError, match="Target event lacks"):
        build_heating_comparison(source, events, climate)


@pytest.mark.parametrize(
    "field,value",
    [
        ("region", "pnw_hotz"),
        ("heat_budget_top_boundary", "500hPa"),
        ("climatology_start_year", 1981),
        ("climatology_event_exclusion", "heatwaves"),
    ],
)
def test_incompatible_climatology_fails(inputs, field, value):
    inputs[2].attrs[field] = value
    with pytest.raises(ValueError):
        build_heating_comparison(*inputs)


def test_missing_climatology_years_cannot_enter_ranking(inputs):
    inputs[2].T_mean_count.loc[{"climatology_time": np.datetime64("2000-06-15")}] = 2
    table = build_heating_comparison(*inputs, integration_days=(4,))
    assert not table.common_cohort.values[[1, 2]].any()
    with pytest.raises(ValueError, match="Target event lacks"):
        build_heating_comparison(*inputs)


@pytest.mark.parametrize("days", [[], [4, 4], [7, 4], [0, 4], [4.5], [True]])
def test_invalid_windows_fail(inputs, days):
    with pytest.raises(ValueError, match="integration_days"):
        build_heating_comparison(*inputs, integration_days=days)


def test_target_identity_and_population_are_verified(inputs):
    with pytest.raises(ValueError, match="Target event ID"):
        build_heating_comparison(*inputs, target_event_id=999)
    with pytest.raises(ValueError, match="exactly one"):
        build_heating_comparison(*inputs, target_peak="2021-06-28")
    source, events, climate = inputs
    events.event_id.values[0] = 999
    with pytest.raises(AssertionError, match="Event population"):
        build_heating_comparison(source, events, climate)


def test_units_and_reference_values_are_verified(inputs):
    inputs[0].dTdt.attrs["units"] = "K day-1"
    with pytest.raises(ValueError, match="K per hour"):
        build_heating_comparison(*inputs)
    inputs[0].dTdt.attrs["units"] = "K hr-1"
    inputs[1].I_dTdt_pre.values[0] += 1
    with pytest.raises(AssertionError, match="Accepted 4-day"):
        build_heating_comparison(*inputs)


def test_competition_ranks_exact_ties():
    ranks, ties = descending_ranks(
        np.array([3.0, 3.0, 1.0, 4.0]), np.array([True, True, True, False])
    )
    np.testing.assert_equal(ranks, [1, 1, 3, np.nan])
    np.testing.assert_equal(ties, [2, 2, 1, 0])


def test_roundtrip_no_overwrite_and_corruption(inputs, tmp_path):
    table = build_heating_comparison(*inputs)
    path = tmp_path / "comparison.nc"
    analysis_io.save_integration_window_ranks(table, path)
    with analysis_io.open_integration_window_ranks(path) as saved:
        xr.testing.assert_equal(saved, table)
    with pytest.raises(FileExistsError):
        analysis_io.save_integration_window_ranks(table, path)
    table.rank_corrected_common.values[3, 0] = 99
    with pytest.raises(AssertionError):
        analysis_io.save_integration_window_ranks(table, tmp_path / "corrupted.nc")


def test_rank_plot_uses_saved_values_and_integer_reversed_axes(inputs, tmp_path):
    first = build_heating_comparison(*inputs)
    second = first.copy(deep=True)
    second.attrs["region"] = "pnw_hotz"
    fig = plot_heating_ranks([first, second])
    try:
        assert len(fig.axes) == 2
        for ax in fig.axes:
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(4, 32))
            np.testing.assert_array_equal(
                ax.lines[0].get_ydata(), first.sel(event=3).rank_raw_common
            )
            np.testing.assert_array_equal(
                ax.lines[1].get_ydata(), first.sel(event=3).rank_corrected_common
            )
            assert ax.get_ylim()[0] > ax.get_ylim()[1]
            assert "Fixed cohort: 3" in ax.get_title()
        plot_style.save_figure(fig, tmp_path / "ranks.png")
        for ax in fig.axes:
            assert all("." not in tick.get_text() for tick in ax.get_yticklabels())
        assert (tmp_path / "ranks.png").stat().st_size > 10000
    finally:
        plt.close(fig)
