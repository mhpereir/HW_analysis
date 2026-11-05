"""Scientific checks for the isolated daily LWA/tendency exploration."""

import hashlib
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from HW_analysis.scripts import plot_hw_lwa_analysis as plot


def make_stage1(days=None, lwa=None, ids=None):
    days = (
        pd.date_range("2001-06-01", periods=4, freq="D")
        if days is None
        else pd.DatetimeIndex(days)
    )
    lwa = np.arange(len(days), dtype=float) ** 3 if lwa is None else np.asarray(lwa)
    ids = np.ones(len(days), dtype=int) if ids is None else np.asarray(ids)
    time = pd.DatetimeIndex(
        np.concatenate(
            [pd.date_range(day, periods=24, freq="h").values for day in days]
        )
    )
    variables = {
        name: ("time", np.zeros(len(time)))
        for name in plot.analysis_io.REQUIRED_HARMONIZED_VARIABLES
    }
    variables.update(
        {
            plot.LWA: ("time", np.repeat(lwa, 24)),
            "hw_event_id": ("time", np.repeat(ids, 24)),
            "hw_flag": ("time", np.repeat(ids > 0, 24).astype(np.int8)),
        }
    )
    # Nontrivial within-day values distinguish daily averages from midnight samples.
    advection = np.repeat(np.arange(len(days)) * 4.0 + 1, 24) + np.tile(
        (np.arange(24) - 11.5) / 24, len(days)
    )
    variables.update(
        {
            "advection": ("time", advection),
            "adiabatic": ("time", -advection / 2),
            "diabatic": ("time", np.full(len(time), 0.25)),
        }
    )
    event_ids = np.unique(ids[ids > 0])
    starts = [days[np.flatnonzero(ids == eid)[0]] for eid in event_ids]
    ends = [
        days[np.flatnonzero(ids == eid)[-1]] + pd.Timedelta(hours=23)
        for eid in event_ids
    ]
    peaks = [
        days[np.flatnonzero(ids == eid)[len(np.flatnonzero(ids == eid)) // 2]]
        for eid in event_ids
    ]
    variables.update(
        {
            "event_id": ("event", event_ids),
            "start_time": ("event", np.asarray(starts, dtype="datetime64[ns]")),
            "end_time": ("event", np.asarray(ends, dtype="datetime64[ns]")),
            "peak_time": ("event", np.asarray(peaks, dtype="datetime64[ns]")),
        }
    )
    ds = xr.Dataset(
        variables,
        coords={"time": time, "event": np.arange(len(event_ids))},
        attrs={
            "pipeline_stage": plot.analysis_io.EXPECTED_PIPELINE_STAGE,
            "stage1_contract_version": 1,
            "analysis_time_resolution": "hourly",
            "event_id_source": "hw_event_id",
            "threshold_variable": "tas",
            "region": "synthetic",
            "heat_budget_bottom_boundary": "surface",
            "heat_budget_top_boundary": "700",
        },
    )
    ds[plot.LWA].attrs = {
        "units": "m2",
        "native_time_resolution": "daily",
        "projection_method": "calendar_day_lookup",
    }
    for term in plot.TERMS:
        ds[term].attrs["units"] = "K hr-1"
    return ds


def test_daily_values_alignment_sign_units_and_immutability():
    ds = make_stage1(lwa=[10, 13, 11, 7])
    before = ds.copy(deep=True)
    triplets, details = plot.prepare_triplets(ds)
    np.testing.assert_array_equal(triplets.d2_lwa_a_dt2, [-5, -2])
    np.testing.assert_allclose(triplets.advection_daily_start, [1, 5])
    np.testing.assert_allclose(triplets.advection_daily_end, [9, 13])
    np.testing.assert_allclose(triplets.advection_daily_center, [5, 9])
    np.testing.assert_allclose(triplets.adiabatic_daily_center, [-2.5, -4.5])
    np.testing.assert_allclose(triplets.diabatic_daily_center, 0.25)
    assert details["tendency_units"] == "K hr-1"
    assert details["n_valid_days"] == 4
    assert details["n_triplets"] == 2
    assert triplets.day_center.dt.day.tolist() == [2, 3]
    assert triplets.step_days.eq(1).all()
    assert details["require_full_event"] is True
    xr.testing.assert_identical(ds, before)


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("offset", [0, 1e12])
def test_polynomial_curvature_and_large_offset_invariance(degree, sign, offset):
    days = pd.date_range("2001-06-01", periods=6)
    t = np.arange(6, dtype=float)
    ds = make_stage1(days, lwa=offset + sign * t**degree)
    triplets, details = plot.prepare_triplets(ds)
    expected = {0: np.zeros(4), 1: np.zeros(4), 2: np.full(4, 2), 3: 6 * t[1:-1]}
    np.testing.assert_array_equal(triplets.d2_lwa_a_dt2, sign * expected[degree])
    assert plot.validate_saved_triplets(ds, triplets, details)["state"] == "passed"


def test_heating_uses_center_day_without_temporal_smoothing_or_differentiation():
    ds = make_stage1()
    ds.advection.values[:] = np.repeat([90, 4, -12, 160], 24)
    triplets, _ = plot.prepare_triplets(ds)
    np.testing.assert_array_equal(triplets.advection_daily_center, [4, -12])


@pytest.mark.parametrize("n_days", [1, 2])
def test_short_events_have_no_one_sided_or_padded_estimates(n_days):
    with pytest.raises(ValueError, match="No complete consecutive daily triplets"):
        plot.prepare_triplets(make_stage1(pd.date_range("2001-06-01", periods=n_days)))


@pytest.mark.parametrize("ids", [[1] * 6, [0] * 6])
def test_summer_gap_never_produces_cross_year_difference_even_with_same_id(ids):
    ds = make_stage1(
        [
            "2000-08-29",
            "2000-08-30",
            "2000-08-31",
            "2001-06-01",
            "2001-06-02",
            "2001-06-03",
        ],
        lwa=[1, 2, 5, 1000, 1003, 1002],
        ids=ids,
    )
    triplets, details = plot.prepare_triplets(ds, all_seasons=True)
    np.testing.assert_array_equal(triplets.d2_lwa_a_dt2, [2, -4])
    assert (triplets.day_end - triplets.day_start).eq(pd.Timedelta(days=2)).all()
    assert details["n_adjacent_links_excluded_gap"] == 1


@pytest.mark.parametrize("ids", [[1] * 6, [0] * 6])
def test_consecutive_calendar_year_boundary_is_also_excluded(ids):
    ds = make_stage1(
        pd.date_range("2000-12-29", periods=6), lwa=[1, 2, 5, 1000, 1003, 1002], ids=ids
    )
    triplets, details = plot.prepare_triplets(ds, all_seasons=True)
    np.testing.assert_array_equal(triplets.d2_lwa_a_dt2, [2, -4])
    assert details["n_adjacent_links_excluded_year"] == 1


def test_missing_day_and_separate_adjacent_events_do_not_join():
    ds = make_stage1(
        [
            "2001-06-01",
            "2001-06-02",
            "2001-06-03",
            "2001-06-05",
            "2001-06-06",
            "2001-06-07",
            "2001-06-08",
        ],
        lwa=[1, 2, 5, 100, 300, 305, 312],
        ids=[1, 1, 1, 1, 2, 2, 2],
    )
    triplets, details = plot.prepare_triplets(ds)
    np.testing.assert_array_equal(triplets.d2_lwa_a_dt2, [2, 2])
    assert triplets.event_id.tolist() == [1, 2]
    assert details["n_adjacent_links_excluded_gap"] == 1
    assert details["n_adjacent_links_excluded_event"] == 1


@pytest.mark.parametrize(
    "defect", ["missing_hour", "nan_lwa", "nan_advection", "inf_diabatic"]
)
def test_incomplete_days_drop_all_touching_triplets_for_every_panel(defect):
    ds = make_stage1(pd.date_range("2001-06-01", periods=7))
    if defect == "missing_hour":
        ds = ds.isel(time=np.delete(np.arange(ds.sizes["time"]), 84))
    else:
        name = {
            "nan_lwa": plot.LWA,
            "nan_advection": "advection",
            "inf_diabatic": "diabatic",
        }[defect]
        ds[name].values[84] = np.inf if defect.startswith("inf") else np.nan
    triplets, details = plot.prepare_triplets(ds)
    assert triplets.day_center.dt.day.tolist() == [2, 6]
    assert np.isfinite(triplets.select_dtypes(include="number")).all().all()
    key = (
        "n_days_excluded_incomplete_hours"
        if defect == "missing_hour"
        else "n_days_excluded_nonfinite"
    )
    assert details[key] == 1


def test_complete_jja_default_and_explicit_peak_or_all_season_selection():
    ds = make_stage1(
        [
            "2001-05-31",
            "2001-06-01",
            "2001-06-02",
            "2001-07-01",
            "2001-07-02",
            "2001-07-03",
        ],
        ids=[1, 1, 1, 2, 2, 2],
    )
    default, _ = plot.prepare_triplets(ds)
    peak, _ = plot.prepare_triplets(ds, require_full_event=False)
    all_seasons, _ = plot.prepare_triplets(ds, all_seasons=True)
    assert default.event_id.tolist() == [2]
    assert peak.event_id.tolist() == [1, 2]
    assert all_seasons.event_id.tolist() == [1, 2]


@pytest.mark.parametrize(
    "defect, match",
    [
        ("duplicate_time", "unique"),
        ("unordered_time", "increasing"),
        ("subhour_time", "exact UTC hours"),
        ("hourly_lwa", "daily native"),
        ("within_day_lwa", "varies within"),
        ("missing_units", "declare units"),
        ("wrong_rate_units", "hourly rate units"),
        ("lwa_population", "TAS-selected"),
        ("bad_event_id", "integer IDs"),
    ],
)
def test_invalid_input_rejected(defect, match):
    ds = make_stage1()
    if defect == "duplicate_time":
        ds = ds.isel(time=[0, *range(ds.sizes["time"])])
    elif defect == "unordered_time":
        ds = ds.isel(time=list(reversed(range(ds.sizes["time"]))))
    elif defect == "subhour_time":
        ds = ds.assign_coords(time=ds.time + np.timedelta64(30, "m"))
    elif defect == "hourly_lwa":
        ds[plot.LWA].attrs["native_time_resolution"] = "hourly"
    elif defect == "within_day_lwa":
        ds[plot.LWA].values[2] += 0.1
    elif defect == "missing_units":
        del ds[plot.LWA].attrs["units"]
    elif defect == "wrong_rate_units":
        ds.adiabatic.attrs["units"] = "K s-1"
    elif defect == "lwa_population":
        ds.attrs["event_id_source"] = "lwa_a_event_id"
    elif defect == "bad_event_id":
        ds["hw_event_id"] = ds.hw_event_id.astype(float)
        ds.hw_event_id.values[0] = 1.5
    with pytest.raises(ValueError, match=match):
        plot.prepare_triplets(ds)


def test_empty_and_constant_populations():
    with pytest.raises(ValueError, match="No complete consecutive"):
        plot.prepare_triplets(make_stage1(pd.date_range("2001-05-01", periods=4)))
    with pytest.raises(ValueError, match="No complete consecutive"):
        plot.prepare_triplets(make_stage1(["2001-06-01", "2001-06-03"]))
    triplets, _ = plot.prepare_triplets(make_stage1(lwa=[5, 5, 5, 5]))
    assert all(
        item["pearson_r"] is None
        for item in plot.correlations(triplets)["heatwave"].values()
    )
    triplets, _ = plot.prepare_triplets(make_stage1())
    stats = plot.correlations(triplets)["heatwave"]
    assert stats["advection"]["pearson_r"] == pytest.approx(1)
    assert stats["adiabatic"]["spearman_rho"] == pytest.approx(-1)
    assert stats["diabatic"]["pearson_r"] is None


def test_single_triplet_and_zero_tendencies_render_without_correlation_or_axis_warnings(
    tmp_path,
):
    ds = make_stage1(pd.date_range("2001-06-01", periods=3), lwa=[0, 0, 0])
    for term in plot.TERMS:
        ds[term].values[:] = 0
    triplets, details = plot.prepare_triplets(ds)
    assert len(triplets) == 1
    assert all(
        item["pearson_r"] is None
        for item in plot.correlations(triplets)["heatwave"].values()
    )
    fig = plot.plot_triplets(triplets, details)
    try:
        plot.plot_style.save_figure(fig, tmp_path / "constant.png")
        assert all(ax.get_xlim()[0] < 0 < ax.get_xlim()[1] for ax in fig.axes)
        assert all(ax.get_ylim()[0] < 0 < ax.get_ylim()[1] for ax in fig.axes)
    finally:
        plt.close(fig)


def test_figure_uses_all_triplets_correct_axes_and_residual_label():
    ds = make_stage1(pd.date_range("2001-06-01", periods=8), ids=[1] * 4 + [0] * 4)
    triplets, details = plot.prepare_triplets(ds)
    fig = plot.plot_triplets(triplets, details)
    try:
        assert len(fig.axes) == 6
        for row, population in enumerate(plot.POPULATIONS):
            group = triplets.loc[triplets.population.eq(population)]
            axes = fig.axes[row * 3 : row * 3 + 3]
            for ax, term in zip(axes, plot.TERMS, strict=True):
                offsets = ax.collections[0].get_offsets()
                np.testing.assert_allclose(offsets[:, 0], group[plot.X_COLUMN])
                np.testing.assert_allclose(offsets[:, 1], group[f"{term}_daily_center"])
                assert "day$^{-2}$" in ax.get_xlabel()
                assert "d^2" in ax.get_xlabel()
                assert ax.xaxis.label.get_visible() == (row == 1)
        assert fig.axes[2].get_title() == "Diabatic residual"
        assert "Heatwaves" in fig.axes[0].get_ylabel()
        assert "Non-heatwave baseline" in fig.axes[3].get_ylabel()
        assert all("K hr" in fig.axes[i].get_ylabel() for i in (0, 3))
        assert len({ax.get_xlim() for ax in fig.axes}) == 1
        assert len({ax.get_ylim() for ax in fig.axes}) == 1
    finally:
        plt.close(fig)


def test_cli_exports_reproducible_triplets_and_never_overwrites(monkeypatch, tmp_path):
    source = tmp_path / "stage1.nc"
    make_stage1(
        pd.date_range("2001-06-01", periods=8), ids=[1] * 4 + [0] * 4
    ).to_netcdf(source, engine="h5netcdf")
    output = tmp_path / "diagnostic"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_hw_lwa_analysis.py",
            "--input-path",
            str(source),
            "--output-dir",
            str(output),
        ],
    )
    assert plot.main() == 0
    summary = json.loads((output / "summary.json").read_text())
    triplets = pd.read_csv(output / "daily_triplets.csv")
    assert len(triplets) == summary["analysis"]["n_triplets"] == 4
    assert triplets.population.value_counts().to_dict() == {
        "heatwave": 2,
        "baseline": 2,
    }
    assert triplets.loc[triplets.population.eq("baseline"), "event_id"].eq(0).all()
    assert summary["input_sha256"] == source_hash
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
    assert summary["source_variable_attrs"][plot.LWA]["units"] == "m2"
    assert summary["analysis"]["season_months"] == [6, 7, 8]
    assert summary["numerical_validation"]["state"] == "passed"
    assert summary["numerical_validation"]["n_checked_triplets"] == 4
    assert summary["derivative_order"] == summary["analysis"]["derivative_order"] == 2
    assert summary["x_column"] == "d2_lwa_a_dt2"
    assert summary["column_units"]["d2_lwa_a_dt2"] == "m2 day-2"
    for population in plot.POPULATIONS:
        assert summary["correlations"][population]["diabatic"]["pearson_r"] is None
        assert summary["correlations"][population]["advection"]["n_triplets"] == 2
        assert summary["analysis"]["populations"][population]["n_triplets"] == 2
    for name, info in summary["outputs"].items():
        assert (
            hashlib.sha256((output / name).read_bytes()).hexdigest() == info["sha256"]
        )
    assert plt.imread(output / "hw_lwa_analysis.png").std() > 0.01
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(FileExistsError, match="fresh path"):
        plot.main()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_cli_all_seasons_and_removed_first_order_switch(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_hw_lwa_analysis.py",
            "--input-path",
            "input.nc",
            "--output-dir",
            str(tmp_path / "new"),
            "--all-seasons",
        ],
    )
    args = plot.parse_args()
    assert args.all_seasons and not args.require_full_event
    assert args.season_months is None
    assert not hasattr(args, "x_axis")
    monkeypatch.setattr("sys.argv", [*sys.argv, "--x-axis", "change"])
    with pytest.raises(SystemExit):
        plot.parse_args()


def test_baseline_is_original_non_heatwave_days_not_excluded_events():
    ds = make_stage1(
        pd.date_range("2001-05-30", periods=13),
        ids=[1] * 4 + [0] * 3 + [2] * 3 + [0] * 3,
    )
    # Short threshold exceedances outside detected heatwaves still have event ID 0.
    ds.hw_flag.values[ds.hw_event_id.values == 0] = 1
    before = ds.copy(deep=True)
    triplets, details = plot.prepare_triplets(ds)
    hw = triplets.loc[triplets.population.eq("heatwave")]
    baseline = triplets.loc[triplets.population.eq("baseline")]
    assert hw.day_center.dt.day.tolist() == [7]
    assert hw.event_id.tolist() == [2]
    assert baseline.day_center.dt.day.tolist() == [4, 10]
    assert baseline.event_id.eq(0).all()
    assert details["n_represented_events"] == details["n_selected_events"] == 1
    assert details["populations"]["baseline"]["n_valid_days"] == 6
    assert details["populations"]["baseline"]["n_triplets"] == 2
    xr.testing.assert_identical(ds, before)


def test_baseline_requires_all_three_dates_in_season_and_honors_overrides():
    ds = make_stage1(pd.date_range("2001-05-29", periods=6), ids=[0] * 6)
    triplets, details = plot.prepare_triplets(ds)
    assert triplets.day_start.tolist() == [pd.Timestamp("2001-06-01")]
    assert details["n_represented_events"] == 0
    assert details["populations"]["heatwave"]["n_triplets"] == 0
    custom, _ = plot.prepare_triplets(ds, season_months=[5])
    assert custom.day_start.tolist() == [pd.Timestamp("2001-05-29")]
    all_triplets, _ = plot.prepare_triplets(ds, all_seasons=True)
    assert len(all_triplets) == 4


@pytest.mark.parametrize(
    "days",
    [
        ["2000-08-30", "2000-08-31", "2001-06-01", "2001-06-02"],
        ["2000-12-30", "2000-12-31", "2001-01-01", "2001-01-02"],
        ["2001-06-01", "2001-06-02", "2001-06-04", "2001-06-05"],
    ],
)
def test_baseline_never_bridges_summer_year_or_missing_day_gaps(days):
    ds = make_stage1(days, lwa=[1, 2, 1000, 1003], ids=[0] * 4)
    with pytest.raises(ValueError, match="No complete consecutive"):
        plot.prepare_triplets(ds, all_seasons=True)


@pytest.mark.parametrize("defect", ["missing_hour", "nan_tendency", "mixed_event_id"])
def test_baseline_requires_complete_finite_days_without_any_event_hours(defect):
    ds = make_stage1(pd.date_range("2001-06-01", periods=6), ids=[0] * 6)
    if defect == "missing_hour":
        ds = ds.isel(time=np.delete(np.arange(ds.sizes["time"]), 60))
    elif defect == "nan_tendency":
        ds.diabatic.values[60] = np.nan
    else:
        ds.hw_event_id.values[60] = 8
    triplets, _ = plot.prepare_triplets(ds)
    assert triplets.day_center.dt.day.tolist() == [5]
    assert triplets.population.eq("baseline").all()


def test_correlations_are_computed_separately_for_each_population():
    ds = make_stage1(pd.date_range("2001-06-01", periods=8), ids=[1] * 4 + [0] * 4)
    ds.advection.values[:] = np.repeat([1, 5, 9, 13, -17, -21, -25, -29], 24)
    triplets, _ = plot.prepare_triplets(ds)
    stats = plot.correlations(triplets)
    assert stats["heatwave"]["advection"]["pearson_r"] == pytest.approx(1)
    assert stats["baseline"]["advection"]["pearson_r"] == pytest.approx(-1)
    assert stats["heatwave"]["advection"]["n_triplets"] == 2
    assert stats["baseline"]["advection"]["n_triplets"] == 2


@pytest.mark.parametrize(
    "ids, empty_population, empty_row",
    [
        ([0] * 4, "heatwave", 0),
        ([1] * 4, "baseline", 1),
    ],
)
def test_one_empty_population_retains_annotated_row(ids, empty_population, empty_row):
    triplets, details = plot.prepare_triplets(make_stage1(ids=ids))
    stats = plot.correlations(triplets)
    assert all(
        item["n_triplets"] == 0 and item["pearson_r"] is None
        for item in stats[empty_population].values()
    )
    fig = plot.plot_triplets(triplets, details)
    try:
        assert len(fig.axes) == 6
        for ax in fig.axes[empty_row * 3 : empty_row * 3 + 3]:
            assert len(ax.collections[0].get_offsets()) == 0
            assert any(
                "No eligible daily triplets" in text.get_text() for text in ax.texts
            )
    finally:
        plt.close(fig)


def test_missing_lwa_units_can_be_declared_with_evidence_without_mutating_input():
    ds = make_stage1()
    del ds[plot.LWA].attrs["units"]
    before = ds.copy(deep=True)
    triplets, details = plot.prepare_triplets(
        ds,
        lwa_units="m2",
        lwa_units_evidence="Height (m) integrated over distance (m).",
    )
    assert len(triplets) == 2
    assert details["lwa_units"] == "m2"
    assert details["lwa_units_source"] == "explicit override"
    assert "distance" in details["lwa_units_evidence"]
    xr.testing.assert_identical(ds, before)
    with pytest.raises(ValueError, match="requires nonempty"):
        plot.prepare_triplets(ds, lwa_units="m2")
    with pytest.raises(ValueError, match="requires --lwa-units"):
        plot.prepare_triplets(ds, lwa_units_evidence="unpaired evidence")
    with pytest.raises(ValueError, match="contradict"):
        plot.prepare_triplets(
            make_stage1(), lwa_units="m hPa", lwa_units_evidence="different units"
        )


@pytest.mark.parametrize("all_seasons", [False, True])
def test_independent_validator_matches_direct_source_across_gaps_and_populations(
    all_seasons,
):
    ds = make_stage1(
        [
            "2000-05-31",
            "2000-06-01",
            "2000-06-02",
            "2000-06-03",
            "2000-06-05",
            "2000-06-06",
            "2000-06-07",
            "2001-06-01",
            "2001-06-02",
            "2001-06-03",
        ],
        ids=[1, 1, 1, 0, 0, 0, 0, 2, 2, 2],
    )
    triplets, details = plot.prepare_triplets(ds, all_seasons=all_seasons)
    report = plot.validate_saved_triplets(ds, triplets, details)
    assert report["state"] == "passed"
    assert report["n_checked_triplets"] == len(triplets)
    assert report["population_counts"]["baseline"] == 1


@pytest.mark.parametrize(
    "corruption",
    [
        "values",
        "curvature",
        "date",
        "center_date",
        "population",
        "missing_triplet",
        "selected_ids",
    ],
)
def test_independent_validator_rejects_corrupted_saved_triplets(corruption):
    ds = make_stage1(pd.date_range("2001-06-01", periods=8), ids=[1] * 4 + [0] * 4)
    triplets, details = plot.prepare_triplets(ds)
    if corruption == "values":
        triplets.loc[0, "advection_daily_center"] += 0.1
    elif corruption == "curvature":
        triplets.loc[0, "d2_lwa_a_dt2"] *= -1
    elif corruption == "date":
        triplets.loc[0, "day_end"] += pd.Timedelta(days=365)
    elif corruption == "center_date":
        triplets.loc[0, "day_center"] += pd.Timedelta(days=1)
    elif corruption == "population":
        triplets.loc[0, "population"] = "baseline"
    elif corruption == "missing_triplet":
        triplets = triplets.iloc[:-1]
    else:
        details["selected_event_ids"] = []
    with pytest.raises((ValueError, AssertionError)):
        plot.validate_saved_triplets(ds, triplets, details)
