"""Decision 010: populations, unchanged histories, and consumer fail-fast checks."""

import argparse
import importlib

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts import plot_composite_timeseries_all as all_plot
from HW_analysis.scripts import plot_composite_timeseries_all_clim_anom as anomaly_all
from HW_analysis.scripts import plot_top_events as top_plot
from HW_analysis.scripts import plot_top_events_clim_anom as anomaly_top
from HW_analysis.scripts.event_features import (
    build_stage2_baseline_features as baseline_builder,
)
from HW_analysis.scripts.event_features import (
    build_stage2_event_features as event_builder,
)
from HW_analysis.src import analysis_io, climatology, season_selection


def seasonal_source():
    """Six events: May, May/June, June, July, August/September and September."""
    time = np.arange("2000-05-01", "2000-10-01", dtype="datetime64[h]").astype(
        "datetime64[ns]"
    )
    starts = np.array(
        [
            "2000-05-24",
            "2000-05-30",
            "2000-06-09",
            "2000-07-13",
            "2000-08-30",
            "2000-09-08",
        ],
        dtype="datetime64[ns]",
    )
    ends = np.array(
        [
            "2000-05-26",
            "2000-06-02",
            "2000-06-11",
            "2000-07-15",
            "2000-09-03",
            "2000-09-10",
        ],
        dtype="datetime64[ns]",
    )
    peaks = np.array(
        [
            "2000-05-25",
            "2000-06-01",
            "2000-06-10",
            "2000-07-14",
            "2000-08-31",
            "2000-09-09",
        ],
        dtype="datetime64[ns]",
    )
    hourly_ids = np.zeros(time.size, dtype=np.int64)
    for event_id, start, end in zip(range(1, 7), starts, ends, strict=True):
        hourly_ids[(time >= start) & (time < end + np.timedelta64(1, "D"))] = event_id
    signal = np.arange(time.size, dtype=float) / 100 + np.sin(np.arange(time.size) / 7)
    ds = xr.Dataset(
        coords={"time": time, "event": np.arange(6)},
        attrs={
            "event_id_source": "hw_event_id",
            "region": "pnw_hotz",
            "heat_budget_bottom_boundary": "surface",
            "heat_budget_top_boundary": "700hPa",
            "stage1_contract_version": 1,
        },
    )
    for index, name in enumerate(top_plot.TOP_EVENT_VARIABLES):
        ds[name] = ("time", signal + index)
    ds["tas_region"] = ("time", 300 + signal)
    ds["tas_climatology"] = ("time", np.full(time.size, 290.0))
    ds["hw_event_id"] = ("time", hourly_ids)
    ds["event_id"] = ("event", np.arange(1, 7))
    ds["start_time"], ds["end_time"], ds["peak_time"] = [
        ("event", dates) for dates in (starts, ends, peaks)
    ]
    ds["tas_peak"] = ("event", [360.0, 350.0, 310.0, 320.0, 340.0, 370.0])
    ds["duration"] = ("event", [3.0, 4.0, 3.0, 3.0, 5.0, 3.0])
    for name in (
        "lwa_a_peak",
        "lwa_c_peak",
        "tas_anom_peak",
        "tas_excess_integral",
        "tas_excess_peak",
    ):
        ds[name] = ("event", np.arange(6, dtype=float))
    return ds


@pytest.mark.parametrize("default_full", [True, False, None])
def test_cli_default_and_explicit_population_modes(monkeypatch, default_full):
    parser = argparse.ArgumentParser()
    season_selection.add_season_arguments(parser, default_full_event=default_full)
    monkeypatch.setattr("sys.argv", ["test"])
    args = season_selection.parse_args(parser)
    assert args.season_months == [6, 7, 8]
    if default_full is not None:
        assert args.require_full_event is default_full
    monkeypatch.setattr("sys.argv", ["test", "--all-seasons"])
    args = season_selection.parse_args(parser)
    assert args.all_seasons and args.season_months is None
    assert not getattr(args, "require_full_event", False)


def test_production_audit_checks_boundary_membership_and_restores_windows(tmp_path):
    from HW_analysis.scripts import validate_season_selection as audit

    source = seasonal_source()
    assert audit.full_jja_ids(source).tolist() == [3, 4]
    original_windows = dict(audit.event_builder.fixed.config.WINDOWS)
    result = audit.audit_stage2(source, tmp_path)
    assert result["events"]["rows"] == 4
    assert result["events"]["early_june_anchors_checked"] == 2
    assert result["baseline"]["early_june_anchors_checked"] > 0
    assert result["full_event_rows"] == 2
    assert result["all_season_event_rows"] == 6
    assert audit.event_builder.fixed.config.WINDOWS == original_windows
    with analysis_io.open_stage2_features(tmp_path / "stage2_events_21day.nc") as ds:
        assert ds.event_id.values.tolist() == [2, 3, 4, 5]


@pytest.mark.parametrize("anomaly", [False, True])
def test_production_audit_captures_real_entrypoints(monkeypatch, tmp_path, anomaly):
    from HW_analysis.scripts import validate_season_selection as audit

    source = seasonal_source()
    for name in audit.top_plot.EXTENDED_TOP_EVENT_VARIABLES:
        if name not in source:
            source[name] = source.diabatic.copy()
    climate = climatology.build_regional_hourly_climatology(
        source, variables=audit.top_plot.EXTENDED_TOP_EVENT_VARIABLES
    )
    monkeypatch.setattr(
        audit.analysis_io, "open_harmonized_timeseries", lambda _, **kwargs: source
    )
    monkeypatch.setattr(
        audit.analysis_io, "open_regional_hourly_climatology", lambda _: climate
    )
    result = audit.audit_reference(
        "pnw_hotz",
        {"input": "synthetic.nc", "climatology": "synthetic_climate.nc"},
        tmp_path,
        anomaly,
        top_n=2,
    )
    assert result["ranked_event_ids"] == [4, 3]
    assert result["smoothed_max_difference"] == 0
    assert result["all_ranked_references_equal"]


@pytest.mark.parametrize(
    "flags",
    [
        ["--all-seasons", "--require-full-event"],
        ["--all-seasons", "--season-months", "6"],
        ["--season-months", "0"],
        ["--season-months", "13"],
    ],
)
def test_contradictory_or_invalid_cli_selection_fails(monkeypatch, flags):
    parser = argparse.ArgumentParser()
    season_selection.add_season_arguments(parser)
    monkeypatch.setattr("sys.argv", ["test", *flags])
    with pytest.raises(SystemExit):
        season_selection.parse_args(parser)


@pytest.mark.parametrize(
    "module_name",
    [
        "plot_top_events",
        "plot_top_events_clim_anom",
        "plot_composite_timeseries_all",
        "plot_composite_timeseries_all_clim_anom",
        "plot_composite_timeseries_split",
        "plot_composite_timeseries_split_clim_anom",
        "plot_event_summary",
        "plot_advection_direction_exploration",
        "plot_advection_direction_exploration_clim_anom",
        "plot_advection_direction_exploration_matched_clim_anom",
    ],
)
def test_plot_entrypoints_share_jja_full_event_defaults(monkeypatch, module_name):
    module = importlib.import_module(f"HW_analysis.scripts.{module_name}")
    flags = [
        "--region",
        "pnw_hotz",
        "--bottom-boundary",
        "surface",
        "--top-boundary",
        "700",
        "--threshold-variable",
        "tas",
        "--quantile",
        "90",
        "--start-year",
        "2000",
        "--end-year",
        "2000",
    ]
    if "split" in module_name:
        flags += ["--split-variable", "duration"]
    if "matched" in module_name:
        flags += ["--event-features-path", "/tmp/unused-stage2.nc"]
    for extra in (
        [],
        ["--all-seasons"],
        ["--no-require-full-event"],
        ["--season-months", "5", "9"],
    ):
        monkeypatch.setattr("sys.argv", [module_name, *flags, *extra])
        args = module.parse_args()
        selected = season_selection.select_event_population(
            seasonal_source(), **season_selection.season_kwargs(args)
        )
        expected = (
            [1, 2, 3, 4, 5, 6]
            if "--all-seasons" in extra
            else (
                [2, 3, 4, 5]
                if "--no-require-full-event" in extra
                else ([1, 6] if "--season-months" in extra else [3, 4])
            )
        )
        assert selected.event_id.values.tolist() == expected
        assert selected.sizes["time"] == seasonal_source().sizes["time"]


@pytest.mark.parametrize("anomaly", [False, True])
def test_all_and_top_entrypoints_produce_identical_reference_curves(
    monkeypatch, tmp_path, anomaly
):
    """Exercise real selection, ranking, reductions and smoothing via both mains."""
    source = seasonal_source()
    baseline_source = source.copy(deep=True)
    for name in top_plot.TOP_EVENT_VARIABLES:
        baseline_source[name] = baseline_source[name] * 0.1
    climate = climatology.build_regional_hourly_climatology(
        baseline_source, variables=top_plot.TOP_EVENT_VARIABLES
    )
    all_module, top_module = (
        (anomaly_all, anomaly_top) if anomaly else (all_plot, top_plot)
    )
    monkeypatch.setattr(
        all_module.analysis_io, "open_harmonized_timeseries", lambda _, **kwargs: source
    )
    monkeypatch.setattr(
        all_module.analysis_io, "open_regional_hourly_climatology", lambda _: climate
    )
    references, selected_ids, plotted_all = [], [], []
    renderer = anomaly_top.absolute_plot if anomaly else top_plot

    def capture_event(ds, event, *, reference_composite, **kwargs):
        references.append(reference_composite)
        selected_ids.append(int(event.event_id))
        return plt.figure()

    def capture_all(composite, path, **kwargs):
        plotted_all.append(composite)
        return [path]

    monkeypatch.setattr(renderer, "plot_one_top_event", capture_event)
    monkeypatch.setattr(renderer.plot_style, "save_figure", lambda *a, **kw: None)
    monkeypatch.setattr(
        all_module.plotting, "write_composite_timeseries_outputs", capture_all
    )
    flags = [
        "--region",
        "pnw_hotz",
        "--bottom-boundary",
        "surface",
        "--top-boundary",
        "700",
        "--threshold-variable",
        "tas",
        "--quantile",
        "90",
        "--start-year",
        "2000",
        "--end-year",
        "2000",
        "--window-days",
        "2",
        "--smoothing-window",
        "24",
    ]
    monkeypatch.setattr(
        "sys.argv", ["all", *flags, "--output-path", str(tmp_path / "all.png")]
    )
    assert all_module.main() == 0
    monkeypatch.setattr(
        "sys.argv",
        ["top", *flags, "--output-dir", str(tmp_path / "top"), "--top-n", "1"],
    )
    assert top_module.main() == 0
    assert selected_ids == [
        4,
        4,
    ]  # May/September and crossing events cannot take this slot.
    assert len(references) == 2
    # The mean and all percentile variables agree, including the diabatic curve.
    for name in plotted_all[0].data_vars:
        xr.testing.assert_equal(references[0][name], plotted_all[0][name])
    smoothed = all_module.plotting.smooth_composite_for_display(
        plotted_all[0], variables=all_plot.SMOOTHED_VARIABLES, smoothing_window=24
    )
    for name in smoothed.data_vars:
        xr.testing.assert_equal(references[1][name], smoothed[name])
    # Independent mean check: both June and July contribute, even for top_n=1.
    values = source.diabatic if not anomaly else source.diabatic * 0.9
    expected = float(values.sel(time=["2000-06-10", "2000-07-14"]).mean())
    assert float(references[0].diabatic.sel(lag_hour=0)) == pytest.approx(expected)


def test_stage2_defaults_admit_crossing_events_and_keep_21_day_may_history(monkeypatch):
    source = seasonal_source()
    for name in ("heat_budget_pre", "lwa_pre_peak"):
        monkeypatch.setitem(event_builder.fixed.config.WINDOWS, name, (-21 * 24, 0))
    events = event_builder.build_event_features(source)
    baseline = baseline_builder.build_baseline_features(source)
    assert events.event_id.values.tolist() == [2, 3, 4, 5]
    assert events.attrs["require_full_event"] == 0
    assert events.attrs["season_selection_rule"] == "anchor_month"
    assert set(baseline.reference_time.dt.month.values) == {6, 7, 8}
    for table, anchor, dim, date in (
        (events, "peak_time", "event", "2000-06-01"),
        (events, "peak_time", "event", "2000-08-31"),
        (baseline, "reference_time", "baseline_day", "2000-06-05"),
        (baseline, "reference_time", "baseline_day", "2000-08-20"),
    ):
        endpoint = np.datetime64(date, "ns")
        row = table.isel(
            {dim: int(np.flatnonzero(table[anchor].values == endpoint)[0])}
        )
        expected = source.diabatic.sel(
            time=slice(endpoint - np.timedelta64(21, "D"), endpoint)
        ).sum()
        assert row.n_samples_heat_budget_pre == 505
        assert float(row.I_diabatic_pre) == pytest.approx(float(expected), rel=1e-13)
    assert event_builder.build_event_features(
        source, require_full_event=True
    ).event_id.values.tolist() == [3, 4]
    assert event_builder.build_event_features(
        source, all_seasons=True
    ).event_id.values.tolist() == [1, 2, 3, 4, 5, 6]
    assert set(
        baseline_builder.build_baseline_features(
            source, all_seasons=True
        ).reference_time.dt.month.values
    ) == {5, 6, 7, 8, 9}
    season_selection.validate_stage2_season(events)
    season_selection.validate_stage2_season(baseline)
    # Isolate an out-of-season heatwave: its May occurrence still contaminates
    # the June baseline's history even though no June timestamps are events.
    source["hw_event_id"] = xr.where(source.hw_event_id == 1, 1, 0)
    baseline = baseline_builder.build_baseline_features(source)
    june5 = baseline.where(
        baseline.reference_time == np.datetime64("2000-06-05"), drop=True
    )
    assert june5.event_adjacent.item() == 1


def stage2_table(kind="event", *, all_seasons=False, full=False):
    dim, anchor = (
        ("event", "peak_time")
        if kind == "event"
        else ("baseline_day", "reference_time")
    )
    dates = np.array(["2000-07-02", "2000-07-12"], dtype="datetime64[ns]")
    ds = xr.Dataset(
        {anchor: (dim, dates)},
        coords={dim: [0, 1]},
        attrs={
            "pipeline_stage": f"stage_2_{kind}_features",
            "all_seasons": int(all_seasons),
            "season_months": "" if all_seasons else "6,7,8",
        },
    )
    if kind == "event":
        ds.attrs["require_full_event"] = int(full)
        ds["start_time"] = (dim, dates - np.timedelta64(1, "D"))
        ds["end_time"] = (dim, dates + np.timedelta64(1, "D"))
    return ds


@pytest.mark.parametrize("kind", ["event", "baseline"])
@pytest.mark.parametrize(
    "problem",
    ["all_seasons", "wrong_dates", "missing_metadata", "false_rule", "custom_months"],
)
def test_stage2_io_rejects_incompatible_or_mislabelled_populations(
    tmp_path, kind, problem
):
    ds = stage2_table(kind, all_seasons=problem == "all_seasons")
    if problem == "wrong_dates":
        anchor = "peak_time" if kind == "event" else "reference_time"
        ds[anchor][0] = np.datetime64("2000-05-12", "ns")
    elif problem == "missing_metadata":
        del ds.attrs["all_seasons"]
    elif problem == "false_rule":
        ds.attrs["season_selection_rule"] = "all_seasons"
    elif problem == "custom_months":
        ds.attrs["season_months"] = "5,6,7,8,9"
    path = tmp_path / "features.nc"
    ds.to_netcdf(path, engine="h5netcdf")
    with pytest.raises(ValueError, match="Stage-2"):
        analysis_io.open_stage2_features(path)


@pytest.mark.parametrize("kind", ["event", "baseline"])
def test_matching_all_season_and_custom_season_io_remains_available(tmp_path, kind):
    for all_seasons, months in ((True, None), (False, [7])):
        ds = stage2_table(kind, all_seasons=all_seasons)
        if months:
            ds.attrs["season_months"] = "7"
        path = tmp_path / "features.nc"
        ds.to_netcdf(path, engine="h5netcdf")
        with analysis_io.open_stage2_features(
            path, all_seasons=all_seasons, season_months=months
        ) as loaded:
            xr.testing.assert_identical(loaded, ds)


def test_legacy_full_event_table_requires_explicit_matching_rule():
    ds = stage2_table(full=True)
    with pytest.raises(ValueError, match="--require-full-event"):
        season_selection.validate_stage2_season(ds)
    season_selection.validate_stage2_season(ds, require_full_event=True)
    ds["start_time"][0] = np.datetime64("2000-05-30", "ns")
    with pytest.raises(ValueError, match="intervals outside"):
        season_selection.validate_stage2_season(ds, require_full_event=True)


@pytest.mark.parametrize(
    "name",
    [
        "event_features.event_feature_grid_plot",
        "event_features.plot_event_feature",
        "event_features.plot_event_feature_split",
        "event_features.plot_event_feature_split_combined",
        "event_features.plot_adiabatic_advection_comparison",
        "event_features.plot_adiabatic_diabatic_advection",
        "event_features.plot_adiabatic_advection_comparison_baseline",
        "event_features.plot_adiabatic_diabatic_advection_baseline",
        "idyn_matching_exploration.explore_idyn_matching",
        "spatial_composites.build_matched_dyn_pre_spatial_composites",
        "plot_advection_direction_exploration_matched_clim_anom",
    ],
)
def test_consumer_loaders_guard_events_and_baselines(tmp_path, name):
    module = importlib.import_module(f"HW_analysis.scripts.{name}")
    for kind in ("event", "baseline"):
        loader = getattr(module, f"open_{kind}_features", None)
        if loader is None:
            continue
        path = tmp_path / f"{kind}.nc"
        stage2_table(kind, all_seasons=True).to_netcdf(path, engine="h5netcdf")
        with pytest.raises(ValueError, match="season mismatch"):
            loader(path)
        with loader(path, all_seasons=True) as ds:
            assert ds.sizes["event" if kind == "event" else "baseline_day"] == 2


@pytest.mark.parametrize(
    "module_name",
    [
        "plot_composite_timeseries_split",
        "plot_composite_timeseries_split_clim_anom",
    ],
)
def test_default_split_quantiles_use_only_the_complete_jja_population(
    monkeypatch, tmp_path, module_name
):
    module = importlib.import_module(f"HW_analysis.scripts.{module_name}")
    source = seasonal_source()
    climate = climatology.build_regional_hourly_climatology(
        source, variables=top_plot.TOP_EVENT_VARIABLES
    )
    monkeypatch.setattr(
        module.analysis_io, "open_harmonized_timeseries", lambda _: source
    )
    monkeypatch.setattr(
        module.analysis_io, "open_regional_hourly_climatology", lambda _: climate
    )
    captured = []

    def capture(composite, output, **kwargs):
        captured.append(composite)
        return [output]

    monkeypatch.setattr(
        module.plotting, "write_split_composite_timeseries_outputs", capture
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            module_name,
            "--region",
            "pnw_hotz",
            "--bottom-boundary",
            "surface",
            "--top-boundary",
            "700",
            "--threshold-variable",
            "tas",
            "--quantile",
            "90",
            "--start-year",
            "2000",
            "--end-year",
            "2000",
            "--split-variable",
            "tas_peak",
            "--split-quantiles",
            "0.5",
            "--output-path",
            str(tmp_path / "split.png"),
        ],
    )
    assert module.main() == 0
    result = captured[0]
    assert result.split_n_events.values.tolist() == [1, 1]
    assert result.split_upper_value.values.tolist() == [315.0, 320.0]


def test_derived_spatial_population_cannot_be_relabelled_as_jja():
    source = stage2_table(all_seasons=True)
    source.attrs["pipeline_stage"] = "daily_dyn_net_spatial_composites"
    with pytest.raises(ValueError, match="season mismatch"):
        season_selection.validate_inherited_event_season(source)
    season_selection.validate_inherited_event_season(source, all_seasons=True)
    assert source.attrs["pipeline_stage"] == "daily_dyn_net_spatial_composites"
