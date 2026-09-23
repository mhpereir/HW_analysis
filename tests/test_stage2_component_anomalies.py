"""Scientific and plotting acceptance for derived Stage-2 anomaly budgets."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from scripts.event_features import event_feature_config as config
from scripts.event_features.build_stage2_baseline_features import (
    build_baseline_features,
)
from scripts.event_features.build_stage2_event_features import build_event_features
from scripts.event_features.plot_adiabatic_advection_comparison_baseline import (
    plot_tendency_scatter,
)
from scripts.integration_window_analysis import build_component_anomalies as runner
from src import analysis_io, climatology, plot_style
from src.stage2_component_anomalies import (
    RAW_TO_ANOMALY,
    TENDENCIES,
    build_component_anomaly_pair,
    comparison_plot_views,
    validate_component_anomalies,
)
from src.stage2_component_anomaly_validation import validate_anomalies_against_sources


@pytest.fixture
def source():
    times = np.concatenate(
        [
            np.arange(f"{year}-05-01", f"{year}-09-02", dtype="datetime64[h]")
            for year in (2019, 2020, 2021)
        ]
    )
    index = pd.DatetimeIndex(times)
    seasonal = (
        (index.month.to_numpy() - 5) * 30
        + index.day.to_numpy()
        + index.hour.to_numpy() / 24
    )
    offset = index.year.to_numpy() - 2020
    adv = -0.3 + 0.003 * seasonal + 0.02 * offset
    adi = 0.25 + 0.002 * seasonal + 0.01 * offset
    dia = 0.08 + 0.001 * seasonal - 0.005 * offset
    peaks = np.asarray(
        [f"{year}-06-29" for year in (2019, 2020, 2021)], dtype="datetime64[ns]"
    )
    flags = np.zeros(times.size, dtype=int)
    for identifier, peak in zip((101, 102, 103), peaks, strict=True):
        flags[(times >= peak) & (times < peak + np.timedelta64(3, "D"))] = identifier
    ds = xr.Dataset(
        {
            "advection": ("time", adv),
            "adiabatic": ("time", adi),
            "diabatic": ("time", dia),
            "dTdt": ("time", adv + adi + dia),
            "tas_region": ("time", 290 + seasonal / 10),
            "tas_climatology": ("time", np.full(times.size, 288.0)),
            "lwa_a_region": ("time", np.ones(times.size)),
            "lwa_c_region": ("time", np.ones(times.size)),
            "hw_event_id": ("time", flags),
            "event_id": ("event", [101, 102, 103]),
            "peak_time": ("event", peaks),
            "start_time": ("event", peaks),
            "end_time": ("event", peaks + np.timedelta64(2, "D")),
            "duration": ("event", [3, 3, 3]),
        },
        coords={"time": times, "event": [10, 12, 14]},
        attrs={
            "event_id_source": "hw_event_id",
            "region": "pnw_bartusek",
            "heat_budget_bottom_boundary": "surface",
            "heat_budget_top_boundary": "700hPa",
            "stage1_contract_version": 2,
            "start_year": 2019,
            "end_year": 2021,
        },
    )
    for name in TENDENCIES:
        ds[name].attrs["units"] = "K hr-1"
    for name in config.EVENT_SUMMARY_FEATURES:
        if name not in ds:
            ds[name] = ("event", [3.0, 4.0, 8.0])
    return ds


def inputs(source, hours=96):
    common = {"season_months": [6, 7, 8], "integration_hours": hours}
    events = build_event_features(source, require_full_event=True, **common)
    baseline = build_baseline_features(source, **common)
    climate = climatology.build_regional_hourly_climatology(
        source, variables=TENDENCIES
    )
    # Production references are opened with explicit CF duration decoding.
    return (
        climate,
        xr.decode_cf(events, decode_timedelta=True),
        xr.decode_cf(baseline, decode_timedelta=True),
    )


@pytest.mark.parametrize("hours", [96, 168, 336, 504])
def test_known_anomalies_all_windows_preserve_both_populations(source, hours, tmp_path):
    climate, events, baseline = inputs(source, hours)
    originals = [d.copy(deep=True) for d in (source, climate, events, baseline)]
    ae, ab = build_component_anomaly_pair(source, climate, events, baseline)
    for variable, rate in (
        ("advection", 0.02),
        ("adiabatic", 0.01),
        ("diabatic", -0.005),
        ("dTdt", 0.025),
        ("dyn", 0.03),
    ):
        np.testing.assert_allclose(
            ae[f"I_{variable}_anom_pre"],
            np.array([-1, 0, 1]) * rate * (hours + 1),
            atol=1e-12,
        )
    for table in (ae, ab):
        assert not set(RAW_TO_ANOMALY).intersection(table.data_vars)
        assert table.attrs["heat_budget_representation"] == "climatological_anomaly"
    report = validate_anomalies_against_sources(
        source, climate, ae, ab, events, baseline
    )
    assert report["expected_hourly_samples"] == hours + 1
    assert report["raw_integrals_recalculated"] is False
    assert report["baseline"]["clean_rows"] == int((baseline.event_adjacent == 0).sum())
    path = analysis_io.save_component_anomalies(ae, tmp_path / "anomalies.nc")
    with analysis_io.open_component_anomalies(path) as saved:
        xr.testing.assert_equal(saved, ae)
    with pytest.raises(FileExistsError):
        analysis_io.save_component_anomalies(ae, path)
    for actual, original in zip(
        (source, climate, events, baseline), originals, strict=True
    ):
        xr.testing.assert_identical(actual, original)


def test_raw_integrals_are_neither_used_nor_recalculated(source):
    climate, events, baseline = inputs(source)
    for table in (events, baseline):
        for name in RAW_TO_ANOMALY:
            table[name].values[:] = np.nan
    ae, _ = build_component_anomaly_pair(source, climate, events, baseline)
    assert ae.I_dTdt_anom_pre.values[-1] == pytest.approx(97 * 0.025)


@pytest.mark.parametrize(
    "problem", ["gap", "nan", "inf", "missing_key", "count", "units", "closure"]
)
def test_incomplete_or_invalid_budget_cannot_change_population_silently(
    source, problem
):
    climate, events, baseline = inputs(source)
    stamp = np.datetime64("2021-06-27T12:00", "ns")
    key = np.datetime64("2000-06-27T12:00", "ns")
    if problem == "gap":
        source = source.drop_sel(time=stamp)
    elif problem in ("nan", "inf"):
        source.adiabatic.loc[{"time": stamp}] = np.nan if problem == "nan" else np.inf
    elif problem == "missing_key":
        climate = climate.drop_sel(climatology_time=key)
    elif problem == "count":
        climate.adiabatic_count.loc[{"climatology_time": key}] = 2
    elif problem == "units":
        climate.adiabatic.attrs["units"] = "K day-1"
    else:
        climate.diabatic.values[:] += 0.01
    with pytest.raises(ValueError):
        build_component_anomaly_pair(source, climate, events, baseline)


@pytest.mark.parametrize(
    "field,value",
    [
        ("region", "pnw_hotz"),
        ("heat_budget_top_boundary", "500hPa"),
        ("climatology_start_year", 1981),
        ("climatology_event_exclusion", "heatwaves"),
    ],
)
def test_incompatible_climatology_is_rejected(source, field, value):
    climate, events, baseline = inputs(source)
    climate.attrs[field] = value
    with pytest.raises(ValueError):
        build_component_anomaly_pair(source, climate, events, baseline)


@pytest.mark.parametrize(
    "problem", ["window", "identity", "peak", "count", "already_anomaly"]
)
def test_incompatible_reference_tables_are_rejected(source, problem):
    climate, events, baseline = inputs(source)
    if problem == "window":
        baseline.attrs["heat_budget_pre_window_hours"] = "-336,0"
    elif problem == "identity":
        events.event_id.values[-1] = 999
    elif problem == "peak":
        events.peak_time.values[-1] += np.timedelta64(1, "h")
    elif problem == "count":
        baseline.n_samples_heat_budget_pre.values[0] -= 1
    else:
        events.attrs["heat_budget_representation"] = "climatological_anomaly"
    with pytest.raises(ValueError):
        build_component_anomaly_pair(source, climate, events, baseline)


def test_independent_validation_catches_shift_even_when_closure_holds(source):
    climate, events, baseline = inputs(source)
    ae, ab = build_component_anomaly_pair(source, climate, events, baseline)
    ae.I_diabatic_anom_pre.values[0] += 1
    ae.I_dTdt_anom_pre.values[0] += 1
    validate_component_anomalies(ae)
    with pytest.raises(AssertionError, match="direct anomaly"):
        validate_anomalies_against_sources(source, climate, ae, ab, events, baseline)


@pytest.mark.parametrize("layout", ["full", "presentation"])
def test_anomaly_figures_use_correct_points_labels_and_colorbar(
    source, layout, tmp_path
):
    climate, events, baseline = inputs(source)
    ae, ab = build_component_anomaly_pair(source, climate, events, baseline)
    original = [d.copy(deep=True) for d in (ae, ab)]
    fig = plot_tendency_scatter(ab, ae, layout=layout)
    try:
        assert "Heating Anomalies" in fig._suptitle.get_text()
        axes = fig.axes[:-1]
        assert all("I'" in ax.get_ylabel() for ax in axes)
        assert fig.axes[-1].get_ylabel() == "Peak TAS Anomaly (K)"
        np.testing.assert_allclose(
            axes[0].collections[1].get_offsets(),
            np.column_stack([ae.I_adiabatic_anom_pre, ae.I_advection_anom_pre]),
        )
        np.testing.assert_array_equal(
            axes[0].collections[1].get_array(), events.tas_anom_peak
        )
        assert len(axes[0].collections[0].get_offsets()) == int(
            (baseline.event_adjacent == 0).sum()
        )
        if layout == "presentation":
            assert "I'_{dT/dt}" in fig._supxlabel.get_text()
            assert "2019-2021 climatology" in fig._supxlabel.get_text()
        plot_style.save_figure(fig, tmp_path / f"{layout}.png")
    finally:
        plt.close(fig)
    for actual, before in zip((ae, ab), original, strict=True):
        xr.testing.assert_identical(actual, before)


def test_plot_rejects_mixed_or_incompatible_anomaly_inputs(source):
    climate, events, baseline = inputs(source)
    ae, ab = build_component_anomaly_pair(source, climate, events, baseline)
    with pytest.raises(ValueError, match="same budget representation"):
        comparison_plot_views(baseline, ae)
    ab.attrs["region"] = "pnw_hotz"
    with pytest.raises(ValueError, match="region"):
        comparison_plot_views(ab, ae)


def test_runner_checks_reference_hashes_and_refuses_overwrite(tmp_path):
    paths = {name: tmp_path / f"{name}_features.nc" for name in ("event", "baseline")}
    paths["reference_manifest"] = tmp_path / "manifest.json"
    paths["reference_manifest"].write_text(
        json.dumps(
            {
                "input_sha256": "source",
                "sha256": {
                    paths["event"].name: "events",
                    paths["baseline"].name: "baseline",
                },
            }
        )
    )
    hashes = {"stage1": "source", "event": "events", "baseline": "baseline"}
    runner.verify_references(paths, hashes)
    with pytest.raises(ValueError, match="reference checksum"):
        runner.verify_references(paths, {**hashes, "event": "wrong"})
    with pytest.raises(ValueError, match="Stage-1 checksum"):
        runner.verify_references(paths, {**hashes, "stage1": "wrong"})
    with pytest.raises(FileExistsError):
        runner.main(
            [
                "--input-path",
                "unused.nc",
                "--climatology-path",
                "unused.nc",
                "--reference-dir",
                str(tmp_path),
                "--integration-hours",
                "96",
                "--output-dir",
                str(tmp_path),
            ]
        )


def test_runner_saves_validated_products_and_both_figure_layouts(
    source, tmp_path, monkeypatch
):
    """Exercise the file handoff using accepted decoded reference tables."""
    climate, events, baseline = inputs(source)
    stage1 = tmp_path / "source.nc"
    source.to_netcdf(stage1, engine="h5netcdf")
    climate.attrs["source_stage1_sha256"] = runner.sha256(stage1)
    climate_path = tmp_path / "climate.nc"
    climate.to_netcdf(climate_path, engine="h5netcdf")
    references = tmp_path / "references"
    references.mkdir()
    for name, table in (("event", events), ("baseline", baseline)):
        table.attrs["input_path"] = str(stage1)
        table.to_netcdf(references / f"{name}_features.nc", engine="h5netcdf")
    (references / "manifest.json").write_text(
        json.dumps(
            {
                "input_sha256": runner.sha256(stage1),
                "sha256": {
                    path.name: runner.sha256(path) for path in references.glob("*.nc")
                },
            }
        )
    )

    # This fixture intentionally omits unrelated Stage-1 diagnostics. All new
    # scientific checks and saved-product readers remain the production paths.
    def open_fixture(path):
        return xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True)

    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", open_fixture)
    monkeypatch.setattr(analysis_io, "open_regional_hourly_climatology", open_fixture)
    monkeypatch.setenv("EXPECTED_COMMIT", "synthetic-test")
    monkeypatch.setenv("PBS_JOBID", "synthetic-test")
    output = tmp_path / "anomaly-output"
    assert (
        runner.main(
            [
                "--input-path",
                str(stage1),
                "--climatology-path",
                str(climate_path),
                "--reference-dir",
                str(references),
                "--integration-hours",
                "96",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["validation"]["raw_integrals_recalculated"] is False
    assert len(manifest["output_sha256"]) == 10
    for name, expected in manifest["output_sha256"].items():
        assert runner.sha256(output / name) == expected
    for kind in ("event", "baseline"):
        with analysis_io.open_component_anomalies(
            output / f"{kind}_features_clim_anom.nc"
        ) as table:
            assert table.attrs["generating_commit"] == "synthetic-test"
    for layout in ("full", "presentation"):
        for suffix in ("png", "pdf"):
            assert (
                output / f"event_vs_clean_baseline_clim_anom_{layout}.{suffix}"
            ).stat().st_size > 1000
