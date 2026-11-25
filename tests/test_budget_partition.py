"""Independent regrouping, product and rendering checks for both partitions."""

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from test_plotting import _make_composite, _make_split_composite
from test_stage2_component_anomalies import source as base_source

from scripts.event_features.build_stage2_baseline_features import (
    build_baseline_features,
)
from scripts.event_features.build_stage2_event_features import build_event_features
from scripts.event_features.plot_adiabatic_advection_comparison_baseline import (
    plot_tendency_scatter,
)
from src import budget_partition as bp
from src import climatology, plotting
from src.budget_partition_validation import validate_products
from src.stage2_component_anomalies import TENDENCIES, build_component_anomaly_pair
from src.stage2_component_anomaly_validation import validate_anomalies_against_sources


@pytest.fixture
def source():
    return base_source.__wrapped__()


def add_faces(source, *, bottom=False):
    ds = source.copy(deep=True)
    # Strongly covarying terms ensure transformed spread is not a sum of spreads.
    for face, fraction in (
        ("west", 0.4),
        ("east", -0.1),
        ("south", 0.2),
        ("north", 0.1),
        ("top", 0.4),
    ):
        ds[f"advection_{face}"] = ds.advection * fraction
        ds[f"advection_{face}"].attrs["units"] = "K hr-1"
    if bottom:
        ds.attrs["heat_budget_bottom_boundary"] = "850hPa"
        ds["advection_bottom"] = ds.advection * -0.2
        ds["advection_top"] = ds.advection * 0.6
        for name in ("advection_top", "advection_bottom"):
            ds[name].attrs["units"] = "K hr-1"
    return ds


@pytest.mark.parametrize("bottom", [False, True])
def test_signed_regrouping_preserves_every_original_array(source, bottom):
    original = add_faces(source, bottom=bottom)
    before = original.copy(deep=True)
    out = bp.add_boundary_flux_fields(original)
    np.testing.assert_allclose(out.adv_horizontal, original.advection * 0.6, atol=1e-14)
    np.testing.assert_allclose(
        out.vertical_contribution, original.adiabatic + original.advection * 0.4
    )
    np.testing.assert_allclose(
        out.dTdt,
        out.adv_horizontal + out.vertical_contribution + out.diabatic,
        atol=1e-14,
    )
    for name in original.variables:
        xr.testing.assert_identical(out[name], original[name])
    xr.testing.assert_identical(original, before)
    xr.testing.assert_identical(bp.add_boundary_flux_fields(out), out)


@pytest.mark.parametrize(
    "fault",
    [
        "missing_top",
        "missing_bottom",
        "closure",
        "units",
        "partial",
        "metadata",
        "nan_mask",
    ],
)
def test_invalid_extensions_fail(source, fault):
    ds = add_faces(source, bottom=True)
    if fault == "missing_top":
        ds = ds.drop_vars("advection_top")
    elif fault == "missing_bottom":
        ds = ds.drop_vars("advection_bottom")
    elif fault == "closure":
        ds.advection_top.values[0] += 1
    elif fault == "units":
        ds.advection_top.attrs["units"] = "K s-1"
    elif fault == "nan_mask":
        ds.advection_top.values[0] = np.nan
    else:
        ds = bp.add_boundary_flux_fields(ds)
        if fault == "partial":
            ds = ds.drop_vars("vertical_contribution")
        else:
            ds.attrs["budget_partition_method"] = "local_gradient"
    with pytest.raises(ValueError):
        bp.add_boundary_flux_fields(ds)


def test_legacy_selection_and_companion_rejection(source):
    xr.testing.assert_equal(bp.select(source).drop_attrs(), source.drop_attrs())
    with pytest.raises(ValueError, match="regenerated"):
        bp.select(source, bp.BOUNDARY_FLUX)
    extended = bp.add_boundary_flux_fields(add_faces(source))
    climate = climatology.build_regional_hourly_climatology(
        source, variables=TENDENCIES
    )
    with pytest.raises(ValueError):
        climatology.apply_regional_hourly_climatology(
            extended, climate, variables=bp.VARIABLES
        )


def test_climatology_and_all_row_integrals_from_samples(source, tmp_path):
    original = source.copy(deep=True)
    extended = bp.add_boundary_flux_fields(add_faces(source))
    names = (*TENDENCIES, *bp.VARIABLES)
    climate = climatology.build_regional_hourly_climatology(extended, variables=names)
    for name in bp.VARIABLES:
        samples = extended[name].values.reshape(3, -1)
        np.testing.assert_allclose(climate[name], samples.mean(axis=0), atol=1e-14)
        np.testing.assert_allclose(
            climate[f"{name}_std"], samples.std(axis=0, ddof=1), atol=1e-14
        )
        np.testing.assert_equal(climate[f"{name}_count"].values, 3)
    common = {"integration_hours": 96}
    events = build_event_features(extended, **common)
    baseline = build_baseline_features(extended, **common)
    references = {
        "event": build_event_features(source, **common),
        "baseline": build_baseline_features(source, **common),
    }
    validation = validate_products(
        add_faces(source), extended, climate, events, baseline, references
    )
    assert validation["event"]["rows"] == 3
    corrupted = events.copy(deep=True)
    corrupted.I_adv_horizontal_pre.values[0] += 0.1
    with pytest.raises(AssertionError):
        validate_products(
            add_faces(source), extended, climate, corrupted, baseline, references
        )
    for table, old, anchor in (
        (events, build_event_features(source, **common), "peak_time"),
        (baseline, build_baseline_features(source, **common), "reference_time"),
    ):
        for name in old.variables:
            xr.testing.assert_identical(table[name], old[name])
        for t, index in zip(
            table[anchor].values, range(table.sizes[table[anchor].dims[0]]), strict=True
        ):
            for name in bp.VARIABLES:
                direct = (
                    extended[name]
                    .sel(time=slice(t - np.timedelta64(96, "h"), t))
                    .values
                )
                assert direct.size == 97
                np.testing.assert_allclose(
                    table[f"I_{name}_pre"].values[index], direct.sum(), atol=1e-12
                )
    ae, ab = build_component_anomaly_pair(
        extended, climate, events, baseline, require_full_event=False
    )
    report = validate_anomalies_against_sources(
        extended, climate, ae, ab, events, baseline
    )
    assert set(bp.VARIABLES) <= set(
        report["event"]["maximum_direct_anomaly_difference_K"]
    )
    for table in (ae, ab):
        bp.validate_integrals(table, anomaly=True)
        path = tmp_path / f"{table.attrs['pipeline_stage']}.nc"
        table.to_netcdf(path, engine="h5netcdf")
        with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as saved:
            bp.validate_integrals(saved, anomaly=True)
    xr.testing.assert_identical(source, original)


def test_stage1_migration_is_additive_and_never_overwrites(source, tmp_path):
    from scripts.build_stage1_budget_partitions import build_product
    from src import analysis_io

    ds = add_faces(source)
    for name in analysis_io.REQUIRED_HARMONIZED_VARIABLES:
        if name not in ds:
            ds[name] = xr.ones_like(ds.advection)
    ds.attrs["pipeline_stage"] = analysis_io.EXPECTED_PIPELINE_STAGE
    source_path, output_path = tmp_path / "old.nc", tmp_path / "new.nc"
    analysis_io.save_harmonized_timeseries(ds, source_path)
    original_bytes = source_path.read_bytes()
    build_product(source_path, output_path)
    with (
        analysis_io.open_harmonized_timeseries(source_path) as original,
        analysis_io.open_harmonized_timeseries(output_path) as actual,
    ):
        for name in original.variables:
            xr.testing.assert_identical(actual[name], original[name])
        bp.validate_stage1(actual)
    assert source_path.read_bytes() == original_bytes
    with pytest.raises(FileExistsError):
        build_product(source_path, output_path)
    with pytest.raises(FileExistsError):
        build_product(source_path, source_path)


@pytest.mark.parametrize("anomaly", [False, True])
def test_all_event_cli_selects_before_reductions(
    source, tmp_path, monkeypatch, anomaly
):
    from scripts import plot_composite_timeseries_all as absolute_cli
    from scripts import plot_composite_timeseries_all_clim_anom as anomaly_cli

    cli = anomaly_cli if anomaly else absolute_cli
    ds = bp.add_boundary_flux_fields(add_faces(source))
    for name in absolute_cli.EXTENDED_COMPOSITE_VARIABLES:
        if name not in ds:
            ds[name] = xr.ones_like(ds.advection)
    variables = (*absolute_cli.EXTENDED_COMPOSITE_VARIABLES, *bp.VARIABLES)
    climate = climatology.build_regional_hourly_climatology(ds, variables=variables)
    captured = {}

    def write(composite, path, **kwargs):
        captured["composite"] = composite
        return [path]

    monkeypatch.setattr(cli.analysis_io, "open_harmonized_timeseries", lambda _: ds)
    monkeypatch.setattr(
        cli.analysis_io, "open_regional_hourly_climatology", lambda _: climate
    )
    monkeypatch.setattr(cli.plotting, "write_composite_timeseries_outputs", write)
    argv = [
        "plot",
        "--region",
        "pnw_bartusek",
        "--bottom-boundary",
        "surface",
        "--top-boundary",
        "700",
        "--threshold-variable",
        "tas",
        "--quantile",
        "90",
        "--start-year",
        "2019",
        "--end-year",
        "2021",
        "--input-path",
        str(tmp_path / "stage1.nc"),
        "--output-path",
        str(tmp_path / "all.png"),
        "--budget-partition",
        "boundary_flux",
        "--plot-extended-variables",
    ]
    if anomaly:
        argv += [
            "--climatology-path",
            str(tmp_path / "climate.nc"),
            "--no-climatology-legend",
        ]
    monkeypatch.setattr("sys.argv", argv)
    cli.main()
    composite = captured["composite"]
    assert composite.attrs["budget_partition"] == bp.BOUNDARY_FLUX
    assert "advection" not in composite and "adiabatic" not in composite
    target_times = ds.peak_time.values[:, None] + np.arange(-168, 169).astype(
        "timedelta64[h]"
    )
    indices = np.searchsorted(ds.time.values, target_times)
    for name in bp.VARIABLES:
        values = ds[name].values
        if anomaly:
            # Each May-Sep calendar appears three times in fixture year order.
            values = values - np.tile(climate[name].values, 3)
        samples = values[indices]
        np.testing.assert_allclose(composite[name], samples.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(
            composite[f"event_percentile_{name}"],
            np.quantile(samples, [0.25, 0.5, 0.75], axis=0),
            atol=1e-12,
        )


@pytest.mark.parametrize("anomaly", [False, True])
@pytest.mark.parametrize("layout", ["full", "presentation"])
def test_integration_figures_select_named_fields(source, anomaly, layout):
    ds = bp.add_boundary_flux_fields(add_faces(source))
    climate = climatology.build_regional_hourly_climatology(
        ds, variables=(*TENDENCIES, *bp.VARIABLES)
    )
    events, baseline = build_event_features(ds), build_baseline_features(ds)
    if anomaly:
        events, baseline = build_component_anomaly_pair(
            ds, climate, events, baseline, require_full_event=False
        )
    before = events.copy(deep=True)
    fig = plot_tendency_scatter(
        baseline, events, partition=bp.BOUNDARY_FLUX, layout=layout
    )
    try:
        suffix = "_anom_pre" if anomaly else "_pre"
        offsets = np.asarray(fig.axes[0].collections[1].get_offsets())
        np.testing.assert_allclose(
            offsets[:, 0], events[f"I_vertical_contribution{suffix}"]
        )
        np.testing.assert_allclose(offsets[:, 1], events[f"I_adv_horizontal{suffix}"])
        assert "adv,h" in fig.axes[0].get_ylabel()
        xaxis = fig.axes[0] if layout == "presentation" else fig.axes[1]
        assert "vertical" in xaxis.get_xlabel()
        assert ("I'" in fig.axes[0].get_ylabel()) == anomaly
        assert len(fig.axes) == (3 if layout == "presentation" else 5)
        fig.canvas.draw()
    finally:
        plt.close(fig)
    xr.testing.assert_identical(events, before)


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("anomaly", [False, True])
@pytest.mark.parametrize("smooth", [False, True])
def test_ten_panel_values_labels_envelopes_and_smoothing(split, anomaly, smooth):
    ds = _make_split_composite() if split else _make_composite()
    # Supply independently prepared new means and envelopes, not transformed quantiles.
    for role, name in bp.ROLE_NAMES.items():
        ds[name] = ds[role] * -0.25
        ds[f"event_percentile_{name}"] = ds[f"event_percentile_{role}"] * 0.17
        ds[name].attrs["climatology_jja_mean"] = 0.123
    ds.attrs["budget_partition"] = bp.BOUNDARY_FLUX
    if anomaly:
        ds.attrs["data_representation"] = "climatological_anomaly"
    if smooth:
        ds = plotting.smooth_composite_for_display(
            ds, variables=plotting.EXTENDED_SMOOTHED_VARIABLES, smoothing_window=3
        )
    fn = (
        plotting.plot_split_composite_timeseries
        if split
        else plotting.plot_composite_timeseries
    )
    fig = fn(ds, plot_extended_variables=True)
    try:
        for index, name, label in (
            (4, "adv_horizontal", "Horizontal advection"),
            (6, "vertical_contribution", "Total vertical contribution"),
        ):
            expected = ds[name].isel(split_bin=0) if split else ds[name]
            np.testing.assert_allclose(
                fig.axes[index].lines[0].get_ydata(), expected, equal_nan=True
            )
            legends = [t.get_text() for t in fig.axes[index].get_legend().get_texts()]
            assert any(label in text for text in legends)
            assert any("JJA clim." in text for text in legends) == anomaly
        assert len(fig.axes) == 10
    finally:
        plt.close(fig)
