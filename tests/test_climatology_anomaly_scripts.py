import xarray as xr

from HW_analysis.scripts import plot_composite_timeseries_all_clim_anom as all_plot
from HW_analysis.scripts import plot_composite_timeseries_split_clim_anom as split_plot


RUN_ARGS = [
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
    "1940",
    "--end-year",
    "2024",
]


def test_all_event_anomalies_are_applied_before_event_stacking(
    monkeypatch,
    tmp_path,
):
    stage1 = xr.Dataset()
    climate = xr.Dataset()
    anomaly_source = xr.Dataset(
        attrs={"data_representation": "climatological_anomaly"}
    )
    composite = xr.Dataset()
    captured = {}
    output = tmp_path / "all.png"

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_composite_timeseries_all_clim_anom.py",
            *RUN_ARGS,
            "--input-path",
            str(tmp_path / "stage1.nc"),
            "--climatology-path",
            str(tmp_path / "climate.nc"),
            "--output-path",
            str(output),
        ],
    )
    monkeypatch.setattr(all_plot.analysis_io, "open_harmonized_timeseries", lambda _: stage1)
    monkeypatch.setattr(
        all_plot.analysis_io,
        "open_regional_hourly_climatology",
        lambda _: climate,
    )

    def fake_apply(ds, baseline, **kwargs):
        captured["anomaly_input"] = ds
        captured["climate"] = baseline
        captured["apply_kwargs"] = kwargs
        return anomaly_source

    def fake_composite(ds, **kwargs):
        captured["composite_input"] = ds
        captured["event_table"] = kwargs["event_table"]
        return composite

    monkeypatch.setattr(all_plot.climatology, "apply_regional_hourly_climatology", fake_apply)
    monkeypatch.setattr(
        all_plot.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )
    monkeypatch.setattr(
        all_plot.plotting,
        "write_composite_timeseries_outputs",
        lambda ds, path, **kwargs: [path, kwargs["smoothed_output_path"]],
    )

    assert all_plot.main() == 0
    assert captured["anomaly_input"] is stage1
    assert captured["climate"] is climate
    assert captured["composite_input"] is anomaly_source
    assert captured["event_table"] is stage1
    assert composite.attrs["data_representation"] == "climatological_anomaly"


def test_split_anomalies_use_absolute_stage1_for_bin_membership(
    monkeypatch,
    tmp_path,
):
    stage1 = xr.Dataset()
    climate = xr.Dataset()
    anomaly_source = xr.Dataset(
        attrs={"data_representation": "climatological_anomaly"}
    )
    composite = xr.Dataset()
    captured = {}
    output = tmp_path / "split.png"

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_composite_timeseries_split_clim_anom.py",
            *RUN_ARGS,
            "--input-path",
            str(tmp_path / "stage1.nc"),
            "--climatology-path",
            str(tmp_path / "climate.nc"),
            "--output-path",
            str(output),
            "--split-variable",
            "duration",
            "--split-quantiles",
            "0.75",
        ],
    )
    monkeypatch.setattr(
        split_plot.analysis_io,
        "open_harmonized_timeseries",
        lambda _: stage1,
    )
    monkeypatch.setattr(
        split_plot.analysis_io,
        "open_regional_hourly_climatology",
        lambda _: climate,
    )
    monkeypatch.setattr(
        split_plot.climatology,
        "apply_regional_hourly_climatology",
        lambda *args, **kwargs: anomaly_source,
    )

    def fake_split(ds, **kwargs):
        captured["composite_input"] = ds
        captured["event_table"] = kwargs["event_table"]
        return composite

    monkeypatch.setattr(
        split_plot.absolute_plot,
        "build_split_quantile_composite",
        fake_split,
    )
    monkeypatch.setattr(
        split_plot.plotting,
        "write_split_composite_timeseries_outputs",
        lambda ds, path, **kwargs: [path, kwargs["smoothed_output_path"]],
    )

    assert split_plot.main() == 0
    assert captured["composite_input"] is anomaly_source
    assert captured["event_table"] is stage1
    assert composite.attrs["data_representation"] == "climatological_anomaly"


def test_anomaly_plot_workflows_refuse_existing_outputs(tmp_path):
    output = tmp_path / "existing.png"
    output.write_bytes(b"existing")

    for check in (all_plot._require_new_outputs, split_plot._require_new_outputs):
        try:
            check(output)
        except FileExistsError as exc:
            assert str(output) in str(exc)
        else:
            raise AssertionError("Expected an existing anomaly plot to be rejected.")
