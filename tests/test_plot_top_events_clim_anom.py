from pathlib import Path

import xarray as xr
from HW_analysis.scripts import plot_top_events_clim_anom as anomaly_plot

RUN_ARGS = [
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
    "1940",
    "--end-year",
    "2024",
]


def _argv(*extra: str) -> list[str]:
    return ["plot_top_events_clim_anom.py", *RUN_ARGS, *extra]


def test_parse_args_builds_isolated_anomaly_paths(monkeypatch):
    monkeypatch.setattr("sys.argv", _argv())

    args = anomaly_plot.parse_args()

    assert args.climatology_path == (
        anomaly_plot.analysis_io.default_regional_hourly_climatology_path(
            region="pnw_hotz",
            bottom_boundary="surface",
            top_boundary="700hPa",
            start_year=1940,
            end_year=2024,
        ).resolve()
    )
    assert args.output_dir == (
        anomaly_plot.REPO_ROOT
        / "results"
        / "plots_top_events_clim_anom"
        / "region_pnw_hotz"
        / "boundary_surface_700hPa"
        / "time_range_1940_2024"
    )


def test_parse_args_builds_isolated_anomaly_presentation_path(monkeypatch):
    monkeypatch.setattr("sys.argv", _argv("--layout", "presentation"))

    args = anomaly_plot.parse_args()

    assert args.output_dir.parts[-4] == "plots_top_events_clim_anom_presentation"
    assert args.layout == "presentation"


def test_main_ranks_absolute_events_and_anomalizes_before_plotting(
    monkeypatch,
    tmp_path,
):
    stage1 = xr.Dataset()
    climate = xr.Dataset()
    selected_events = xr.Dataset()
    anomaly_source = xr.Dataset(
        attrs={
            "data_representation": "climatological_anomaly",
            "climatology_start_year": 1940,
            "climatology_end_year": 2024,
        }
    )
    captured = {}
    output_dir = tmp_path / "top-events-anomaly"
    input_path = tmp_path / "stage1.nc"
    climatology_path = tmp_path / "climatology.nc"
    monkeypatch.setattr(
        "sys.argv",
        _argv(
            "--input-path",
            str(input_path),
            "--climatology-path",
            str(climatology_path),
            "--output-dir",
            str(output_dir),
            "--top-n",
            "2",
            "--plot-extended-variables",
        ),
    )
    monkeypatch.setattr(
        anomaly_plot.analysis_io,
        "open_harmonized_timeseries",
        lambda path: stage1,
    )
    monkeypatch.setattr(
        anomaly_plot.analysis_io,
        "open_regional_hourly_climatology",
        lambda path: climate,
    )
    monkeypatch.setattr(
        anomaly_plot.absolute_plot,
        "describe_harmonized_dataset",
        lambda ds: None,
    )

    def fake_select(source, *, n):
        captured["selection_source"] = source
        captured["top_n"] = n
        return selected_events

    def fake_apply(source, baseline, *, variables):
        captured["anomaly_source"] = source
        captured["climate"] = baseline
        captured["variables"] = variables
        return anomaly_source

    def fake_write(source, events, **kwargs):
        captured["plot_source"] = source
        captured["selected_events"] = events
        captured["plot_kwargs"] = kwargs
        return [output_dir / "raw.png", output_dir / "smoothed.png"]

    monkeypatch.setattr(
        anomaly_plot.absolute_plot, "select_top_tas_events", fake_select
    )
    monkeypatch.setattr(
        anomaly_plot.climatology,
        "apply_regional_hourly_climatology",
        fake_apply,
    )
    monkeypatch.setattr(
        anomaly_plot.absolute_plot,
        "write_top_event_plots",
        fake_write,
    )

    assert anomaly_plot.main() == 0
    assert captured["selection_source"] is stage1
    assert captured["top_n"] == 2
    assert captured["anomaly_source"] is stage1
    assert captured["climate"] is climate
    assert captured["variables"] == (
        anomaly_plot.absolute_plot.EXTENDED_TOP_EVENT_VARIABLES
    )
    assert captured["plot_source"] is anomaly_source
    assert captured["selected_events"] is selected_events
    assert captured["plot_kwargs"]["event_table"] is stage1
    assert captured["plot_kwargs"]["filename_tag"] == "clim_anom"
    assert anomaly_source.attrs["climatology_path"] == str(climatology_path.resolve())


def test_anomaly_workflow_refuses_existing_output_directory(tmp_path):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()

    try:
        anomaly_plot._require_new_output_dir(output_dir)
    except FileExistsError as exc:
        assert str(output_dir) in str(exc)
    else:
        raise AssertionError("Expected an existing anomaly output to be rejected.")


def test_explicit_climatology_path_is_resolved(monkeypatch, tmp_path):
    climatology_path = tmp_path / "climate.nc"
    monkeypatch.setattr(
        "sys.argv",
        _argv("--climatology-path", str(climatology_path)),
    )

    args = anomaly_plot.parse_args()

    assert args.climatology_path == climatology_path.resolve()
    assert isinstance(args.climatology_path, Path)
