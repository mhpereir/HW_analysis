from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import xarray as xr
from HW_analysis.scripts import (
    build_stage1_advection_exploration as build_script,
)
from HW_analysis.scripts import (
    plot_advection_direction_exploration as plot_script,
)
from HW_analysis.scripts import (
    plot_advection_direction_exploration_clim_anom as anomaly_script,
)
from HW_analysis.scripts import (
    plot_advection_direction_exploration_matched_clim_anom as matched_script,
)
from HW_analysis.src import advection_direction_plotting

STAGE1_CLI_ARGS = [
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


def test_builder_defaults_to_isolated_stage1_subfolder():
    args = build_script.finalize_args(_base_args())

    assert args.input_path.parent.name == "stage1"
    assert args.output_path.parent.name == "advection_direction_exploration"
    assert args.output_path.name == args.input_path.name
    assert args.output_path != args.input_path


def test_builder_rejects_output_equal_to_input():
    args = _base_args()
    shared = Path("/tmp/shared-stage1.nc")
    args.input_path = shared
    args.output_path = shared

    try:
        build_script.finalize_args(args)
    except ValueError as exc:
        assert "must differ" in str(exc)
    else:
        raise AssertionError("Expected equal input and output paths to fail.")


def test_plot_defaults_to_enhanced_stage1_and_separate_plot_tree():
    args = _base_args()
    args.window_days = 7
    args.season_months = [6, 7, 8]
    args.require_full_event = True

    out = plot_script.finalize_args(args)

    assert out.input_path.parent.name == "advection_direction_exploration"
    assert "plots_advection_direction_exploration" in out.output_path.parts
    assert out.output_path.name == "advection_face_contributions_two_panel.png"
    assert out.smoothing_window == 24


def test_plot_smoothed_output_is_a_distinct_sibling():
    output = Path("/tmp/advection_face_contributions_two_panel.png")

    smoothed = advection_direction_plotting.smoothed_output_path(output)

    assert smoothed.name == "advection_face_contributions_two_panel_smoothed.png"
    assert smoothed != output


def test_all_three_plot_entrypoints_default_to_24_hour_smoothing(
    monkeypatch,
    tmp_path,
):
    entrypoints = (
        (plot_script, []),
        (anomaly_script, []),
        (
            matched_script,
            ["--event-features-path", str(tmp_path / "features.nc")],
        ),
    )
    for module, extra_args in entrypoints:
        monkeypatch.setattr(
            "sys.argv",
            [module.__file__, *STAGE1_CLI_ARGS, *extra_args],
        )

        args = module.parse_args()

        assert args.smoothing_window == 24


def test_absolute_plot_main_routes_raw_and_smoothed_outputs(monkeypatch, tmp_path):
    output = tmp_path / "absolute.png"
    args = Namespace(
        input_path=tmp_path / "stage1.nc",
        output_path=output,
        smoothing_window=24,
        overwrite=False,
    )
    stage1 = xr.Dataset()
    composite = xr.Dataset()
    captured = {}

    monkeypatch.setattr(plot_script, "parse_args", lambda: args)
    monkeypatch.setattr(
        plot_script.analysis_io,
        "open_harmonized_timeseries",
        lambda path: stage1,
    )
    monkeypatch.setattr(plot_script, "build_composite", lambda ds, parsed: composite)

    def fake_write(ds, path, **kwargs):
        captured.update(ds=ds, path=path, kwargs=kwargs)
        return [path, kwargs["smoothed_output_path"]]

    monkeypatch.setattr(
        plot_script.advection_direction_plotting,
        "write_advection_direction_exploration_outputs",
        fake_write,
    )

    assert plot_script.main() == 0
    assert captured == {
        "ds": composite,
        "path": output,
        "kwargs": {
            "smoothed_output_path": output.with_name("absolute_smoothed.png"),
            "smoothing_window": 24,
        },
    }


def test_anomaly_plot_main_routes_raw_and_smoothed_outputs(monkeypatch, tmp_path):
    output = tmp_path / "anomaly.png"
    args = Namespace(
        input_path=tmp_path / "stage1.nc",
        climatology_path=tmp_path / "climatology.nc",
        output_path=output,
        window_days=7,
        smoothing_window=24,
        season_months=None,
        require_full_event=False,
    )
    stage1 = _face_source()
    climate = xr.Dataset()
    anomaly_source = stage1.copy(deep=True)
    composite = xr.Dataset()
    captured = {}

    monkeypatch.setattr(anomaly_script, "parse_args", lambda: args)
    monkeypatch.setattr(
        anomaly_script.analysis_io,
        "open_harmonized_timeseries",
        lambda path: stage1,
    )
    monkeypatch.setattr(
        anomaly_script.analysis_io,
        "open_regional_hourly_climatology",
        lambda path: climate,
    )
    monkeypatch.setattr(
        anomaly_script.climatology,
        "apply_regional_hourly_climatology",
        lambda *positional, **kwargs: anomaly_source,
    )
    monkeypatch.setattr(
        anomaly_script.composites,
        "all_event_peak_aligned_composite",
        lambda *positional, **kwargs: composite,
    )

    def fake_write(ds, path, **kwargs):
        captured.update(ds=ds, path=path, kwargs=kwargs)
        return [path, kwargs["smoothed_output_path"]]

    monkeypatch.setattr(
        anomaly_script.advection_direction_plotting,
        "write_advection_direction_exploration_outputs",
        fake_write,
    )

    assert anomaly_script.main() == 0
    assert captured["ds"] is composite
    assert captured["path"] == output
    assert captured["kwargs"] == {
        "smoothed_output_path": output.with_name("anomaly_smoothed.png"),
        "smoothing_window": 24,
    }


def test_matched_plot_main_routes_raw_and_smoothed_outputs(monkeypatch, tmp_path):
    output = tmp_path / "matched.png"
    input_paths = {
        name: tmp_path / name
        for name in ("stage1.nc", "climatology.nc", "features.nc", "settings.json")
    }
    for path in input_paths.values():
        path.write_bytes(b"test")
    args = Namespace(
        input_path=input_paths["stage1.nc"],
        climatology_path=input_paths["climatology.nc"],
        event_features_path=input_paths["features.nc"],
        matching_settings_path=input_paths["settings.json"],
        matching_specification="peak_anomaly_0p20",
        output_path=output,
        window_days=7,
        smoothing_window=24,
    )
    stage1 = _face_source()
    climate = xr.Dataset()
    event_features = xr.Dataset()
    negative = xr.Dataset()
    positive = xr.Dataset()
    settings = SimpleNamespace(sha256="settings-sha")
    prepared = SimpleNamespace(
        negative=negative,
        positive=positive,
        specification=SimpleNamespace(identifier="peak_anomaly_0p20"),
        match=SimpleNamespace(pair_count=4),
    )
    captured = {}

    monkeypatch.setattr(matched_script, "parse_args", lambda: args)
    monkeypatch.setattr(
        matched_script.matching_settings,
        "load_matching_settings",
        lambda path: settings,
    )
    monkeypatch.setattr(
        matched_script.analysis_io,
        "open_harmonized_timeseries",
        lambda path: stage1,
    )
    monkeypatch.setattr(
        matched_script.analysis_io,
        "open_regional_hourly_climatology",
        lambda path: climate,
    )
    monkeypatch.setattr(
        matched_script, "open_event_features", lambda path: event_features
    )
    monkeypatch.setattr(matched_script, "sha256_file", lambda path: "features-sha")
    monkeypatch.setattr(
        matched_script.climatology,
        "apply_regional_hourly_climatology",
        lambda *positional, **kwargs: stage1,
    )
    monkeypatch.setattr(
        matched_script,
        "build_matched_composites",
        lambda *positional, **kwargs: prepared,
    )

    def fake_write(negative_ds, positive_ds, path, **kwargs):
        captured.update(
            negative=negative_ds,
            positive=positive_ds,
            path=path,
            kwargs=kwargs,
        )
        return [path, kwargs["smoothed_output_path"]]

    monkeypatch.setattr(
        matched_script.advection_direction_plotting,
        "write_matched_advection_direction_exploration_outputs",
        fake_write,
    )

    assert matched_script.main() == 0
    assert captured["negative"] is negative
    assert captured["positive"] is positive
    assert captured["path"] == output
    assert captured["kwargs"] == {
        "smoothed_output_path": output.with_name("matched_smoothed.png"),
        "smoothing_window": 24,
    }


def _face_source() -> xr.Dataset:
    return xr.Dataset(
        {
            "advection": ("time", [0.0]),
            "advection_west": ("time", [0.0]),
            "advection_east": ("time", [0.0]),
            "advection_south": ("time", [0.0]),
            "advection_north": ("time", [0.0]),
            "advection_top": ("time", [0.0]),
        }
    )


def _base_args() -> Namespace:
    return Namespace(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary="700",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=2024,
        input_path=None,
        output_path=None,
        start_year_ehb=1940,
        end_year_ehb=2025,
        heat_budget_root=None,
        smoothing_window=24,
        overwrite=False,
    )
