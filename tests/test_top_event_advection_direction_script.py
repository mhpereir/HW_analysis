import argparse

import numpy as np
import pytest
import xarray as xr
from HW_analysis.scripts import (
    plot_advection_direction_exploration_top_events_clim_anom as script,
)
from HW_analysis.src import analysis_io

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


def test_parse_args_builds_isolated_default_paths(monkeypatch):
    monkeypatch.setattr("sys.argv", [script.__file__, *RUN_ARGS])

    args = script.parse_args()

    assert args.input_path == analysis_io.default_harmonized_timeseries_path(
        region="pnw_hotz",
        bottom_boundary="surface",
        top_boundary="700hPa",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=2024,
    )
    assert args.climatology_path == (
        analysis_io.default_regional_hourly_climatology_path(
            region="pnw_hotz",
            bottom_boundary="surface",
            top_boundary="700hPa",
            start_year=1940,
            end_year=2024,
        ).resolve()
    )
    assert args.output_dir == (
        script.REPO_ROOT
        / "results"
        / "plots_advection_direction_exploration_top_events_clim_anom"
        / "region_pnw_hotz"
        / "boundary_surface_700hPa"
        / "time_range_1940_2024"
    )
    assert args.top_n == 10
    assert args.window_days == 7
    assert args.smoothing_window == 24


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("top_n", 0, "--top-n"),
        ("window_days", 0, "--window-days"),
        ("smoothing_window", 0, "--smoothing-window"),
    ],
)
def test_validate_args_rejects_nonpositive_values(attribute, value, message):
    args = argparse.Namespace(
        top_n=10,
        window_days=7,
        smoothing_window=24,
        season_months=None,
        require_full_event=False,
    )
    setattr(args, attribute, value)

    with pytest.raises(ValueError, match=message):
        script.validate_args(args)


def test_build_inputs_uses_filtered_absolute_events_for_rank_and_reference(
    monkeypatch,
):
    stage1 = _make_stage1()
    climate = xr.Dataset()
    anomaly_source = stage1.copy(deep=True)
    anomaly_source.attrs["data_representation"] = "climatological_anomaly"
    reference = xr.Dataset(attrs={"n_events": 2})
    stacked = xr.Dataset(
        {
            "event_id": ("event", [3, 1]),
            "selection_rank": ("event", [1, 2]),
        },
        coords={"event": [3, 1]},
    )
    captured = {}

    def fake_apply(source, climatology, *, variables):
        captured["apply"] = (source, climatology, variables)
        return anomaly_source

    def fake_build(plot_source, absolute_stage1, **kwargs):
        captured["build"] = (plot_source, absolute_stage1, kwargs)
        return reference, stacked

    monkeypatch.setattr(
        script.climatology,
        "apply_regional_hourly_climatology",
        fake_apply,
    )
    monkeypatch.setattr(
        script.advection_direction_top_events,
        "build_top_event_inputs",
        fake_build,
    )

    out_reference, out_events = script.build_top_event_anomaly_inputs(
        stage1,
        climate,
        top_n=2,
        window_days=5,
        season_months=[6, 7, 8],
        require_full_event=True,
    )

    apply_source, apply_climate, variables = captured["apply"]
    assert apply_source is stage1
    assert apply_climate is climate
    assert variables == (
        "advection",
        "advection_west",
        "advection_east",
        "advection_south",
        "advection_north",
        "advection_top",
    )
    plot_source, absolute_source, build_kwargs = captured["build"]
    assert plot_source is anomaly_source
    assert absolute_source is stage1
    assert build_kwargs == {
        "data_representation": "climatological_anomaly",
        "top_n": 2,
        "window_days": 5,
        "season_months": [6, 7, 8],
        "require_full_event": True,
    }
    assert out_reference is reference
    assert out_events is stacked


def test_require_new_output_dir_rejects_existing_directory(tmp_path):
    with pytest.raises(FileExistsError, match="already exists"):
        script._require_new_output_dir(tmp_path)


def _make_stage1() -> xr.Dataset:
    time = np.arange(
        np.datetime64("2000-06-01T00"),
        np.datetime64("2000-06-01T03"),
        np.timedelta64(1, "h"),
    )
    faces = {
        "advection_west": np.array([0.1, 0.2, 0.3]),
        "advection_east": np.array([-0.1, -0.1, -0.1]),
        "advection_south": np.array([0.05, 0.05, 0.05]),
        "advection_north": np.array([-0.02, -0.02, -0.02]),
        "advection_top": np.array([0.01, 0.01, 0.01]),
    }
    return xr.Dataset(
        {
            **{name: ("time", values) for name, values in faces.items()},
            "advection": ("time", sum(faces.values())),
            "event_id": ("event", [1, 2, 3]),
            "tas_peak": ("event", [300.0, 301.0, 305.0]),
            "peak_time": (
                "event",
                np.array(
                    [
                        "2000-06-01T00",
                        "2000-06-01T01",
                        "2000-06-01T02",
                    ],
                    dtype="datetime64[h]",
                ),
            ),
        },
        coords={"time": time, "event": [0, 1, 2]},
    )
