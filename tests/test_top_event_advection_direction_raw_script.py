import argparse

import pytest
import xarray as xr
from HW_analysis.scripts import (
    plot_advection_direction_exploration_top_events as script,
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
    assert args.output_dir == (
        script.REPO_ROOT
        / "results"
        / "plots_advection_direction_exploration_top_events"
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


def test_build_inputs_marks_values_as_absolute(monkeypatch):
    stage1 = xr.Dataset()
    reference = xr.Dataset()
    events = xr.Dataset()
    captured = {}

    def fake_build(plot_source, absolute_stage1, **kwargs):
        captured["call"] = (plot_source, absolute_stage1, kwargs)
        return reference, events

    monkeypatch.setattr(
        script.advection_direction_top_events,
        "build_top_event_inputs",
        fake_build,
    )

    result = script.build_top_event_inputs(
        stage1,
        top_n=4,
        window_days=3,
        season_months=[6, 7, 8],
        require_full_event=True,
    )

    assert result == (reference, events)
    plot_source, absolute_source, kwargs = captured["call"]
    assert plot_source is stage1
    assert absolute_source is stage1
    assert kwargs == {
        "data_representation": "absolute",
        "top_n": 4,
        "window_days": 3,
        "season_months": [6, 7, 8],
        "require_full_event": True,
    }


def test_require_new_output_dir_rejects_existing_directory(tmp_path):
    with pytest.raises(FileExistsError, match="already exists"):
        script._require_new_output_dir(tmp_path)
