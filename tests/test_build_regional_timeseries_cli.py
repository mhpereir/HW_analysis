from argparse import Namespace

import numpy as np
import pytest
import xarray as xr

from scripts import build_regional_timeseries
from src import analysis_io


def test_parse_args_requires_start_and_end_year(monkeypatch):
    monkeypatch.setattr("sys.argv", ["build_regional_timeseries.py"])

    with pytest.raises(SystemExit) as excinfo:
        build_regional_timeseries.parse_args()

    assert excinfo.value.code == 2


def test_parse_args_rejects_descending_year_range(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "2024",
            "--end-year",
            "1940",
        ],
    )

    with pytest.raises(SystemExit) as excinfo:
        build_regional_timeseries.parse_args()

    assert excinfo.value.code == 2


def test_parse_args_builds_inclusive_analysis_years(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "1940",
            "--end-year",
            "1942",
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.start_year == 1940
    assert args.end_year == 1942
    assert args.analysis_years == [1940, 1941, 1942]
    assert args.bottom_boundary == "surface"
    assert args.top_boundary == "700hPa"
    assert args.start_year_ehb == 1940
    assert args.end_year_ehb == 2025
    assert args.heat_budget_root == (
        build_regional_timeseries.data_io.era5_heat_budget_annual_root(
            region="pnw_bartusek",
            bottom_boundary="surface",
            top_boundary="700hPa",
            start_year_ehb=1940,
            end_year_ehb=2025,
        )
    )
    assert args.threshold_variable == "tas"
    assert args.add_full_diagnostics is False
    assert args.output_path == analysis_io.default_harmonized_timeseries_path(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary="700hPa",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=1942,
    )
    assert args.output_path.name == (
        "harmonized_regional_timeseries_pnw_bartusek_surface_700hPa_tas_q90_1940_1942.nc"
    )


def test_parse_args_builds_run_specific_output_path(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--region",
            "western_canada",
            "--quantile",
            "97p5",
            "--threshold-variable",
            "lwa_c",
            "--bottom-boundary",
            "700hPa",
            "--top-boundary",
            "500",
            "--start-year",
            "1950",
            "--end-year",
            "1951",
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.output_path.name == (
        "harmonized_regional_timeseries_western_canada_700hPa_500hPa_lwa_c_q97p5_1950_1951.nc"
    )
    assert args.heat_budget_root.name == "annual"
    assert args.heat_budget_root.parent.name == "western_canada_700hPa_500hPa_1940_2025"


def test_parse_args_accepts_custom_output_path(monkeypatch, tmp_path):
    output_path = tmp_path / "custom_stage1.nc"
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "1940",
            "--end-year",
            "1940",
            "--output-path",
            str(output_path),
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.output_path == output_path


def test_parse_args_accepts_add_full_diagnostics_flag(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "1940",
            "--end-year",
            "1940",
            "--add-full-diagnostics",
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.add_full_diagnostics is True


def test_parse_args_accepts_custom_ehb_year_tokens(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "build_regional_timeseries.py",
            "--start-year",
            "1950",
            "--end-year",
            "1951",
            "--start-year-ehb",
            "1940",
            "--end-year-ehb",
            "2026",
        ],
    )

    args = build_regional_timeseries.parse_args()

    assert args.heat_budget_root.parent.name == "pnw_bartusek_surface_700hPa_1940_2026"


def test_load_era5_inputs_loads_full_diagnostics_only_when_requested(monkeypatch):
    calls = []
    heat_budget_calls = []

    monkeypatch.setattr(build_regional_timeseries.data_io, "open_era5_tas", lambda **kwargs: xr.Dataset())
    monkeypatch.setattr(build_regional_timeseries.data_io, "open_era5_lwa", lambda **kwargs: xr.Dataset())
    monkeypatch.setattr(build_regional_timeseries.data_io, "open_era5_lwa_threshold", lambda **kwargs: xr.Dataset())
    monkeypatch.setattr(build_regional_timeseries.data_io, "open_era5_hw_threshold", lambda **kwargs: xr.Dataset())
    def fake_open_heat_budget(**kwargs):
        heat_budget_calls.append(kwargs)
        return xr.Dataset()

    monkeypatch.setattr(build_regional_timeseries.data_io, "open_era5_heat_budget", fake_open_heat_budget)

    def fake_load_full(args):
        calls.append(args)
        return {"pbl_p": xr.Dataset()}

    monkeypatch.setattr(build_regional_timeseries, "load_full_diagnostic_inputs", fake_load_full)

    args = Namespace(
        analysis_years=[1940],
        region="pnw_bartusek",
        quantile="90",
        zg_level=500,
        heat_budget_root="/data/heat_budget/annual",
        add_full_diagnostics=False,
    )
    datasets = build_regional_timeseries.load_era5_inputs(args)

    assert calls == []
    assert heat_budget_calls[-1] == {
        "years": [1940],
        "heat_budget_root": "/data/heat_budget/annual",
    }
    assert "pbl_p" not in datasets

    args.add_full_diagnostics = True
    datasets = build_regional_timeseries.load_era5_inputs(args)

    assert calls == [args]
    assert "pbl_p" in datasets


def test_append_event_summary_table_defaults_to_tas_events():
    ds = _make_harmonized_timeseries_for_events()

    out = build_regional_timeseries.append_event_summary_table(ds)

    assert out.sizes["time"] == 4
    assert out.sizes["event"] == 2
    assert {"event_id", "start_time", "end_time", "duration", "tas_anom_peak"} <= set(out)
    np.testing.assert_array_equal(out["event_id"].values, [1, 2])
    np.testing.assert_array_equal(out["duration"].values, [1, 1])
    np.testing.assert_allclose(out["tas_anom_peak"].values, [5.0, 5.0])
    assert out.attrs["event_id_source"] == "hw_event_id"
    assert out.attrs["peak_variable"] == "tas_region"


def test_append_hw_event_summary_table_keeps_legacy_tas_behavior():
    ds = _make_harmonized_timeseries_for_events()

    out = build_regional_timeseries.append_hw_event_summary_table(ds)

    assert out.attrs["event_id_source"] == "hw_event_id"
    assert out.attrs["peak_variable"] == "tas_region"


@pytest.mark.parametrize(
    ("threshold_variable", "event_id_source", "peak_variable", "expected_event_ids"),
    [
        ("lwa", "lwa_event_id", "lwa_region", [7]),
        ("lwa_a", "lwa_a_event_id", "lwa_a_region", [11]),
        ("lwa_c", "lwa_c_event_id", "lwa_c_region", [21]),
    ],
)
def test_append_event_summary_table_selects_threshold_variable_products(
    threshold_variable,
    event_id_source,
    peak_variable,
    expected_event_ids,
):
    ds = _make_harmonized_timeseries_for_events()

    out = build_regional_timeseries.append_event_summary_table(
        ds,
        threshold_variable=threshold_variable,
    )

    assert out.attrs["event_id_source"] == event_id_source
    assert out.attrs["peak_variable"] == peak_variable
    np.testing.assert_array_equal(out["event_id"].values, expected_event_ids)


def test_event_summary_variables_rejects_unknown_threshold_variable():
    with pytest.raises(ValueError, match="Unsupported threshold variable"):
        build_regional_timeseries.event_summary_variables("other")


def _make_harmonized_timeseries_for_events() -> xr.Dataset:
    time = np.array(
        [
            "2000-05-01T00:00",
            "2000-05-01T12:00",
            "2000-05-02T00:00",
            "2000-05-03T00:00",
        ],
        dtype="datetime64[m]",
    )
    return xr.Dataset(
        data_vars={
            "hw_event_id": ("time", np.array([1, 1, 0, 2], dtype=np.int64)),
            "tas_region": ("time", np.array([300.0, 301.0, 280.0, 305.0])),
            "tas_climatology": ("time", np.array([295.0, 295.0, 275.0, 300.0])),
            "hw_threshold": ("time", np.array([298.0, 298.0, 278.0, 303.0])),
            "lwa_region": ("time", np.array([1.0, 4.0, 0.5, 2.0])),
            "lwa_event_id": ("time", np.array([0, 7, 7, 0], dtype=np.int64)),
            "lwa_a_region": ("time", np.array([1.0, 2.0, 0.5, 3.0])),
            "lwa_a_event_id": ("time", np.array([11, 11, 0, 0], dtype=np.int64)),
            "lwa_c_region": ("time", np.array([4.0, 5.0, 0.5, 6.0])),
            "lwa_c_event_id": ("time", np.array([0, 0, 21, 21], dtype=np.int64)),
        },
        coords={"time": time},
    )
