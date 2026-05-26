import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from scripts import plot_diurnal_cycle
from src import analysis_io


RUN_ARGS = [
    "--region", "pnw_hotz",
    "--bottom-boundary", "surface",
    "--top-boundary", "700",
    "--threshold-variable", "tas",
    "--quantile", "90",
    "--start-year", "1940",
    "--end-year", "2024",
]


def _argv(*extra: str) -> list[str]:
    return ["plot_diurnal_cycle.py", *RUN_ARGS, *extra]


def test_parse_args_builds_default_paths(monkeypatch):
    monkeypatch.setattr("sys.argv", _argv())

    args = plot_diurnal_cycle.parse_args()

    assert args.input_path == analysis_io.default_harmonized_timeseries_path(
        region="pnw_hotz",
        bottom_boundary="surface",
        top_boundary="700hPa",
        threshold_variable="tas",
        quantile="90",
        start_year=1940,
        end_year=2024,
    )
    assert args.output_path == (
        plot_diurnal_cycle.REPO_ROOT
        / "results"
        / "plots_diurnal_cycle"
        / "region_pnw_hotz"
        / "boundary_surface_700hPa"
        / "time_range_1940_2024"
        / "hw_non_hw_diurnal_cycle_jja_local.png"
    )
    assert args.season_months == [6, 7, 8]
    assert args.local_utc_offset_hours == -7


def test_parse_args_accepts_custom_options(monkeypatch, tmp_path):
    input_path = tmp_path / "stage1.nc"
    output_path = tmp_path / "diurnal.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            *_argv(
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--season-months",
            "7",
            "8",
            "--local-utc-offset-hours",
            "-8",
            ),
        ],
    )

    args = plot_diurnal_cycle.parse_args()

    assert args.input_path == input_path
    assert args.output_path == output_path
    assert args.season_months == [7, 8]
    assert args.local_utc_offset_hours == -8


def test_validate_args_rejects_invalid_months():
    args = argparse.Namespace(season_months=[6, 13], local_utc_offset_hours=-7)

    with pytest.raises(ValueError, match="season-months"):
        plot_diurnal_cycle.validate_args(args)


def test_validate_args_rejects_invalid_utc_offset_type():
    args = argparse.Namespace(season_months=[6, 7, 8], local_utc_offset_hours="-7")

    with pytest.raises(ValueError, match="local-utc-offset-hours"):
        plot_diurnal_cycle.validate_args(args)


def test_validate_args_rejects_invalid_utc_offset_range():
    args = argparse.Namespace(season_months=[6, 7, 8], local_utc_offset_hours=-24)

    with pytest.raises(ValueError, match="between -23 and 23"):
        plot_diurnal_cycle.validate_args(args)


def test_utc_to_local_time_values_crosses_prior_day_and_month():
    time = xr.DataArray(
        np.array(["2000-06-01T02:00"], dtype="datetime64[m]"),
        dims=("time",),
    )

    local = plot_diurnal_cycle.utc_to_local_time_values(time, -7)

    np.testing.assert_array_equal(
        local,
        np.array(["2000-05-31T19:00"], dtype="datetime64[ns]"),
    )


def test_build_diurnal_composite_filters_by_native_month_before_local_shift():
    ds = _make_diurnal_dataset()

    composite = plot_diurnal_cycle.build_diurnal_composite(
        ds,
        season_months=[6],
        local_utc_offset_hours=-7,
    )

    non_hw = composite.sel(hw_class="Non-heatwave days")
    assert int(composite.attrs["n_non_hw_samples"]) == 4
    np.testing.assert_allclose(non_hw["T_mean"].sel(local_hour=23).item(), 999.0)
    np.testing.assert_allclose(non_hw["T_mean"].sel(local_hour=0).item(), 12.0)


def test_build_diurnal_composite_splits_hw_and_non_hw_by_event_id():
    ds = _make_diurnal_dataset()

    composite = plot_diurnal_cycle.build_diurnal_composite(
        ds,
        season_months=[6],
        local_utc_offset_hours=-7,
    )

    hw = composite.sel(hw_class="Heatwave days")
    non_hw = composite.sel(hw_class="Non-heatwave days")
    np.testing.assert_allclose(hw["T_mean"].sel(local_hour=0).item(), 32.0)
    np.testing.assert_allclose(non_hw["T_mean"].sel(local_hour=0).item(), 12.0)
    assert int(composite.attrs["n_hw_samples"]) == 3
    assert int(composite.attrs["n_non_hw_samples"]) == 4


def test_build_diurnal_composite_computes_iqr_by_local_hour():
    ds = _make_diurnal_dataset()

    composite = plot_diurnal_cycle.build_diurnal_composite(
        ds,
        season_months=[6],
        local_utc_offset_hours=-7,
    )

    hw_iqr = composite["sample_percentile_T_mean"].sel(
        hw_class="Heatwave days",
        local_hour=0,
    )
    np.testing.assert_allclose(hw_iqr.sel(quantile=0.25).item(), 31.0)
    np.testing.assert_allclose(hw_iqr.sel(quantile=0.5).item(), 32.0)
    np.testing.assert_allclose(hw_iqr.sel(quantile=0.75).item(), 33.0)


def test_write_diurnal_cycle_plot_writes_png_and_plot_draws_iqr_lines(tmp_path):
    composite = plot_diurnal_cycle.build_diurnal_composite(
        _make_diurnal_dataset(),
        season_months=[6],
        local_utc_offset_hours=-7,
    )

    path = plot_diurnal_cycle.write_diurnal_cycle_plot(composite, tmp_path / "diurnal.png")
    fig = plot_diurnal_cycle.plot_diurnal_cycle(composite)
    try:
        assert path.exists()
        assert path.name == "diurnal.png"
        assert any(
            line.get_alpha() == 0.28
            for ax in fig.axes
            for line in ax.lines
        )
        assert fig.axes[4].get_legend().get_texts()[-1].get_text() == "IQR bounds"
    finally:
        plt.close(fig)


def test_main_orchestrates_open_composite_write_and_close(monkeypatch, tmp_path, capsys):
    input_path = tmp_path / "stage1.nc"
    output_path = tmp_path / "diurnal.png"
    opened = _ClosableDataset()
    composite = xr.Dataset()
    captured = {}

    def fake_open(path: Path):
        captured["input_path"] = path
        return opened

    def fake_composite(ds, **kwargs):
        captured["composite_ds"] = ds
        captured["composite_kwargs"] = kwargs
        return composite

    def fake_write(ds, path):
        captured["plot_ds"] = ds
        captured["output_path"] = path
        return path

    monkeypatch.setattr(
        "sys.argv",
        [
            *_argv(
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--season-months",
            "6",
            "--local-utc-offset-hours",
            "-7",
            ),
        ],
    )
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", fake_open)
    monkeypatch.setattr(plot_diurnal_cycle, "build_diurnal_composite", fake_composite)
    monkeypatch.setattr(plot_diurnal_cycle, "write_diurnal_cycle_plot", fake_write)

    result = plot_diurnal_cycle.main()

    assert result == 0
    assert opened.closed
    assert captured["input_path"] == input_path
    assert captured["composite_ds"] is opened
    assert captured["composite_kwargs"] == {
        "season_months": [6],
        "local_utc_offset_hours": -7,
    }
    assert captured["plot_ds"] is composite
    assert captured["output_path"] == output_path
    assert "Wrote HW/non-HW local diurnal-cycle figure:" in capsys.readouterr().out


class _ClosableDataset:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


def _make_diurnal_dataset() -> xr.Dataset:
    time = np.array(
        [
            "2000-06-01T06:00",  # local May 31 23:00, excluded from JJA-local.
            "2000-06-01T07:00",  # local Jun 01 00:00, non-HW.
            "2000-06-02T07:00",  # local Jun 02 00:00, non-HW.
            "2000-06-03T07:00",  # local Jun 03 00:00, HW.
            "2000-06-04T07:00",  # local Jun 04 00:00, HW.
            "2000-06-01T08:00",  # local Jun 01 01:00, HW.
            "2000-06-02T08:00",  # local Jun 02 01:00, non-HW.
        ],
        dtype="datetime64[m]",
    )
    values = np.array([999.0, 10.0, 14.0, 30.0, 34.0, 20.0, 40.0])
    data_vars = {
        name: ("time", values.copy())
        for name in plot_diurnal_cycle.DIURNAL_VARIABLES
    }
    data_vars["hw_event_id"] = (
        "time",
        np.array([0, 0, 0, 1, 2, 3, 0], dtype=np.int64),
    )
    return xr.Dataset(
        data_vars=data_vars,
        coords={"time": time},
        attrs={"region": "pnw_bartusek"},
    )
