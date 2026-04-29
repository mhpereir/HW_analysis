import pytest

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
    assert args.output_path == analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH


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
