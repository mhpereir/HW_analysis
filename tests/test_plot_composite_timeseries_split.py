import argparse

import numpy as np
import pytest
import xarray as xr

from scripts import plot_composite_timeseries_split as plot_split
from src import analysis_io


def test_parse_args_defaults_to_base_plot(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_split.py",
        "--split-variable",
        "duration",
        "--split-quantiles",
        "0.5",
    ])

    args = plot_split.parse_args()

    assert not args.plot_extended_variables


def test_parse_args_accepts_plot_extended_variables(monkeypatch):
    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_split.py",
        "--split-variable",
        "duration",
        "--split-quantiles",
        "0.5",
        "--plot-extended-variables",
    ])

    args = plot_split.parse_args()

    assert args.plot_extended_variables


def test_build_split_quantile_composite_one_quantile_makes_two_bins(monkeypatch):
    captured = []

    def fake_composite(ds, **kwargs):
        captured.append(kwargs["event_table"]["event_id"].values.copy())
        return _make_composite(float(len(captured)))

    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )

    out = plot_split.build_split_quantile_composite(
        xr.Dataset(),
        event_table=_make_event_table(),
        split_variable="duration",
        split_quantiles=[0.5],
        composite_kwargs={
            "variables": plot_split.COMPOSITE_VARIABLES,
            "pre_days": 1,
            "post_days": 1,
            "event_percentiles": (0.25, 0.5, 0.75),
        },
    )

    assert out.sizes["split_bin"] == 2
    np.testing.assert_allclose(out["split_qmin"].values, [0.0, 0.5])
    np.testing.assert_allclose(out["split_qmax"].values, [0.5, 1.0])
    np.testing.assert_array_equal(out["split_n_events"].values, [2, 2])
    np.testing.assert_array_equal(captured, [np.array([1, 2]), np.array([3, 4])])
    assert out.attrs["split_variable"] == "duration"
    assert out.attrs["split_quantiles"] == "0.5"


def test_build_split_quantile_composite_assigns_boundary_ties_to_lower_bin(monkeypatch):
    captured = []

    def fake_composite(ds, **kwargs):
        captured.append(kwargs["event_table"]["event_id"].values.copy())
        return _make_composite(float(len(captured)))

    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )

    out = plot_split.build_split_quantile_composite(
        xr.Dataset(),
        event_table=_make_event_table(
            event_ids=np.arange(1, 7),
            duration=np.array([1.0, 2.0, 2.0, 2.0, 3.0, 4.0]),
        ),
        split_variable="duration",
        split_quantiles=[0.5],
        composite_kwargs={
            "variables": plot_split.COMPOSITE_VARIABLES,
            "pre_days": 1,
            "post_days": 1,
            "event_percentiles": (0.25, 0.5, 0.75),
        },
    )

    np.testing.assert_array_equal(captured[0], [1, 2, 3, 4])
    np.testing.assert_array_equal(captured[1], [5, 6])
    np.testing.assert_array_equal(out["split_n_events"].values, [4, 2])
    assert int(out["split_n_events"].sum().item()) == 6


def test_build_split_quantile_composite_sorts_multiple_quantiles(monkeypatch):
    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        lambda ds, **kwargs: _make_composite(float(kwargs["event_table"].sizes["event"])),
    )

    out = plot_split.build_split_quantile_composite(
        xr.Dataset(),
        event_table=_make_event_table(
            event_ids=np.arange(1, 9),
            duration=np.arange(1.0, 9.0),
        ),
        split_variable="duration",
        split_quantiles=[0.75, 0.25],
        composite_kwargs={
            "variables": plot_split.COMPOSITE_VARIABLES,
            "pre_days": 1,
            "post_days": 1,
            "event_percentiles": (0.25, 0.5, 0.75),
        },
    )

    np.testing.assert_allclose(out["split_qmin"].values, [0.0, 0.25, 0.75])
    np.testing.assert_allclose(out["split_qmax"].values, [0.25, 0.75, 1.0])
    np.testing.assert_array_equal(out["split_n_events"].values, [2, 4, 2])


def test_build_split_year_composite_one_cutoff_makes_two_bins(monkeypatch):
    captured = []

    def fake_composite(ds, **kwargs):
        captured.append(kwargs["event_table"]["event_id"].values.copy())
        return _make_composite(float(len(captured)))

    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )

    out = plot_split.build_split_year_composite(
        xr.Dataset(),
        event_table=_make_event_table(
            event_ids=np.arange(1, 5),
            peak_time=np.array(
                ["1940-06-01", "1979-07-01", "1980-06-01", "2024-07-01"],
                dtype="datetime64[ns]",
            ),
        ),
        split_years=[1980],
        composite_kwargs={
            "variables": plot_split.COMPOSITE_VARIABLES,
            "pre_days": 1,
            "post_days": 1,
            "event_percentiles": (0.25, 0.5, 0.75),
        },
    )

    assert out.sizes["split_bin"] == 2
    np.testing.assert_array_equal(out["split_start_year"].values, [1940, 1980])
    np.testing.assert_array_equal(out["split_end_year"].values, [1979, 2024])
    np.testing.assert_array_equal(out["split_n_events"].values, [2, 2])
    np.testing.assert_array_equal(captured, [np.array([1, 2]), np.array([3, 4])])
    assert out.attrs["split_variable"] == "peak_time"
    assert out.attrs["split_type"] == "year_bin"
    assert out.attrs["split_years"] == "1980"


def test_build_split_year_composite_sorts_multiple_cutoffs(monkeypatch):
    captured = []

    def fake_composite(ds, **kwargs):
        captured.append(kwargs["event_table"]["event_id"].values.copy())
        return _make_composite(float(len(captured)))

    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )

    out = plot_split.build_split_year_composite(
        xr.Dataset(),
        event_table=_make_event_table(
            event_ids=np.arange(1, 6),
            peak_time=np.array(
                [
                    "1979-06-01",
                    "1980-06-01",
                    "1999-06-01",
                    "2000-06-01",
                    "2024-06-01",
                ],
                dtype="datetime64[ns]",
            ),
        ),
        split_years=[2000, 1980],
        composite_kwargs={
            "variables": plot_split.COMPOSITE_VARIABLES,
            "pre_days": 1,
            "post_days": 1,
            "event_percentiles": (0.25, 0.5, 0.75),
        },
    )

    np.testing.assert_array_equal(out["split_start_year"].values, [1979, 1980, 2000])
    np.testing.assert_array_equal(out["split_end_year"].values, [1979, 1999, 2024])
    np.testing.assert_array_equal(captured[0], [1])
    np.testing.assert_array_equal(captured[1], [2, 3])
    np.testing.assert_array_equal(captured[2], [4, 5])


def test_build_split_year_composite_cutoff_year_starts_next_bin(monkeypatch):
    captured = []

    def fake_composite(ds, **kwargs):
        captured.append(kwargs["event_table"]["event_id"].values.copy())
        return _make_composite(float(len(captured)))

    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        fake_composite,
    )

    plot_split.build_split_year_composite(
        xr.Dataset(),
        event_table=_make_event_table(
            event_ids=np.array([1, 2, 3]),
            peak_time=np.array(
                ["1979-06-01", "1980-06-01", "1981-06-01"],
                dtype="datetime64[ns]",
            ),
        ),
        split_years=[1980],
        composite_kwargs={},
    )

    np.testing.assert_array_equal(captured[0], [1])
    np.testing.assert_array_equal(captured[1], [2, 3])


def test_main_filters_event_table_before_quantile_splitting(monkeypatch, tmp_path):
    opened = _make_event_table(
        event_ids=np.array([1, 2, 3]),
        duration=np.array([10.0, 20.0, 30.0]),
        peak_time=np.array(
            ["2000-06-01", "2000-07-01", "2000-09-01"],
            dtype="datetime64[ns]",
        ),
    )
    captured = {}

    def fake_build(ds, **kwargs):
        captured["event_table"] = kwargs["event_table"]
        captured["split_variable"] = kwargs["split_variable"]
        captured["split_quantiles"] = kwargs["split_quantiles"]
        captured["composite_kwargs"] = kwargs["composite_kwargs"]
        return xr.Dataset(
            data_vars={"T_mean": (("split_bin", "lag_hour"), np.ones((2, 1)))},
            coords={"split_bin": ["low", "high"], "lag_hour": [0]},
        )

    def fake_write(ds, output, **kwargs):
        captured["plot_ds"] = ds
        captured["output_path"] = output
        captured["plot_kwargs"] = kwargs
        captured["smoothed_output_path"] = kwargs["smoothed_output_path"]
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_split.py",
        "--input-path",
        str(tmp_path / "stage1.nc"),
        "--output-path",
        str(tmp_path / "split.png"),
        "--split-variable",
        "duration",
        "--split-quantiles",
        "0.5",
        "--season-months",
        "6",
        "7",
    ])
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", lambda path: opened)
    monkeypatch.setattr(plot_split, "build_split_quantile_composite", fake_build)
    monkeypatch.setattr(
        plot_split.plotting,
        "write_split_composite_timeseries_outputs",
        fake_write,
    )

    assert plot_split.main() == 0
    np.testing.assert_array_equal(captured["event_table"]["event_id"].values, [1, 2])
    assert captured["split_variable"] == "duration"
    assert captured["split_quantiles"] == [0.5]
    assert captured["composite_kwargs"]["variables"] == plot_split.COMPOSITE_VARIABLES
    assert captured["plot_kwargs"]["smoothed_variables"] == plot_split.SMOOTHED_VARIABLES
    assert not captured["plot_kwargs"]["plot_extended_variables"]
    assert captured["output_path"].name == "split_duration.png"
    assert captured["smoothed_output_path"].name == "split_duration_smoothed.png"


def test_main_uses_extended_variables_when_requested(monkeypatch, tmp_path):
    opened = _make_event_table()
    captured = {}

    def fake_build(ds, **kwargs):
        captured["composite_kwargs"] = kwargs["composite_kwargs"]
        return xr.Dataset(
            data_vars={"T_mean": (("split_bin", "lag_hour"), np.ones((2, 1)))},
            coords={"split_bin": ["low", "high"], "lag_hour": [0]},
        )

    def fake_write(ds, output, **kwargs):
        captured["plot_kwargs"] = kwargs
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_split.py",
        "--input-path",
        str(tmp_path / "stage1.nc"),
        "--output-path",
        str(tmp_path / "split.png"),
        "--split-variable",
        "duration",
        "--split-quantiles",
        "0.5",
        "--plot-extended-variables",
    ])
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", lambda path: opened)
    monkeypatch.setattr(plot_split, "build_split_quantile_composite", fake_build)
    monkeypatch.setattr(
        plot_split.plotting,
        "write_split_composite_timeseries_outputs",
        fake_write,
    )

    assert plot_split.main() == 0
    assert captured["composite_kwargs"]["variables"] == (
        plot_split.EXTENDED_COMPOSITE_VARIABLES
    )
    assert captured["plot_kwargs"]["smoothed_variables"] == (
        plot_split.EXTENDED_SMOOTHED_VARIABLES
    )
    assert captured["plot_kwargs"]["plot_extended_variables"]


def test_main_filters_event_table_before_year_splitting(monkeypatch, tmp_path):
    opened = _make_event_table(
        event_ids=np.array([1, 2, 3]),
        duration=np.array([10.0, 20.0, 30.0]),
        peak_time=np.array(
            ["2000-06-01", "2001-07-01", "2002-09-01"],
            dtype="datetime64[ns]",
        ),
    )
    captured = {}

    def fake_build(ds, **kwargs):
        captured["event_table"] = kwargs["event_table"]
        captured["split_years"] = kwargs["split_years"]
        return xr.Dataset(
            data_vars={"T_mean": (("split_bin", "lag_hour"), np.ones((2, 1)))},
            coords={"split_bin": ["2000-2000", "2001-2001"], "lag_hour": [0]},
        )

    def fake_write(ds, output, **kwargs):
        captured["plot_ds"] = ds
        captured["output_path"] = output
        captured["smoothed_output_path"] = kwargs["smoothed_output_path"]
        return [output, kwargs["smoothed_output_path"]]

    monkeypatch.setattr("sys.argv", [
        "plot_composite_timeseries_split.py",
        "--input-path",
        str(tmp_path / "stage1.nc"),
        "--output-path",
        str(tmp_path / "split.png"),
        "--split-variable",
        "peak_time",
        "--split-years",
        "2001",
        "--season-months",
        "6",
        "7",
    ])
    monkeypatch.setattr(analysis_io, "open_harmonized_timeseries", lambda path: opened)
    monkeypatch.setattr(plot_split, "build_split_year_composite", fake_build)
    monkeypatch.setattr(
        plot_split.plotting,
        "write_split_composite_timeseries_outputs",
        fake_write,
    )

    assert plot_split.main() == 0
    np.testing.assert_array_equal(captured["event_table"]["event_id"].values, [1, 2])
    assert captured["split_years"] == [2001]
    assert captured["output_path"].name == "split_peak_time.png"
    assert captured["smoothed_output_path"].name == "split_peak_time_smoothed.png"


def test_validate_split_variable_rejects_missing_variable():
    with pytest.raises(ValueError, match="missing split variable"):
        plot_split.build_split_quantile_composite(
            xr.Dataset(),
            event_table=_make_event_table(),
            split_variable="not_a_metric",
            split_quantiles=[0.5],
            composite_kwargs={},
        )


def test_validate_split_variable_rejects_non_numeric_variable():
    event_table = _make_event_table()
    event_table["category"] = ("event", np.array(["a", "b", "c", "d"]))

    with pytest.raises(TypeError, match="must be numeric"):
        plot_split.build_split_quantile_composite(
            xr.Dataset(),
            event_table=event_table,
            split_variable="category",
            split_quantiles=[0.5],
            composite_kwargs={},
        )


def test_validate_peak_time_variable_rejects_missing_variable():
    with pytest.raises(ValueError, match="missing split variable"):
        plot_split.build_split_year_composite(
            xr.Dataset(),
            event_table=_make_event_table().drop_vars("peak_time"),
            split_years=[1980],
            composite_kwargs={},
        )


def test_validate_peak_time_variable_rejects_non_datetime_variable():
    event_table = _make_event_table()
    event_table["peak_time"] = ("event", np.array([1.0, 2.0, 3.0, 4.0]))

    with pytest.raises(TypeError, match="must be datetime64"):
        plot_split.build_split_year_composite(
            xr.Dataset(),
            event_table=event_table,
            split_years=[1980],
            composite_kwargs={},
        )


def test_empty_quantile_bin_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        lambda ds, **kwargs: _make_composite(1.0),
    )

    with pytest.raises(ValueError, match="contains no events"):
        plot_split.build_split_quantile_composite(
            xr.Dataset(),
            event_table=_make_event_table(duration=np.ones(4)),
            split_variable="duration",
            split_quantiles=[0.5],
            composite_kwargs={},
        )


def test_empty_year_bin_raises_clear_error(monkeypatch):
    monkeypatch.setattr(
        plot_split.composites,
        "all_event_peak_aligned_composite",
        lambda ds, **kwargs: _make_composite(1.0),
    )

    with pytest.raises(ValueError, match="contains no events"):
        plot_split.build_split_year_composite(
            xr.Dataset(),
            event_table=_make_event_table(
                event_ids=np.array([1, 2]),
                peak_time=np.array(
                    ["1940-06-01", "1942-06-01"],
                    dtype="datetime64[ns]",
                ),
            ),
            split_years=[1941, 1942],
            composite_kwargs={},
        )


def test_split_year_outside_filtered_peak_year_range_raises_clear_error():
    with pytest.raises(ValueError, match="filtered peak-time year range"):
        plot_split.build_split_year_composite(
            xr.Dataset(),
            event_table=_make_event_table(
                event_ids=np.array([1, 2]),
                peak_time=np.array(
                    ["1940-06-01", "2024-06-01"],
                    dtype="datetime64[ns]",
                ),
            ),
            split_years=[2025],
            composite_kwargs={},
        )


def test_validate_args_rejects_missing_split_quantiles():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        split_variable="duration",
        split_quantiles=None,
        split_years=None,
        season_months=None,
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="--split-quantiles"):
        plot_split.validate_args(args)


def test_validate_args_rejects_duplicate_split_quantiles():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        split_variable="duration",
        split_quantiles=[0.5, 0.5],
        split_years=None,
        season_months=None,
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="duplicate"):
        plot_split.validate_args(args)


def test_validate_args_rejects_missing_split_years_for_peak_time():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        split_variable="peak_time",
        split_quantiles=None,
        split_years=None,
        season_months=None,
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="--split-years"):
        plot_split.validate_args(args)


def test_validate_args_rejects_duplicate_split_years():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        split_variable="peak_time",
        split_quantiles=None,
        split_years=[1980, 1980],
        season_months=None,
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="duplicate"):
        plot_split.validate_args(args)


def test_validate_args_rejects_out_of_range_split_years():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        split_variable="peak_time",
        split_quantiles=None,
        split_years=[0],
        season_months=None,
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="between 1 and 9999"):
        plot_split.validate_args(args)


def test_validate_args_rejects_split_quantiles_for_peak_time():
    args = argparse.Namespace(
        window_days=7,
        smoothing_window=24,
        split_variable="peak_time",
        split_quantiles=[0.5],
        split_years=[1980],
        season_months=None,
        require_full_event=False,
    )

    with pytest.raises(ValueError, match="uses --split-years"):
        plot_split.validate_args(args)


def test_split_output_path_adds_split_variable_to_filename():
    path = plot_split._split_output_path(
        plot_split.DEFAULT_OUTPUT_PATH,
        "tas_excess_integral",
    )

    assert path.name == "hw_events_composite_tas_excess_integral.png"
    assert plot_split._smoothed_output_path(path).name == (
        "hw_events_composite_tas_excess_integral_smoothed.png"
    )


def test_split_output_path_does_not_duplicate_existing_token():
    path = plot_split._split_output_path(
        plot_split.DEFAULT_OUTPUT_PATH.with_name("custom_duration.png"),
        "duration",
    )

    assert path.name == "custom_duration.png"


def _make_event_table(
    *,
    event_ids: np.ndarray | None = None,
    duration: np.ndarray | None = None,
    peak_time: np.ndarray | None = None,
) -> xr.Dataset:
    if event_ids is None:
        event_ids = np.array([1, 2, 3, 4], dtype=np.int64)
    if duration is None:
        duration = np.arange(1.0, float(event_ids.size) + 1.0)
    if peak_time is None:
        peak_time = np.arange(
            np.datetime64("2000-06-01"),
            np.datetime64("2000-06-01") + np.timedelta64(event_ids.size, "D"),
            dtype="datetime64[D]",
        ).astype("datetime64[ns]")
    event = np.arange(event_ids.size)
    return xr.Dataset(
        data_vars={
            "event_id": ("event", event_ids),
            "duration": ("event", duration),
            "peak_time": ("event", peak_time),
            "start_time": ("event", peak_time),
            "end_time": ("event", peak_time),
        },
        coords={"event": event},
    )


def _make_composite(offset: float) -> xr.Dataset:
    lag_hour = np.arange(-1, 2)
    data_vars = {}
    for name in plot_split.COMPOSITE_VARIABLES:
        values = np.asarray([1.0, 2.0, 3.0]) + offset
        data_vars[name] = ("lag_hour", values)
        data_vars[f"event_percentile_{name}"] = (
            ("quantile", "lag_hour"),
            np.vstack([values - 0.5, values, values + 0.5]),
        )
    return xr.Dataset(
        data_vars=data_vars,
        coords={"lag_hour": lag_hour, "quantile": [0.25, 0.5, 0.75]},
        attrs={"pre_days": 1, "post_days": 1},
    )
