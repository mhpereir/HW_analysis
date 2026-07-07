import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import plot_event_feature_split as plot_split


def test_validate_args_rejects_bad_quantile_and_non_panel_variable():
    valid = argparse.Namespace(
        point_size=12.0,
        alpha=0.5,
        selection_variable="duration",
        selection_quantile=0.5,
    )
    plot_split.validate_args(valid)

    invalid_quantile = argparse.Namespace(
        point_size=12.0,
        alpha=0.5,
        selection_variable="duration",
        selection_quantile=1.0,
    )
    with pytest.raises(ValueError, match="selection-quantile"):
        plot_split.validate_args(invalid_quantile)

    invalid_variable = argparse.Namespace(
        point_size=12.0,
        alpha=0.5,
        selection_variable=plot_split.X_VARIABLE,
        selection_quantile=0.5,
    )
    with pytest.raises(ValueError, match="selection-variable"):
        plot_split.validate_args(invalid_variable)


def test_build_quantile_split_uses_threshold_and_puts_ties_in_low_group():
    features = _make_feature_table()

    split = plot_split.build_quantile_split(
        features,
        selection_variable="duration",
        selection_quantile=0.5,
    )

    assert split.threshold == 3.0
    np.testing.assert_array_equal(split.low_mask, [True, True, True, True, False, False])
    np.testing.assert_array_equal(split.high_mask, [False, False, False, False, True, True])


def test_build_quantile_split_supports_derived_selection_variables():
    features = _make_feature_table()
    expected = plot_split.feature_values(features, "f_diabatic_pre")

    split = plot_split.build_quantile_split(
        features,
        selection_variable="f_diabatic_pre",
        selection_quantile=0.5,
    )

    assert split.threshold == np.quantile(expected, 0.5)
    np.testing.assert_array_equal(split.low_mask, expected <= split.threshold)
    np.testing.assert_array_equal(split.high_mask, expected > split.threshold)


def test_selection_panel_draws_black_quantile_divider():
    features = _make_feature_table()

    fig = plot_split.plot_tendency_scatter(
        features,
        selection_variable="duration",
        selection_quantile=0.5,
    )
    try:
        duration_axis = fig.axes[list(plot_split.Y_VARIABLES).index("duration")]
        threshold_lines = [
            line for line in duration_axis.lines if line.get_gid() == "selection_threshold"
        ]
        assert len(threshold_lines) == 1
        assert threshold_lines[0].get_color() == "black"
        np.testing.assert_allclose(threshold_lines[0].get_ydata(), [3.0, 3.0])
    finally:
        plot_split.plt.close(fig)


def test_each_panel_draws_group_mean_lines_and_std_bands():
    features = _make_feature_table()

    fig = plot_split.plot_tendency_scatter(
        features,
        selection_variable="duration",
        selection_quantile=0.5,
    )
    try:
        for ax in fig.axes:
            mean_lines = [
                line
                for line in ax.lines
                if line.get_gid() in {"low_mean", "high_mean"}
            ]
            std_bands = [
                patch
                for patch in ax.patches
                if patch.get_gid() in {"low_std", "high_std"}
            ]
            assert len(mean_lines) == 2
            assert len(std_bands) == 2
    finally:
        plot_split.plt.close(fig)


def test_tas_peak_panel_uses_data_driven_y_limits():
    features = _make_feature_table()

    fig = plot_split.plot_tendency_scatter(
        features,
        selection_variable="tas_peak",
        selection_quantile=0.5,
    )
    try:
        tas_peak_axis = fig.axes[list(plot_split.Y_VARIABLES).index("tas_peak")]
        ymin, ymax = tas_peak_axis.get_ylim()
        assert ymin > 250.0
        assert ymax < 310.0
    finally:
        plot_split.plt.close(fig)


def test_main_writes_raw_and_standardized_split_outputs(monkeypatch, tmp_path):
    input_path = tmp_path / "features.nc"
    output_path = tmp_path / "event_feature_tendency_scatter.png"
    written = []

    def fake_open(path):
        assert path == input_path
        return _make_feature_table()

    def fake_write(
        features,
        path,
        *,
        selection_variable,
        selection_quantile,
        point_size,
        alpha,
        standardized=False,
    ):
        written.append(
            (
                Path(path).name,
                selection_variable,
                selection_quantile,
                point_size,
                alpha,
                standardized,
            )
        )
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_event_feature_split.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--selection-variable",
            "duration",
            "--selection-quantile",
            "0.5",
            "--point-size",
            "10",
            "--alpha",
            "0.6",
        ],
    )
    monkeypatch.setattr(plot_split, "open_event_features", fake_open)
    monkeypatch.setattr(plot_split, "write_tendency_scatter_plot", fake_write)

    assert plot_split.main() == 0
    assert written == [
        (
            "event_feature_tendency_scatter_duration.png",
            "duration",
            0.5,
            10.0,
            0.6,
            False,
        ),
        (
            "event_feature_tendency_scatter_duration_standardized.png",
            "duration",
            0.5,
            10.0,
            0.6,
            True,
        ),
    ]


def _make_feature_table() -> xr.Dataset:
    event = np.arange(6)
    return xr.Dataset(
        data_vars={
            "I_dTdt_pre": ("event", np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])),
            "I_adiabatic_pre": ("event", np.ones(6)),
            "I_diabatic_pre": ("event", np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])),
            "I_advection_pre": ("event", -np.ones(6)),
            "I_lwa_a_pre_peak": ("event", np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0])),
            "T_anom_mean_ant": ("event", np.array([0.5, 0.7, 0.9, 1.1, 1.3, 1.5])),
            "days_from_solstice": (
                "event",
                np.array([0, 30, 60, 90, 120, 150], dtype="timedelta64[D]"),
            ),
            "duration": (
                "event",
                np.array([1, 2, 3, 3, 4, 5], dtype="timedelta64[D]"),
            ),
            "tas_peak": ("event", np.array([300.0, 300.5, 301.0, 301.5, 302.0, 302.5])),
            "tas_anom_peak": ("event", np.array([5.0, 5.5, 6.0, 6.5, 7.0, 7.5])),
            "tas_excess_integral": (
                "event",
                np.array([10.0, 20.0, 40.0, 80.0, 160.0, 320.0]),
            ),
        },
        coords={"event": event},
    )
