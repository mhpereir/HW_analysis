import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import plot_event_feature_split_combined as plot_combined


def test_parse_and_validate_args_do_not_require_split_cli(monkeypatch, tmp_path):
    input_path = tmp_path / "features.nc"
    output_path = tmp_path / "combined.png"
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_event_feature_split_combined.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
        ],
    )

    args = plot_combined.parse_args()
    plot_combined.validate_args(args)

    assert args.input_path == input_path
    assert args.output_path == output_path
    assert not hasattr(args, "selection_variable")
    assert not hasattr(args, "selection_quantile")


def test_validate_args_rejects_paths_without_filenames():
    args = argparse.Namespace(input_path=Path("features.nc"), output_path=Path("plots"))

    with pytest.raises(ValueError, match="output-path"):
        plot_combined.validate_args(args)


def test_build_quantile_split_uses_threshold_and_puts_ties_in_low_group():
    features = _make_feature_table()

    split = plot_combined.build_quantile_split(
        features,
        split_spec=("duration", 0.5),
    )

    assert split.threshold == 3.0
    np.testing.assert_array_equal(split.low_mask, [True, True, True, True, False, False])
    np.testing.assert_array_equal(split.high_mask, [False, False, False, False, True, True])


def test_split_specs_support_direct_and_derived_variables():
    features = _make_feature_table()

    splits = plot_combined.build_quantile_splits(
        features,
        split_specs=(
            ("tas_excess_integral", 0.5),
            ("f_diabatic_pre", 0.5),
            ("sqrt_I_lwa_a_pre_peak", 0.5),
        ),
    )

    assert [split.variable for split in splits] == [
        "tas_excess_integral",
        "f_diabatic_pre",
        "sqrt_I_lwa_a_pre_peak",
    ]
    expected = plot_combined.feature_values(features, "f_diabatic_pre")
    assert splits[1].threshold == np.quantile(expected, 0.5)


def test_split_validation_rejects_invalid_variable_quantile_and_empty_groups():
    features = _make_feature_table()

    with pytest.raises(ValueError, match="split variables"):
        plot_combined.validate_feature_variables(
            features,
            split_specs=(("not_a_feature", 0.5),),
        )

    with pytest.raises(ValueError, match="strictly between"):
        plot_combined.normalize_split_specs((("duration", 1.0),))

    features = features.copy(deep=True)
    features["constant_split"] = ("event", np.ones(features.sizes["event"]))
    with pytest.raises(ValueError, match="does not create two non-empty groups"):
        plot_combined.build_quantile_split(
            features,
            split_spec=("constant_split", 0.5),
        )


def test_plot_split_violin_combined_draws_one_row_per_y_variable():
    features = _make_feature_table()
    split_specs = (("duration", 0.5), ("tas_excess_integral", 0.5))

    fig = plot_combined.plot_split_violin_combined(
        features,
        split_specs=split_specs,
    )
    try:
        assert len(fig.axes) == len(plot_combined.Y_VARIABLES)
        expected_violins_per_row = 1 + 2 * len(split_specs)
        for ax in fig.axes:
            violin_artists = [
                artist
                for artist in ax.collections
                if str(artist.get_gid()).startswith("violin_")
            ]
            assert len(violin_artists) == expected_violins_per_row
    finally:
        plot_combined.plt.close(fig)


def test_main_writes_one_raw_combined_output(monkeypatch, tmp_path):
    input_path = tmp_path / "features.nc"
    output_path = tmp_path / "combined.png"
    written = []

    def fake_open(path):
        assert path == input_path
        return _make_feature_table()

    def fake_write(features, path, *, split_specs=plot_combined.SPLIT_SPECS):
        written.append((Path(path).name, split_specs))
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_event_feature_split_combined.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
        ],
    )
    monkeypatch.setattr(plot_combined, "open_event_features", fake_open)
    monkeypatch.setattr(plot_combined, "write_split_violin_plot", fake_write)

    assert plot_combined.main() == 0
    assert written == [("combined.png", plot_combined.SPLIT_SPECS)]


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
