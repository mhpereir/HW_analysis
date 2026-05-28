import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from scripts.event_features import build_stage3_event_feature_pca as build_event_feature_pca


def test_build_event_feature_pca_uses_default_derived_features_and_metadata():
    feature_table = _make_feature_table()

    out = build_event_feature_pca.build_event_feature_pca(
        feature_table,
        input_path="features.nc",
    )

    assert out.sizes["event"] == 4
    assert out.sizes["feature"] == len(build_event_feature_pca.DEFAULT_PCA_FEATURES)
    assert out.sizes["pc"] == 4
    np.testing.assert_array_equal(out["event"].values, [1, 2, 3, 4])
    np.testing.assert_array_equal(out["pc"].values, ["PC1", "PC2", "PC3", "PC4"])
    np.testing.assert_array_equal(
        out["feature"].values,
        list(build_event_feature_pca.DEFAULT_PCA_FEATURES),
    )

    expected_first_row = np.array(
            [
                10.0,
                1.0 / 6.0,
                -2.0 / 6.0,
                2.0,
                1.0,
            np.cos(-30.0 * 2.0 * np.pi / 365.0),
            2.0,
        ]
    )
    np.testing.assert_allclose(out["feature_matrix"].values[0], expected_first_row)
    np.testing.assert_allclose(out["feature_matrix_scaled"].values.mean(axis=0), 0.0, atol=1e-12)
    np.testing.assert_allclose(out["feature_matrix_scaled"].values.std(axis=0), 1.0, atol=1e-12)
    np.testing.assert_array_equal(out["valid_event_mask_original"].values, [1, 1, 1, 1])
    np.testing.assert_array_equal(out["event_id"].values, [1, 2, 3, 4])
    assert "log10_tas_excess_integral" in out
    np.testing.assert_allclose(out["log10_tas_excess_integral"].values, [1.0, 2.0, 3.0, 4.0])
    assert out["pc_score"].dims == ("event", "pc")
    assert out["pc_loading"].dims == ("pc", "feature")
    assert out.attrs["pipeline_stage"] == "stage_3_event_feature_pca"
    assert out.attrs["source_feature_table"] == "features.nc"
    assert out.attrs["scaler"] == "standard"
    assert out.attrs["clustering_performed"] == 0


def test_robust_scaler_uses_median_and_iqr():
    feature_table = _make_feature_table()

    out = build_event_feature_pca.build_event_feature_pca(
        feature_table,
        feature_names=["I_dTdt_pre", "sqrt_I_lwa_a_pre_peak"],
        n_components=2,
        scaler="robust",
    )

    np.testing.assert_allclose(out["feature_center"].values, [25.0, 3.5])
    np.testing.assert_allclose(out["feature_scale"].values, [15.0, 1.5])
    np.testing.assert_allclose(
        out["feature_matrix_scaled"].values[:, 0],
        np.array([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]),
    )
    assert out.attrs["scaler"] == "robust"


def test_nonfinite_derived_features_drop_events_and_record_original_mask():
    feature_table = _make_feature_table()
    feature_table["I_lwa_a_pre_peak"] = (
        "event",
        np.array([4.0, 9.0, -1.0, 25.0]),
    )

    out = build_event_feature_pca.build_event_feature_pca(
        feature_table,
        n_components=2,
    )

    np.testing.assert_array_equal(out["event_id"].values, [1, 2, 4])
    np.testing.assert_array_equal(out["valid_event_mask_original"].values, [1, 1, 0, 1])
    assert out.attrs["n_input_events"] == 4
    assert out.attrs["n_valid_events"] == 3
    assert out.attrs["n_dropped_events"] == 1


def test_missing_source_variable_raises_clear_error():
    feature_table = _make_feature_table().drop_vars("I_lwa_a_pre_peak")

    with pytest.raises(ValueError, match="missing required source variables"):
        build_event_feature_pca.build_event_feature_pca(feature_table)


def test_zero_variance_feature_raises_clear_error():
    feature_table = _make_feature_table()
    feature_table["constant"] = ("event", np.ones(feature_table.sizes["event"]))

    with pytest.raises(ValueError, match="zero-variance"):
        build_event_feature_pca.build_event_feature_pca(
            feature_table,
            feature_names=["I_dTdt_pre", "constant"],
        )


def test_n_components_cannot_exceed_matrix_rank_limit():
    feature_table = _make_feature_table()

    with pytest.raises(ValueError, match="n_components cannot exceed"):
        build_event_feature_pca.build_event_feature_pca(
            feature_table,
            feature_names=["I_dTdt_pre", "sqrt_I_lwa_a_pre_peak"],
            n_components=3,
        )


def test_fewer_than_two_valid_events_raises_clear_error():
    feature_table = _make_feature_table()
    feature_table["I_lwa_a_pre_peak"] = (
        "event",
        np.array([4.0, -1.0, -1.0, -1.0]),
    )

    with pytest.raises(ValueError, match="Fewer than two events"):
        build_event_feature_pca.build_event_feature_pca(feature_table)


def test_validate_args_rejects_existing_output_without_overwrite(tmp_path):
    output_path = tmp_path / "pca.nc"
    output_path.write_text("exists")
    args = argparse.Namespace(
        output_path=output_path,
        overwrite=False,
        n_components=None,
        features=["I_dTdt_pre", "sqrt_I_lwa_a_pre_peak"],
    )

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_event_feature_pca.validate_args(args)


def test_write_pca_output_writes_netcdf(tmp_path):
    pca_ds = build_event_feature_pca.build_event_feature_pca(
        _make_feature_table(),
        n_components=2,
    )
    output_path = tmp_path / "pca.nc"

    written = build_event_feature_pca.write_pca_output(pca_ds, output_path)

    assert written == output_path.resolve()
    assert output_path.exists()
    reopened = xr.open_dataset(output_path, engine="h5netcdf", decode_timedelta=True)
    try:
        assert "pc_score" in reopened
        assert reopened.sizes["pc"] == 2
    finally:
        reopened.close()


def test_main_orchestrates_open_build_and_write(monkeypatch, tmp_path):
    feature_table = _make_feature_table()
    pca_ds = xr.Dataset()
    captured = {}
    input_path = tmp_path / "features.nc"
    output_path = tmp_path / "pca.nc"

    def fake_open(path: str | Path):
        captured["input_path"] = path
        return feature_table

    def fake_build(ds, **kwargs):
        captured["build_ds"] = ds
        captured["build_kwargs"] = kwargs
        return pca_ds

    def fake_write(ds, path):
        captured["write_ds"] = ds
        captured["output_path"] = path
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage3_event_feature_pca.py",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--features",
            "I_dTdt_pre",
            "sqrt_I_lwa_a_pre_peak",
            "--n-components",
            "2",
            "--scaler",
            "robust",
        ],
    )
    monkeypatch.setattr(build_event_feature_pca, "open_event_features", fake_open)
    monkeypatch.setattr(build_event_feature_pca, "build_event_feature_pca", fake_build)
    monkeypatch.setattr(build_event_feature_pca, "write_pca_output", fake_write)

    result = build_event_feature_pca.main()

    assert result == 0
    assert captured["input_path"] == input_path
    assert captured["build_ds"] is feature_table
    assert captured["build_kwargs"] == {
        "feature_names": ["I_dTdt_pre", "sqrt_I_lwa_a_pre_peak"],
        "n_components": 2,
        "scaler": "robust",
        "input_path": input_path,
    }
    assert captured["write_ds"] is pca_ds
    assert captured["output_path"] == output_path


def _make_feature_table() -> xr.Dataset:
    event = np.array([100, 101, 102, 103], dtype=np.int64)
    return xr.Dataset(
        data_vars={
            "event_id": ("event", np.array([1, 2, 3, 4], dtype=np.int64)),
            "start_time": (
                "event",
                np.array(
                    ["2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04"],
                    dtype="datetime64[ns]",
                ),
            ),
            "end_time": (
                "event",
                np.array(
                    ["2000-06-03", "2000-06-04", "2000-06-05", "2000-06-06"],
                    dtype="datetime64[ns]",
                ),
            ),
            "peak_time": (
                "event",
                np.array(
                    ["2000-06-02", "2000-06-03", "2000-06-04", "2000-06-05"],
                    dtype="datetime64[ns]",
                ),
            ),
            "duration": (
                "event",
                np.array([2, 3, 5, 8], dtype="timedelta64[D]"),
            ),
            "tas_peak": ("event", np.array([300.0, 301.0, 303.0, 306.0])),
            "tas_anom_peak": ("event", np.array([5.0, 6.0, 8.0, 11.0])),
            "tas_excess_peak": ("event", np.array([2.0, 3.0, 5.0, 8.0])),
            "tas_excess_integral": ("event", np.array([10.0, 100.0, 1000.0, 10000.0])),
            "lwa_a_peak": ("event", np.array([1.0, 2.0, 3.0, 4.0])),
            "lwa_c_peak": ("event", np.array([4.0, 3.0, 2.0, 1.0])),
            "I_dTdt_pre": ("event", np.array([10.0, 20.0, 30.0, 40.0])),
            "I_adiabatic_pre": ("event", np.array([1.0, 2.0, 4.0, 8.0])),
            "I_diabatic_pre": ("event", np.array([3.0, 1.0, 2.0, 4.0])),
            "I_advection_pre": ("event", np.array([-2.0, -4.0, -1.0, -3.0])),
            "I_lwa_a_pre_peak": ("event", np.array([4.0, 9.0, 16.0, 25.0])),
            "T_anom_mean_ant": ("event", np.array([1.0, 2.0, 4.0, 7.0])),
            "days_from_solstice": (
                "event",
                np.array([-30, -10, 10, 30], dtype="timedelta64[D]"),
            ),
        },
        coords={"event": event},
        attrs={"pipeline_stage": "stage_2_event_features"},
    )
