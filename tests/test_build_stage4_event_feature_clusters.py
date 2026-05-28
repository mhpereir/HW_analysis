import argparse
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import (
    build_stage4_event_feature_clusters as build_stage4_clusters,
)
from HW_analysis.scripts.event_features import plot_event_feature_clusters


def test_ward_clustering_adds_labels_tracked_values_and_summaries():
    pca = _make_pca_dataset()

    out = build_stage4_clusters.cluster_pc_scores(
        pca,
        method="ward",
        n_clusters=2,
    )

    assert out["cluster_label"].dims == ("event",)
    assert set(out["cluster_label"].values) == {0, 1}
    assert "cluster_probability" not in out
    assert out["tracked_variable_value"].dims == ("event", "tracked_variable")
    assert out.sizes["tracked_variable"] == len(
        build_stage4_clusters.DEFAULT_TRACKED_VARIABLES
    )
    assert out["cluster_count"].dims == ("cluster",)
    assert int(out["cluster_count"].sum()) == pca.sizes["event"]
    assert out["cluster_variable_mean"].dims == ("cluster", "tracked_variable")
    assert out["cluster_variable_n_finite"].dims == ("cluster", "tracked_variable")
    assert out.attrs["pipeline_stage"] == "stage_4_event_feature_clusters"
    assert out.attrs["cluster_method"] == "ward"
    assert out.attrs["cluster_n"] == 2
    assert out.attrs["cluster_pcs"] == "PC1,PC2,PC3"
    assert out.attrs["clustering_performed"] == 1
    assert "silhouette_score" in out.attrs


def test_kmeans_uses_random_state_and_records_metadata():
    pca = _make_pca_dataset()

    out = build_stage4_clusters.cluster_pc_scores(
        pca,
        method="kmeans",
        n_clusters=2,
        random_state=42,
        tracked_variables=("PC1", "tas_anom_peak"),
    )

    assert out.attrs["cluster_method"] == "kmeans"
    assert out.attrs["random_state"] == 42
    assert out.attrs["cluster_tracked_variables"] == "PC1,tas_anom_peak"
    np.testing.assert_array_equal(
        out["tracked_variable"].values,
        ["PC1", "tas_anom_peak"],
    )


def test_gmm_adds_probabilities_with_rows_that_sum_to_one():
    pca = _make_pca_dataset()

    out = build_stage4_clusters.cluster_pc_scores(
        pca,
        method="gmm",
        n_clusters=2,
        random_state=7,
    )

    assert out["cluster_probability"].dims == ("event", "cluster")
    assert out.sizes["cluster"] == 2
    np.testing.assert_allclose(out["cluster_probability"].sum("cluster").values, 1.0)


def test_default_tracked_variables_cover_cluster_plot_variables():
    plot_variables = {
        plot_event_feature_clusters.X_VARIABLE,
        *plot_event_feature_clusters.Y_VARIABLES,
    }

    assert plot_variables <= set(build_stage4_clusters.DEFAULT_TRACKED_VARIABLES)


def test_cluster_plot_feature_values_read_tracked_variables():
    clusters = _make_cluster_plot_dataset()

    plot_event_feature_clusters.validate_feature_variables(clusters)
    np.testing.assert_allclose(
        plot_event_feature_clusters.feature_values(clusters, "I_dTdt_pre"),
        [10.0, 11.0, 12.0, 30.0, 31.0, 32.0],
    )
    np.testing.assert_allclose(
        plot_event_feature_clusters.feature_values(clusters, "f_diabatic_pre"),
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    )
    np.testing.assert_allclose(
        plot_event_feature_clusters.feature_values(clusters, "duration"),
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )


def test_cluster_plot_validation_requires_cluster_labels():
    clusters = _make_cluster_plot_dataset().drop_vars("cluster_label")

    with pytest.raises(ValueError, match="cluster_label"):
        plot_event_feature_clusters.validate_feature_variables(clusters)


def test_cluster_plot_uses_integer_cluster_colorbar_ticks():
    clusters = _make_cluster_plot_dataset()

    fig = plot_event_feature_clusters.plot_tendency_scatter(clusters)
    try:
        colorbar_axis = fig.axes[-1]
        np.testing.assert_array_equal(colorbar_axis.get_yticks(), [0, 1, 2])
        assert [text.get_text() for text in colorbar_axis.get_yticklabels()] == [
            "0",
            "1",
            "2",
        ]
    finally:
        plot_event_feature_clusters.plt.close(fig)


def test_cluster_plot_main_writes_raw_and_standardized_for_each_method(
    monkeypatch,
    tmp_path,
):
    input_dir = tmp_path / "clusters"
    output_dir = tmp_path / "diagnostics"
    input_dir.mkdir()
    ward_path = input_dir / "hw_event_feature_clusters_demo_ward_k2_pc1-pc2.nc"
    kmeans_path = input_dir / "hw_event_feature_clusters_demo_kmeans_k2_pc1-pc2.nc"
    ward_path.write_text("placeholder")
    kmeans_path.write_text("placeholder")
    opened = []
    written = []

    def fake_open(path):
        opened.append(Path(path))
        method = "ward" if "ward" in Path(path).name else "kmeans"
        return _make_cluster_plot_dataset(method=method)

    def fake_write(
        ds,
        path,
        *,
        point_size,
        alpha,
        standardized=False,
    ):
        written.append(
            (
                ds.attrs["cluster_method"],
                Path(path).name,
                point_size,
                alpha,
                standardized,
            )
        )
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_event_feature_clusters.py",
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--methods",
            "ward",
            "kmeans",
            "--n-clusters",
            "2",
            "--pcs",
            "PC1",
            "PC2",
            "--point-size",
            "12",
            "--alpha",
            "0.5",
        ],
    )
    monkeypatch.setattr(plot_event_feature_clusters, "open_cluster_dataset", fake_open)
    monkeypatch.setattr(
        plot_event_feature_clusters,
        "write_tendency_scatter_plot",
        fake_write,
    )

    result = plot_event_feature_clusters.main()

    assert result == 0
    assert opened == [ward_path.resolve(), kmeans_path.resolve()]
    assert written == [
        (
            "ward",
            "hw_event_feature_clusters_demo_ward_k2_pc1-pc2_tendency_scatter.png",
            12.0,
            0.5,
            False,
        ),
        (
            "ward",
            "hw_event_feature_clusters_demo_ward_k2_pc1-pc2_tendency_scatter_standardized.png",
            12.0,
            0.5,
            True,
        ),
        (
            "kmeans",
            "hw_event_feature_clusters_demo_kmeans_k2_pc1-pc2_tendency_scatter.png",
            12.0,
            0.5,
            False,
        ),
        (
            "kmeans",
            "hw_event_feature_clusters_demo_kmeans_k2_pc1-pc2_tendency_scatter_standardized.png",
            12.0,
            0.5,
            True,
        ),
    ]


def test_resolve_tracked_variable_reads_pcs_direct_variables_and_derived_variables():
    pca = _make_pca_dataset()

    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "PC2"),
        pca["pc_score"].sel(pc="PC2").values,
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "I_dTdt_pre"),
        [10.0, 11.0, 12.0, 30.0, 31.0, 32.0],
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "duration"),
        [2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "tas_excess_integral"),
        [100.0, 110.0, 120.0, 300.0, 310.0, 320.0],
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(
            pca,
            "log10_tas_excess_integral",
        ),
        np.log10([100.0, 110.0, 120.0, 300.0, 310.0, 320.0]),
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "sqrt_I_lwa_a_pre_peak"),
        [2.0, 3.0, 4.0, 5.0, np.nan, 6.0],
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "cos_days_from_solstice"),
        np.cos(
            np.array([0.0, 30.0, 60.0, 90.0, 120.0, 150.0])
            * 2.0
            * np.pi
            / 365.0
        ),
    )

    denominator = np.array([6.0, 7.0, 10.0, 15.0, 16.0, 20.0])
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "f_adiabatic_pre"),
        np.array([1.0, 2.0, 4.0, 5.0, 6.0, 8.0]) / denominator,
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "f_diabatic_pre"),
        np.array([3.0, 1.0, 2.0, 4.0, 7.0, 9.0]) / denominator,
    )
    np.testing.assert_allclose(
        build_stage4_clusters.resolve_tracked_variable(pca, "f_advection_pre"),
        np.array([-2.0, -4.0, -4.0, -6.0, -3.0, -3.0]) / denominator,
    )


def test_summarize_clusters_uses_finite_values_and_handles_empty_clusters():
    tracked = xr.DataArray(
        np.array(
            [
                [1.0, 10.0],
                [3.0, np.nan],
                [5.0, 30.0],
            ]
        ),
        dims=("event", "tracked_variable"),
        coords={
            "event": [1, 2, 3],
            "tracked_variable": ["a", "b"],
        },
    )

    out = build_stage4_clusters.summarize_clusters(
        tracked,
        np.array([0, 0, 1], dtype=np.int64),
        3,
    )

    np.testing.assert_array_equal(out["cluster_count"].values, [2, 1, 0])
    np.testing.assert_allclose(out["cluster_variable_mean"].sel(cluster=0).values, [2.0, 10.0])
    np.testing.assert_array_equal(out["cluster_variable_n_finite"].sel(cluster=0).values, [2, 1])
    assert np.isnan(out["cluster_variable_mean"].sel(cluster=2, tracked_variable="a").item())


def test_missing_tracked_variable_raises_clear_error():
    with pytest.raises(ValueError, match="tracked variable"):
        build_stage4_clusters.resolve_tracked_variable(_make_pca_dataset(), "not_a_variable")


def test_missing_pc_raises_clear_error():
    with pytest.raises(ValueError, match="Requested PCs"):
        build_stage4_clusters.cluster_pc_scores(
            _make_pca_dataset(),
            pcs=("PC1", "PC9"),
            n_clusters=2,
        )


def test_invalid_cluster_counts_raise_clear_errors():
    pca = _make_pca_dataset()

    with pytest.raises(ValueError, match="n_clusters must be >= 2"):
        build_stage4_clusters.cluster_pc_scores(pca, n_clusters=1)

    with pytest.raises(ValueError, match="cannot exceed number of events"):
        build_stage4_clusters.cluster_pc_scores(pca, n_clusters=pca.sizes["event"] + 1)


def test_nonfinite_score_matrix_raises_clear_error():
    pca = _make_pca_dataset()
    pca["pc_score"][0, 0] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        build_stage4_clusters.cluster_pc_scores(pca, n_clusters=2)


def test_write_cluster_output_respects_overwrite_flag(tmp_path):
    out = build_stage4_clusters.cluster_pc_scores(
        _make_pca_dataset(),
        method="ward",
        n_clusters=2,
    )
    output_path = tmp_path / "clusters.nc"
    output_path.write_text("exists")

    with pytest.raises(FileExistsError, match="--overwrite"):
        build_stage4_clusters.write_cluster_output(out, output_path)

    written = build_stage4_clusters.write_cluster_output(out, output_path, overwrite=True)

    assert written == output_path.resolve()
    assert output_path.exists()


def test_cluster_output_path_uses_expected_filename(tmp_path):
    path = build_stage4_clusters.cluster_output_path(
        "hw_event_feature_pca_pnw_bartusek_tas_q90_1940_2024.nc",
        tmp_path,
        method="kmeans",
        pcs=("PC1", "PC2", "PC3"),
        n_clusters=3,
    )

    assert path == (
        tmp_path
        / "hw_event_feature_clusters_pnw_bartusek_tas_q90_1940_2024_kmeans_k3_pc1-pc2-pc3.nc"
    )


def test_main_loops_over_requested_methods_and_writes_outputs(monkeypatch, tmp_path):
    pca = _make_pca_dataset()
    input_path = tmp_path / "pca.nc"
    output_dir = tmp_path / "clusters"
    captured = []

    def fake_open(path):
        assert path == input_path
        return pca

    def fake_write(ds, path, *, overwrite):
        captured.append((ds.attrs["cluster_method"], Path(path).name, overwrite))
        return Path(path)

    monkeypatch.setattr(
        "sys.argv",
        [
            "build_stage4_event_feature_clusters.py",
            "--input-path",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--methods",
            "ward",
            "kmeans",
            "--pcs",
            "PC1",
            "PC2",
            "--n-clusters",
            "2",
            "--random-state",
            "11",
            "--tracked-variables",
            "PC1",
            "tas_anom_peak",
            "--overwrite",
        ],
    )
    monkeypatch.setattr(build_stage4_clusters, "open_pca_dataset", fake_open)
    monkeypatch.setattr(build_stage4_clusters, "write_cluster_output", fake_write)

    result = build_stage4_clusters.main()

    assert result == 0
    assert [method for method, _, _ in captured] == ["ward", "kmeans"]
    assert all(overwrite for _, _, overwrite in captured)
    assert captured[0][1] == "hw_event_feature_clusters_pca_ward_k2_pc1-pc2.nc"
    assert captured[1][1] == "hw_event_feature_clusters_pca_kmeans_k2_pc1-pc2.nc"


def test_validate_args_rejects_empty_inputs():
    with pytest.raises(ValueError, match="n-clusters"):
        build_stage4_clusters.validate_args(
            argparse.Namespace(
                n_clusters=1,
                methods=["ward"],
                pcs=["PC1"],
                tracked_variables=["PC1"],
            )
        )
    with pytest.raises(ValueError, match="At least one clustering method"):
        build_stage4_clusters.validate_args(
            argparse.Namespace(
                n_clusters=2,
                methods=[],
                pcs=["PC1"],
                tracked_variables=["PC1"],
            )
        )
    with pytest.raises(ValueError, match="At least one PC"):
        build_stage4_clusters.validate_args(
            argparse.Namespace(
                n_clusters=2,
                methods=["ward"],
                pcs=[],
                tracked_variables=["PC1"],
            )
        )
    with pytest.raises(ValueError, match="At least one tracked variable"):
        build_stage4_clusters.validate_args(
            argparse.Namespace(
                n_clusters=2,
                methods=["ward"],
                pcs=["PC1"],
                tracked_variables=[],
            )
        )


def _make_pca_dataset() -> xr.Dataset:
    event = np.array([101, 102, 103, 104, 105, 106], dtype=np.int64)
    pc = np.array(["PC1", "PC2", "PC3", "PC4"], dtype=object)
    scores = np.array(
        [
            [0.0, 0.0, 0.0, 0.2],
            [0.1, 0.2, 0.0, 0.1],
            [-0.2, -0.1, 0.1, 0.0],
            [5.0, 5.0, 5.0, -0.1],
            [5.2, 4.9, 5.1, -0.2],
            [4.8, 5.1, 4.9, -0.3],
        ],
        dtype=float,
    )
    feature = np.array(["I_dTdt_pre"], dtype=object)
    return xr.Dataset(
        data_vars={
            "pc_score": (("event", "pc"), scores),
            "pc_loading": (("pc", "feature"), np.ones((4, 1))),
            "feature_matrix": (
                ("event", "feature"),
                np.array([[10.0], [11.0], [12.0], [30.0], [31.0], [32.0]]),
            ),
            "I_dTdt_pre": ("event", np.array([10.0, 11.0, 12.0, 30.0, 31.0, 32.0])),
            "I_adiabatic_pre": ("event", np.array([1.0, 2.0, 4.0, 5.0, 6.0, 8.0])),
            "I_diabatic_pre": ("event", np.array([3.0, 1.0, 2.0, 4.0, 7.0, 9.0])),
            "I_advection_pre": ("event", np.array([-2.0, -4.0, -4.0, -6.0, -3.0, -3.0])),
            "I_lwa_a_pre_peak": ("event", np.array([4.0, 9.0, 16.0, 25.0, -1.0, 36.0])),
            "T_anom_mean_ant": ("event", np.array([1.0, 1.2, 1.4, 3.0, 3.2, 3.4])),
            "days_from_solstice": (
                "event",
                np.array([0, 30, 60, 90, 120, 150], dtype="timedelta64[D]"),
            ),
            "duration": ("event", np.array([2, 3, 4, 5, 6, 7], dtype="timedelta64[D]")),
            "tas_anom_peak": ("event", np.array([5.0, 5.2, 5.4, 8.0, 8.2, 8.4])),
            "tas_excess_integral": (
                "event",
                np.array([100.0, 110.0, 120.0, 300.0, 310.0, 320.0]),
            ),
        },
        coords={
            "event": event,
            "pc": pc,
            "feature": feature,
        },
    )


def _make_cluster_plot_dataset(method: str = "ward") -> xr.Dataset:
    event = np.array([101, 102, 103, 104, 105, 106], dtype=np.int64)
    tracked_variable = np.array(
        [
            plot_event_feature_clusters.X_VARIABLE,
            *plot_event_feature_clusters.Y_VARIABLES,
        ],
        dtype=object,
    )
    tracked_values = np.column_stack(
        [
            np.array([10.0, 11.0, 12.0, 30.0, 31.0, 32.0]),
            np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
            np.array([0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
            np.array([-0.3, -0.3, -0.3, -0.3, -0.3, -0.3]),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            np.array([1.0, 1.2, 1.4, 3.0, 3.2, 3.4]),
            np.array([1.0, 0.9, 0.8, 0.7, 0.6, 0.5]),
            np.array([2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            np.array([5.0, 5.2, 5.4, 8.0, 8.2, 8.4]),
            np.array([2.0, 2.1, 2.2, 2.4, 2.5, 2.6]),
        ]
    )
    return xr.Dataset(
        data_vars={
            "cluster_label": ("event", np.array([0, 1, 2, 0, 1, 2])),
            "tracked_variable_value": (
                ("event", "tracked_variable"),
                tracked_values,
            ),
        },
        coords={
            "event": event,
            "tracked_variable": tracked_variable,
        },
        attrs={"cluster_method": method},
    )
