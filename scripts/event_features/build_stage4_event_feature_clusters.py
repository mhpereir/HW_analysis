"""Cluster heatwave events in PCA score space."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage3_event_feature_pca"
    / "hw_event_feature_pca_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "stage4_event_feature_clusters"

EVENT_DIM = "event"
PC_DIM = "pc"
CLUSTER_DIM = "cluster"
TRACKED_VARIABLE_DIM = "tracked_variable"

DEFAULT_PCS = ("PC1", "PC2", "PC3")
DEFAULT_METHODS = ("ward", "kmeans", "gmm")
DEFAULT_TRACKED_VARIABLES = (
    "PC1",
    "PC2",
    "PC3",
    "I_dTdt_pre",
    "I_adiabatic_pre",
    "I_diabatic_pre",
    "I_advection_pre",
    "f_adiabatic_pre",
    "f_diabatic_pre",
    "f_advection_pre",
    "sqrt_I_lwa_a_pre_peak",
    "T_anom_mean_ant",
    "cos_days_from_solstice",
    "duration",
    "tas_anom_peak",
    "log10_tas_excess_integral",
    "tas_excess_integral",
)
INTEGRATED_HEAT_BUDGET_VARIABLES = (
    "I_adiabatic_pre",
    "I_diabatic_pre",
    "I_advection_pre",
)
HEAT_BUDGET_FRACTION_NUMERATORS = {
    "f_adiabatic_pre": "I_adiabatic_pre",
    "f_diabatic_pre": "I_diabatic_pre",
    "f_advection_pre": "I_advection_pre",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for PCA score clustering."""
    parser = argparse.ArgumentParser(
        description="Cluster heatwave events using principal-component scores."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the PCA NetCDF product.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where clustering NetCDF files will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
        help="Clustering methods to run.",
    )
    parser.add_argument(
        "--pcs",
        nargs="+",
        default=list(DEFAULT_PCS),
        help="Principal components used as clustering coordinates.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=3,
        help="Number of clusters/components.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Random seed for stochastic clustering methods.",
    )
    parser.add_argument(
        "--tracked-variables",
        nargs="+",
        default=list(DEFAULT_TRACKED_VARIABLES),
        help="Event-level variables to retain and summarize by cluster.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow output files to replace existing files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI options that do not require opening the dataset."""
    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be >= 2.")
    if not args.methods:
        raise ValueError("At least one clustering method is required.")
    if not args.pcs:
        raise ValueError("At least one PC is required.")
    if not args.tracked_variables:
        raise ValueError("At least one tracked variable is required.")


def main() -> int:
    """Open the PCA dataset, run requested clustering methods, and write outputs."""
    args = parse_args()
    validate_args(args)

    pca = open_pca_dataset(args.input_path)
    try:
        written = []
        for method in args.methods:
            clustered = cluster_pc_scores(
                pca,
                pcs=tuple(args.pcs),
                method=method,
                n_clusters=args.n_clusters,
                random_state=args.random_state,
                tracked_variables=tuple(args.tracked_variables),
            )
            clustered.attrs["source_pca_path"] = str(args.input_path)
            output_path = cluster_output_path(
                args.input_path,
                args.output_dir,
                method=method,
                pcs=tuple(args.pcs),
                n_clusters=args.n_clusters,
            )
            written.append(
                write_cluster_output(clustered, output_path, overwrite=args.overwrite)
            )

        print("Wrote PCA clustering datasets:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        pca.close()
    return 0


def open_pca_dataset(path: str | Path) -> xr.Dataset:
    """Open a PCA NetCDF product."""
    input_path = Path(path).expanduser().resolve()
    try:
        return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)
    except TypeError as exc:
        if "decode_timedelta" not in str(exc):
            raise
        return xr.open_dataset(input_path, engine="h5netcdf")


def cluster_pc_scores(
    pca_ds: xr.Dataset,
    *,
    pcs: tuple[str, ...] = DEFAULT_PCS,
    method: str = "ward",
    n_clusters: int = 3,
    random_state: int = 0,
    tracked_variables: tuple[str, ...] = DEFAULT_TRACKED_VARIABLES,
) -> xr.Dataset:
    """Return a PCA dataset copy with cluster assignments and summaries attached."""
    validate_cluster_inputs(pca_ds, pcs=pcs, method=method, n_clusters=n_clusters)
    if not tracked_variables:
        raise ValueError("At least one tracked variable is required.")
    score_matrix = clustering_score_matrix(pca_ds, pcs)

    if method == "ward":
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage="ward",
        )
        labels = model.fit_predict(score_matrix)
        probabilities = None
    elif method == "kmeans":
        model = KMeans(
            n_clusters=n_clusters,
            n_init=50,
            random_state=random_state,
        )
        labels = model.fit_predict(score_matrix)
        probabilities = None
    elif method == "gmm":
        model = GaussianMixture(
            n_components=n_clusters,
            covariance_type="full",
            random_state=random_state,
        )
        labels = model.fit_predict(score_matrix)
        probabilities = model.predict_proba(score_matrix)
    else:
        raise ValueError(f"Unknown clustering method: {method}")

    out = pca_ds.copy(deep=False)
    out = out.assign_coords({CLUSTER_DIM: np.arange(n_clusters, dtype=np.int64)})
    out["cluster_label"] = (EVENT_DIM, labels.astype(np.int64))
    out["cluster_label"].attrs["description"] = (
        "Zero-based cluster label assigned in PCA score space."
    )

    if probabilities is not None:
        out["cluster_probability"] = (
            (EVENT_DIM, CLUSTER_DIM),
            np.asarray(probabilities, dtype=float),
        )
        out["cluster_probability"].attrs["description"] = (
            "Posterior cluster membership probabilities from GaussianMixture."
        )

    tracked = build_tracked_variable_array(out, tracked_variables)
    out["tracked_variable_value"] = tracked
    out["tracked_variable_value"].attrs["description"] = (
        "Event-level variables retained for cluster interpretation."
    )

    summary = summarize_clusters(tracked, labels, n_clusters)
    for name, da in summary.data_vars.items():
        out[name] = da

    add_cluster_attrs(
        out,
        labels=labels,
        score_matrix=score_matrix,
        pcs=pcs,
        method=method,
        n_clusters=n_clusters,
        random_state=random_state,
        tracked_variables=tracked_variables,
    )
    return out


def validate_cluster_inputs(
    pca_ds: xr.Dataset,
    *,
    pcs: tuple[str, ...],
    method: str,
    n_clusters: int,
) -> None:
    """Fail clearly when a PCA dataset cannot support requested clustering."""
    if EVENT_DIM not in pca_ds.sizes:
        raise ValueError(f"PCA dataset is missing required {EVENT_DIM!r} dimension.")
    if PC_DIM not in pca_ds.coords:
        raise ValueError(f"PCA dataset is missing required {PC_DIM!r} coordinate.")
    if "pc_score" not in pca_ds:
        raise ValueError("PCA dataset is missing required variable: pc_score.")
    if method not in DEFAULT_METHODS:
        raise ValueError(f"Unknown clustering method: {method}")
    if not pcs:
        raise ValueError("At least one PC is required.")
    if n_clusters < 2:
        raise ValueError("n_clusters must be >= 2.")

    n_events = pca_ds.sizes[EVENT_DIM]
    if n_clusters > n_events:
        raise ValueError(
            f"n_clusters cannot exceed number of events; got {n_clusters}, "
            f"n_events is {n_events}."
        )

    available_pcs = {str(value) for value in pca_ds[PC_DIM].values}
    missing_pcs = [pc for pc in pcs if pc not in available_pcs]
    if missing_pcs:
        raise ValueError(f"Requested PCs are not present: {', '.join(missing_pcs)}.")


def clustering_score_matrix(pca_ds: xr.Dataset, pcs: tuple[str, ...]) -> np.ndarray:
    """Return finite PCA score matrix for requested clustering coordinates."""
    scores = pca_ds["pc_score"].sel({PC_DIM: list(pcs)}).transpose(EVENT_DIM, PC_DIM)
    matrix = np.asarray(scores.values, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("pc_score selection must produce a 2D event-by-pc matrix.")
    if not np.isfinite(matrix).all():
        raise ValueError("Cluster score matrix contains non-finite values.")
    return matrix


def resolve_tracked_variable(pca_ds: xr.Dataset, name: str) -> np.ndarray:
    """Return one tracked variable as a float event vector."""
    if name in pc_names(pca_ds):
        if "pc_score" not in pca_ds:
            raise ValueError("PCA dataset is missing required variable: pc_score.")
        return np.asarray(pca_ds["pc_score"].sel({PC_DIM: name}).values, dtype=float)

    if name in pca_ds:
        da = pca_ds[name]
        if da.dims != (EVENT_DIM,):
            raise ValueError(
                f"Tracked variable {name!r} must be 1D over {EVENT_DIM!r}; "
                f"got dims {da.dims!r}."
            )
        return data_array_to_float_vector(da)

    if has_feature_matrix_variable(pca_ds, name):
        values = pca_ds["feature_matrix"].sel(feature=name).values
        return np.asarray(values, dtype=float)

    if name in HEAT_BUDGET_FRACTION_NUMERATORS:
        return heat_budget_fraction_values(pca_ds, name)

    if name == "sqrt_I_lwa_a_pre_peak" and "I_lwa_a_pre_peak" in pca_ds:
        source = data_array_to_float_vector(pca_ds["I_lwa_a_pre_peak"])
        out = np.full(source.shape, np.nan, dtype=float)
        valid = np.isfinite(source) & (source >= 0.0)
        out[valid] = np.sqrt(source[valid])
        return out

    if name == "cos_days_from_solstice" and "days_from_solstice" in pca_ds:
        source = data_array_to_float_vector(pca_ds["days_from_solstice"])
        return np.cos(source * 2.0 * np.pi / 365.0)

    if name == "log10_tas_excess_integral" and "tas_excess_integral" in pca_ds:
        source = data_array_to_float_vector(pca_ds["tas_excess_integral"])
        out = np.full(source.shape, np.nan, dtype=float)
        valid = np.isfinite(source) & (source > 0.0)
        out[valid] = np.log10(source[valid])
        return out

    raise ValueError(f"PCA dataset does not contain tracked variable {name!r}.")


def pc_names(pca_ds: xr.Dataset) -> set[str]:
    """Return available PC coordinate names, or an empty set when absent."""
    if PC_DIM not in pca_ds.coords:
        return set()
    return {str(value) for value in pca_ds[PC_DIM].values}


def data_array_to_float_vector(da: xr.DataArray) -> np.ndarray:
    """Return a 1D event DataArray as floats, converting timedeltas to days."""
    values = da.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return values / np.timedelta64(1, "D")
    return np.asarray(values, dtype=float)


def has_feature_matrix_variable(pca_ds: xr.Dataset, name: str) -> bool:
    """Return True when a variable is available as a row in feature_matrix."""
    if "feature_matrix" not in pca_ds or "feature" not in pca_ds.coords:
        return False
    return name in {str(value) for value in pca_ds["feature"].values}


def heat_budget_fraction_values(pca_ds: xr.Dataset, variable: str) -> np.ndarray:
    """Return one heat-budget term as a fraction of summed absolute budget terms."""
    missing = [name for name in INTEGRATED_HEAT_BUDGET_VARIABLES if name not in pca_ds]
    if missing:
        raise ValueError(
            f"Cannot derive {variable!r}; missing source variables: "
            f"{', '.join(missing)}."
        )

    numerator_name = HEAT_BUDGET_FRACTION_NUMERATORS[variable]
    numerator = data_array_to_float_vector(pca_ds[numerator_name])
    denominator = np.zeros_like(numerator, dtype=float)
    for source_name in INTEGRATED_HEAT_BUDGET_VARIABLES:
        denominator = denominator + np.abs(data_array_to_float_vector(pca_ds[source_name]))
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0.0)
    return out


def build_tracked_variable_array(
    pca_ds: xr.Dataset,
    tracked_variables: tuple[str, ...] = DEFAULT_TRACKED_VARIABLES,
) -> xr.DataArray:
    """Return tracked event-level values with event and tracked-variable dims."""
    columns = [resolve_tracked_variable(pca_ds, name) for name in tracked_variables]
    values = np.column_stack(columns).astype(float)
    return xr.DataArray(
        values,
        dims=(EVENT_DIM, TRACKED_VARIABLE_DIM),
        coords={
            EVENT_DIM: pca_ds[EVENT_DIM].values,
            TRACKED_VARIABLE_DIM: np.asarray(tracked_variables, dtype=object),
        },
    )


def summarize_clusters(
    tracked_values: xr.DataArray,
    labels: np.ndarray,
    n_clusters: int,
) -> xr.Dataset:
    """Return finite-value summary statistics by cluster and tracked variable."""
    values = np.asarray(tracked_values.values, dtype=float)
    labels = np.asarray(labels, dtype=np.int64)
    if values.shape[0] != labels.size:
        raise ValueError("Tracked values and cluster labels must have the same event length.")

    n_variables = values.shape[1]
    shape = (n_clusters, n_variables)
    mean = np.full(shape, np.nan, dtype=float)
    median = np.full(shape, np.nan, dtype=float)
    std = np.full(shape, np.nan, dtype=float)
    minimum = np.full(shape, np.nan, dtype=float)
    maximum = np.full(shape, np.nan, dtype=float)
    n_finite = np.zeros(shape, dtype=np.int64)
    cluster_count = np.zeros(n_clusters, dtype=np.int64)

    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        cluster_count[cluster_id] = int(cluster_mask.sum())
        if not cluster_mask.any():
            continue
        cluster_values = values[cluster_mask, :]
        finite = np.isfinite(cluster_values)
        n_finite[cluster_id, :] = finite.sum(axis=0)
        for variable_idx in range(n_variables):
            valid_values = cluster_values[finite[:, variable_idx], variable_idx]
            if valid_values.size == 0:
                continue
            mean[cluster_id, variable_idx] = float(np.mean(valid_values))
            median[cluster_id, variable_idx] = float(np.median(valid_values))
            std[cluster_id, variable_idx] = float(np.std(valid_values))
            minimum[cluster_id, variable_idx] = float(np.min(valid_values))
            maximum[cluster_id, variable_idx] = float(np.max(valid_values))

    coords = {
        CLUSTER_DIM: np.arange(n_clusters, dtype=np.int64),
        TRACKED_VARIABLE_DIM: tracked_values[TRACKED_VARIABLE_DIM].values,
    }
    summary = xr.Dataset(coords=coords)
    summary["cluster_count"] = (CLUSTER_DIM, cluster_count)
    summary["cluster_variable_mean"] = (
        (CLUSTER_DIM, TRACKED_VARIABLE_DIM),
        mean,
    )
    summary["cluster_variable_median"] = (
        (CLUSTER_DIM, TRACKED_VARIABLE_DIM),
        median,
    )
    summary["cluster_variable_std"] = (
        (CLUSTER_DIM, TRACKED_VARIABLE_DIM),
        std,
    )
    summary["cluster_variable_min"] = (
        (CLUSTER_DIM, TRACKED_VARIABLE_DIM),
        minimum,
    )
    summary["cluster_variable_max"] = (
        (CLUSTER_DIM, TRACKED_VARIABLE_DIM),
        maximum,
    )
    summary["cluster_variable_n_finite"] = (
        (CLUSTER_DIM, TRACKED_VARIABLE_DIM),
        n_finite,
    )
    add_summary_attrs(summary)
    return summary


def add_summary_attrs(summary: xr.Dataset) -> None:
    """Attach metadata to cluster summary variables."""
    summary["cluster_count"].attrs["description"] = "Number of events assigned to each cluster."
    summary["cluster_variable_mean"].attrs["description"] = (
        "Cluster mean of each tracked variable over finite event values."
    )
    summary["cluster_variable_median"].attrs["description"] = (
        "Cluster median of each tracked variable over finite event values."
    )
    summary["cluster_variable_std"].attrs["description"] = (
        "Cluster population standard deviation of each tracked variable over finite event values."
    )
    summary["cluster_variable_min"].attrs["description"] = (
        "Cluster minimum of each tracked variable over finite event values."
    )
    summary["cluster_variable_max"].attrs["description"] = (
        "Cluster maximum of each tracked variable over finite event values."
    )
    summary["cluster_variable_n_finite"].attrs["description"] = (
        "Number of finite event values used for each cluster-variable summary."
    )


def add_cluster_attrs(
    out: xr.Dataset,
    *,
    labels: np.ndarray,
    score_matrix: np.ndarray,
    pcs: tuple[str, ...],
    method: str,
    n_clusters: int,
    random_state: int,
    tracked_variables: tuple[str, ...],
) -> None:
    """Attach clustering method metadata and valid internal scores."""
    out.attrs.update(
        {
            "pipeline_stage": "stage_4_event_feature_clusters",
            "source_pca_path": str(out.attrs.get("source_pca_path", "")),
            "cluster_method": method,
            "cluster_n": int(n_clusters),
            "cluster_pcs": ",".join(pcs),
            "cluster_tracked_variables": ",".join(tracked_variables),
            "random_state": int(random_state),
            "clustering_performed": 1,
        }
    )
    unique_labels = np.unique(labels)
    if 1 < unique_labels.size < score_matrix.shape[0]:
        out.attrs["silhouette_score"] = float(silhouette_score(score_matrix, labels))
        out.attrs["davies_bouldin_score"] = float(
            davies_bouldin_score(score_matrix, labels)
        )
        out.attrs["calinski_harabasz_score"] = float(
            calinski_harabasz_score(score_matrix, labels)
        )


def write_cluster_output(
    clustered_ds: xr.Dataset,
    output_path: str | Path,
    *,
    overwrite: bool = False,
) -> Path:
    """Write a PCA clustering dataset to NetCDF."""
    path = Path(output_path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output path already exists: {path}. Pass --overwrite.")
    path.parent.mkdir(parents=True, exist_ok=True)
    clustered_ds.to_netcdf(path, engine="h5netcdf")
    return path


def cluster_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    method: str,
    pcs: tuple[str, ...],
    n_clusters: int,
) -> Path:
    """Return the output path for one method/cluster-count/PC selection."""
    input_stem = Path(input_path).stem
    run_stem = input_stem.removeprefix("hw_event_feature_pca_")
    pc_slug = "-".join(pc.lower() for pc in pcs)
    filename = (
        f"hw_event_feature_clusters_{run_stem}_{method}_k{n_clusters}_{pc_slug}.nc"
    )
    return Path(output_dir) / filename


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
