"""Build PCA products from fixed-window event-feature tables."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from sklearn.decomposition import PCA


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / "hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage3_event_feature_pca"
    / "hw_event_feature_pca_pnw_bartusek_tas_q90_1940_2024.nc"
)

EVENT_DIM = "event"
ORIGINAL_EVENT_DIM = "event_original"
PC_DIM = "pc"
FEATURE_DIM = "feature"

INTEGRATED_HEAT_BUDGET_VARIABLES = (
    "I_adiabatic_pre",
    "I_diabatic_pre",
    "I_advection_pre",
)
HEAT_BUDGET_FRACTION_FEATURES = (
    "f_adiabatic_pre",
    "f_diabatic_pre",
    "f_advection_pre",
)
DEFAULT_PCA_FEATURES = (
    "I_dTdt_pre",
    "f_adiabatic_pre",
    "f_advection_pre",
    "sqrt_I_lwa_a_pre_peak",
    "T_anom_mean_ant",
    "cos_days_from_solstice",
    "duration",
)
DERIVED_FEATURE_SOURCES = {
    "f_adiabatic_pre": INTEGRATED_HEAT_BUDGET_VARIABLES,
    "f_diabatic_pre": INTEGRATED_HEAT_BUDGET_VARIABLES,
    "f_advection_pre": INTEGRATED_HEAT_BUDGET_VARIABLES,
    "sqrt_I_lwa_a_pre_peak": "I_lwa_a_pre_peak",
    "cos_days_from_solstice": "days_from_solstice",
    "log10_tas_excess_integral": "tas_excess_integral",
}
HEAT_BUDGET_FRACTION_NUMERATORS = {
    "f_adiabatic_pre": "I_adiabatic_pre",
    "f_diabatic_pre": "I_diabatic_pre",
    "f_advection_pre": "I_advection_pre",
}
EVENT_METADATA_VARIABLES = (
    "event_id",
    "start_time",
    "end_time",
    "peak_time",
    "duration",
    "tas_peak",
    "tas_anom_peak",
    "tas_excess_peak",
    "tas_excess_integral",
    "lwa_a_peak",
    "lwa_c_peak",
)
DIAGNOSTIC_VARIABLES = (
    "I_dTdt_pre",
    "I_adiabatic_pre",
    "I_diabatic_pre",
    "I_advection_pre",
    "I_lwa_a_pre_peak",
    "T_anom_mean_ant",
    "days_from_solstice",
)
DERIVED_DIAGNOSTIC_VARIABLES = (
    "log10_tas_excess_integral",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for event-feature PCA construction."""
    parser = argparse.ArgumentParser(
        description="Build a PCA NetCDF product from an event-feature table."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the event-feature NetCDF table.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the PCA NetCDF product will be written.",
    )
    parser.add_argument(
        "--features",
        nargs="+",
        default=list(DEFAULT_PCA_FEATURES),
        help="PCA input feature names. Derived feature names are supported.",
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of principal components. Defaults to min(n_valid_events, n_features).",
    )
    parser.add_argument(
        "--scaler",
        choices=("standard", "robust"),
        default="standard",
        help="Column scaling applied before PCA.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow output files to replace existing files.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate PCA CLI arguments."""
    if args.n_components is not None and args.n_components < 1:
        raise ValueError("--n-components must be >= 1.")
    if len(args.features) < 2:
        raise ValueError("At least two PCA features are required.")
    if args.output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output path already exists: {args.output_path}. Pass --overwrite."
        )


def main() -> int:
    """Load an event-feature table, fit PCA, and write the PCA product."""
    args = parse_args()
    validate_args(args)

    features = open_event_features(args.input_path)
    try:
        pca_ds = build_event_feature_pca(
            features,
            feature_names=args.features,
            n_components=args.n_components,
            scaler=args.scaler,
            input_path=args.input_path,
        )
        written = write_pca_output(pca_ds, args.output_path)
        print("Wrote event-feature PCA dataset:")
        print(f"  {_display_path(written)}")
    finally:
        features.close()
    return 0


def open_event_features(path: str | Path) -> xr.Dataset:
    """Open an event-feature NetCDF table."""
    input_path = Path(path).expanduser().resolve()
    return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)


def build_event_feature_pca(
    feature_table: xr.Dataset,
    *,
    feature_names: Sequence[str] = DEFAULT_PCA_FEATURES,
    n_components: int | None = None,
    scaler: str = "standard",
    input_path: str | Path | None = None,
) -> xr.Dataset:
    """Return a PCA dataset built from selected event features."""
    validate_pca_inputs(feature_table, feature_names=feature_names, scaler=scaler)
    matrix_all = build_feature_matrix(feature_table, feature_names)
    valid_event_mask = np.asarray(np.isfinite(matrix_all).all(axis=1))
    valid_idx = np.flatnonzero(valid_event_mask)

    if valid_idx.size < 2:
        raise ValueError("Fewer than two events remain after finite-value filtering.")
    if len(feature_names) < 2:
        raise ValueError("At least two PCA features are required.")

    matrix = matrix_all[valid_idx, :]
    center, scale = fit_scaler(matrix, scaler=scaler)
    matrix_scaled = (matrix - center) / scale
    if not np.isfinite(matrix_scaled).all():
        raise ValueError("Scaled PCA feature matrix contains non-finite values.")

    max_components = min(matrix_scaled.shape)
    if n_components is None:
        n_components = max_components
    if n_components > max_components:
        raise ValueError(
            "n_components cannot exceed min(n_valid_events, n_features); "
            f"got {n_components}, maximum is {max_components}."
        )

    model = PCA(n_components=n_components)
    scores = model.fit_transform(matrix_scaled)

    out = make_pca_dataset(
        feature_table,
        feature_names=tuple(feature_names),
        valid_idx=valid_idx,
        valid_event_mask=valid_event_mask,
        matrix=matrix,
        matrix_scaled=matrix_scaled,
        center=center,
        scale=scale,
        scores=scores,
        model=model,
    )
    add_global_attrs(
        out,
        input_path=input_path,
        feature_names=feature_names,
        scaler=scaler,
        n_input_events=feature_table.sizes[EVENT_DIM],
        n_valid_events=valid_idx.size,
        n_components=n_components, # type: ignore
    )
    return out


def validate_pca_inputs(
    feature_table: xr.Dataset,
    *,
    feature_names: Sequence[str],
    scaler: str,
) -> None:
    """Validate event-feature table and requested PCA feature names."""
    if EVENT_DIM not in feature_table.dims:
        raise ValueError(f"Input feature table has no {EVENT_DIM!r} dimension.")
    if len(feature_names) < 2:
        raise ValueError("At least two PCA features are required.")
    if scaler not in {"standard", "robust"}:
        raise ValueError("scaler must be either 'standard' or 'robust'.")

    required = unique_required_sources(feature_names)
    missing = [name for name in required if name not in feature_table]
    if missing:
        raise ValueError(
            "Event-feature table is missing required source variables: "
            f"{', '.join(missing)}."
        )
    for name in required:
        da = feature_table[name]
        if da.dims != (EVENT_DIM,):
            raise ValueError(
                f"Source variable {name!r} must be 1D over {EVENT_DIM!r}; "
                f"got dims {da.dims!r}."
            )


def build_feature_matrix(
    feature_table: xr.Dataset,
    feature_names: Sequence[str],
) -> np.ndarray:
    """Return the unscaled PCA matrix for every input event."""
    columns = [feature_values(feature_table, name) for name in feature_names]
    return np.column_stack(columns).astype(float)


def feature_values(feature_table: xr.Dataset, feature_name: str) -> np.ndarray:
    """Return one requested PCA feature as a float event vector."""
    if feature_name in HEAT_BUDGET_FRACTION_FEATURES:
        return heat_budget_fraction_values(feature_table, feature_name)

    values = feature_table[source_variable_name(feature_name)].values
    if np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, "D")
    else:
        out = np.asarray(values, dtype=float)

    if feature_name == "sqrt_I_lwa_a_pre_peak":
        transformed = np.full(out.shape, np.nan, dtype=float)
        valid = out >= 0.0
        transformed[valid] = np.sqrt(out[valid])
        out = transformed
    elif feature_name == "cos_days_from_solstice":
        out = np.cos(out * 2.0 * np.pi / 365.0)
    elif feature_name == "log10_tas_excess_integral":
        transformed = np.full(out.shape, np.nan, dtype=float)
        valid = out > 0.0
        transformed[valid] = np.log10(out[valid])
        out = transformed
    return out


def heat_budget_fraction_values(feature_table: xr.Dataset, feature_name: str) -> np.ndarray:
    """Return one heat-budget term as a fraction of summed absolute budget terms."""
    numerator = np.asarray(
        feature_table[HEAT_BUDGET_FRACTION_NUMERATORS[feature_name]].values,
        dtype=float,
    )
    denominator = np.zeros_like(numerator, dtype=float)
    for source_name in INTEGRATED_HEAT_BUDGET_VARIABLES:
        denominator = denominator + np.abs(np.asarray(feature_table[source_name].values, dtype=float))
    return np.where(denominator != 0.0, numerator / denominator, np.nan)


def fit_scaler(matrix: np.ndarray, *, scaler: str) -> tuple[np.ndarray, np.ndarray]:
    """Return fitted center and scale vectors for a finite feature matrix."""
    if scaler == "standard":
        center = np.mean(matrix, axis=0)
        scale = np.std(matrix, axis=0)
    elif scaler == "robust":
        center = np.median(matrix, axis=0)
        q25 = np.percentile(matrix, 25, axis=0)
        q75 = np.percentile(matrix, 75, axis=0)
        scale = q75 - q25
    else:
        raise ValueError("scaler must be either 'standard' or 'robust'.")

    bad = (~np.isfinite(scale)) | (scale == 0.0)
    if bad.any():
        bad_idx = ", ".join(str(idx) for idx in np.flatnonzero(bad))
        raise ValueError(
            "Selected PCA features include zero-variance or zero-scale columns "
            f"after filtering; column indices: {bad_idx}."
        )
    return center.astype(float), scale.astype(float)


def make_pca_dataset(
    feature_table: xr.Dataset,
    *,
    feature_names: tuple[str, ...],
    valid_idx: np.ndarray,
    valid_event_mask: np.ndarray,
    matrix: np.ndarray,
    matrix_scaled: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    scores: np.ndarray,
    model: PCA,
) -> xr.Dataset:
    """Assemble the PCA output dataset."""
    event_coord = retained_event_coordinate(feature_table, valid_idx)
    pc_coord = np.asarray([f"PC{i}" for i in range(1, scores.shape[1] + 1)], dtype=object)
    feature_coord = np.asarray(feature_names, dtype=object)
    event_original_coord = np.asarray(feature_table[EVENT_DIM].values)

    out = xr.Dataset(
        coords={
            EVENT_DIM: event_coord,
            PC_DIM: pc_coord,
            FEATURE_DIM: feature_coord,
            ORIGINAL_EVENT_DIM: event_original_coord,
        }
    )
    out["pc_score"] = ((EVENT_DIM, PC_DIM), scores)
    out["pc_loading"] = ((PC_DIM, FEATURE_DIM), model.components_)
    out["explained_variance"] = (PC_DIM, model.explained_variance_)
    out["explained_variance_ratio"] = (PC_DIM, model.explained_variance_ratio_)
    out["cumulative_explained_variance_ratio"] = (
        PC_DIM,
        np.cumsum(model.explained_variance_ratio_),
    )
    out["feature_center"] = (FEATURE_DIM, center)
    out["feature_scale"] = (FEATURE_DIM, scale)
    out["feature_matrix"] = ((EVENT_DIM, FEATURE_DIM), matrix)
    out["feature_matrix_scaled"] = ((EVENT_DIM, FEATURE_DIM), matrix_scaled)
    out["valid_event_mask_original"] = (
        ORIGINAL_EVENT_DIM,
        valid_event_mask.astype(np.int8),
    )
    copy_event_variables(out, feature_table, valid_idx)
    copy_diagnostic_variables(out, feature_table, valid_idx)
    add_variable_attrs(out)
    return out


def retained_event_coordinate(feature_table: xr.Dataset, valid_idx: np.ndarray) -> np.ndarray:
    """Return event coordinate values for retained events, preferring event_id."""
    if "event_id" in feature_table:
        return np.asarray(feature_table["event_id"].isel({EVENT_DIM: valid_idx}).values)
    return np.asarray(feature_table[EVENT_DIM].isel({EVENT_DIM: valid_idx}).values)


def copy_event_variables(
    out: xr.Dataset,
    feature_table: xr.Dataset,
    valid_idx: np.ndarray,
) -> None:
    """Copy retained event metadata variables when present."""
    for name in EVENT_METADATA_VARIABLES:
        if name in feature_table:
            copy_1d_event_variable(out, feature_table, name, valid_idx)


def copy_diagnostic_variables(
    out: xr.Dataset,
    feature_table: xr.Dataset,
    valid_idx: np.ndarray,
) -> None:
    """Copy raw and derived diagnostic variables useful for interpreting PCs."""
    for name in DIAGNOSTIC_VARIABLES:
        if name in feature_table and name not in out:
            copy_1d_event_variable(out, feature_table, name, valid_idx)
    for name in DERIVED_DIAGNOSTIC_VARIABLES:
        required = source_variable_names(name)
        if all(source in feature_table for source in required) and name not in out:
            values = feature_values(feature_table, name)[valid_idx]
            out[name] = (EVENT_DIM, values)
            out[name].attrs.update(
                {
                    "source_variable": ",".join(required),
                    "operation": "derived_diagnostic",
                }
            )


def copy_1d_event_variable(
    out: xr.Dataset,
    feature_table: xr.Dataset,
    name: str,
    valid_idx: np.ndarray,
) -> None:
    """Copy a 1D event variable without coordinate alignment surprises."""
    da = feature_table[name]
    if da.dims != (EVENT_DIM,):
        return
    out[name] = (EVENT_DIM, np.asarray(da.isel({EVENT_DIM: valid_idx}).values))
    out[name].attrs = dict(da.attrs)


def add_variable_attrs(out: xr.Dataset) -> None:
    """Attach core PCA variable metadata."""
    out["pc_score"].attrs["description"] = (
        "Principal-component scores for each retained event."
    )
    out["pc_loading"].attrs["description"] = (
        "Feature loading vectors for standardized PCA input variables."
    )
    out["explained_variance"].attrs["description"] = (
        "Variance explained by each principal component."
    )
    out["explained_variance_ratio"].attrs["description"] = (
        "Fraction of standardized feature variance explained by each PC."
    )
    out["cumulative_explained_variance_ratio"].attrs["description"] = (
        "Cumulative fraction of standardized feature variance explained."
    )
    out["feature_center"].attrs["description"] = (
        "Centering value subtracted from each PCA input feature."
    )
    out["feature_scale"].attrs["description"] = (
        "Scaling value used for each PCA input feature."
    )
    out["feature_matrix"].attrs["description"] = (
        "Unscaled PCA input matrix after derived-variable calculation and event filtering."
    )
    out["feature_matrix_scaled"].attrs["description"] = (
        "Standardized PCA input matrix used to fit PCA."
    )
    out["valid_event_mask_original"].attrs.update(
        {
            "description": "Mask over original input events; 1 retained, 0 dropped.",
            "missing_event_policy": "drop_missing_events",
        }
    )


def add_global_attrs(
    out: xr.Dataset,
    *,
    input_path: str | Path | None,
    feature_names: Sequence[str],
    scaler: str,
    n_input_events: int,
    n_valid_events: int,
    n_components: int,
) -> None:
    """Attach PCA provenance and method metadata."""
    out.attrs.update(
        {
            "pipeline_stage": "stage_3_event_feature_pca",
            "input_path": "" if input_path is None else str(input_path),
            "source_feature_table": "" if input_path is None else str(input_path),
            "pca_features": ",".join(feature_names),
            "scaler": scaler,
            "pca_implementation": "sklearn.decomposition.PCA",
            "n_input_events": int(n_input_events),
            "n_valid_events": int(n_valid_events),
            "n_dropped_events": int(n_input_events - n_valid_events),
            "n_features": int(len(feature_names)),
            "n_components": int(n_components),
            "missing_event_policy": "drop_missing_events",
            "clustering_performed": 0,
        }
    )


def write_pca_output(pca_ds: xr.Dataset, output_path: str | Path) -> Path:
    """Write the PCA dataset to NetCDF."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    pca_ds.to_netcdf(path, engine="h5netcdf")
    return path


def unique_required_sources(feature_names: Sequence[str]) -> tuple[str, ...]:
    """Return source variable names required to resolve requested features."""
    names: list[str] = []
    for feature_name in feature_names:
        for source_name in source_variable_names(feature_name):
            if source_name not in names:
                names.append(source_name)
    return tuple(names)


def source_variable_name(feature_name: str) -> str:
    """Return the single source variable needed for a raw or simple derived feature."""
    source = DERIVED_FEATURE_SOURCES.get(feature_name, feature_name)
    if isinstance(source, tuple):
        raise ValueError(f"{feature_name!r} maps to multiple source variables.")
    return source


def source_variable_names(feature_name: str) -> tuple[str, ...]:
    """Return all source variables needed for a raw or derived feature."""
    source = DERIVED_FEATURE_SOURCES.get(feature_name, feature_name)
    if isinstance(source, tuple):
        return source
    return (source,)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
