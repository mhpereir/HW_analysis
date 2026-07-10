"""Plot Stage-4 event-feature relationships colored by cluster label."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib
import numpy as np
import xarray as xr
from matplotlib.colors import BoundaryNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src import plot_style

REGION = "pnw_bartusek"

DEFAULT_INPUT_DIR = REPO_ROOT / "results" / "stage4_event_feature_clusters" / REGION
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_DIR / "diagnostics" / REGION
DEFAULT_METHODS = ("ward", "kmeans", "gmm")
DEFAULT_PCS = ("PC1", "PC2", "PC3")
DEFAULT_N_CLUSTERS = 3

EVENT_DIM = "event"
FEATURE_DIM = "feature"
TRACKED_VARIABLE_DIM = "tracked_variable"
FEATURE_MATRIX_VARIABLE = "feature_matrix"
TRACKED_VALUE_VARIABLE = "tracked_variable_value"
CLUSTER_VARIABLE = "cluster_label"

X_VARIABLE = "I_dTdt_pre"
INTEGRATED_HEAT_BUDGET_VARIABLES = (
    "I_adiabatic_pre",
    "I_diabatic_pre",
    "I_advection_pre",
)
HEAT_BUDGET_Y_VARIABLES = (
    "f_adiabatic_pre",
    "f_diabatic_pre",
    "f_advection_pre",
)
HEAT_BUDGET_FRACTION_NUMERATORS = {
    "f_adiabatic_pre": "I_adiabatic_pre",
    "f_diabatic_pre": "I_diabatic_pre",
    "f_advection_pre": "I_advection_pre",
}
Y_VARIABLES = (
    *HEAT_BUDGET_Y_VARIABLES,
    "sqrt_I_lwa_a_pre_peak",
    "T_anom_mean_ant",
    "cos_days_from_solstice",
    "duration",
    "tas_anom_peak",
    "log10_tas_excess_integral",
)
VARIABLE_LABELS = {
    "I_dTdt_pre": "Integrated dT/dt (K)",
    "f_adiabatic_pre": "I_adiabatic / sum(|budget terms|)",
    "f_diabatic_pre": "I_diabatic / sum(|budget terms|)",
    "f_advection_pre": "I_advection / sum(|budget terms|)",
    "sqrt_I_lwa_a_pre_peak": "sqrt(integrated anticyclonic LWA exposure)",
    "T_anom_mean_ant": "Antecedent T anomaly (K)",
    "cos_days_from_solstice": "cos(days from solstice * 2pi / 365)",
    "duration": "Duration (days)",
    "tas_anom_peak": "Peak TAS anomaly (K)",
    "log10_tas_excess_integral": "log10(TAS excess integral)",
    CLUSTER_VARIABLE: "Cluster label",
}
PANEL_TITLES = {
    "f_adiabatic_pre": "Adiabatic Fraction",
    "f_diabatic_pre": "Diabatic Fraction",
    "f_advection_pre": "Advection Fraction",
    "sqrt_I_lwa_a_pre_peak": "Sqrt LWA Exposure",
    "T_anom_mean_ant": "Antecedent T Anomaly",
    "cos_days_from_solstice": "Season Phase",
    "duration": "Duration",
    "tas_anom_peak": "Peak TAS Anomaly",
    "log10_tas_excess_integral": "Log TAS Excess Integral",
}
DERIVED_VARIABLE_SOURCES = {
    "sqrt_I_lwa_a_pre_peak": "I_lwa_a_pre_peak",
    "cos_days_from_solstice": "days_from_solstice",
    "log10_tas_excess_integral": "tas_excess_integral",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for cluster-colored feature diagnostics."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot Stage-4 event-feature relationships colored by cluster label."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Stage-4 event-feature cluster NetCDF products.",
    )
    parser.add_argument(
        "--input-paths",
        nargs="+",
        type=Path,
        default=None,
        help="Explicit Stage-4 cluster products. Bypasses method glob resolution.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where scatter-plot PNG files will be written.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
        help="Clustering methods to plot when resolving products from --input-dir.",
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=DEFAULT_N_CLUSTERS,
        help="Cluster count used in input-product filename resolution.",
    )
    parser.add_argument(
        "--pcs",
        nargs="+",
        default=list(DEFAULT_PCS),
        help="Principal components used in input-product filename resolution.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=24.0,
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.75,
        help="Scatter marker opacity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0.")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must satisfy 0 < alpha <= 1.")
    if not args.methods:
        raise ValueError("At least one clustering method is required.")
    if args.n_clusters < 2:
        raise ValueError("--n-clusters must be >= 2.")
    if not args.pcs:
        raise ValueError("At least one PC is required.")


def main() -> int:
    """Load Stage-4 cluster products and write cluster-colored diagnostics."""
    args = parse_args()
    validate_args(args)

    input_paths = resolve_input_paths(
        args.input_dir,
        methods=tuple(args.methods),
        n_clusters=args.n_clusters,
        pcs=tuple(args.pcs),
        explicit_paths=args.input_paths,
    )

    written = []
    for input_path in input_paths:
        clusters = open_cluster_dataset(input_path)
        try:
            written.append(
                write_tendency_scatter_plot(
                    clusters,
                    output_path_for_input(input_path, args.output_dir),
                    point_size=args.point_size,
                    alpha=args.alpha,
                )
            )
            written.append(
                write_tendency_scatter_plot(
                    clusters,
                    output_path_for_input(
                        input_path,
                        args.output_dir,
                        standardized=True,
                    ),
                    point_size=args.point_size,
                    alpha=args.alpha,
                    standardized=True,
                )
            )
        finally:
            clusters.close()

    print("Wrote cluster-colored event-feature tendency scatter figures:")
    for path in written:
        print(f"  {_display_path(path)}")
    return 0


def resolve_input_paths(
    input_dir: str | Path,
    *,
    methods: tuple[str, ...],
    n_clusters: int,
    pcs: tuple[str, ...],
    explicit_paths: list[Path] | None = None,
) -> tuple[Path, ...]:
    """Return Stage-4 cluster product paths to plot."""
    if explicit_paths is not None:
        return tuple(path.expanduser().resolve() for path in explicit_paths)

    directory = Path(input_dir).expanduser().resolve()
    paths = []
    for method in methods:
        pattern = cluster_product_pattern(
            method=method,
            n_clusters=n_clusters,
            pcs=pcs,
        )
        matches = sorted(directory.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                "No Stage-4 cluster product found for "
                f"method {method!r} with pattern {pattern!r} in {directory}."
            )
        if len(matches) > 1:
            display_matches = ", ".join(str(path) for path in matches)
            raise ValueError(
                "Multiple Stage-4 cluster products found for "
                f"method {method!r} with pattern {pattern!r}: {display_matches}."
            )
        paths.append(matches[0].resolve())
    return tuple(paths)


def cluster_product_pattern(
    *,
    method: str,
    n_clusters: int,
    pcs: tuple[str, ...],
) -> str:
    """Return the Stage-4 filename pattern for one cluster method."""
    pc_slug = "-".join(pc.lower() for pc in pcs)
    return f"*_{method}_k{n_clusters}_{pc_slug}.nc"


def open_cluster_dataset(path: str | Path) -> xr.Dataset:
    """Open a Stage-4 event-feature cluster NetCDF product."""
    input_path = Path(path).expanduser().resolve()
    try:
        return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)
    except TypeError as exc:
        if "decode_timedelta" not in str(exc):
            raise
        return xr.open_dataset(input_path, engine="h5netcdf")


def write_tendency_scatter_plot(
    clusters: xr.Dataset,
    output_path: str | Path,
    *,
    point_size: float = 24.0,
    alpha: float = 0.75,
    standardized: bool = False,
) -> Path:
    """Write the cluster-colored integrated tendency scatter figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tendency_scatter(
        clusters,
        point_size=point_size,
        alpha=alpha,
        standardized=standardized,
    )
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_tendency_scatter(
    clusters: xr.Dataset,
    *,
    point_size: float = 24.0,
    alpha: float = 0.75,
    standardized: bool = False,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return a multi-panel feature scatter figure colored by cluster label."""
    validate_feature_variables(clusters)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=plot_style.publication_figsize("full", aspect=1.0),
        sharex=True,
        constrained_layout=True,
    )
    cluster_values = feature_values(clusters, CLUSTER_VARIABLE)
    cluster_labels, cmap, norm = cluster_color_mapping(cluster_values)

    mappable = None
    for ax, y_variable in zip(np.ravel(axes), Y_VARIABLES):
        mappable = plot_one_tendency_panel(
            ax,
            clusters,
            y_variable,
            cluster_values=cluster_values,
            cluster_cmap=cmap,
            cluster_norm=norm,
            point_size=point_size,
            alpha=alpha,
            standardized=standardized,
        )

    if mappable is not None:
        cbar = fig.colorbar(
            mappable,
            ax=np.ravel(axes),
            ticks=cluster_labels,
            shrink=0.86,
        )
        cbar.set_label(variable_label(CLUSTER_VARIABLE))
        cbar.ax.set_yticklabels([str(label) for label in cluster_labels])

    method = clusters.attrs.get("cluster_method")
    title = "Event Fixed-Window Heat-Budget Feature Relationships"
    if method:
        title = f"{title} ({method} clusters)"
    if standardized:
        title = f"{title} (Standardized)"
    fig.suptitle(title)
    return fig


def plot_one_tendency_panel(
    ax: Axes,
    clusters: xr.Dataset,
    y_variable: str,
    *,
    cluster_values: np.ndarray,
    cluster_cmap,
    cluster_norm: BoundaryNorm,
    point_size: float,
    alpha: float,
    standardized: bool,
):
    """Plot one y-variable against integrated temperature tendency."""
    x_values = feature_values(clusters, X_VARIABLE)
    y_values = feature_values(clusters, y_variable)
    if standardized:
        x_values = standardized_values(x_values)
        y_values = standardized_values(y_values)
    finite = (
        np.isfinite(x_values)
        & np.isfinite(y_values)
        & np.isfinite(cluster_values)
    )

    mappable = ax.scatter(
        x_values[finite],
        y_values[finite],
        c=cluster_values[finite],
        cmap=cluster_cmap,
        norm=cluster_norm,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )

    add_zero_reference_lines(ax)
    if y_variable in INTEGRATED_HEAT_BUDGET_VARIABLES:
        add_one_to_one_line(ax, x_values[finite], y_values[finite])
    ax.set_title(PANEL_TITLES[y_variable])
    ax.set_xlabel(variable_label(X_VARIABLE, standardized=standardized))
    ax.set_ylabel(variable_label(y_variable, standardized=standardized))
    ax.text(
        0.03,
        0.97,
        f"n = {int(finite.sum())}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    plot_style.style_axis(ax)
    return mappable


def validate_feature_variables(clusters: xr.Dataset) -> None:
    """Fail clearly when the cluster product lacks required plot variables."""
    required = [X_VARIABLE, *Y_VARIABLES, CLUSTER_VARIABLE]
    missing = [name for name in required if not can_resolve_feature(clusters, name)]
    if missing:
        raise ValueError(
            "Stage-4 cluster product is missing required variables: "
            f"{', '.join(missing)}."
        )


def can_resolve_feature(clusters: xr.Dataset, variable: str) -> bool:
    """Return True when a feature can be resolved to one event vector."""
    try:
        feature_values(clusters, variable)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def feature_values(clusters: xr.Dataset, variable: str | None) -> np.ndarray:
    """Return one event-level variable as a float array."""
    if variable is None:
        raise ValueError("variable must not be None.")

    stored = stored_feature_values(clusters, variable)
    if stored is not None:
        return stored

    if variable in HEAT_BUDGET_Y_VARIABLES:
        return heat_budget_fraction_values(clusters, variable)

    if variable in DERIVED_VARIABLE_SOURCES:
        source_values = feature_values(clusters, DERIVED_VARIABLE_SOURCES[variable])
        return transform_derived_values(variable, source_values)

    raise ValueError(f"Stage-4 cluster product does not contain variable {variable!r}.")


def stored_feature_values(clusters: xr.Dataset, variable: str) -> np.ndarray | None:
    """Return a stored event vector from direct, tracked, or feature-matrix storage."""
    if variable in clusters:
        da = clusters[variable]
        if da.dims == (EVENT_DIM,):
            return data_array_to_float_vector(da)

    if has_tracked_variable(clusters, variable):
        da = clusters[TRACKED_VALUE_VARIABLE].sel({TRACKED_VARIABLE_DIM: variable})
        return data_array_to_float_vector(da)

    if has_feature_matrix_variable(clusters, variable):
        da = clusters[FEATURE_MATRIX_VARIABLE].sel({FEATURE_DIM: variable})
        return data_array_to_float_vector(da)

    return None


def has_tracked_variable(clusters: xr.Dataset, variable: str) -> bool:
    """Return True when a variable is available in tracked_variable_value."""
    if TRACKED_VALUE_VARIABLE not in clusters:
        return False
    if TRACKED_VARIABLE_DIM not in clusters.coords:
        return False
    if TRACKED_VARIABLE_DIM not in clusters[TRACKED_VALUE_VARIABLE].dims:
        return False
    return variable in {str(value) for value in clusters[TRACKED_VARIABLE_DIM].values}


def has_feature_matrix_variable(clusters: xr.Dataset, variable: str) -> bool:
    """Return True when a variable is available as a feature_matrix row."""
    if FEATURE_MATRIX_VARIABLE not in clusters:
        return False
    if FEATURE_DIM not in clusters.coords:
        return False
    if FEATURE_DIM not in clusters[FEATURE_MATRIX_VARIABLE].dims:
        return False
    return variable in {str(value) for value in clusters[FEATURE_DIM].values}


def data_array_to_float_vector(da: xr.DataArray) -> np.ndarray:
    """Return a one-dimensional event DataArray as floats."""
    if EVENT_DIM not in da.dims or len(da.dims) != 1:
        raise ValueError(f"Expected a one-dimensional {EVENT_DIM!r} variable.")
    values = da.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return values / np.timedelta64(1, "D")
    return np.asarray(values, dtype=float)


def heat_budget_fraction_values(clusters: xr.Dataset, variable: str) -> np.ndarray:
    """Return one heat-budget term as a fraction of summed absolute budget terms."""
    numerator = feature_values(clusters, HEAT_BUDGET_FRACTION_NUMERATORS[variable])
    denominator = np.zeros_like(numerator, dtype=float)
    for source_name in INTEGRATED_HEAT_BUDGET_VARIABLES:
        denominator = denominator + np.abs(feature_values(clusters, source_name))

    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0.0)
    return out


def transform_derived_values(variable: str, values: np.ndarray) -> np.ndarray:
    """Apply the derived-variable transform to source event values."""
    if variable == "sqrt_I_lwa_a_pre_peak":
        return np.where(values >= 0.0, np.sqrt(values), np.nan)
    if variable == "cos_days_from_solstice":
        return np.cos(values * 2.0 * np.pi / 365.0)
    if variable == "log10_tas_excess_integral":
        return np.where(values > 0.0, np.log10(values), np.nan)
    raise ValueError(f"Unknown derived variable: {variable}")


def standardized_values(values: np.ndarray) -> np.ndarray:
    """Return z-scored values using finite-sample mean and standard deviation."""
    out = np.asarray(values, dtype=float)
    finite = np.isfinite(out)
    if not finite.any():
        return np.full(out.shape, np.nan, dtype=float)
    mean = float(np.nanmean(out[finite]))
    std = float(np.nanstd(out[finite]))
    if std == 0.0 or not np.isfinite(std):
        return np.full(out.shape, np.nan, dtype=float)
    return (out - mean) / std


def cluster_color_mapping(cluster_values: np.ndarray):
    """Return labels, cmap, and norm for categorical cluster-label coloring."""
    finite_values = np.asarray(cluster_values, dtype=float)
    finite = np.isfinite(finite_values)
    if not finite.any():
        raise ValueError("No finite cluster labels are available for plotting.")

    labels = np.array(sorted({int(value) for value in finite_values[finite]}))
    cmap = plt.get_cmap("tab20", labels.size)
    boundaries = np.concatenate(
        [
            labels.astype(float) - 0.5,
            np.array([float(labels[-1]) + 0.5]),
        ]
    )
    norm = BoundaryNorm(boundaries, cmap.N)
    return labels, cmap, norm


def variable_label(variable: str, *, standardized: bool = False) -> str:
    """Return a readable axis or colorbar label."""
    label = VARIABLE_LABELS.get(variable, variable)
    if standardized:
        return f"standardized {label}"
    return label


def add_zero_reference_lines(ax: Axes) -> None:
    """Add horizontal and vertical zero lines."""
    ax.axhline(
        0.0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        linestyle="--",
        zorder=0,
    )
    ax.axvline(
        0.0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        linestyle="--",
        zorder=0,
    )


def add_one_to_one_line(ax: Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    """Add a y=x reference line over the finite data extent."""
    if x_values.size == 0 or y_values.size == 0:
        return
    lower = float(np.nanmin([np.nanmin(x_values), np.nanmin(y_values)]))
    upper = float(np.nanmax([np.nanmax(x_values), np.nanmax(y_values)]))
    ax.plot(
        [lower, upper],
        [lower, upper],
        color=plot_style.COLORS["benchmark"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        linestyle=":",
        zorder=0,
        label="1:1",
    )


def output_path_for_input(
    input_path: str | Path,
    output_dir: str | Path,
    *,
    standardized: bool = False,
) -> Path:
    """Return the figure output path for one Stage-4 cluster product."""
    suffix = (
        "_tendency_scatter_standardized.png"
        if standardized
        else "_tendency_scatter.png"
    )
    return Path(output_dir).expanduser().resolve() / f"{Path(input_path).stem}{suffix}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
