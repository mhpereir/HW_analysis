"""Explore PCA products before clustering.

This script loads the PCA NetCDF product produced by ``build_event_feature_pca.py``
and writes a small set of diagnostic figures:

- explained-variance scree plot,
- PC loading heatmap,
- PC score scatter plots colored by event diagnostics.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "event_features"
    / "hw_event_feature_pca_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "event_features" / "pca_clustering"
DEFAULT_COLOR_VARIABLES = (
    "tas_anom_peak",
    "log10_tas_excess_integral",
    "duration",
    "I_dTdt_pre",
)
DEFAULT_CORRELATION_DIAGNOSTICS = (
    "tas_anom_peak",
    "tas_excess_integral",
    "duration",
    "I_dTdt_pre",
    "I_advection_pre",
    "I_diabatic_pre",
    "I_adiabatic_pre"
)
DEFAULT_CORRELATION_PCS = ("PC1", "PC2", "PC3", "PC4")
DEFAULT_PC_X = "PC1"
DEFAULT_PC_Y = "PC2"

VARIABLE_LABELS = {
    "tas_anom_peak": "Peak TAS anomaly (K)",
    "log10_tas_excess_integral": "log10(TAS excess integral)",
    "duration": "Duration (days)",
    "I_dTdt_pre": "Integrated dT/dt (K)",
    "I_diabatic_pre": "Diabatic Heating (K)",
    "tas_excess_integral": "TAS excess integral",
    "tas_peak": "Peak TAS",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for PCA diagnostic plotting."""
    parser = argparse.ArgumentParser(
        description="Plot diagnostic figures from an event-feature PCA NetCDF product."
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
        help="Directory where diagnostic PNG files will be written.",
    )
    parser.add_argument(
        "--pc-x",
        default=DEFAULT_PC_X,
        help="Principal component for the scatter x-axis, e.g. PC1.",
    )
    parser.add_argument(
        "--pc-y",
        default=DEFAULT_PC_Y,
        help="Principal component for the scatter y-axis, e.g. PC2.",
    )
    parser.add_argument(
        "--color-variables",
        nargs="+",
        default=list(DEFAULT_COLOR_VARIABLES),
        help="Event-level variables used to color PC score scatter panels.",
    )
    parser.add_argument(
        "--max-loadings-pcs",
        type=int,
        default=6,
        help="Maximum number of PCs to show in the loading heatmap.",
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
        default=0.78,
        help="Scatter marker opacity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI options."""
    if args.max_loadings_pcs < 1:
        raise ValueError("--max-loadings-pcs must be >= 1.")
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0.")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must satisfy 0 < alpha <= 1.")


def main() -> int:
    """Open the PCA dataset and write exploratory diagnostic figures."""
    args = parse_args()
    validate_args(args)

    pca = open_pca_dataset(args.input_path)
    try:
        print_pc1_diagnostic_correlations(pca)
        written = write_pca_diagnostic_plots(
            pca,
            args.output_dir,
            pc_x=args.pc_x,
            pc_y=args.pc_y,
            color_variables=args.color_variables,
            max_loadings_pcs=args.max_loadings_pcs,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        print("Wrote PCA diagnostic figures:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        pca.close()
    return 0


def open_pca_dataset(path: str | Path) -> xr.Dataset:
    """Open a PCA NetCDF product."""
    input_path = Path(path).expanduser().resolve()
    return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)


def write_pca_diagnostic_plots(
    pca: xr.Dataset,
    output_dir: str | Path,
    *,
    pc_x: str = DEFAULT_PC_X,
    pc_y: str = DEFAULT_PC_Y,
    color_variables: tuple[str, ...] | list[str] = DEFAULT_COLOR_VARIABLES,
    max_loadings_pcs: int = 6,
    point_size: float = 24.0,
    alpha: float = 0.78,
) -> list[Path]:
    """Write the default set of PCA diagnostic plots."""
    validate_pca_dataset(pca, pc_x=pc_x, pc_y=pc_y, color_variables=color_variables)
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    figure_specs = [
        (plot_explained_variance, output_path / "pca_explained_variance.png", {}),
        (
            plot_loading_heatmap,
            output_path / "pca_loading_heatmap.png",
            {"max_pcs": max_loadings_pcs},
        ),
        (
            plot_pc_score_diagnostics,
            output_path / f"pca_scores_{pc_x.lower()}_{pc_y.lower()}.png",
            {
                "pc_x": pc_x,
                "pc_y": pc_y,
                "color_variables": tuple(color_variables),
                "point_size": point_size,
                "alpha": alpha,
            },
        ),
    ]
    for plotter, path, kwargs in figure_specs:
        fig = plotter(pca, **kwargs)
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def validate_pca_dataset(
    pca: xr.Dataset,
    *,
    pc_x: str,
    pc_y: str,
    color_variables: tuple[str, ...] | list[str],
) -> None:
    """Validate that the PCA dataset supports the requested diagnostic plots."""
    required = {
        "pc_score",
        "pc_loading",
        "explained_variance_ratio",
        "cumulative_explained_variance_ratio",
    }
    missing = sorted(name for name in required if name not in pca)
    if missing:
        raise ValueError(f"PCA dataset is missing required variables: {', '.join(missing)}.")

    pcs = set(str(value) for value in pca["pc"].values)
    for pc_name in (pc_x, pc_y):
        if pc_name not in pcs:
            raise ValueError(f"Requested PC {pc_name!r} is not present in the dataset.")

    missing_colors = [name for name in color_variables if not has_event_variable(pca, name)]
    if missing_colors:
        raise ValueError(
            "PCA dataset is missing requested color variables: "
            f"{', '.join(missing_colors)}."
        )


def plot_explained_variance(pca: xr.Dataset) -> plt.Figure:  # type: ignore[type-arg]
    """Return a scree plot with individual and cumulative explained variance."""
    pc_labels = [str(value) for value in pca["pc"].values]
    x = np.arange(len(pc_labels))
    ratio = np.asarray(pca["explained_variance_ratio"].values, dtype=float)
    cumulative = np.asarray(
        pca["cumulative_explained_variance_ratio"].values,
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.bar(x, ratio, color="tab:blue", alpha=0.78, label="Individual")
    ax.plot(x, cumulative, color="tab:red", marker="o", linewidth=1.8, label="Cumulative")
    ax.set_xticks(x)
    ax.set_xticklabels(pc_labels, rotation=45, ha="right")
    ax.set_ylim(0.0, min(1.05, max(1.0, float(np.nanmax(cumulative)) + 0.05)))
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA Explained Variance")
    ax.grid(True, axis="y", color="0.88", linewidth=0.8)
    ax.legend(frameon=False)
    return fig


def plot_loading_heatmap(
    pca: xr.Dataset,
    *,
    max_pcs: int = 6,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return a heatmap of PC loadings by input feature."""
    n_pcs = min(max_pcs, pca.sizes["pc"])
    loadings = np.asarray(pca["pc_loading"].isel(pc=slice(0, n_pcs)).values, dtype=float)
    pc_labels = [str(value) for value in pca["pc"].isel(pc=slice(0, n_pcs)).values]
    feature_labels = [str(value) for value in pca["feature"].values]
    vmax = max(0.1, float(np.nanmax(np.abs(loadings))))

    fig_width = max(8.0, 0.75 * len(feature_labels))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), constrained_layout=True)
    image = ax.imshow(loadings, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(feature_labels)))
    ax.set_xticklabels(feature_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(pc_labels)))
    ax.set_yticklabels(pc_labels)
    ax.set_xlabel("PCA input feature")
    ax.set_ylabel("Principal component")
    ax.set_title("PCA Loading Vectors")
    cbar = fig.colorbar(image, ax=ax, shrink=0.86)
    cbar.set_label("Loading")

    for row in range(loadings.shape[0]):
        for col in range(loadings.shape[1]):
            ax.text(
                col,
                row,
                f"{loadings[row, col]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black" if abs(loadings[row, col]) < 0.55 * vmax else "white",
            )
    return fig


def plot_pc_score_diagnostics(
    pca: xr.Dataset,
    *,
    pc_x: str,
    pc_y: str,
    color_variables: tuple[str, ...],
    point_size: float = 24.0,
    alpha: float = 0.78,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return PC-score scatter panels colored by selected event diagnostics."""
    n_panels = len(color_variables)
    ncols = 2 if n_panels > 1 else 1
    nrows = math.ceil(n_panels / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(6.4 * ncols, 5.3 * nrows),
        squeeze=False,
        constrained_layout=True,
    )
    x_values = pc_score_values(pca, pc_x)
    y_values = pc_score_values(pca, pc_y)

    for ax, color_variable in zip(axes.ravel(), color_variables):
        plot_one_score_panel(
            ax,
            pca,
            x_values,
            y_values,
            pc_x=pc_x,
            pc_y=pc_y,
            color_variable=color_variable,
            point_size=point_size,
            alpha=alpha,
            fig=fig,
        )

    for ax in axes.ravel()[n_panels:]:
        ax.axis("off")

    fig.suptitle(f"PCA Scores: {pc_x} vs {pc_y}", fontsize=13)
    return fig


def plot_one_score_panel(
    ax: Axes,
    pca: xr.Dataset,
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    pc_x: str,
    pc_y: str,
    color_variable: str,
    point_size: float,
    alpha: float,
    fig: plt.Figure,  # type: ignore[type-arg]
) -> None:
    """Plot one PC-score scatter panel."""
    color_values = event_variable_values(pca, color_variable)
    finite = np.isfinite(x_values) & np.isfinite(y_values) & np.isfinite(color_values)
    scatter = ax.scatter(
        x_values[finite],
        y_values[finite],
        c=color_values[finite],
        cmap="viridis",
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )
    ax.axhline(0.0, color="0.62", linewidth=0.9, linestyle="--", zorder=0)
    ax.axvline(0.0, color="0.62", linewidth=0.9, linestyle="--", zorder=0)
    ax.set_xlabel(pc_axis_label(pca, pc_x))
    ax.set_ylabel(pc_axis_label(pca, pc_y))
    ax.set_title(variable_label(color_variable))
    ax.grid(True, color="0.88", linewidth=0.8)
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
    cbar = fig.colorbar(scatter, ax=ax, shrink=0.86)
    cbar.set_label(variable_label(color_variable))


def pc_score_values(pca: xr.Dataset, pc_name: str) -> np.ndarray:
    """Return PC score values for one component."""
    return np.asarray(pca["pc_score"].sel(pc=pc_name).values, dtype=float)


def event_variable_values(pca: xr.Dataset, variable: str) -> np.ndarray:
    """Return one event-level diagnostic variable as float values."""
    if variable in pca:
        values = pca[variable].values
        if np.issubdtype(values.dtype, np.timedelta64):
            return values / np.timedelta64(1, "D")
        return np.asarray(values, dtype=float)

    if "feature" in pca.coords and variable in set(str(value) for value in pca["feature"].values):
        return np.asarray(pca["feature_matrix"].sel(feature=variable).values, dtype=float)

    raise ValueError(f"PCA dataset does not contain diagnostic variable {variable!r}.")


def has_event_variable(pca: xr.Dataset, variable: str) -> bool:
    """Return True when a diagnostic can be read directly or from feature_matrix."""
    if variable in pca:
        return True
    if "feature" not in pca.coords or "feature_matrix" not in pca:
        return False
    return variable in set(str(value) for value in pca["feature"].values)


def print_pc1_diagnostic_correlations(
    pca: xr.Dataset,
    diagnostics: tuple[str, ...] = DEFAULT_CORRELATION_DIAGNOSTICS,
    pc_names: tuple[str, ...] = DEFAULT_CORRELATION_PCS,
) -> None:
    """Print Pearson correlations between selected PC scores and diagnostics."""
    print("PC diagnostic correlations:")
    for pc_name in pc_names:
        pc_scores = pc_score_values(pca, pc_name)
        print(f"{pc_name}:")
        for name in diagnostics:
            values = event_variable_values(pca, name)
            r = finite_correlation(pc_scores, values)
            print(f"  {name} {r:.6f}")


def finite_correlation(x_values: np.ndarray, y_values: np.ndarray) -> float:
    """Return Pearson correlation over finite paired values."""
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if finite.sum() < 2:
        return np.nan
    return float(np.corrcoef(x_values[finite], y_values[finite])[0, 1])


def pc_axis_label(pca: xr.Dataset, pc_name: str) -> str:
    """Return a PC axis label with explained variance percentage when available."""
    ratio = float(pca["explained_variance_ratio"].sel(pc=pc_name).values)
    return f"{pc_name} ({100.0 * ratio:.1f}% variance)"


def variable_label(variable: str) -> str:
    """Return a human-readable label for a PCA event diagnostic."""
    return VARIABLE_LABELS.get(variable, variable)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
