"""Plot adiabatic and advective pre-peak tendencies against adiabatic tendency.

The diagnostic loads a Stage-2 event-feature table and writes a multi-panel
scatter figure. Each panel uses integrated adiabatic tendency on the x-axis,
and points are colored by peak temperature anomaly by default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib
import numpy as np
import xarray as xr
from matplotlib.colors import Normalize

try:
    from matplotlib.colors import TwoSlopeNorm
except ImportError:  # pragma: no cover - compatibility with older Matplotlib.
    TwoSlopeNorm = None  # type: ignore[assignment]

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src import plot_style

REGION = "pnw_bartusek"
THRESHOLD_VARIABLE = "lwa_a"
QUANTILE_THRESHOLD = "q90"

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / f"hw_event_features_fixed_windows_{REGION}_{THRESHOLD_VARIABLE}_{QUANTILE_THRESHOLD}_1940_2024.nc"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / "diagnostics"
    / REGION
    / THRESHOLD_VARIABLE
    / "diabatic_advection_vs_adiabatic_scatter.png"
)

X_VARIABLE = "I_adiabatic_pre"
COLOR_VARIABLE = "tas_anom_peak"
COLOR_MAP = "gist_heat_r"
Y_VARIABLES = (
    "I_diabatic_pre",
    "I_advection_pre",
    "sqrt_lwa_a_peak",
)
VARIABLE_LABELS = {
    "tas_peak": "Peak TAS (K)",
    "tas_anom_peak": "Peak TAS Anomaly (K)",
    "lwa_a_peak": "LWA a ([hPa m])",
    "sqrt_lwa_a_peak": "sqrt(LWA_a [hPa m])",
    "I_diabatic_pre": "I_diabatic (K)",
    "I_advection_pre": "I_advective (K)",
    "I_adiabatic_pre": "I_adiabatic (K)",
    "I_lwa_a_pre_peak": "I_LWA_a",
    "sqrt_I_lwa_a_pre_peak": "sqrt(I_LWA_a)",
}
PANEL_TITLES = {
    "I_diabatic_pre": "Diabatic vs Adiabatic",
    "I_advection_pre": "Advection vs Adiabatic",
    "sqrt_I_lwa_a_pre_peak": "Sqrt LWA a Exposure",
    "sqrt_lwa_a_peak": "Sqrt LWA a Peak",
}
DERIVED_VARIABLE_SOURCES = {
    "sqrt_I_lwa_a_pre_peak": "I_lwa_a_pre_peak",
    "sqrt_lwa_a_peak": "lwa_a_peak",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the adiabatic/advection diagnostic."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot integrated adiabatic and advective tendencies against "
            "integrated adiabatic tendency from an event-feature table."
        )
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
        help="Path where the scatter-plot PNG will be written.",
    )
    parser.add_argument(
        "--color-variable",
        type=str,
        default=COLOR_VARIABLE,
        help="Event-level feature used to color points.",
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


def main() -> int:
    """Load event features and write the adiabatic/advection scatter diagnostic."""
    args = parse_args()
    validate_args(args)

    features = open_event_features(args.input_path)
    try:
        written = write_tendency_scatter_plot(
            features,
            args.output_path,
            color_variable=args.color_variable,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        print("Wrote adiabatic/advection tendency scatter figure:")
        print(f"  {_display_path(written)}")
    finally:
        features.close()
    return 0


def open_event_features(path: str | Path) -> xr.Dataset:
    """Open an event-feature NetCDF table."""
    input_path = Path(path).expanduser().resolve()
    return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)


def write_tendency_scatter_plot(
    features: xr.Dataset,
    output_path: str | Path,
    *,
    color_variable: str | None = COLOR_VARIABLE,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> Path:
    """Write the adiabatic/advection/LWA scatter figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tendency_scatter(
        features,
        color_variable=color_variable,
        point_size=point_size,
        alpha=alpha,
    )
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_tendency_scatter(
    features: xr.Dataset,
    *,
    color_variable: str | None = COLOR_VARIABLE,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return an adiabatic/advection/LWA scatter figure."""
    validate_feature_variables(features, color_variable=color_variable)

    fig, axes = plt.subplots(
        nrows=len(Y_VARIABLES),
        ncols=1,
        figsize=plot_style.publication_figsize("single", aspect=1.6),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    color_values = feature_values(features, color_variable) if color_variable else None
    # color_norm = color_norm_for_values(color_values)

    color_norm = None

    mappable = None
    for index, (ax, y_variable) in enumerate(zip(axes, Y_VARIABLES)):
        mappable = plot_one_tendency_panel(
            ax,
            features,
            y_variable,
            color_values=color_values,
            color_norm=color_norm,
            color_variable=color_variable,
            point_size=point_size,
            alpha=alpha,
            show_xlabel=index == len(Y_VARIABLES) - 1,
        )
    set_shared_x_data_limits(axes, feature_values(features, X_VARIABLE))

    if color_variable is not None and mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.92)
        cbar.set_label(variable_label(color_variable))

    fig.suptitle(
        "Adiabatic and Advective Tendencies vs Adiabatic Tendency",
    )
    return fig


def plot_one_tendency_panel(
    ax: Axes,
    features: xr.Dataset,
    y_variable: str,
    *,
    color_values: np.ndarray | None,
    color_norm: Normalize | None,
    color_variable: str | None,
    point_size: float,
    alpha: float,
    show_xlabel: bool,
):
    """Plot one tendency variable against integrated adiabatic tendency."""
    x_values = feature_values(features, X_VARIABLE)
    y_values = feature_values(features, y_variable)
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if color_values is not None:
        finite &= np.isfinite(color_values)

    kwargs = {
        "s": point_size,
        "alpha": alpha,
        "edgecolors": "none",
    }
    if color_values is None:
        kwargs["color"] = plot_style.COLORS["volume"]
        mappable = ax.scatter(x_values[finite], y_values[finite], **kwargs)
    else:
        mappable = ax.scatter(
            x_values[finite],
            y_values[finite],
            c=color_values[finite],
            cmap=COLOR_MAP,
            norm=color_norm,
            **kwargs,
        )

    add_zero_reference_lines(ax)
    add_one_to_one_line(ax, x_values[finite], y_values[finite])
    if y_variable == "I_advection_pre":
        add_one_to_negative_one_line(ax, x_values[finite], y_values[finite])
    ax.set_title(PANEL_TITLES[y_variable])
    ax.set_ylabel(variable_label(y_variable))
    if show_xlabel:
        ax.set_xlabel(variable_label(X_VARIABLE))
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
    if color_variable is None:
        return None
    return mappable


def validate_feature_variables(
    features: xr.Dataset,
    *,
    color_variable: str | None,
) -> None:
    """Fail clearly when the requested feature table lacks required variables."""
    required = [X_VARIABLE]
    for name in Y_VARIABLES:
        required.extend(source_variable_names(name))
    if color_variable is not None:
        required.extend(source_variable_names(color_variable))
    missing = [name for name in dict.fromkeys(required) if name not in features]
    if missing:
        raise ValueError(
            "Event-feature table is missing required variables: "
            f"{', '.join(missing)}."
        )


def feature_values(features: xr.Dataset, variable: str | None) -> np.ndarray:
    """Return a feature variable as a float array."""
    if variable is None:
        raise ValueError("variable must not be None.")
    values = features[source_variable_name(variable)].values
    if np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, "D")
    else:
        out = np.asarray(values, dtype=float)
    if variable in {"sqrt_I_lwa_a_pre_peak", "sqrt_lwa_a_peak"}:
        transformed = np.full(out.shape, np.nan, dtype=float)
        valid = out >= 0.0
        transformed[valid] = np.sqrt(out[valid])
        out = transformed
    return np.asarray(out, dtype=float)


def source_variable_name(variable: str) -> str:
    """Return the dataset variable needed for a plot variable."""
    return DERIVED_VARIABLE_SOURCES.get(variable, variable)


def source_variable_names(variable: str) -> tuple[str, ...]:
    """Return all dataset variables needed for a plot variable."""
    return (source_variable_name(variable),)


def color_norm_for_values(values: np.ndarray | None) -> Normalize | None:
    """Return a shared color normalization centered on zero."""
    if values is None:
        return None
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None
    max_abs = float(np.nanmax(np.abs(finite)))
    if max_abs == 0.0:
        max_abs = 1.0
    if TwoSlopeNorm is None:
        return Normalize(vmin=-max_abs, vmax=max_abs)
    return TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)


def set_shared_x_data_limits(axes: np.ndarray, x_values: np.ndarray) -> None:
    """Limit shared x-axes to the finite extent of the x-data."""
    finite = np.asarray(x_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return
    xmin = float(np.nanmin(finite))
    xmax = float(np.nanmax(finite))
    if xmin == xmax:
        padding = 0.5 if xmin == 0.0 else abs(xmin) * 0.05
        xmin -= padding
        xmax += padding
    for ax in axes:
        ax.set_xlim(xmin, xmax)


def variable_label(variable: str) -> str:
    """Return a readable axis or colorbar label."""
    return VARIABLE_LABELS.get(variable, variable)


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
        gid="one_to_one",
    )


def add_one_to_negative_one_line(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """Add a y=-x reference line over the finite data extent."""
    if x_values.size == 0 or y_values.size == 0:
        return
    lower = float(np.nanmin([np.nanmin(x_values), np.nanmin(-y_values)]))
    upper = float(np.nanmax([np.nanmax(x_values), np.nanmax(-y_values)]))
    ax.plot(
        [lower, upper],
        [-lower, -upper],
        color=plot_style.COLORS["benchmark"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        linestyle=":",
        zorder=0,
        label="1:-1",
        gid="one_to_negative_one",
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
