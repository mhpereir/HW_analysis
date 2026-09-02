"""Compare advection with adiabatic heating and net dynamical contribution.

The diagnostic loads a Stage-2 event-feature table and writes either the full
2x2 scatter figure or a presentation-oriented 2x1 subset. Points are colored
by peak temperature anomaly by default.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, TwoSlopeNorm

from src import plot_style

REGION = "pnw_hotz"
THRESHOLD_VARIABLE = "tas"
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
    / "adiabatic_advection_distance_comparison.png"
)
DEFAULT_PRESENTATION_OUTPUT_PATH = DEFAULT_OUTPUT_PATH.with_name(
    "adiabatic_advection_distance_comparison_presentation.png"
)

X_VARIABLE = "I_adiabatic_pre"
ADVECTION_VARIABLE = "I_advection_pre"
DYNAMICAL_VARIABLE = "I_dyn_pre"
TEMPERATURE_CHANGE_VARIABLE = "I_dTdt_pre"
DIABATIC_VARIABLE = "I_diabatic_pre"
COLOR_VARIABLE = plot_style.EVENT_SEVERITY_VARIABLE
COLOR_MAP = plot_style.EVENT_SEVERITY_COLOR_MAP
# Set to False to use Matplotlib's default color normalization.
USE_CENTERED_COLOR_NORMALIZATION = False
COLOR_NORMALIZATION_CENTER = 3.0
FULL_LAYOUT = "full"
PRESENTATION_LAYOUT = "presentation"
LAYOUT_CHOICES = (FULL_LAYOUT, PRESENTATION_LAYOUT)
NET_DYNAMICAL_LABEL = r"$I_{dyn,net}$ (K)"
VARIABLE_LABELS = {
    "tas_anom_peak": plot_style.EVENT_SEVERITY_LABEL,
    "tas_peak": "Peak TAS (K)",
    "lwa_a_peak": "Peak LWA A [hPa m]",
    "I_advection_pre": r"$I_{advective}$(K)",
    "I_adiabatic_pre": r"$I_{adiabatic}$ (K)",
    "I_dyn_pre": r"$I_{dyn,net}$ (K)",
    "I_dTdt_pre": r"$I_{dT/dt}$ (K)",
    "I_diabatic_pre": r"$I_{diabatic}$ (K)",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the adiabatic/advection diagnostic."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot integrated advection and net dynamical contribution against "
            "integrated adiabatic tendency."
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
        default=None,
        help=(
            "Path where the scatter-plot PNG will be written. Defaults to a "
            "layout-specific filename."
        ),
    )
    parser.add_argument(
        "--layout",
        choices=LAYOUT_CHOICES,
        default=FULL_LAYOUT,
        help="Use the full four-panel or presentation two-panel layout.",
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


def validate_layout(layout: str) -> None:
    """Reject unknown figure layouts for programmatic callers."""
    if layout not in LAYOUT_CHOICES:
        choices = ", ".join(LAYOUT_CHOICES)
        raise ValueError(f"layout must be one of: {choices}.")


def default_output_path(layout: str) -> Path:
    """Return the non-overlapping default output for a figure layout."""
    validate_layout(layout)
    if layout == PRESENTATION_LAYOUT:
        return DEFAULT_PRESENTATION_OUTPUT_PATH
    return DEFAULT_OUTPUT_PATH


def main() -> int:
    """Load event features and write the adiabatic/advection comparison."""
    args = parse_args()
    validate_args(args)

    features = open_event_features(args.input_path)
    try:
        output_path = args.output_path or default_output_path(args.layout)
        written = write_tendency_scatter_plot(
            features,
            output_path,
            layout=args.layout,
            color_variable=args.color_variable,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        print("Wrote adiabatic/advection net dynamical comparison figure:")
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
    layout: str = FULL_LAYOUT,
    color_variable: str | None = COLOR_VARIABLE,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> Path:
    """Write the adiabatic/advection net dynamical comparison figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tendency_scatter(
        features,
        layout=layout,
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
    layout: str = FULL_LAYOUT,
    color_variable: str | None = COLOR_VARIABLE,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the requested adiabatic/advection comparison layout."""
    validate_layout(layout)
    validate_feature_variables(features, color_variable=color_variable)

    x_values = feature_values(features, X_VARIABLE)
    advection_values = feature_values(features, ADVECTION_VARIABLE)
    net_dynamical_values = feature_values(features, DYNAMICAL_VARIABLE)
    temperature_change_values = feature_values(features, TEMPERATURE_CHANGE_VARIABLE)
    diabatic_values = feature_values(features, DIABATIC_VARIABLE)
    color_values = feature_values(features, color_variable) if color_variable else None
    color_norm = color_norm_for_values(color_values)
    if color_values is not None and color_norm is None:
        raise ValueError(
            f"Event color variable {color_variable!r} contains no finite values."
        )

    finite_adiabatic = (
        np.isfinite(x_values)
        & np.isfinite(advection_values)
        & np.isfinite(net_dynamical_values)
    )
    if color_values is not None:
        finite_adiabatic &= np.isfinite(color_values)
    finite_temperature_change = finite_adiabatic & np.isfinite(
        temperature_change_values
    )
    finite_diabatic = finite_adiabatic & np.isfinite(diabatic_values)

    if layout == PRESENTATION_LAYOUT:
        return plot_presentation_tendency_scatter(
            x_values,
            advection_values,
            net_dynamical_values,
            diabatic_values,
            finite_adiabatic,
            finite_diabatic,
            color_variable=color_variable,
            color_values=color_values,
            color_norm=color_norm,
            point_size=point_size,
            alpha=alpha,
        )

    return plot_full_tendency_scatter(
        x_values,
        advection_values,
        net_dynamical_values,
        temperature_change_values,
        diabatic_values,
        finite_adiabatic,
        finite_temperature_change,
        finite_diabatic,
        color_variable=color_variable,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )


def plot_full_tendency_scatter(
    x_values: np.ndarray,
    advection_values: np.ndarray,
    net_dynamical_values: np.ndarray,
    temperature_change_values: np.ndarray,
    diabatic_values: np.ndarray,
    finite_adiabatic: np.ndarray,
    finite_temperature_change: np.ndarray,
    finite_diabatic: np.ndarray,
    *,
    color_variable: str | None,
    color_values: np.ndarray | None,
    color_norm: Normalize | None,
    point_size: float,
    alpha: float,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the full 2x2 event-only comparison."""
    fig = plt.figure(
        figsize=plot_style.publication_figsize(
            "full",
            aspect=plot_style.TWO_PANEL_STACK_ASPECT,
        ),
        constrained_layout=True,
    )
    grid = fig.add_gridspec(nrows=2, ncols=2)
    axes = np.array(
        [
            fig.add_subplot(grid[0, 0]),
            fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[0, 1]),
            fig.add_subplot(grid[1, 1]),
        ]
    )
    axes[1].sharex(axes[0])
    axes[3].sharex(axes[2])

    plot_scatter_panel(
        axes[0],
        x_values,
        advection_values,
        finite_adiabatic,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )
    add_one_to_negative_one_line(
        axes[0],
        x_values[finite_adiabatic],
        advection_values[finite_adiabatic],
    )
    axes[0].set_title("Advection vs Adiabatic Heating")
    axes[0].set_ylabel(variable_label(ADVECTION_VARIABLE))

    mappable = plot_scatter_panel(
        axes[1],
        x_values,
        net_dynamical_values,
        finite_adiabatic,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )
    axes[1].set_title("Net Dynamical Contribution")
    axes[1].set_ylabel(NET_DYNAMICAL_LABEL)
    axes[1].set_xlabel(variable_label(X_VARIABLE))

    plot_scatter_panel(
        axes[2],
        net_dynamical_values,
        temperature_change_values,
        finite_temperature_change,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )
    axes[2].set_title(r"Integrated dT/dt vs $I_{dyn,net}$")
    axes[2].set_ylabel(variable_label(TEMPERATURE_CHANGE_VARIABLE))

    plot_scatter_panel(
        axes[3],
        net_dynamical_values,
        diabatic_values,
        finite_diabatic,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )
    axes[3].set_title(r"Diabatic Heating vs $I_{dyn,net}$")
    axes[3].set_ylabel(variable_label(DIABATIC_VARIABLE))
    axes[3].set_xlabel(NET_DYNAMICAL_LABEL)

    set_shared_x_data_limits(axes[:2], x_values[finite_adiabatic])
    set_shared_x_data_limits(axes[2:], net_dynamical_values[finite_adiabatic])

    if color_variable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.92)
        cbar.set_label(variable_label(color_variable))

    fig.suptitle("Advection and Net Dynamical Contribution")
    return fig


def plot_presentation_tendency_scatter(
    x_values: np.ndarray,
    advection_values: np.ndarray,
    net_dynamical_values: np.ndarray,
    diabatic_values: np.ndarray,
    finite_adiabatic: np.ndarray,
    finite_diabatic: np.ndarray,
    *,
    color_variable: str | None,
    color_values: np.ndarray | None,
    color_norm: Normalize | None,
    point_size: float,
    alpha: float,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the presentation 2x1 event-only comparison."""
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=plot_style.publication_figsize(
            "single",
            aspect=plot_style.TWO_PANEL_COLUMN_ASPECT,
        ),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    plot_scatter_panel(
        axes[0],
        x_values,
        advection_values,
        finite_adiabatic,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )
    add_one_to_negative_one_line(
        axes[0],
        x_values[finite_adiabatic],
        advection_values[finite_adiabatic],
    )
    axes[0].set_title("Advection vs Adiabatic Heating")
    axes[0].set_ylabel(variable_label(ADVECTION_VARIABLE))
    axes[0].set_xlabel(variable_label(X_VARIABLE))

    mappable = plot_scatter_panel(
        axes[1],
        net_dynamical_values,
        diabatic_values,
        finite_diabatic,
        color_values=color_values,
        color_norm=color_norm,
        point_size=point_size,
        alpha=alpha,
    )
    axes[1].set_title(r"Diabatic Heating vs $I_{dyn,net}$")
    axes[1].set_ylabel(variable_label(DIABATIC_VARIABLE))
    axes[1].set_xlabel(NET_DYNAMICAL_LABEL)

    set_shared_x_data_limits(np.array([axes[0]]), x_values[finite_adiabatic])
    set_shared_x_data_limits(
        np.array([axes[1]]),
        net_dynamical_values[finite_diabatic],
    )

    if color_variable is not None:
        cbar = fig.colorbar(mappable, ax=axes, shrink=0.92)
        cbar.set_label(variable_label(color_variable))

    fig.suptitle("Advection and Net Dynamical Contribution")
    return fig


def plot_scatter_panel(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    finite: np.ndarray,
    *,
    color_values: np.ndarray | None,
    color_norm: Normalize | None,
    point_size: float,
    alpha: float,
):
    """Plot one panel using a shared finite-event mask."""
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


def color_norm_for_values(color_values: np.ndarray | None) -> Normalize | None:
    """Return a shared color norm centered on the configured value when enabled."""
    if color_values is None:
        return None
    if not USE_CENTERED_COLOR_NORMALIZATION:
        return plot_style.finite_range_color_norm(color_values)

    finite = np.asarray(color_values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return None

    if np.all(finite == COLOR_NORMALIZATION_CENTER):
        padding = (
            0.5
            if COLOR_NORMALIZATION_CENTER == 0.0
            else abs(COLOR_NORMALIZATION_CENTER) * 0.05
        )
        return TwoSlopeNorm(
            vmin=COLOR_NORMALIZATION_CENTER - padding,
            vcenter=COLOR_NORMALIZATION_CENTER,
            vmax=COLOR_NORMALIZATION_CENTER + padding,
        )

    norm = TwoSlopeNorm(vcenter=COLOR_NORMALIZATION_CENTER)
    norm.autoscale_None(finite)
    return norm


def validate_feature_variables(
    features: xr.Dataset,
    *,
    color_variable: str | None,
) -> None:
    """Fail clearly when the event-feature table lacks required variables."""
    required = [
        X_VARIABLE,
        ADVECTION_VARIABLE,
        DYNAMICAL_VARIABLE,
        TEMPERATURE_CHANGE_VARIABLE,
        DIABATIC_VARIABLE,
    ]
    if color_variable is not None:
        required.append(color_variable)
    missing = [name for name in dict.fromkeys(required) if name not in features]
    if missing:
        raise ValueError(
            f"Event-feature table is missing required variables: {', '.join(missing)}."
        )


def feature_values(features: xr.Dataset, variable: str | None) -> np.ndarray:
    """Return a feature variable as a float array."""
    if variable is None:
        raise ValueError("variable must not be None.")
    values = features[variable].values
    if np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, "D")
    else:
        out = np.asarray(values, dtype=float)
    return np.asarray(out, dtype=float)


def set_shared_x_data_limits(axes: np.ndarray, x_values: np.ndarray) -> None:
    """Limit shared x-axes to the finite extent of the plotted x-data."""
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
        linestyle="-",
        zorder=0,
    )
    ax.axvline(
        0.0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        linestyle="--",
        zorder=0,
    )


def add_one_to_one_line(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """Add a y=x reference line over the finite data extent."""
    if x_values.size == 0 or y_values.size == 0:
        return
    lower = float(np.nanmin([np.nanmin(x_values), np.nanmin(y_values)]))
    upper = float(np.nanmax([np.nanmax(x_values), np.nanmax(y_values)]))
    ax.plot(
        [lower, upper],
        [lower, upper],
        scalex=False,
        scaley=False,
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
