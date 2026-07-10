"""Explore event-feature relationships before clustering.

This script intentionally starts with lightweight 2D diagnostics rather than
clustering. It loads the event-feature table and plots the integrated heat-budget
terms against the integrated temperature tendency over the same pre-peak window.
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

from src import plot_style


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / "hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / "diagnostics"
    / "event_feature_tendency_scatter.png"
)

X_VARIABLE = "I_dTdt_pre"
HEAT_BUDGET_Y_VARIABLES = (
    "I_adiabatic_pre",
    "I_diabatic_pre",
    "I_advection_pre",
)
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
    "I_adiabatic_pre": "Integrated adiabatic tendency (K)",
    "I_diabatic_pre": "Integrated diabatic tendency (K)",
    "I_advection_pre": "Integrated advective tendency (K)",
    "sqrt_I_lwa_a_pre_peak": "sqrt(integrated anticyclonic LWA exposure)",
    "T_anom_mean_ant": "Antecedent T anomaly (K)",
    "cos_days_from_solstice": "cos(days from solstice * 2pi / 365)",
    "duration": "Duration (days)",
    "tas_anom_peak": "Peak TAS anomaly (K)",
    "log10_tas_excess_integral": "log10(TAS excess integral)",
}
PANEL_TITLES = {
    "I_adiabatic_pre": "Adiabatic",
    "I_diabatic_pre": "Diabatic",
    "I_advection_pre": "Advection",
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
    """Parse command-line options for feature-space diagnostic plots."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot integrated heat-budget terms against integrated temperature "
            "change from an event-feature table."
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
        default=None,
        help="Optional event-level feature used to color points.",
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
    """Load event features and write tendency scatter diagnostics."""
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
        print("Wrote event-feature tendency scatter figure:")
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
    color_variable: str | None = None,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> Path:
    """Write the integrated tendency scatter figure."""
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
    color_variable: str | None = None,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return a multi-panel feature scatter figure."""
    validate_feature_variables(features, color_variable=color_variable)

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=plot_style.publication_figsize("full", aspect=1.0),
        sharex=True,
        constrained_layout=True,
    )
    color_values = feature_values(features, color_variable) if color_variable else None

    mappable = None
    for ax, y_variable in zip(np.ravel(axes), Y_VARIABLES):
        mappable = plot_one_tendency_panel(
            ax,
            features,
            y_variable,
            color_values=color_values,
            color_variable=color_variable,
            point_size=point_size,
            alpha=alpha,
        )

    if color_variable is not None and mappable is not None:
        cbar = fig.colorbar(mappable, ax=np.ravel(axes), shrink=0.86)
        cbar.set_label(variable_label(color_variable))

    fig.suptitle("Event Fixed-Window Heat-Budget Feature Relationships")
    return fig


def plot_one_tendency_panel(
    ax: Axes,
    features: xr.Dataset,
    y_variable: str,
    *,
    color_values: np.ndarray | None,
    color_variable: str | None,
    point_size: float,
    alpha: float,
):
    """Plot one y-variable against integrated temperature tendency."""
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
            cmap="viridis",
            **kwargs,
        )

    add_zero_reference_lines(ax)
    if y_variable in HEAT_BUDGET_Y_VARIABLES:
        add_one_to_one_line(ax, x_values[finite], y_values[finite])
    ax.set_title(PANEL_TITLES[y_variable])
    ax.set_xlabel(variable_label(X_VARIABLE))
    ax.set_ylabel(variable_label(y_variable))
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
    required.extend(source_variable_name(name) for name in Y_VARIABLES)
    if color_variable is not None:
        required.append(source_variable_name(color_variable))
    missing = [name for name in required if name not in features]
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
    if variable == "sqrt_I_lwa_a_pre_peak":
        out = np.where(out >= 0.0, np.sqrt(out), np.nan)
    elif variable == "cos_days_from_solstice":
        out = np.cos(out * 2.0 * np.pi / 365.0)
    elif variable == "log10_tas_excess_integral":
        out = np.where(out > 0.0, np.log10(out), np.nan)
    return out


def source_variable_name(variable: str) -> str:
    """Return the dataset variable needed for a plot variable."""
    return DERIVED_VARIABLE_SOURCES.get(variable, variable)


def variable_label(variable: str) -> str:
    """Return a readable axis or colorbar label."""
    if variable in VARIABLE_LABELS:
        return VARIABLE_LABELS[variable]
    units = ""
    return variable if not units else f"{variable} ({units})"


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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
