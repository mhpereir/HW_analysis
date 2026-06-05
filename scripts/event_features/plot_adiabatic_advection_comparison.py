"""Compare advection with adiabatic heating and net dynamical contribution.

The diagnostic loads a Stage-2 event-feature table and writes a four-panel
scatter figure. The first two panels use integrated adiabatic tendency on the
x-axis, while the final two use net dynamical contribution on the x-axis.
Points are colored by peak temperature anomaly by default.
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

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
})

REGION = "pnw_bartusek"
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

X_VARIABLE = "I_adiabatic_pre"
ADVECTION_VARIABLE = "I_advection_pre"
TEMPERATURE_CHANGE_VARIABLE = "I_dTdt_pre"
DIABATIC_VARIABLE = "I_diabatic_pre"
COLOR_VARIABLE = "tas_anom_peak"
COLOR_MAP = "gist_heat_r"
NET_DYNAMICAL_LABEL = r"$I_{dyn,net}$ (K)"
VARIABLE_LABELS = {
    "tas_anom_peak": "Peak TAS Anomaly (K)",
    "tas_peak": "Peak TAS (K)",
    "I_advection_pre": "I_advective (K)",
    "I_adiabatic_pre": "I_adiabatic (K)",
    "I_dTdt_pre": "I_dT/dt (K)",
    "I_diabatic_pre": "I_diabatic (K)",
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
    """Load event features and write the adiabatic/advection comparison."""
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
    color_variable: str | None = COLOR_VARIABLE,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> Path:
    """Write the adiabatic/advection net dynamical comparison figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tendency_scatter(
        features,
        color_variable=color_variable,
        point_size=point_size,
        alpha=alpha,
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_tendency_scatter(
    features: xr.Dataset,
    *,
    color_variable: str | None = COLOR_VARIABLE,
    point_size: float = 24.0,
    alpha: float = 0.75,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the four-panel adiabatic/advection net dynamical comparison."""
    validate_feature_variables(features, color_variable=color_variable)

    x_values = feature_values(features, X_VARIABLE)
    advection_values = feature_values(features, ADVECTION_VARIABLE)
    net_dynamical_values = net_dynamical_contribution(
        x_values,
        advection_values,
    )
    temperature_change_values = feature_values(features, TEMPERATURE_CHANGE_VARIABLE)
    diabatic_values = feature_values(features, DIABATIC_VARIABLE)
    color_values = feature_values(features, color_variable) if color_variable else None

    finite_adiabatic = np.isfinite(x_values) & np.isfinite(advection_values)
    if color_values is not None:
        finite_adiabatic &= np.isfinite(color_values)
    finite_temperature_change = finite_adiabatic & np.isfinite(
        temperature_change_values
    )
    finite_diabatic = finite_adiabatic & np.isfinite(diabatic_values)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(7.5, 12.8),
        sharex=False,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)
    axes[1].sharex(axes[0])
    axes[3].sharex(axes[2])

    plot_scatter_panel(
        axes[0],
        x_values,
        advection_values,
        finite_adiabatic,
        color_values=color_values,
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
        point_size=point_size,
        alpha=alpha,
    )
    axes[2].set_title(r"Integrated dT/dt vs $I_{dyn,net}$")
    axes[2].set_ylabel(variable_label(TEMPERATURE_CHANGE_VARIABLE))
    axes[2].set_xlabel(NET_DYNAMICAL_LABEL)

    plot_scatter_panel(
        axes[3],
        net_dynamical_values,
        diabatic_values,
        finite_diabatic,
        color_values=color_values,
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


def plot_scatter_panel(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
    finite: np.ndarray,
    *,
    color_values: np.ndarray | None,
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
        kwargs["color"] = "tab:blue"
        mappable = ax.scatter(x_values[finite], y_values[finite], **kwargs)
    else:
        mappable = ax.scatter(
            x_values[finite],
            y_values[finite],
            c=color_values[finite],
            cmap=COLOR_MAP,
            **kwargs,
        )

    add_zero_reference_lines(ax)
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
    return mappable


def validate_feature_variables(
    features: xr.Dataset,
    *,
    color_variable: str | None,
) -> None:
    """Fail clearly when the event-feature table lacks required variables."""
    required = [
        X_VARIABLE,
        ADVECTION_VARIABLE,
        TEMPERATURE_CHANGE_VARIABLE,
        DIABATIC_VARIABLE,
    ]
    if color_variable is not None:
        required.append(color_variable)
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
    values = features[variable].values
    if np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, "D")
    else:
        out = np.asarray(values, dtype=float)
    return np.asarray(out, dtype=float)


def net_dynamical_contribution(
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> np.ndarray:
    """Return the net dynamical contribution from adiabatic and advective terms."""
    return x_values + y_values


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
    ax.axhline(0.0, color="0.35", linewidth=2.0, linestyle="-", zorder=0)
    ax.axvline(0.0, color="0.55", linewidth=0.9, linestyle="--", zorder=0)


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
        color="0.25",
        linewidth=1.0,
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
        color="0.25",
        linewidth=1.0,
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
