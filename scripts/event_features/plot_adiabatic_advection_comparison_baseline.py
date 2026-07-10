"""Compare selected events with clean baseline-day tendencies.

The diagnostic loads matching Stage-2 baseline-day and event-feature tables.
Clean baseline days are plotted as a translucent background and the selected
event population is plotted as a foreground layer.
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

REGION = "pnw_hotz"
THRESHOLD_VARIABLE = "tas"
QUANTILE_THRESHOLD = "q90"
TIME_START = 1940
TIME_END = 2024

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_baseline_features"
    / (
        "non_event_day_features_fixed_windows_"
        f"{REGION}_{THRESHOLD_VARIABLE}_{QUANTILE_THRESHOLD}.nc"
    )
)
DEFAULT_EVENT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / (
        "hw_event_features_fixed_windows_"
        f"{REGION}_{THRESHOLD_VARIABLE}_{QUANTILE_THRESHOLD}_"
        f"{TIME_START}_{TIME_END}.nc"
    )
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_baseline_features"
    / "diagnostics"
    / REGION
    / THRESHOLD_VARIABLE
    / "event_vs_clean_baseline_adiabatic_advection_comparison.png"
)

X_VARIABLE = "I_adiabatic_pre"
ADVECTION_VARIABLE = "I_advection_pre"
TEMPERATURE_CHANGE_VARIABLE = "I_dTdt_pre"
DIABATIC_VARIABLE = "I_diabatic_pre"
EVENT_ADJACENT_VARIABLE = "event_adjacent"
EVENT_DIM = "event"
PLOTTED_VARIABLES = (
    X_VARIABLE,
    ADVECTION_VARIABLE,
    TEMPERATURE_CHANGE_VARIABLE,
    DIABATIC_VARIABLE,
)
BASELINE_POINT_COLOR = plot_style.COLORS["volume"]
EVENT_POINT_COLOR = plot_style.COLORS["diabatic"]
EVENT_EDGE_COLOR = "white"
NET_DYNAMICAL_LABEL = r"$I_{dyn,net}$ (K)"
VARIABLE_LABELS = {
    "I_advection_pre": r"$I_{advective}$(K)",
    "I_adiabatic_pre": r"$I_{adiabatic}$ (K)",
    "I_dTdt_pre": r"$I_{dT/dt}$ (K)",
    "I_diabatic_pre": r"$I_{diabatic}$ (K)",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the event-versus-baseline diagnostic."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot selected events and clean baseline days in the "
            "adiabatic/advection net dynamical comparison."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the Stage-2 baseline-day feature NetCDF table.",
    )
    parser.add_argument(
        "--event-input-path",
        type=Path,
        default=DEFAULT_EVENT_INPUT_PATH,
        help="Path to the matching Stage-2 event-feature NetCDF table.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the scatter-plot PNG will be written.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=24.0,
        help="Baseline scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Baseline scatter marker opacity.",
    )
    parser.add_argument(
        "--event-point-size",
        type=float,
        default=40.0,
        help="Event scatter marker size.",
    )
    parser.add_argument(
        "--event-alpha",
        type=float,
        default=0.9,
        help="Event scatter marker opacity.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if args.point_size <= 0:
        raise ValueError("--point-size must be > 0.")
    if not 0 < args.alpha <= 1:
        raise ValueError("--alpha must satisfy 0 < alpha <= 1.")
    if args.event_point_size <= 0:
        raise ValueError("--event-point-size must be > 0.")
    if not 0 < args.event_alpha <= 1:
        raise ValueError("--event-alpha must satisfy 0 < value <= 1.")


def main() -> int:
    """Load matching feature tables and write the comparison diagnostic."""
    args = parse_args()
    validate_args(args)

    baseline_features = open_baseline_features(args.input_path)
    try:
        event_features = open_event_features(args.event_input_path)
        try:
            written = write_tendency_scatter_plot(
                baseline_features,
                event_features,
                args.output_path,
                point_size=args.point_size,
                alpha=args.alpha,
                event_point_size=args.event_point_size,
                event_alpha=args.event_alpha,
            )
            print("Wrote event-versus-clean-baseline comparison figure:")
            print(f"  {_display_path(written)}")
        finally:
            event_features.close()
    finally:
        baseline_features.close()
    return 0


def open_baseline_features(path: str | Path) -> xr.Dataset:
    """Open a Stage-2 baseline-day feature NetCDF table."""
    input_path = Path(path).expanduser().resolve()
    return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)


def open_event_features(path: str | Path) -> xr.Dataset:
    """Open a Stage-2 event-feature NetCDF table."""
    input_path = Path(path).expanduser().resolve()
    return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)


def write_tendency_scatter_plot(
    baseline_features: xr.Dataset,
    event_features: xr.Dataset,
    output_path: str | Path,
    *,
    point_size: float = 24.0,
    alpha: float = 0.2,
    event_point_size: float = 40.0,
    event_alpha: float = 0.9,
) -> Path:
    """Write the selected event-versus-clean-baseline comparison figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tendency_scatter(
        baseline_features,
        event_features,
        point_size=point_size,
        alpha=alpha,
        event_point_size=event_point_size,
        event_alpha=event_alpha,
    )
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_tendency_scatter(
    baseline_features: xr.Dataset,
    event_features: xr.Dataset,
    *,
    point_size: float = 24.0,
    alpha: float = 0.2,
    event_point_size: float = 40.0,
    event_alpha: float = 0.9,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return the selected event-versus-clean-baseline comparison figure."""
    validate_feature_variables(baseline_features, event_features)
    clean = clean_baseline_mask(baseline_features)

    baseline = tendency_values(baseline_features)
    events = tendency_values(event_features)

    fig = plt.figure(
        figsize=plot_style.publication_figsize("full", aspect=0.62),
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

    plot_comparison_panel(
        axes[0],
        baseline["adiabatic"],
        baseline["advection"],
        clean,
        events["adiabatic"],
        events["advection"],
        point_size=point_size,
        alpha=alpha,
        event_point_size=event_point_size,
        event_alpha=event_alpha,
        show_legend=True,
    )
    add_one_to_negative_one_line_from_panel(axes[0])
    axes[0].set_title("Advection vs Adiabatic Heating")
    axes[0].set_ylabel(variable_label(ADVECTION_VARIABLE))

    plot_comparison_panel(
        axes[1],
        baseline["adiabatic"],
        baseline["net_dynamical"],
        clean,
        events["adiabatic"],
        events["net_dynamical"],
        point_size=point_size,
        alpha=alpha,
        event_point_size=event_point_size,
        event_alpha=event_alpha,
        show_legend=False,
    )
    axes[1].set_title("Net Dynamical Contribution")
    axes[1].set_ylabel(NET_DYNAMICAL_LABEL)
    axes[1].set_xlabel(variable_label(X_VARIABLE))

    plot_comparison_panel(
        axes[2],
        baseline["net_dynamical"],
        baseline["temperature_change"],
        clean,
        events["net_dynamical"],
        events["temperature_change"],
        point_size=point_size,
        alpha=alpha,
        event_point_size=event_point_size,
        event_alpha=event_alpha,
        show_legend=False,
    )
    axes[2].set_title(r"Integrated dT/dt vs $I_{dyn,net}$")
    axes[2].set_ylabel(variable_label(TEMPERATURE_CHANGE_VARIABLE))

    plot_comparison_panel(
        axes[3],
        baseline["net_dynamical"],
        baseline["diabatic"],
        clean,
        events["net_dynamical"],
        events["diabatic"],
        point_size=point_size,
        alpha=alpha,
        event_point_size=event_point_size,
        event_alpha=event_alpha,
        show_legend=False,
    )
    axes[3].set_title(r"Diabatic Heating vs $I_{dyn,net}$")
    axes[3].set_ylabel(variable_label(DIABATIC_VARIABLE))
    axes[3].set_xlabel(NET_DYNAMICAL_LABEL)

    set_shared_x_data_limits(axes[:2], panel_x_values(axes[:2]))
    set_shared_x_data_limits(axes[2:], panel_x_values(axes[2:]))

    fig.suptitle("Events vs Clean Baseline-Day Tendencies")
    return fig


def plot_comparison_panel(
    ax: Axes,
    baseline_x: np.ndarray,
    baseline_y: np.ndarray,
    clean: np.ndarray,
    event_x: np.ndarray,
    event_y: np.ndarray,
    *,
    point_size: float,
    alpha: float,
    event_point_size: float,
    event_alpha: float,
    show_legend: bool,
) -> None:
    """Plot clean baseline and selected event layers in one panel."""
    baseline_finite = clean & np.isfinite(baseline_x) & np.isfinite(baseline_y)
    event_finite = np.isfinite(event_x) & np.isfinite(event_y)

    baseline_scatter = ax.scatter(
        baseline_x[baseline_finite],
        baseline_y[baseline_finite],
        s=point_size,
        alpha=alpha,
        edgecolors="none",
        color=BASELINE_POINT_COLOR,
        label="Clean baseline days",
        zorder=1,
    )
    event_scatter = ax.scatter(
        event_x[event_finite],
        event_y[event_finite],
        s=event_point_size,
        alpha=event_alpha,
        edgecolors=EVENT_EDGE_COLOR,
        linewidths=0.45,
        color=EVENT_POINT_COLOR,
        label="Events",
        zorder=3,
    )

    add_zero_reference_lines(ax)
    if show_legend:
        ax.legend(
            handles=[baseline_scatter, event_scatter],
            loc="lower right",
            **plot_style.legend_kwargs(),
        )
    ax.text(
        0.03,
        0.97,
        (
            f"baseline n = {int(baseline_finite.sum())}\n"
            f"events n = {int(event_finite.sum())}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    plot_style.style_axis(ax)


def validate_feature_variables(
    baseline_features: xr.Dataset,
    event_features: xr.Dataset,
) -> None:
    """Fail clearly when either feature table lacks required variables."""
    validate_required_variables(
        baseline_features,
        [EVENT_ADJACENT_VARIABLE, *PLOTTED_VARIABLES],
        table_label="Baseline-day",
    )
    validate_required_variables(
        event_features,
        list(PLOTTED_VARIABLES),
        table_label="Event-feature",
    )
    validate_event_rows(event_features)


def validate_required_variables(
    features: xr.Dataset,
    required: list[str],
    *,
    table_label: str,
) -> None:
    """Fail clearly when a feature table lacks required variables."""
    missing = [name for name in dict.fromkeys(required) if name not in features]
    if missing:
        raise ValueError(
            f"{table_label} table is missing required variables: "
            f"{', '.join(missing)}."
        )


def validate_event_rows(features: xr.Dataset) -> None:
    """Fail clearly when an event-feature table contains no event rows."""
    if EVENT_DIM not in features.sizes or features.sizes[EVENT_DIM] == 0:
        raise ValueError("Event-feature table contains no event rows.")


def clean_baseline_mask(features: xr.Dataset) -> np.ndarray:
    """Return clean baseline rows, failing when none are available."""
    clean = np.asarray(features[EVENT_ADJACENT_VARIABLE].values) == 0
    if not np.any(clean):
        raise ValueError(
            "Baseline-day feature table contains no clean rows where "
            "event_adjacent == 0."
        )
    return clean


def tendency_values(features: xr.Dataset) -> dict[str, np.ndarray]:
    """Return arrays used in the four comparison panels."""
    adiabatic = feature_values(features, X_VARIABLE)
    advection = feature_values(features, ADVECTION_VARIABLE)
    net_dynamical = net_dynamical_contribution(adiabatic, advection)
    return {
        "adiabatic": adiabatic,
        "advection": advection,
        "net_dynamical": net_dynamical,
        "temperature_change": feature_values(features, TEMPERATURE_CHANGE_VARIABLE),
        "diabatic": feature_values(features, DIABATIC_VARIABLE),
    }


def feature_values(features: xr.Dataset, variable: str) -> np.ndarray:
    """Return a feature variable as a float array."""
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


def panel_x_values(axes: np.ndarray) -> np.ndarray:
    """Return the x-values from every scatter collection in the given axes."""
    values = []
    for ax in axes:
        for collection in ax.collections:
            offsets = np.asarray(collection.get_offsets())
            if offsets.size:
                values.append(np.asarray(offsets[:, 0], dtype=float))
    if not values:
        return np.array([], dtype=float)
    return np.concatenate(values)


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
    """Return a readable axis label."""
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


def add_one_to_negative_one_line_from_panel(ax: Axes) -> None:
    """Add a y=-x reference line using the panel's plotted scatter values."""
    x_values = []
    y_values = []
    for collection in ax.collections:
        offsets = np.asarray(collection.get_offsets())
        if offsets.size:
            x_values.append(np.asarray(offsets[:, 0], dtype=float))
            y_values.append(np.asarray(offsets[:, 1], dtype=float))
    if not x_values:
        return
    add_one_to_negative_one_line(
        ax,
        np.concatenate(x_values),
        np.concatenate(y_values),
    )


def add_one_to_negative_one_line(
    ax: Axes,
    x_values: np.ndarray,
    y_values: np.ndarray,
) -> None:
    """Add a y=-x reference line over the finite data extent."""
    finite = np.isfinite(x_values) & np.isfinite(y_values)
    if not np.any(finite):
        return
    x_values = x_values[finite]
    y_values = y_values[finite]
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
