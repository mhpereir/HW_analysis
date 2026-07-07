"""Compare event tendencies with clean baseline-day tendencies.

The diagnostic loads matching Stage-2 baseline-day and event-feature tables.
Clean baseline days are plotted as a translucent background and events are
plotted as a foreground layer.
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
THRESHOLD_VARIABLE = "lwa_a"
QUANTILE_THRESHOLD = "q90"

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
        f"{REGION}_{THRESHOLD_VARIABLE}_{QUANTILE_THRESHOLD}_1940_2024.nc"
    )
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_baseline_features"
    / "diagnostics"
    / REGION
    / THRESHOLD_VARIABLE
    / "event_vs_clean_baseline_diabatic_advection_scatter.png"
)

X_VARIABLE = "I_adiabatic_pre"
EVENT_ADJACENT_VARIABLE = "event_adjacent"
EVENT_DIM = "event"
BASELINE_POINT_COLOR = "tab:blue"
EVENT_POINT_COLOR = "tab:red"
EVENT_EDGE_COLOR = "white"
BASELINE_Y_VARIABLES = (
    "I_diabatic_pre",
    "I_advection_pre",
    "sqrt_I_lwa_a_pre_reference",
)
EVENT_Y_VARIABLES = {
    "I_diabatic_pre": "I_diabatic_pre",
    "I_advection_pre": "I_advection_pre",
    "sqrt_I_lwa_a_pre_reference": "sqrt_I_lwa_a_pre_peak",
}
VARIABLE_LABELS = {
    "I_diabatic_pre": "I_diabatic (K)",
    "I_advection_pre": "I_advective (K)",
    "I_adiabatic_pre": "I_adiabatic (K)",
    "I_lwa_a_pre_reference": "I_LWA_a",
    "sqrt_I_lwa_a_pre_reference": "sqrt(I_LWA_a)",
    "I_lwa_a_pre_peak": "I_LWA_a",
    "sqrt_I_lwa_a_pre_peak": "sqrt(I_LWA_a)",
}
PANEL_TITLES = {
    "I_diabatic_pre": "Diabatic vs Adiabatic",
    "I_advection_pre": "Advection vs Adiabatic",
    "sqrt_I_lwa_a_pre_reference": "Sqrt LWA a Exposure",
}
DERIVED_VARIABLE_SOURCES = {
    "sqrt_I_lwa_a_pre_reference": "I_lwa_a_pre_reference",
    "sqrt_I_lwa_a_pre_peak": "I_lwa_a_pre_peak",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the event-versus-baseline diagnostic."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare event and clean baseline-day integrated diabatic, "
            "advective, and LWA quantities against integrated adiabatic tendency."
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
        help="Scatter marker size.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        help="Scatter marker opacity.",
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
    """Load matching feature tables and write the combined scatter diagnostic."""
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
            print("Wrote event-versus-clean-baseline scatter figure:")
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
    """Write the event-versus-clean-baseline scatter figure."""
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
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    """Return the event-versus-clean-baseline scatter figure."""
    validate_feature_variables(baseline_features, event_features)
    clean = clean_baseline_mask(baseline_features)

    fig, axes = plt.subplots(
        nrows=len(BASELINE_Y_VARIABLES),
        ncols=1,
        figsize=(7.5, 3.2 * len(BASELINE_Y_VARIABLES)),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for index, (ax, y_variable) in enumerate(zip(axes, BASELINE_Y_VARIABLES)):
        plot_one_tendency_panel(
            ax,
            baseline_features,
            event_features,
            y_variable,
            clean=clean,
            point_size=point_size,
            alpha=alpha,
            event_point_size=event_point_size,
            event_alpha=event_alpha,
            show_xlabel=index == len(BASELINE_Y_VARIABLES) - 1,
            show_legend=index == 0,
        )

    baseline_x = feature_values(baseline_features, X_VARIABLE)
    event_x = feature_values(event_features, X_VARIABLE)
    combined_x = np.concatenate(
        (
            baseline_x[clean & np.isfinite(baseline_x)],
            event_x[np.isfinite(event_x)],
        )
    )
    set_shared_x_data_limits(axes, combined_x)
    fig.suptitle(
        "Events vs Clean Baseline-Day Tendencies",
    )
    return fig


def plot_one_tendency_panel(
    ax: Axes,
    baseline_features: xr.Dataset,
    event_features: xr.Dataset,
    baseline_y_variable: str,
    *,
    clean: np.ndarray,
    point_size: float,
    alpha: float,
    event_point_size: float,
    event_alpha: float,
    show_xlabel: bool,
    show_legend: bool,
) -> None:
    """Plot event foreground points over clean baseline-day background points."""
    event_y_variable = EVENT_Y_VARIABLES[baseline_y_variable]
    baseline_x = feature_values(baseline_features, X_VARIABLE)
    baseline_y = feature_values(baseline_features, baseline_y_variable)
    event_x = feature_values(event_features, X_VARIABLE)
    event_y = feature_values(event_features, event_y_variable)
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
    reference_x = np.concatenate(
        (baseline_x[baseline_finite], event_x[event_finite])
    )
    reference_y = np.concatenate(
        (baseline_y[baseline_finite], event_y[event_finite])
    )
    add_zero_reference_lines(ax)
    add_one_to_one_line(ax, reference_x, reference_y)
    if baseline_y_variable == "I_advection_pre":
        add_one_to_negative_one_line(ax, reference_x, reference_y)
    ax.set_title(PANEL_TITLES[baseline_y_variable])
    ax.set_ylabel(variable_label(baseline_y_variable))
    if show_xlabel:
        ax.set_xlabel(variable_label(X_VARIABLE))
    ax.grid(True, color="0.88", linewidth=0.8)
    if show_legend:
        ax.legend(
            handles=[baseline_scatter, event_scatter],
            loc="lower right",
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


def validate_feature_variables(
    baseline_features: xr.Dataset,
    event_features: xr.Dataset,
) -> None:
    """Fail clearly when either feature table lacks required variables."""
    baseline_required = [EVENT_ADJACENT_VARIABLE, X_VARIABLE]
    for name in BASELINE_Y_VARIABLES:
        baseline_required.extend(source_variable_names(name))
    event_required = [X_VARIABLE]
    for name in EVENT_Y_VARIABLES.values():
        event_required.extend(source_variable_names(name))

    validate_required_variables(
        baseline_features,
        baseline_required,
        table_label="Baseline-day",
    )
    validate_required_variables(
        event_features,
        event_required,
        table_label="Event-feature",
    )
    if EVENT_DIM not in event_features.sizes or event_features.sizes[EVENT_DIM] == 0:
        raise ValueError("Event-feature table contains no event rows.")


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
            f"{table_label} feature table is missing required variables: "
            f"{', '.join(missing)}."
        )


def clean_baseline_mask(features: xr.Dataset) -> np.ndarray:
    """Return clean baseline rows, failing when none are available."""
    clean = np.asarray(features[EVENT_ADJACENT_VARIABLE].values) == 0
    if not np.any(clean):
        raise ValueError(
            "Baseline-day feature table contains no clean rows where "
            "event_adjacent == 0."
        )
    return clean


def feature_values(features: xr.Dataset, variable: str) -> np.ndarray:
    """Return a plot variable as a float array."""
    values = features[source_variable_name(variable)].values
    if np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, "D")
    else:
        out = np.asarray(values, dtype=float)
    if variable in {"sqrt_I_lwa_a_pre_reference", "sqrt_I_lwa_a_pre_peak"}:
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


def set_shared_x_data_limits(axes: np.ndarray, x_values: np.ndarray) -> None:
    """Limit shared x-axes to the finite combined x-data extent."""
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
    ax.axhline(0.0, color="0.55", linewidth=0.9, linestyle="--", zorder=0)
    ax.axvline(0.0, color="0.55", linewidth=0.9, linestyle="--", zorder=0)


def add_one_to_one_line(ax: Axes, x_values: np.ndarray, y_values: np.ndarray) -> None:
    """Add a y=x reference line over the finite data extent."""
    if x_values.size == 0 or y_values.size == 0:
        return
    lower = float(np.nanmin([np.nanmin(x_values), np.nanmin(y_values)]))
    upper = float(np.nanmax([np.nanmax(x_values), np.nanmax(y_values)]))
    ax.plot(
        [lower, upper],
        [lower, upper],
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
