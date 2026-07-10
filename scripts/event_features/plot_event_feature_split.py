"""Plot Stage-2 event-feature relationships split by one feature quantile."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
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

REGION = "pnw_bartusek"

DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / f"hw_event_features_fixed_windows_{REGION}_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_event_features"
    / "diagnostics"
    / REGION
    / "event_feature_tendency_scatter.png"
)

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
Y_VARIABLES = (
    *HEAT_BUDGET_Y_VARIABLES,
    "sqrt_I_lwa_a_pre_peak",
    "T_anom_mean_ant",
    "cos_days_from_solstice",
    "duration",
    "tas_anom_peak",
    "tas_peak",
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
    "tas_peak": "Peak TAS (K)",
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
    "tas_peak": "Peak TAS",
}
DERIVED_VARIABLE_SOURCES = {
    "f_adiabatic_pre": INTEGRATED_HEAT_BUDGET_VARIABLES,
    "f_diabatic_pre": INTEGRATED_HEAT_BUDGET_VARIABLES,
    "f_advection_pre": INTEGRATED_HEAT_BUDGET_VARIABLES,
    "sqrt_I_lwa_a_pre_peak": "I_lwa_a_pre_peak",
    "cos_days_from_solstice": "days_from_solstice",
}
SPLIT_GROUPS = (
    ("low", plot_style.COLORS["volume"], "Low"),
    ("high", plot_style.COLORS["diabatic"], "High"),
)
SUMMARY_ALPHA = 0.20


@dataclass(frozen=True)
class QuantileSplit:
    """Event masks and metadata for one split-variable quantile threshold."""

    variable: str
    quantile: float
    threshold: float
    values: np.ndarray
    low_mask: np.ndarray
    high_mask: np.ndarray


def parse_args() -> argparse.Namespace:
    """Parse command-line options for split feature-space diagnostics."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot integrated heat-budget terms against integrated temperature "
            "change from a Stage-2 event-feature table, split by one feature "
            "quantile."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Path to the Stage-2 event-feature NetCDF table.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Base path where the split scatter-plot PNG will be written.",
    )
    parser.add_argument(
        "--selection-variable",
        "--split-variable",
        dest="selection_variable",
        required=True,
        help="One 3x3 grid y-variable used to split events into two groups.",
    )
    parser.add_argument(
        "--selection-quantile",
        "--split-quantile",
        dest="selection_quantile",
        type=float,
        required=True,
        help="Quantile threshold in (0, 1) used to split events.",
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
    validate_selection_quantile(args.selection_quantile)
    if args.selection_variable not in Y_VARIABLES:
        allowed = ", ".join(Y_VARIABLES)
        raise ValueError(
            "--selection-variable must be one of the plotted y-variables: "
            f"{allowed}."
        )


def main() -> int:
    """Load event features and write split tendency scatter diagnostics."""
    args = parse_args()
    validate_args(args)

    features = open_event_features(args.input_path)
    try:
        output_path = _split_output_path(args.output_path, args.selection_variable)
        written = write_tendency_scatter_plot(
            features,
            output_path,
            selection_variable=args.selection_variable,
            selection_quantile=args.selection_quantile,
            point_size=args.point_size,
            alpha=args.alpha,
        )
        standardized_written = write_tendency_scatter_plot(
            features,
            standardized_output_path(output_path),
            selection_variable=args.selection_variable,
            selection_quantile=args.selection_quantile,
            point_size=args.point_size,
            alpha=args.alpha,
            standardized=True,
        )
        print("Wrote split event-feature tendency scatter figures:")
        print(f"  {_display_path(written)}")
        print(f"  {_display_path(standardized_written)}")
    finally:
        features.close()
    return 0


def open_event_features(path: str | Path) -> xr.Dataset:
    """Open a Stage-2 event-feature NetCDF table."""
    input_path = Path(path).expanduser().resolve()
    try:
        return xr.open_dataset(input_path, engine="h5netcdf", decode_timedelta=True)
    except TypeError as exc:
        if "decode_timedelta" not in str(exc):
            raise
        return xr.open_dataset(input_path, engine="h5netcdf")


def write_tendency_scatter_plot(
    features: xr.Dataset,
    output_path: str | Path,
    *,
    selection_variable: str,
    selection_quantile: float,
    point_size: float = 24.0,
    alpha: float = 0.75,
    standardized: bool = False,
) -> Path:
    """Write the split integrated tendency scatter figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_tendency_scatter(
        features,
        selection_variable=selection_variable,
        selection_quantile=selection_quantile,
        point_size=point_size,
        alpha=alpha,
        standardized=standardized,
    )
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_tendency_scatter(
    features: xr.Dataset,
    *,
    selection_variable: str,
    selection_quantile: float,
    point_size: float = 24.0,
    alpha: float = 0.75,
    standardized: bool = False,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return a multi-panel feature scatter figure split by one quantile."""
    validate_feature_variables(features, selection_variable=selection_variable)
    split = build_quantile_split(
        features,
        selection_variable=selection_variable,
        selection_quantile=selection_quantile,
    )

    fig, axes = plt.subplots(
        nrows=3,
        ncols=3,
        figsize=plot_style.publication_figsize("full", aspect=1.0),
        sharex=True,
        constrained_layout=True,
    )
    for ax, y_variable in zip(np.ravel(axes), Y_VARIABLES):
        plot_one_tendency_panel(
            ax,
            features,
            y_variable,
            split=split,
            point_size=point_size,
            alpha=alpha,
            standardized=standardized,
        )

    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        axes.flat[2].legend(
            handles,
            labels,
            loc="upper right",
            **plot_style.legend_kwargs(),
        )

    title = (
        "Event Fixed-Window Heat-Budget Feature Relationships "
        f"(split by {variable_label(selection_variable)})"
    )
    if standardized:
        title = f"{title} (Standardized)"
    fig.suptitle(title)
    return fig


def plot_one_tendency_panel(
    ax: Axes,
    features: xr.Dataset,
    y_variable: str,
    *,
    split: QuantileSplit,
    point_size: float,
    alpha: float,
    standardized: bool,
) -> None:
    """Plot one y-variable against integrated temperature tendency."""
    x_values = feature_values(features, X_VARIABLE)
    y_values = feature_values(features, y_variable)
    threshold_value = split.threshold if y_variable == split.variable else None

    if standardized:
        x_values = standardized_values(x_values)
        y_values, threshold_value = standardized_panel_values(
            y_values,
            threshold_value,
        )

    finite = np.isfinite(x_values) & np.isfinite(y_values)

    for group_name, color, label in SPLIT_GROUPS:
        group_mask = split_group_mask(split, group_name) & finite
        ax.scatter(
            x_values[group_mask],
            y_values[group_mask],
            s=point_size,
            alpha=alpha,
            color=color,
            edgecolors="none",
            label=f"{label} split",
        )
        add_group_summary_overlay(
            ax,
            y_values,
            group_mask,
            color=color,
            group_name=group_name,
            label=label,
        )

    add_zero_reference_lines(ax)
    if y_variable in INTEGRATED_HEAT_BUDGET_VARIABLES:
        add_one_to_one_line(ax, x_values[finite], y_values[finite])
    if threshold_value is not None and np.isfinite(threshold_value):
        line = ax.axhline(
            threshold_value,
            color=plot_style.COLORS["calculated"],
            linewidth=plot_style.LINE_WIDTH_PT,
            linestyle="-",
            zorder=4,
        )
        line.set_gid("selection_threshold")

    ax.set_title(PANEL_TITLES[y_variable])
    ax.set_xlabel(variable_label(X_VARIABLE, standardized=standardized))
    ax.set_ylabel(variable_label(y_variable, standardized=standardized))
    if y_variable == "tas_peak" and not standardized:
        set_data_driven_y_limits(ax, y_values[finite])
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


def validate_feature_variables(
    features: xr.Dataset,
    *,
    selection_variable: str,
) -> None:
    """Fail clearly when the requested feature table lacks required variables."""
    required = [X_VARIABLE]
    for name in Y_VARIABLES:
        required.extend(source_variable_names(name))
    required.extend(source_variable_names(selection_variable))
    missing = [name for name in dict.fromkeys(required) if name not in features]
    if missing:
        raise ValueError(
            "Event-feature table is missing required variables: "
            f"{', '.join(missing)}."
        )


def validate_selection_quantile(value: float | None) -> float:
    """Return a valid selection quantile."""
    if value is None:
        raise ValueError("--selection-quantile is required.")
    quantile = float(value)
    if not 0.0 < quantile < 1.0:
        raise ValueError("--selection-quantile must be strictly between 0 and 1.")
    return quantile


def build_quantile_split(
    features: xr.Dataset,
    *,
    selection_variable: str,
    selection_quantile: float,
) -> QuantileSplit:
    """Return low/high event masks split by one feature quantile."""
    if selection_variable not in Y_VARIABLES:
        allowed = ", ".join(Y_VARIABLES)
        raise ValueError(
            "selection_variable must be one of the plotted y-variables: "
            f"{allowed}."
        )
    quantile = validate_selection_quantile(selection_quantile)
    values = feature_values(features, selection_variable)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(
            f"Selection variable {selection_variable!r} has no finite values."
        )

    threshold = float(np.nanquantile(values[finite], quantile))
    low_mask = finite & (values <= threshold)
    high_mask = finite & (values > threshold)
    if not low_mask.any() or not high_mask.any():
        raise ValueError(
            f"Selection quantile {quantile:g} for {selection_variable!r} "
            "does not create two non-empty groups."
        )
    return QuantileSplit(
        variable=selection_variable,
        quantile=quantile,
        threshold=threshold,
        values=values,
        low_mask=low_mask,
        high_mask=high_mask,
    )


def split_group_mask(split: QuantileSplit, group_name: str) -> np.ndarray:
    """Return the event mask for one split group."""
    if group_name == "low":
        return split.low_mask
    if group_name == "high":
        return split.high_mask
    raise ValueError(f"Unknown split group: {group_name}")


def split_group_statistics(values: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    """Return mean, standard deviation, and count for finite masked values."""
    group_values = np.asarray(values, dtype=float)[mask]
    finite_values = group_values[np.isfinite(group_values)]
    if finite_values.size == 0:
        return np.nan, np.nan, 0
    return (
        float(np.mean(finite_values)),
        float(np.std(finite_values)),
        int(finite_values.size),
    )


def add_group_summary_overlay(
    ax: Axes,
    y_values: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    group_name: str,
    label: str,
) -> None:
    """Add horizontal group mean and mean +/- std band to one panel."""
    mean, std, count = split_group_statistics(y_values, mask)
    if count == 0 or not np.isfinite(mean):
        return

    band_lower = mean - std if np.isfinite(std) else mean
    band_upper = mean + std if np.isfinite(std) else mean
    band = ax.axhspan(
        band_lower,
        band_upper,
        color=color,
        alpha=SUMMARY_ALPHA,
        linewidth=0,
        zorder=1,
    )
    band.set_gid(f"{group_name}_std")
    line = ax.axhline(
        mean,
        color=color,
        linewidth=plot_style.LINE_WIDTH_PT,
        linestyle="-",
        zorder=3,
        label=f"{label} mean",
    )
    line.set_gid(f"{group_name}_mean")


def feature_values(features: xr.Dataset, variable: str | None) -> np.ndarray:
    """Return a feature variable as a float array."""
    if variable is None:
        raise ValueError("variable must not be None.")
    if variable in HEAT_BUDGET_Y_VARIABLES:
        return heat_budget_fraction_values(features, variable)

    values = features[source_variable_name(variable)].values
    if np.issubdtype(values.dtype, np.timedelta64):
        out = values / np.timedelta64(1, "D")
    else:
        out = np.asarray(values, dtype=float)
    if variable == "sqrt_I_lwa_a_pre_peak":
        out = np.where(out >= 0.0, np.sqrt(out), np.nan)
    elif variable == "cos_days_from_solstice":
        out = np.cos(out * 2.0 * np.pi / 365.0)
    return out


def heat_budget_fraction_values(features: xr.Dataset, variable: str) -> np.ndarray:
    """Return one heat-budget term as a fraction of summed absolute budget terms."""
    numerator_name = {
        "f_adiabatic_pre": "I_adiabatic_pre",
        "f_diabatic_pre": "I_diabatic_pre",
        "f_advection_pre": "I_advection_pre",
    }[variable]
    numerator = np.asarray(features[numerator_name].values, dtype=float)
    denominator = np.zeros_like(numerator, dtype=float)
    for source_name in INTEGRATED_HEAT_BUDGET_VARIABLES:
        denominator = denominator + np.abs(
            np.asarray(features[source_name].values, dtype=float)
        )
    out = np.full(numerator.shape, np.nan, dtype=float)
    np.divide(numerator, denominator, out=out, where=denominator != 0.0)
    return out


def standardized_panel_values(
    values: np.ndarray,
    threshold: float | None,
) -> tuple[np.ndarray, float | None]:
    """Return standardized values and optional threshold on the same scale."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    if not finite.any():
        return np.full(values.shape, np.nan, dtype=float), np.nan
    mean = float(np.nanmean(values[finite]))
    std = float(np.nanstd(values[finite]))
    if std == 0.0 or not np.isfinite(std):
        return np.full(values.shape, np.nan, dtype=float), np.nan
    out = (values - mean) / std
    if threshold is None:
        return out, None
    return out, (float(threshold) - mean) / std


def standardized_values(values: np.ndarray) -> np.ndarray:
    """Return z-scored values using finite-sample mean and standard deviation."""
    out, _ = standardized_panel_values(values, None)
    return out


def source_variable_name(variable: str) -> str:
    """Return the dataset variable needed for a plot variable."""
    source = DERIVED_VARIABLE_SOURCES.get(variable, variable)
    if isinstance(source, tuple):
        raise ValueError(f"{variable!r} maps to multiple source variables.")
    return source


def source_variable_names(variable: str) -> tuple[str, ...]:
    """Return all dataset variables needed for a plot variable."""
    source = DERIVED_VARIABLE_SOURCES.get(variable, variable)
    if isinstance(source, tuple):
        return source
    return (source,)


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


def set_data_driven_y_limits(ax: Axes, values: np.ndarray, *, pad_fraction: float = 0.08) -> None:
    """Set y-limits from finite data values, ignoring reference-line artists."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    span = upper - lower
    if span == 0.0:
        pad = max(abs(upper) * pad_fraction, 1.0)
    else:
        pad = span * pad_fraction
    ax.set_ylim(lower - pad, upper + pad)


def _split_output_path(output_path: Path, selection_variable: str) -> Path:
    """Return output path with the selection-variable token added to the stem."""
    token = _filename_token(selection_variable)
    if output_path.stem.endswith(f"_{token}"):
        return output_path
    return output_path.with_name(f"{output_path.stem}_{token}{output_path.suffix}")


def _filename_token(value: str) -> str:
    """Return a conservative filename token."""
    token = "".join(
        char.lower() if char.isascii() and char.isalnum() else "_"
        for char in value.strip()
    )
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("selection variable must contain a filename-safe character.")
    return token


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def standardized_output_path(path: str | Path) -> Path:
    """Return an output path with _standardized before the file suffix."""
    output_path = Path(path)
    return output_path.with_name(f"{output_path.stem}_standardized{output_path.suffix}")


if __name__ == "__main__":
    raise SystemExit(main())
