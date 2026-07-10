"""Plot Stage-2 event-feature distributions split by configured feature quantiles."""

from __future__ import annotations

import argparse
import sys
import textwrap
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
from matplotlib.patches import Patch

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
    / "event_feature_split_violin_combined.png"
)

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

SPLIT_SPECS = tuple((variable, 0.9) for variable in Y_VARIABLES)

TOTAL_COLOR = plot_style.COLORS["calculated"]
LOW_COLOR = plot_style.COLORS["volume"]
HIGH_COLOR = plot_style.COLORS["diabatic"]
VIOLIN_ALPHA = 0.72
TOTAL_POSITION = 0.0
SPLIT_POSITION_OFFSET = 0.18
TOTAL_VIOLIN_WIDTH = 0.36
SPLIT_VIOLIN_WIDTH = 0.30
TICK_LABEL_WRAP_WIDTH = 18


@dataclass(frozen=True)
class SplitSpec:
    """One configured split variable and quantile."""

    variable: str
    quantile: float


@dataclass(frozen=True)
class QuantileSplit:
    """Event masks and metadata for one split-variable quantile threshold."""

    spec: SplitSpec
    threshold: float
    values: np.ndarray
    low_mask: np.ndarray
    high_mask: np.ndarray

    @property
    def variable(self) -> str:
        """Return the split variable name."""
        return self.spec.variable

    @property
    def quantile(self) -> float:
        """Return the split quantile."""
        return self.spec.quantile


def parse_args() -> argparse.Namespace:
    """Parse command-line options for combined split-violin diagnostics."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot event-feature distributions as row-wise violin plots. Split "
            "variables and quantiles are configured in SPLIT_SPECS."
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
        help="Path where the combined split-violin PNG will be written.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate CLI arguments."""
    if not Path(args.input_path).expanduser().suffix:
        raise ValueError("--input-path must include a filename.")
    if not Path(args.output_path).expanduser().suffix:
        raise ValueError("--output-path must include a filename.")


def main() -> int:
    """Load event features and write combined split-violin diagnostics."""
    args = parse_args()
    validate_args(args)

    features = open_event_features(args.input_path)
    try:
        written = write_split_violin_plot(features, args.output_path)
        print("Wrote combined split event-feature violin figure:")
        print(f"  {_display_path(written)}")
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


def write_split_violin_plot(
    features: xr.Dataset,
    output_path: str | Path,
    *,
    split_specs: tuple[SplitSpec | tuple[str, float], ...] = SPLIT_SPECS,
) -> Path:
    """Write the combined split-violin figure."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_split_violin_combined(features, split_specs=split_specs)
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_split_violin_combined(
    features: xr.Dataset,
    *,
    split_specs: tuple[SplitSpec | tuple[str, float], ...] = SPLIT_SPECS,
) -> plt.Figure:  # type: ignore[type-arg]
    """Return one violin-plot row per configured y variable."""
    splits = build_quantile_splits(features, split_specs=split_specs)
    validate_feature_variables(features, split_specs=tuple(split.spec for split in splits))

    nrows = len(Y_VARIABLES)
    ncols = max(len(splits) + 1, 1)
    fig_width = plot_style.FULL_TWO_COLUMN_WIDTH_IN
    fig_height = max(12.0, 1.75 * nrows + 1.4)
    fig, axes_grid = plt.subplots(
        nrows=nrows,
        ncols=1,
        figsize=(fig_width, fig_height),
        sharex=True,
        constrained_layout=True,
        squeeze=False,
    )
    axes = axes_grid.ravel()

    for index, (ax, y_variable) in enumerate(zip(axes, Y_VARIABLES)):
        plot_one_split_violin_row(
            ax,
            features,
            y_variable,
            splits=splits,
            show_xlabel=index == nrows - 1,
        )

    handles = legend_handles(splits)
    if handles:
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=len(handles),
            **plot_style.legend_kwargs(),
        )
    fig.suptitle("Event Fixed-Window Feature Distributions by Split Population")
    return fig


def plot_one_split_violin_row(
    ax: Axes,
    features: xr.Dataset,
    y_variable: str,
    *,
    splits: tuple[QuantileSplit, ...],
    show_xlabel: bool,
) -> None:
    """Plot one y-variable distribution row across all split populations."""
    y_values = feature_values(features, y_variable)
    finite_y = np.isfinite(y_values)

    add_violin(
        ax,
        y_values[finite_y],
        position=TOTAL_POSITION,
        width=TOTAL_VIOLIN_WIDTH,
        color=TOTAL_COLOR,
        gid=f"violin_total_{y_variable}",
    )

    for split_index, split in enumerate(splits, start=1):
        low_mask = finite_y & split.low_mask
        high_mask = finite_y & split.high_mask
        add_violin(
            ax,
            y_values[low_mask],
            position=split_index - SPLIT_POSITION_OFFSET,
            width=SPLIT_VIOLIN_WIDTH,
            color=LOW_COLOR,
            gid=f"violin_low_{split.variable}_{y_variable}",
        )
        add_violin(
            ax,
            y_values[high_mask],
            position=split_index + SPLIT_POSITION_OFFSET,
            width=SPLIT_VIOLIN_WIDTH,
            color=HIGH_COLOR,
            gid=f"violin_high_{split.variable}_{y_variable}",
        )

    add_horizontal_zero_line(ax)
    ax.set_title(PANEL_TITLES[y_variable], loc="left")
    ax.set_ylabel(variable_label(y_variable))
    ax.set_axisbelow(True)
    ax.text(
        0.01,
        0.94,
        f"n = {int(finite_y.sum())}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.85},
    )
    configure_x_axis(ax, splits=splits, show_xlabel=show_xlabel)
    if y_variable == "tas_peak":
        set_data_driven_y_limits(ax, y_values[finite_y])
    plot_style.style_axis(ax)


def validate_feature_variables(
    features: xr.Dataset,
    *,
    split_specs: tuple[SplitSpec | tuple[str, float], ...] = SPLIT_SPECS,
) -> None:
    """Fail clearly when the feature table lacks required plot or split variables."""
    missing_y = [name for name in Y_VARIABLES if not can_resolve_feature(features, name)]
    if missing_y:
        raise ValueError(
            "Event-feature table is missing required plotted variables: "
            f"{', '.join(missing_y)}."
        )

    invalid_splits = []
    for spec in normalize_split_specs(split_specs):
        if not can_resolve_feature(features, spec.variable):
            invalid_splits.append(spec.variable)
    if invalid_splits:
        raise ValueError(
            "Event-feature table is missing required split variables: "
            f"{', '.join(invalid_splits)}."
        )


def normalize_split_specs(
    split_specs: tuple[SplitSpec | tuple[str, float], ...],
) -> tuple[SplitSpec, ...]:
    """Return configured split specs as validated dataclass instances."""
    specs: list[SplitSpec] = []
    for raw_spec in split_specs:
        if isinstance(raw_spec, SplitSpec):
            variable = raw_spec.variable
            quantile = raw_spec.quantile
        else:
            try:
                variable, quantile = raw_spec
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "Each split spec must be a SplitSpec or a "
                    "(variable, quantile) pair."
                ) from exc
        if not str(variable).strip():
            raise ValueError("Split variable names must be non-empty.")
        specs.append(SplitSpec(str(variable), validate_split_quantile(quantile)))
    return tuple(specs)


def validate_split_quantile(value: float | None) -> float:
    """Return a valid split quantile."""
    if value is None:
        raise ValueError("Split quantile is required.")
    quantile = float(value)
    if not 0.0 < quantile < 1.0:
        raise ValueError("Split quantile must be strictly between 0 and 1.")
    return quantile


def build_quantile_splits(
    features: xr.Dataset,
    *,
    split_specs: tuple[SplitSpec | tuple[str, float], ...] = SPLIT_SPECS,
) -> tuple[QuantileSplit, ...]:
    """Return low/high event masks for all configured split specs."""
    validate_feature_variables(features, split_specs=split_specs)
    return tuple(
        build_quantile_split(features, split_spec=spec)
        for spec in normalize_split_specs(split_specs)
    )


def build_quantile_split(
    features: xr.Dataset,
    *,
    split_spec: SplitSpec | tuple[str, float],
) -> QuantileSplit:
    """Return low/high event masks split by one feature quantile."""
    spec = normalize_split_specs((split_spec,))[0]
    values = feature_values(features, spec.variable)
    finite = np.isfinite(values)
    if not finite.any():
        raise ValueError(f"Split variable {spec.variable!r} has no finite values.")

    threshold = float(np.nanquantile(values[finite], spec.quantile))
    low_mask = finite & (values <= threshold)
    high_mask = finite & (values > threshold)
    if not low_mask.any() or not high_mask.any():
        raise ValueError(
            f"Split quantile {spec.quantile:g} for {spec.variable!r} "
            "does not create two non-empty groups."
        )
    return QuantileSplit(
        spec=spec,
        threshold=threshold,
        values=values,
        low_mask=low_mask,
        high_mask=high_mask,
    )


def can_resolve_feature(features: xr.Dataset, variable: str) -> bool:
    """Return True when a feature can be resolved to a finite-compatible vector."""
    try:
        feature_values(features, variable)
    except (KeyError, TypeError, ValueError):
        return False
    return True


def feature_values(features: xr.Dataset, variable: str | None) -> np.ndarray:
    """Return a feature variable as a one-dimensional float array."""
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
    out = np.asarray(out, dtype=float)
    if out.ndim != 1:
        raise ValueError(f"Feature variable {variable!r} is not one-dimensional.")
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
    if out.ndim != 1:
        raise ValueError(f"Feature variable {variable!r} is not one-dimensional.")
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


def variable_label(variable: str) -> str:
    """Return a readable axis or tick label."""
    return VARIABLE_LABELS.get(variable, variable)


def add_violin(
    ax: Axes,
    values: np.ndarray,
    *,
    position: float,
    width: float,
    color: str,
    gid: str,
) -> None:
    """Add one violin, or a collapsed line for a degenerate distribution."""
    finite_values = np.asarray(values, dtype=float)
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size == 0:
        return

    if finite_values.size < 2 or np.nanmin(finite_values) == np.nanmax(finite_values):
        value = float(finite_values[0])
        line = ax.hlines(
            value,
            position - width / 2.0,
            position + width / 2.0,
            color=color,
            linewidth=plot_style.LINE_WIDTH_PT,
            zorder=3,
        )
        line.set_gid(gid)
        return

    parts = ax.violinplot(
        [finite_values],
        positions=[position],
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    body = parts["bodies"][0] #type: ignore
    body.set_facecolor(color)
    body.set_edgecolor(plot_style.COLORS["calculated"])
    body.set_alpha(VIOLIN_ALPHA)
    body.set_linewidth(0.7)
    body.set_gid(gid)


def add_horizontal_zero_line(ax: Axes) -> None:
    """Add a horizontal zero reference line."""
    ax.axhline(
        0.0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        linestyle="--",
        zorder=0,
    )


def configure_x_axis(
    ax: Axes,
    *,
    splits: tuple[QuantileSplit, ...],
    show_xlabel: bool,
) -> None:
    """Configure shared split-population x ticks."""
    tick_positions = np.arange(len(splits) + 1, dtype=float)
    tick_labels = ["All events", *[split_tick_label(split) for split in splits]]
    ax.set_xlim(-0.55, max(len(splits), 0) + 0.55)
    ax.set_xticks(tick_positions)
    if show_xlabel:
        ax.set_xticklabels(tick_labels, rotation=35, ha="right", rotation_mode="anchor")
        ax.set_xlabel("Split population")
    else:
        ax.tick_params(labelbottom=False)


def split_tick_label(split: QuantileSplit) -> str:
    """Return a wrapped tick label for one split variable."""
    label = f"{variable_label(split.variable)} (q={split.quantile:g})"
    return "\n".join(textwrap.wrap(label, width=TICK_LABEL_WRAP_WIDTH))


def legend_handles(splits: tuple[QuantileSplit, ...]) -> list[Patch]:
    """Return figure legend handles for total, lower, and upper populations."""
    handles = [
        Patch(
            facecolor=TOTAL_COLOR,
            edgecolor=plot_style.COLORS["calculated"],
            label="Total population",
        )
    ]
    if not splits:
        return handles

    quantiles = {split.quantile for split in splits}
    if len(quantiles) == 1:
        quantile = next(iter(quantiles))
        low_label, high_label = percentile_group_labels(quantile)
    else:
        low_label = "Lower percentile group"
        high_label = "Upper percentile group"
    handles.extend(
        [
            Patch(
                facecolor=LOW_COLOR,
                edgecolor=plot_style.COLORS["calculated"],
                label=low_label,
            ),
            Patch(
                facecolor=HIGH_COLOR,
                edgecolor=plot_style.COLORS["calculated"],
                label=high_label,
            ),
        ]
    )
    return handles


def percentile_group_labels(quantile: float) -> tuple[str, str]:
    """Return labels for lower and upper percentile groups."""
    percentile = quantile * 100.0
    percentile_text = f"{percentile:g}"
    return (
        f"0-{percentile_text}th percentile",
        f"{percentile_text}-100th percentile",
    )


def set_data_driven_y_limits(
    ax: Axes,
    values: np.ndarray,
    *,
    pad_fraction: float = 0.08,
) -> None:
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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
