"""Compare signed pre-peak heat-budget fractions across Stage-2 regions."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

from src import diagnostics, plot_style


EVENT_DIM = "event"
BASELINE_DIM = "baseline_day"
EVENT_ADJACENT_VARIABLE = "event_adjacent"
EVENT_PIPELINE_STAGE = "stage_2_event_features"
BASELINE_PIPELINE_STAGE = "stage_2_baseline_features"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "results"
    / "stage2_baseline_features"
    / "diagnostics"
    / "inter_region"
    / "inter_region_budget_fractions.png"
)
DEFAULT_TITLE = "Regional Pre-Peak Heat-Budget Composition"
COMPATIBILITY_ATTRIBUTES = (
    "integral_method",
    "window_endpoint_inclusion",
    "heat_budget_pre_window_hours",
)
FRACTION_PANELS = (
    ("f_adiabatic", "Adiabatic", plot_style.COLORS["adiabatic"]),
    ("f_advection", "Advection", plot_style.COLORS["advection"]),
    ("f_dyn", "Net dynamical", plot_style.COLORS["calculated"]),
    ("f_diabatic", "Diabatic", plot_style.COLORS["diabatic"]),
)
BASELINE_COLOR = "#7A7A7A"
EVENT_OFFSET = -0.13
BASELINE_OFFSET = 0.13


@dataclass(frozen=True)
class RegionInput:
    """Explicit Stage-2 event and baseline input paths for one region."""

    region: str
    event_path: Path
    baseline_path: Path


@dataclass(frozen=True)
class RegionalBudgetSummary:
    """Event and clean-baseline budget summaries for one region."""

    region: str
    events: xr.Dataset
    baseline: xr.Dataset
    compatibility_signature: tuple[tuple[str, str], ...]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse explicit regional Stage-2 inputs and the output path."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot signed gross-activity-normalized heat-budget fractions "
            "for multiple Stage-2 regions."
        )
    )
    parser.add_argument(
        "--region-input",
        action="append",
        nargs=3,
        required=True,
        metavar=("REGION", "EVENT_PATH", "BASELINE_PATH"),
        help=(
            "Region identifier followed by its Stage-2 event and baseline "
            "NetCDF paths. Repeat once per region in the desired row order."
        ),
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT_PATH}).",
    )
    parser.add_argument(
        "--title",
        default=DEFAULT_TITLE,
        help=f"Figure title (default: {DEFAULT_TITLE!r}).",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Load Stage-2 tables and write the inter-region comparison figure."""
    args = parse_args(argv)
    inputs = normalize_region_inputs(args.region_input)
    summaries = load_regional_budget_summaries(inputs)
    written = write_inter_region_budget_figure(
        summaries,
        args.output_path,
        title=args.title,
    )
    print(f"Wrote inter-region heat-budget fraction figure: {written}")
    return 0


def normalize_region_inputs(raw_inputs: Sequence[Sequence[str]]) -> tuple[RegionInput, ...]:
    """Validate repeated CLI triplets while preserving the requested order."""
    normalized: list[RegionInput] = []
    seen: set[str] = set()
    for raw in raw_inputs:
        if len(raw) != 3:
            raise ValueError(
                "Each --region-input requires REGION EVENT_PATH BASELINE_PATH."
            )
        region, event_path, baseline_path = raw
        if region not in plot_style.REGION_NAME_MAPPING:
            available = ", ".join(plot_style.REGION_NAME_MAPPING)
            raise ValueError(
                f"Unknown region {region!r}; expected one of: {available}."
            )
        if region in seen:
            raise ValueError(f"Region {region!r} was supplied more than once.")
        seen.add(region)
        normalized.append(
            RegionInput(
                region=region,
                event_path=Path(event_path).expanduser().resolve(),
                baseline_path=Path(baseline_path).expanduser().resolve(),
            )
        )
    if not normalized:
        raise ValueError("At least one --region-input is required.")
    return tuple(normalized)


def load_regional_budget_summaries(
    inputs: Sequence[RegionInput],
) -> tuple[RegionalBudgetSummary, ...]:
    """Load and reduce each explicit regional pair, then check compatibility."""
    summaries: list[RegionalBudgetSummary] = []
    for region_input in inputs:
        with open_stage2_table(region_input.event_path) as event_features:
            with open_stage2_table(region_input.baseline_path) as baseline_features:
                summaries.append(
                    prepare_regional_budget_summary(
                        region_input.region,
                        event_features,
                        baseline_features,
                    )
                )
    validate_compatible_regional_summaries(summaries)
    return tuple(summaries)


def open_stage2_table(path: str | Path) -> xr.Dataset:
    """Open one explicit Stage-2 NetCDF table."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Stage-2 input does not exist: {resolved}")
    return xr.open_dataset(resolved, engine="h5netcdf", decode_timedelta=True)


def prepare_regional_budget_summary(
    region: str,
    event_features: xr.Dataset,
    baseline_features: xr.Dataset,
) -> RegionalBudgetSummary:
    """Build event and clean-baseline summaries for one region."""
    validate_pipeline_stage(
        event_features,
        expected=EVENT_PIPELINE_STAGE,
        table_label="Event-feature",
    )
    validate_pipeline_stage(
        baseline_features,
        expected=BASELINE_PIPELINE_STAGE,
        table_label="Baseline-day",
    )
    event_signature = budget_compatibility_signature(event_features)
    baseline_signature = budget_compatibility_signature(baseline_features)
    if event_signature != baseline_signature:
        raise ValueError(
            f"Event and baseline Stage-2 budget metadata differ for {region!r}: "
            f"event={dict(event_signature)}, baseline={dict(baseline_signature)}."
        )

    if EVENT_DIM not in event_features.sizes or event_features.sizes[EVENT_DIM] == 0:
        raise ValueError(f"Event-feature table for {region!r} contains no rows.")
    if BASELINE_DIM not in baseline_features.sizes:
        raise ValueError(
            f"Baseline-day table for {region!r} is missing {BASELINE_DIM!r}."
        )
    if EVENT_ADJACENT_VARIABLE not in baseline_features:
        raise ValueError(
            f"Baseline-day table for {region!r} is missing "
            f"{EVENT_ADJACENT_VARIABLE!r}."
        )
    if baseline_features[EVENT_ADJACENT_VARIABLE].dims != (BASELINE_DIM,):
        raise ValueError(
            f"{EVENT_ADJACENT_VARIABLE} must have only the {BASELINE_DIM!r} "
            "dimension."
        )

    clean_mask = np.asarray(
        baseline_features[EVENT_ADJACENT_VARIABLE].values
    ) == 0
    if not np.any(clean_mask):
        raise ValueError(f"Baseline-day table for {region!r} has no clean rows.")
    clean_baseline = baseline_features.isel(
        {BASELINE_DIM: np.flatnonzero(clean_mask)}
    )

    event_fractions = diagnostics.derive_signed_budget_fractions(
        event_features,
        row_dim=EVENT_DIM,
    )
    baseline_fractions = diagnostics.derive_signed_budget_fractions(
        clean_baseline,
        row_dim=BASELINE_DIM,
    )
    return RegionalBudgetSummary(
        region=region,
        events=diagnostics.summarize_budget_fraction_distributions(
            event_fractions,
            row_dim=EVENT_DIM,
        ).load(),
        baseline=diagnostics.summarize_budget_fraction_distributions(
            baseline_fractions,
            row_dim=BASELINE_DIM,
        ).load(),
        compatibility_signature=event_signature,
    )


def validate_pipeline_stage(
    features: xr.Dataset,
    *,
    expected: str,
    table_label: str,
) -> None:
    """Require the documented Stage-2 product marker."""
    actual = features.attrs.get("pipeline_stage")
    if actual != expected:
        raise ValueError(
            f"{table_label} table must have pipeline_stage={expected!r}; "
            f"found {actual!r}."
        )


def budget_compatibility_signature(
    features: xr.Dataset,
) -> tuple[tuple[str, str], ...]:
    """Return the Stage-2 metadata that must match across plotted regions."""
    values: list[tuple[str, str]] = []
    for attribute in COMPATIBILITY_ATTRIBUTES:
        value = features.attrs.get(attribute)
        if value in {None, ""}:
            raise ValueError(
                f"Stage-2 table is missing compatibility attribute {attribute!r}."
            )
        values.append((attribute, str(value)))
    return tuple(values)


def validate_compatible_regional_summaries(
    summaries: Sequence[RegionalBudgetSummary],
) -> None:
    """Require one common Stage-2 window and integration contract."""
    if not summaries:
        raise ValueError("At least one regional summary is required.")
    reference = summaries[0]
    for summary in summaries[1:]:
        if summary.compatibility_signature != reference.compatibility_signature:
            raise ValueError(
                "Regional Stage-2 budget metadata are incompatible: "
                f"{reference.region}={dict(reference.compatibility_signature)}, "
                f"{summary.region}={dict(summary.compatibility_signature)}."
            )


def write_inter_region_budget_figure(
    summaries: Sequence[RegionalBudgetSummary],
    output_path: str | Path,
    *,
    title: str = DEFAULT_TITLE,
) -> Path:
    """Write a non-overwriting inter-region budget-composition PNG."""
    resolved = Path(output_path).expanduser().resolve()
    if resolved.exists():
        raise FileExistsError(f"Output path already exists: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    figure = plot_inter_region_budget_fractions(summaries, title=title)
    try:
        plot_style.save_figure(figure, resolved)
    finally:
        plt.close(figure)
    return resolved


def plot_inter_region_budget_fractions(
    summaries: Sequence[RegionalBudgetSummary],
    *,
    title: str = DEFAULT_TITLE,
) -> Figure:
    """Return the common-axis regional fraction and activity figure."""
    validate_compatible_regional_summaries(summaries)
    region_names = [summary.region for summary in summaries]
    if len(region_names) != len(set(region_names)):
        raise ValueError("Regional summaries must contain unique region names.")

    figure, axes = plt.subplots(
        nrows=1,
        ncols=5,
        sharey=True,
        figsize=plot_style.publication_figsize("full", aspect=0.56),
        gridspec_kw={"width_ratios": (1.0, 1.0, 1.0, 1.0, 1.15)},
        constrained_layout=False,
    )
    figure.subplots_adjust(
        left=0.17,
        right=0.985,
        bottom=0.17,
        top=0.78,
        wspace=0.24,
    )
    axes = np.asarray(axes)
    y_positions = np.arange(len(summaries), dtype=float)

    for axis, (variable, panel_title, color) in zip(axes[:4], FRACTION_PANELS):
        plot_summary_panel(
            axis,
            summaries,
            variable=variable,
            color=color,
            y_positions=y_positions,
        )
        axis.axvline(
            0.0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            zorder=0,
        )
        axis.set_xlim(-1.05, 1.05)
        axis.set_xticks((-1.0, 0.0, 1.0))
        axis.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        plot_style.use_default_numeric_formatter(axis.xaxis)
        axis.set_title(panel_title, color=color)
        axis.set_xlabel("Signed fraction")
        plot_style.style_axis(axis)

    activity_axis = axes[4]
    plot_summary_panel(
        activity_axis,
        summaries,
        variable=diagnostics.GROSS_BUDGET_ACTIVITY,
        color=plot_style.COLORS["mass"],
        y_positions=y_positions,
    )
    activity_values = np.concatenate(
        [
            np.asarray(summary.events[diagnostics.GROSS_BUDGET_ACTIVITY].values)
            for summary in summaries
        ]
        + [
            np.asarray(summary.baseline[diagnostics.GROSS_BUDGET_ACTIVITY].values)
            for summary in summaries
        ]
    )
    activity_limits = plot_style.padded_data_limits(
        activity_values,
        required_values=(0.0,),
    )
    if activity_limits is not None:
        activity_axis.set_xlim(0.0, activity_limits[1])
    activity_axis.set_title("Gross activity", color=plot_style.COLORS["mass"])
    activity_axis.set_xlabel("K")
    plot_style.style_axis(activity_axis)

    display_names = [plot_style.REGION_NAME_MAPPING[name] for name in region_names]
    axes[0].set_yticks(y_positions, labels=display_names)
    axes[0].invert_yaxis()
    axes[0].set_ylabel("Region")
    for axis in axes:
        plot_style.use_default_numeric_formatter(axis.yaxis)
    figure.suptitle(title, y=0.965)
    figure.legend(
        handles=population_legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=2,
        **plot_style.legend_kwargs(frameon=False),
    )
    figure.text(
        0.5,
        0.025,
        (
            "Dots are medians; thick intervals are P25-P75; thin intervals "
            "are P10-P90. Positive fractions heat and negative fractions cool."
        ),
        ha="center",
        va="bottom",
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
    )
    return figure


def plot_summary_panel(
    axis: Axes,
    summaries: Sequence[RegionalBudgetSummary],
    *,
    variable: str,
    color: str,
    y_positions: np.ndarray,
) -> None:
    """Draw event and clean-baseline quantile glyphs for one quantity."""
    for y_position, summary in zip(y_positions, summaries):
        plot_quantile_glyph(
            axis,
            summary.events[variable],
            y=y_position + EVENT_OFFSET,
            color=color,
            filled=True,
            zorder=3,
        )
        plot_quantile_glyph(
            axis,
            summary.baseline[variable],
            y=y_position + BASELINE_OFFSET,
            color=BASELINE_COLOR,
            filled=False,
            zorder=2,
        )


def plot_quantile_glyph(
    axis: Axes,
    summary: xr.DataArray,
    *,
    y: float,
    color: str,
    filled: bool,
    zorder: int,
) -> None:
    """Draw P10-P90, P25-P75, and median for one distribution."""
    values = quantile_values(summary)
    axis.hlines(
        y,
        values[0],
        values[4],
        color=color,
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
        zorder=zorder,
    )
    axis.hlines(
        y,
        values[1],
        values[3],
        color=color,
        linewidth=4.0,
        zorder=zorder,
    )
    axis.scatter(
        values[2],
        y,
        s=38.0,
        facecolors=color if filled else "white",
        edgecolors=color,
        linewidths=1.1,
        zorder=zorder + 1,
    )


def quantile_values(summary: xr.DataArray) -> np.ndarray:
    """Return the documented five summary quantiles in canonical order."""
    if summary.dims != ("quantile",):
        raise ValueError(
            f"Summary must use only the 'quantile' dimension; found {summary.dims}."
        )
    quantiles = np.asarray(summary["quantile"].values, dtype=float)
    expected = np.asarray(diagnostics.BUDGET_SUMMARY_QUANTILES)
    if not np.array_equal(quantiles, expected):
        raise ValueError(
            f"Expected summary quantiles {expected.tolist()}; "
            f"found {quantiles.tolist()}."
        )
    values = np.asarray(summary.values, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Summary quantiles must all be finite.")
    return values


def population_legend_handles() -> tuple[Line2D, Line2D]:
    """Return shared population glyphs for the figure legend."""
    events = Line2D(
        [],
        [],
        color=plot_style.COLORS["calculated"],
        marker="o",
        markerfacecolor=plot_style.COLORS["calculated"],
        markersize=6,
        linewidth=3.0,
        label="Heatwave events",
    )
    baseline = Line2D(
        [],
        [],
        color=BASELINE_COLOR,
        marker="o",
        markerfacecolor="white",
        markersize=6,
        linewidth=3.0,
        label="Clean baseline days",
    )
    return events, baseline


if __name__ == "__main__":
    raise SystemExit(main())
