"""Explore deterministic matching of positive and negative I_dyn_pre events.

This is an isolated Stage-2 exploration. It consumes an existing event-feature
table, reads its canonical ``I_dyn_pre`` variable, and compares the
unmatched sign populations with deterministic one-to-one optimal matches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.Idyn_matching_exploration import matching_settings  # noqa: E402
from src import plot_style, selectors  # noqa: E402


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results/stage2_event_features"
    / "hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/Idyn_matching_exploration"

OVERVIEW_FILENAME = "idyn_population_overview.png"
MATCHING_FILENAME = "tas_anom_matching_diagnostics.png"
BALANCE_FILENAME = "covariate_balance_and_sensitivity.png"
COMPARISON_FILENAME = "matching_specification_tradeoff.png"

POSITIVE_COLOR = plot_style.COLORS["diabatic"]
NEGATIVE_COLOR = plot_style.COLORS["advection"]
MATCHED_COLOR = plot_style.COLORS["temperature_tendency"]

VARIABLE_LABELS = {
    "tas_anom_peak": "Peak temperature anomaly",
    "tas_peak": "Peak temperature",
    "tas_excess_peak": "Peak threshold excess",
    "tas_excess_integral": "Integrated threshold excess",
    "duration": "Duration",
    "days_from_solstice": "Days from June 21",
    "T_anom_mean_ant": "Antecedent mean anomaly",
    "I_dTdt_pre": "Integrated temperature change",
}


@dataclass(frozen=True)
class Exploration:
    """Prepared arrays, primary matches, and balance diagnostics."""

    settings: matching_settings.MatchingSettings
    event_ids: np.ndarray
    values: Mapping[str, np.ndarray]
    i_dyn: np.ndarray
    negative_indices: np.ndarray
    positive_indices: np.ndarray
    primary_match: selectors.SignMatchResult
    balance: Mapping[str, Mapping[str, float]]
    specification_matches: Mapping[str, selectors.SignMatchResult]
    frontier_matches: Mapping[
        str,
        Mapping[float, selectors.SignMatchResult],
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore matching positive and negative Stage-2 I_dyn_pre events."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help="Stage-2 event-feature NetCDF product.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for exploratory PNG figures.",
    )
    parser.add_argument(
        "--settings-path",
        type=Path,
        default=matching_settings.DEFAULT_SETTINGS_PATH,
        help="Tracked JSON file defining matching methods and SD calipers.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing exploratory figures.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.input_path.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")
    if not args.settings_path.expanduser().resolve().is_file():
        raise FileNotFoundError(
            f"Matching settings file does not exist: {args.settings_path}"
        )

    existing = [
        path
        for path in output_paths(args.output_dir).values()
        if path.exists()
    ]
    if existing and not args.overwrite:
        listed = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Output figure(s) already exist: {listed}. Pass --overwrite."
        )


def output_paths(output_dir: str | Path) -> dict[str, Path]:
    root = Path(output_dir).expanduser().resolve()
    return {
        "overview": root / OVERVIEW_FILENAME,
        "matching": root / MATCHING_FILENAME,
        "balance": root / BALANCE_FILENAME,
        "comparison": root / COMPARISON_FILENAME,
    }


def open_event_features(path: str | Path) -> xr.Dataset:
    input_path = Path(path).expanduser().resolve()
    with xr.open_dataset(
        input_path,
        engine="h5netcdf",
        decode_timedelta=True,
    ) as ds:
        return ds.load()


def prepare_exploration(
    features: xr.Dataset,
    *,
    settings: matching_settings.MatchingSettings,
) -> Exploration:
    """Validate a Stage-2 table and prepare the matching diagnostics."""
    configured_variables = {
        variable
        for family in settings.families.values()
        for variable in family.variables
    }
    analysis_variables = set(settings.balance_variables).union(configured_variables)
    required = {
        "event_id",
        settings.group_variable,
        *analysis_variables,
    }
    missing = sorted(required.difference(features.data_vars))
    if missing:
        raise ValueError(
            "Event-feature table is missing required variables: "
            + ", ".join(missing)
        )
    if "event" not in features.dims:
        raise ValueError("Event-feature table is missing the 'event' dimension.")

    event_ids = np.asarray(features["event_id"].values)
    if event_ids.ndim != 1 or event_ids.size != features.sizes["event"]:
        raise ValueError("event_id must be one-dimensional on the event axis.")
    if np.unique(event_ids).size != event_ids.size:
        raise ValueError("event_id values must be unique.")

    values = {
        name: event_numeric_values(features[name], name=name)
        for name in sorted(analysis_variables)
    }
    group_metric = features[settings.group_variable]
    i_dyn = event_numeric_values(group_metric, name=settings.group_variable)
    if not np.isfinite(i_dyn).all():
        raise ValueError("I_dyn_pre contains non-finite values.")

    negative_indices = np.flatnonzero(i_dyn < 0)
    positive_indices = np.flatnonzero(i_dyn > 0)
    if negative_indices.size == 0 or positive_indices.size == 0:
        raise ValueError(
            "Both negative and positive nonzero I_dyn_pre events are required."
        )

    specification_matches = {
        identifier: match_specification(
            features,
            group_metric,
            settings=settings,
            specification=specification,
        )
        for identifier, specification in settings.specifications.items()
    }
    primary_match = specification_matches[settings.primary_specification]
    if primary_match.pair_count == 0:
        raise ValueError("The primary specification produced no matched pairs.")

    balance = {
        name: variable_balance(
            values[name],
            negative_indices=negative_indices,
            positive_indices=positive_indices,
            match=primary_match,
        )
        for name in settings.balance_variables
    }
    frontier_matches = {
        family_id: {
            frontier_caliper: selectors.match_events_by_metric_sign(
                features,
                group_metric,
                match_variables=settings.family(family_id).variables,
                caliper_sd=frontier_caliper,
                reference_sign=settings.reference_sign,
            )
            for frontier_caliper in settings.frontier_calipers_sd
        }
        for family_id in settings.frontier_families
    }
    return Exploration(
        settings=settings,
        event_ids=event_ids,
        values=values,
        i_dyn=i_dyn,
        negative_indices=negative_indices,
        positive_indices=positive_indices,
        primary_match=primary_match,
        balance=balance,
        specification_matches=specification_matches,
        frontier_matches=frontier_matches,
    )


def match_specification(
    features: xr.Dataset,
    group_metric: xr.DataArray,
    *,
    settings: matching_settings.MatchingSettings,
    specification: matching_settings.MatchingSpecification,
) -> selectors.SignMatchResult:
    """Execute one validated named matching specification."""
    family = settings.family(specification.family)
    return selectors.match_events_by_metric_sign(
        features,
        group_metric,
        match_variables=family.variables,
        caliper_sd=specification.caliper_sd,
        reference_sign=settings.reference_sign,
    )


def event_numeric_values(da: xr.DataArray, *, name: str) -> np.ndarray:
    """Return one event variable as floats in its analysis-scale units."""
    values = np.asarray(da.values)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional on the event axis.")
    if np.issubdtype(values.dtype, np.timedelta64):
        values = values / np.timedelta64(1, "D")
    out = np.asarray(values, dtype=float)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} contains non-finite values.")
    return out


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = selectors.pooled_standard_deviation(left, right)
    mean_difference = float(np.mean(right) - np.mean(left))
    if scale == 0:
        return 0.0 if mean_difference == 0 else float(np.sign(mean_difference) * np.inf)
    return mean_difference / scale


def variable_balance(
    values: np.ndarray,
    *,
    negative_indices: np.ndarray,
    positive_indices: np.ndarray,
    match: selectors.SignMatchResult,
) -> dict[str, float]:
    return {
        "negative_mean_before": float(np.mean(values[negative_indices])),
        "positive_mean_before": float(np.mean(values[positive_indices])),
        "smd_before": standardized_mean_difference(
            values[negative_indices],
            values[positive_indices],
        ),
        "negative_mean_after": float(np.mean(values[match.negative_indices])),
        "positive_mean_after": float(np.mean(values[match.positive_indices])),
        "smd_after": standardized_mean_difference(
            values[match.negative_indices],
            values[match.positive_indices],
        ),
    }


def balance_for_match(
    exploration: Exploration,
    match: selectors.SignMatchResult,
) -> dict[str, dict[str, float]]:
    """Return the common balance audit for one matching specification."""
    if match.pair_count < 2:
        raise ValueError("Balance requires at least two retained pairs.")
    return {
        name: variable_balance(
            exploration.values[name],
            negative_indices=exploration.negative_indices,
            positive_indices=exploration.positive_indices,
            match=match,
        )
        for name in exploration.settings.balance_variables
    }


def balance_score(
    exploration: Exploration,
    match: selectors.SignMatchResult,
) -> dict[str, float | int | None]:
    """Summarize average, worst-case, and variable-wise SMD improvement."""
    if match.pair_count < 2:
        return {
            "matched_pairs": match.pair_count,
            "mean_absolute_smd": None,
            "maximum_absolute_smd": None,
            "variables_improved": 0,
        }
    balance = balance_for_match(exploration, match)
    variables = exploration.settings.balance_variables
    before = np.array([balance[name]["smd_before"] for name in variables])
    after = np.array([balance[name]["smd_after"] for name in variables])
    return {
        "matched_pairs": match.pair_count,
        "mean_absolute_smd": float(np.mean(np.abs(after))),
        "maximum_absolute_smd": float(np.max(np.abs(after))),
        "variables_improved": int(np.count_nonzero(np.abs(after) < np.abs(before))),
    }


def metrics_summary(exploration: Exploration) -> dict[str, object]:
    anomaly = exploration.values["tas_anom_peak"]
    match = exploration.primary_match
    pair_differences = np.abs(
        anomaly[match.negative_indices] - anomaly[match.positive_indices]
    )
    return {
        "candidate_events": int(exploration.event_ids.size),
        "negative_events": int(exploration.negative_indices.size),
        "positive_events": int(exploration.positive_indices.size),
        "zero_i_dyn_events": int(np.count_nonzero(exploration.i_dyn == 0)),
        "correlation_i_dyn_tas_anom_peak": float(
            np.corrcoef(exploration.i_dyn, anomaly)[0, 1]
        ),
        "correlation_i_dyn_I_dTdt_pre": float(
            np.corrcoef(exploration.i_dyn, exploration.values["I_dTdt_pre"])[0, 1]
        ),
        "I_dTdt_pre_smd_before": standardized_mean_difference(
            exploration.values["I_dTdt_pre"][exploration.negative_indices],
            exploration.values["I_dTdt_pre"][exploration.positive_indices],
        ),
        "primary_match_variables": list(match.match_variables),
        "primary_specification": exploration.settings.primary_specification,
        "calipers_pooled_sd": dict(match.calipers_sd),
        "matching_method": match.method,
        "matching_settings_path": str(exploration.settings.source_path),
        "matching_settings_sha256": exploration.settings.sha256,
        "matching_settings_schema_version": exploration.settings.schema_version,
        "matched_pairs": match.pair_count,
        "unmatched_negative_events": int(
            exploration.negative_indices.size - match.pair_count
        ),
        "unmatched_positive_events": int(
            exploration.positive_indices.size - match.pair_count
        ),
        "mean_absolute_pair_difference_K": float(np.mean(pair_differences)),
        "max_absolute_pair_difference_K": float(np.max(pair_differences)),
        "balance": exploration.balance,
        "sensitivity_pair_counts": {
            identifier: exploration.specification_matches[identifier].pair_count
            for identifier in exploration.settings.retention_sensitivity
        },
        "selected_specification_comparisons": {
            identifier: balance_score(
                exploration,
                exploration.specification_matches[identifier],
            )
            for identifier in exploration.settings.summary_specifications
        },
    }


def write_figures(
    exploration: Exploration,
    output_dir: str | Path,
) -> dict[str, Path]:
    paths = output_paths(output_dir)
    Path(output_dir).expanduser().resolve().mkdir(parents=True, exist_ok=True)
    plot_population_overview(exploration, paths["overview"])
    plot_matching_diagnostics(exploration, paths["matching"])
    plot_balance_and_sensitivity(exploration, paths["balance"])
    plot_matching_specification_tradeoff(exploration, paths["comparison"])
    return paths


def plot_population_overview(exploration: Exploration, path: str | Path) -> None:
    anomaly = exploration.values["tas_anom_peak"]
    negative = exploration.negative_indices
    positive = exploration.positive_indices

    fig, axes = plt.subplots(
        1,
        2,
        figsize=plot_style.publication_figsize("full", aspect=0.38),
    )
    scatter_ax, ecdf_ax = axes

    scatter_ax.scatter(
        anomaly[negative],
        exploration.i_dyn[negative],
        s=plot_style.SCATTER_SIZE_PT2,
        alpha=0.58,
        color=NEGATIVE_COLOR,
        label=f"Negative (n={negative.size})",
    )
    scatter_ax.scatter(
        anomaly[positive],
        exploration.i_dyn[positive],
        s=plot_style.SCATTER_SIZE_PT2,
        alpha=0.52,
        color=POSITIVE_COLOR,
        label=f"Positive (n={positive.size})",
    )
    scatter_ax.axhline(
        0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
    )
    scatter_ax.set_xlabel("Peak temperature anomaly [K]")
    scatter_ax.set_ylabel(r"$I_{dyn}$ [K]")
    correlation = np.corrcoef(exploration.i_dyn, anomaly)[0, 1]
    scatter_ax.set_title(f"Event-level relationship\nr = {correlation:.3f}")
    scatter_ax.legend(**plot_style.legend_kwargs(loc="upper left"))

    plot_ecdf(
        ecdf_ax,
        anomaly[negative],
        color=NEGATIVE_COLOR,
        label=f"Negative, mean={np.mean(anomaly[negative]):.2f} K",
    )
    plot_ecdf(
        ecdf_ax,
        anomaly[positive],
        color=POSITIVE_COLOR,
        label=f"Positive, mean={np.mean(anomaly[positive]):.2f} K",
    )
    ecdf_ax.set_xlabel("Peak temperature anomaly [K]")
    ecdf_ax.set_ylabel("Empirical cumulative probability")
    ecdf_ax.set_title("Unmatched severity distributions")
    ecdf_ax.set_ylim(0, 1.02)
    ecdf_ax.legend(**plot_style.legend_kwargs(loc="lower right"))

    for ax in axes:
        plot_style.style_axis(ax)
    fig.suptitle("PNW Bartusek heatwaves split by integrated dynamical heating")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.17, top=0.82, wspace=0.25)
    plot_style.save_figure(fig, path)
    plt.close(fig)


def plot_matching_diagnostics(exploration: Exploration, path: str | Path) -> None:
    anomaly = exploration.values["tas_anom_peak"]
    negative = exploration.negative_indices
    positive = exploration.positive_indices
    match = exploration.primary_match

    fig, axes = plt.subplots(
        1,
        3,
        figsize=plot_style.publication_figsize("full", aspect=0.34),
    )
    before_ax, after_ax, pair_ax = axes

    plot_ecdf(
        before_ax,
        anomaly[negative],
        color=NEGATIVE_COLOR,
        label="Negative",
    )
    plot_ecdf(
        before_ax,
        anomaly[positive],
        color=POSITIVE_COLOR,
        label="Positive",
    )
    before_smd = exploration.balance["tas_anom_peak"]["smd_before"]
    before_ax.set_title(f"Before matching\nSMD = {before_smd:.3f}")

    plot_ecdf(
        after_ax,
        anomaly[match.negative_indices],
        color=NEGATIVE_COLOR,
        label="Negative",
    )
    plot_ecdf(
        after_ax,
        anomaly[match.positive_indices],
        color=POSITIVE_COLOR,
        label="Matched positive",
    )
    after_smd = exploration.balance["tas_anom_peak"]["smd_after"]
    after_ax.set_title(f"After matching\nSMD = {after_smd:.3f}")

    for ax in (before_ax, after_ax):
        ax.set_xlabel("Peak temperature anomaly [K]")
        ax.set_ylabel("Empirical cumulative probability")
        ax.set_ylim(0, 1.02)
        ax.legend(**plot_style.legend_kwargs(loc="lower right"))
        plot_style.style_axis(ax)

    negative_values = anomaly[match.negative_indices]
    positive_values = anomaly[match.positive_indices]
    pair_ax.scatter(
        negative_values,
        positive_values,
        s=plot_style.SCATTER_SIZE_PT2 * 1.3,
        alpha=0.65,
        color=MATCHED_COLOR,
    )
    lower, upper = plot_style.one_to_one_limits(negative_values, positive_values)
    pair_ax.plot(
        [lower, upper],
        [lower, upper],
        color=plot_style.COLORS["calculated"],
        linestyle="--",
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
    )
    pair_ax.set_xlim(lower, upper)
    pair_ax.set_ylim(lower, upper)
    pair_ax.set_xlabel("Negative anomaly [K]")
    pair_ax.set_ylabel("Matched positive anomaly [K]")
    pair_ax.set_title(f"Pair agreement\nn = {match.pair_count} pairs")
    plot_style.format_one_to_one_axis(pair_ax)

    primary_specification = exploration.settings.specification(
        exploration.settings.primary_specification
    )
    primary_family = exploration.settings.family(primary_specification.family)
    fig.suptitle(
        f"One-to-one optimal matching on {primary_family.label.lower()} "
        f"(caliper = {primary_specification.caliper_sd:.2f} pooled SD)"
    )
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.76, wspace=0.34)
    plot_style.save_figure(fig, path)
    plt.close(fig)


def plot_balance_and_sensitivity(
    exploration: Exploration,
    path: str | Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=plot_style.publication_figsize("full", aspect=0.43),
    )
    balance_ax, sensitivity_ax = axes

    variables = list(exploration.settings.balance_variables)
    positions = np.arange(len(variables))
    before = np.array(
        [exploration.balance[name]["smd_before"] for name in variables]
    )
    after = np.array(
        [exploration.balance[name]["smd_after"] for name in variables]
    )
    balance_ax.axvspan(-0.1, 0.1, color=plot_style.COLORS["grid"], alpha=0.7)
    balance_ax.axvline(
        0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
    )
    for position, start, end in zip(positions, before, after, strict=True):
        balance_ax.plot(
            [start, end],
            [position, position],
            color="#A9A9A9",
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            zorder=1,
        )
    balance_ax.scatter(
        before,
        positions,
        color=POSITIVE_COLOR,
        marker="o",
        s=34,
        label="Before",
        zorder=2,
    )
    balance_ax.scatter(
        after,
        positions,
        color=MATCHED_COLOR,
        marker="s",
        s=32,
        label="After anomaly match",
        zorder=3,
    )
    balance_ax.set_yticks(positions)
    balance_ax.set_yticklabels(
        [VARIABLE_LABELS.get(name, name) for name in variables]
    )
    plot_style.use_default_numeric_formatter(balance_ax.yaxis)
    balance_ax.invert_yaxis()
    balance_ax.set_xlabel("Standardized mean difference\n(positive minus negative)")
    balance_ax.set_title("Balance audit")
    balance_ax.legend(**plot_style.legend_kwargs(loc="upper left"))
    balance_limit = max(0.65, float(np.max(np.abs(np.concatenate([before, after])))))
    balance_ax.set_xlim(-balance_limit - 0.10, balance_limit + 0.10)
    plot_style.style_axis(balance_ax, grid=False)
    balance_ax.grid(True, axis="x", color=plot_style.COLORS["grid"], linewidth=0.6)

    sensitivity_specifications = [
        exploration.settings.specification(identifier)
        for identifier in exploration.settings.retention_sensitivity
    ]
    labels = [
        exploration.settings.family(specification.family).label
        for specification in sensitivity_specifications
    ]
    counts = np.array(
        [
            exploration.specification_matches[specification.identifier].pair_count
            for specification in sensitivity_specifications
        ]
    )
    bar_positions = np.arange(len(labels))
    bar_colors = [MATCHED_COLOR, NEGATIVE_COLOR, POSITIVE_COLOR, "#777777"]
    bars = sensitivity_ax.barh(
        bar_positions,
        counts,
        color=[bar_colors[index % len(bar_colors)] for index in bar_positions],
        alpha=0.88,
    )
    sensitivity_ax.set_yticks(bar_positions)
    sensitivity_ax.set_yticklabels(labels)
    plot_style.use_default_numeric_formatter(sensitivity_ax.yaxis)
    sensitivity_ax.invert_yaxis()
    sensitivity_ax.set_xlabel("Matched pairs")
    sensitivity_calipers = {
        specification.caliper_sd
        for specification in sensitivity_specifications
    }
    if len(sensitivity_calipers) == 1:
        sensitivity_subtitle = (
            f"{next(iter(sensitivity_calipers)):.2f} SD per-variable caliper"
        )
    else:
        sensitivity_subtitle = "Configured per-variable SD calipers"
    sensitivity_ax.set_title(f"Retention sensitivity\n{sensitivity_subtitle}")
    maximum_count = max(exploration.negative_indices.size, counts.max())
    sensitivity_ax.set_xlim(0, maximum_count * 1.12)
    for bar, count in zip(bars, counts, strict=True):
        sensitivity_ax.text(
            count + 1.2,
            bar.get_y() + bar.get_height() / 2,
            str(int(count)),
            va="center",
            fontsize=plot_style.LEGEND_FONT_SIZE_PT,
        )
    plot_style.format_integer_axis(sensitivity_ax.xaxis, spacing=10)
    plot_style.style_axis(sensitivity_ax)

    fig.suptitle("What peak-anomaly matching balances and what it leaves different")
    fig.subplots_adjust(left=0.18, right=0.98, bottom=0.17, top=0.78, wspace=0.52)
    plot_style.save_figure(fig, path)
    plt.close(fig)


def plot_matching_specification_tradeoff(
    exploration: Exploration,
    path: str | Path,
) -> None:
    """Compare the proposed match with current and high-retention designs."""
    fig, axes = plt.subplots(
        1,
        2,
        figsize=plot_style.publication_figsize("full", aspect=0.47),
    )
    balance_ax, frontier_ax = axes

    selected: list[tuple[str, selectors.SignMatchResult | None, str, str]] = [
        (
            "Before matching",
            None,
            plot_style.COLORS["residual"],
            "x",
        )
    ]
    comparison_colors = (MATCHED_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR)
    comparison_markers = ("s", "^", "o")
    for index, identifier in enumerate(exploration.settings.balance_comparison):
        specification = exploration.settings.specification(identifier)
        family = exploration.settings.family(specification.family)
        selected.append(
            (
                f"{family.label} ({specification.caliper_sd:.2f} SD)",
                exploration.specification_matches[identifier],
                comparison_colors[index % len(comparison_colors)],
                comparison_markers[index % len(comparison_markers)],
            )
        )

    variables = exploration.settings.balance_variables
    positions = np.arange(len(variables))
    balance_ax.axvspan(-0.1, 0.1, color=plot_style.COLORS["grid"], alpha=0.7)
    balance_ax.axvline(
        0,
        color=plot_style.COLORS["zero"],
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
    )
    for label, match, color, marker in selected:
        if match is None:
            smds = np.array(
                [exploration.balance[name]["smd_before"] for name in variables]
            )
        elif match.pair_count < 2:
            smds = np.full(len(variables), np.nan)
        else:
            balance = balance_for_match(exploration, match)
            smds = np.array(
                [balance[name]["smd_after"] for name in variables]
            )
        balance_ax.scatter(
            smds,
            positions,
            color=color,
            marker=marker,
            s=33,
            label=label,
            zorder=3,
        )
    balance_ax.set_yticks(positions)
    balance_ax.set_yticklabels(
        [VARIABLE_LABELS.get(name, name) for name in variables]
    )
    plot_style.use_default_numeric_formatter(balance_ax.yaxis)
    balance_ax.invert_yaxis()
    balance_ax.set_xlabel("Standardized mean difference\n(positive minus negative)")
    balance_ax.set_title("Common seven-variable balance audit")
    balance_ax.set_xlim(-1.15, 1.15)
    plot_style.style_axis(balance_ax, grid=False)
    balance_ax.grid(True, axis="x", color=plot_style.COLORS["grid"], linewidth=0.6)

    frontier_colors = (MATCHED_COLOR, POSITIVE_COLOR, NEGATIVE_COLOR)
    frontier_markers = ("s", "^", "o")
    for index, family_id in enumerate(exploration.settings.frontier_families):
        color = frontier_colors[index % len(frontier_colors)]
        marker = frontier_markers[index % len(frontier_markers)]
        family = exploration.settings.family(family_id)
        matches = exploration.frontier_matches[family_id]
        pair_counts = []
        worst_smds = []
        for frontier_caliper in exploration.settings.frontier_calipers_sd:
            match = matches[frontier_caliper]
            pair_counts.append(match.pair_count)
            worst_smd = balance_score(exploration, match)["maximum_absolute_smd"]
            worst_smds.append(np.nan if worst_smd is None else worst_smd)
        frontier_ax.plot(
            pair_counts,
            worst_smds,
            color=color,
            marker=marker,
            linewidth=plot_style.LINE_WIDTH_PT,
            markersize=5,
            label=family.label,
        )
        annotation_groups: list[tuple[int, float, list[float]]] = []
        for pair_count, worst_smd, comparison_caliper in zip(
            pair_counts,
            worst_smds,
            exploration.settings.frontier_calipers_sd,
            strict=True,
        ):
            if not np.isfinite(worst_smd):
                continue
            existing_index = next(
                (
                    index
                    for index, (group_count, group_smd, _) in enumerate(
                        annotation_groups
                    )
                    if group_count == pair_count and np.isclose(group_smd, worst_smd)
                ),
                None,
            )
            if existing_index is None:
                annotation_groups.append(
                    (pair_count, worst_smd, [comparison_caliper])
                )
            else:
                annotation_groups[existing_index][2].append(comparison_caliper)
        for pair_count, worst_smd, group_calipers in annotation_groups:
            if len(group_calipers) > 2:
                caliper_label = f"{group_calipers[0]:g}-{group_calipers[-1]:g}"
            else:
                caliper_label = ", ".join(f"{value:g}" for value in group_calipers)
            right_aligned = pair_count > 0.9 * exploration.negative_indices.size
            frontier_ax.annotate(
                caliper_label,
                (pair_count, worst_smd),
                xytext=(-4 if right_aligned else 4, 4),
                textcoords="offset points",
                ha="right" if right_aligned else "left",
                fontsize=7,
                color=color,
            )
    frontier_ax.axhline(
        0.1,
        color=plot_style.COLORS["zero"],
        linestyle="--",
        linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
    )
    frontier_ax.text(
        2,
        0.115,
        "0.1 SMD target",
        color=plot_style.COLORS["zero"],
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
    )
    frontier_ax.set_xlabel("Matched pairs")
    frontier_ax.set_ylabel(
        f"Worst absolute SMD across {len(variables)} audit variables"
    )
    frontier_ax.set_title(
        "Retention versus worst-case balance\n"
        "(labels are calipers in pooled SD)"
    )
    frontier_ax.set_xlim(0, exploration.negative_indices.size * 1.05)
    frontier_ax.set_ylim(0, 1.18)
    plot_style.format_integer_axis(frontier_ax.xaxis, spacing=10)
    plot_style.style_axis(frontier_ax)

    fig.suptitle("Alternative matching specifications change both balance and estimand")
    handles, labels = balance_ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.90),
        ncol=2,
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
        frameon=True,
    )
    fig.subplots_adjust(left=0.18, right=0.985, bottom=0.16, top=0.69, wspace=0.43)
    plot_style.save_figure(fig, path)
    plt.close(fig)


def plot_ecdf(ax, values: np.ndarray, *, color: str, label: str) -> None:
    ordered = np.sort(np.asarray(values, dtype=float))
    cumulative = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    ax.step(ordered, cumulative, where="post", color=color, label=label)


def main() -> int:
    args = parse_args()
    validate_args(args)
    settings = matching_settings.load_matching_settings(args.settings_path)
    features = open_event_features(args.input_path)
    exploration = prepare_exploration(features, settings=settings)
    written = write_figures(exploration, args.output_dir)

    print(json.dumps(metrics_summary(exploration), indent=2, sort_keys=True))
    print("Wrote exploratory figures:")
    for path in written.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
