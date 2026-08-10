"""Explore deterministic matching of positive and negative I_dyn events.

This is an isolated Stage-2 exploration. It consumes an existing event-feature
table, derives ``I_dyn = I_adiabatic_pre + I_advection_pre``, and compares the
unmatched sign populations with deterministic one-to-one optimal matches.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import plot_style  # noqa: E402


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "results/stage2_event_features"
    / "hw_event_features_fixed_windows_pnw_bartusek_tas_q90_1940_2024.nc"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/Idyn_matching_exploration"
DEFAULT_CALIPER = 0.2

OVERVIEW_FILENAME = "idyn_population_overview.png"
MATCHING_FILENAME = "tas_anom_matching_diagnostics.png"
BALANCE_FILENAME = "covariate_balance_and_sensitivity.png"

POSITIVE_COLOR = plot_style.COLORS["diabatic"]
NEGATIVE_COLOR = plot_style.COLORS["advection"]
MATCHED_COLOR = plot_style.COLORS["temperature_tendency"]

PRIMARY_MATCH_VARIABLES = ("tas_anom_peak",)
BALANCE_VARIABLES = (
    "tas_anom_peak",
    "tas_peak",
    "tas_excess_peak",
    "tas_excess_integral",
    "duration",
    "days_from_solstice",
    "T_anom_mean_ant",
)
SENSITIVITY_SPECS = {
    "Peak anomaly": ("tas_anom_peak",),
    "Anomaly + season timing": ("tas_anom_peak", "days_from_solstice"),
    "Anomaly + duration": ("tas_anom_peak", "duration"),
    "Anomaly + timing + duration": (
        "tas_anom_peak",
        "days_from_solstice",
        "duration",
    ),
}

VARIABLE_LABELS = {
    "tas_anom_peak": "Peak temperature anomaly",
    "tas_peak": "Peak temperature",
    "tas_excess_peak": "Peak threshold excess",
    "tas_excess_integral": "Integrated threshold excess",
    "duration": "Duration",
    "days_from_solstice": "Days from June 21",
    "T_anom_mean_ant": "Antecedent mean anomaly",
}


@dataclass(frozen=True)
class MatchResult:
    """One-to-one event matches and their standardized distances."""

    negative_indices: np.ndarray
    positive_indices: np.ndarray
    distances: np.ndarray
    match_variables: tuple[str, ...]
    pooled_scales: Mapping[str, float]
    caliper: float

    @property
    def pair_count(self) -> int:
        return int(self.negative_indices.size)


@dataclass(frozen=True)
class Exploration:
    """Prepared arrays, primary matches, and balance diagnostics."""

    event_ids: np.ndarray
    values: Mapping[str, np.ndarray]
    i_dyn: np.ndarray
    negative_indices: np.ndarray
    positive_indices: np.ndarray
    primary_match: MatchResult
    balance: Mapping[str, Mapping[str, float]]
    sensitivity: Mapping[str, MatchResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore matching positive and negative Stage-2 I_dyn events."
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
        "--caliper",
        type=float,
        default=DEFAULT_CALIPER,
        help="Per-variable caliper in pooled-standard-deviation units.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing exploratory figures.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not np.isfinite(args.caliper) or args.caliper <= 0:
        raise ValueError("--caliper must be a finite positive number.")
    if not args.input_path.expanduser().resolve().is_file():
        raise FileNotFoundError(f"Input file does not exist: {args.input_path}")

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
    caliper: float = DEFAULT_CALIPER,
) -> Exploration:
    """Validate a Stage-2 table and prepare the matching diagnostics."""
    required = {
        "event_id",
        "I_advection_pre",
        "I_adiabatic_pre",
        *BALANCE_VARIABLES,
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
        for name in BALANCE_VARIABLES
    }
    advection = event_numeric_values(
        features["I_advection_pre"],
        name="I_advection_pre",
    )
    adiabatic = event_numeric_values(
        features["I_adiabatic_pre"],
        name="I_adiabatic_pre",
    )
    i_dyn = advection + adiabatic
    if not np.isfinite(i_dyn).all():
        raise ValueError("I_dyn contains non-finite values.")

    negative_indices = np.flatnonzero(i_dyn < 0)
    positive_indices = np.flatnonzero(i_dyn > 0)
    if negative_indices.size == 0 or positive_indices.size == 0:
        raise ValueError("Both negative and positive nonzero I_dyn events are required.")

    primary_match = optimal_sign_match(
        event_ids,
        i_dyn,
        values,
        match_variables=PRIMARY_MATCH_VARIABLES,
        caliper=caliper,
    )
    if primary_match.pair_count == 0:
        raise ValueError("The primary specification produced no matched pairs.")

    balance = {
        name: variable_balance(
            values[name],
            negative_indices=negative_indices,
            positive_indices=positive_indices,
            match=primary_match,
        )
        for name in BALANCE_VARIABLES
    }
    sensitivity = {
        label: optimal_sign_match(
            event_ids,
            i_dyn,
            values,
            match_variables=variables,
            caliper=caliper,
        )
        for label, variables in SENSITIVITY_SPECS.items()
    }
    return Exploration(
        event_ids=event_ids,
        values=values,
        i_dyn=i_dyn,
        negative_indices=negative_indices,
        positive_indices=positive_indices,
        primary_match=primary_match,
        balance=balance,
        sensitivity=sensitivity,
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


def optimal_sign_match(
    event_ids: np.ndarray,
    i_dyn: np.ndarray,
    values: Mapping[str, np.ndarray],
    *,
    match_variables: Sequence[str],
    caliper: float,
) -> MatchResult:
    """Return maximum-cardinality, minimum-distance sign matches.

    Negative events define the reference population. Each reference event can
    be paired with at most one positive event. Every matched pair must satisfy
    the caliper for every standardized matching variable.
    """
    if not match_variables:
        raise ValueError("At least one matching variable is required.")
    if not np.isfinite(caliper) or caliper <= 0:
        raise ValueError("caliper must be a finite positive number.")

    event_ids = np.asarray(event_ids)
    i_dyn = np.asarray(i_dyn, dtype=float)
    negative = np.flatnonzero(i_dyn < 0)
    positive = np.flatnonzero(i_dyn > 0)
    if negative.size == 0 or positive.size == 0:
        raise ValueError("Both negative and positive I_dyn events are required.")

    negative = negative[np.argsort(event_ids[negative], kind="mergesort")]
    positive = positive[np.argsort(event_ids[positive], kind="mergesort")]

    standardized_differences = []
    scales: dict[str, float] = {}
    for name in match_variables:
        if name not in values:
            raise KeyError(f"Unknown matching variable: {name}")
        variable = np.asarray(values[name], dtype=float)
        if variable.shape != i_dyn.shape:
            raise ValueError(f"{name} does not align with the event axis.")
        if not np.isfinite(variable).all():
            raise ValueError(f"{name} contains non-finite values.")
        scale = pooled_standard_deviation(variable[negative], variable[positive])
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(
                f"Matching variable {name!r} has no finite positive pooled scale."
            )
        scales[name] = scale
        standardized_differences.append(
            np.abs(variable[negative, None] - variable[positive][None, :]) / scale
        )

    difference_cube = np.stack(standardized_differences, axis=2)
    valid = np.all(difference_cube <= caliper, axis=2)
    distance = np.sqrt(np.mean(difference_cube**2, axis=2))

    # A dummy column allows each negative event to remain unmatched. The dummy
    # penalty exceeds the maximum possible change in total valid-edge cost, so
    # the assignment first maximizes pair count and then minimizes distance.
    dummy_penalty = (negative.size + 1) * (caliper + 1.0)
    invalid_penalty = 2.0 * dummy_penalty
    augmented_cost = np.concatenate(
        [
            np.where(valid, distance, invalid_penalty),
            np.full((negative.size, negative.size), dummy_penalty),
        ],
        axis=1,
    )
    row_indices, column_indices = linear_sum_assignment(augmented_cost)
    matched = column_indices < positive.size
    rows = row_indices[matched]
    columns = column_indices[matched]

    return MatchResult(
        negative_indices=negative[rows],
        positive_indices=positive[columns],
        distances=distance[rows, columns],
        match_variables=tuple(match_variables),
        pooled_scales=scales,
        caliper=float(caliper),
    )


def pooled_standard_deviation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.size < 2 or right.size < 2:
        return float("nan")
    numerator = (left.size - 1) * np.var(left, ddof=1) + (
        right.size - 1
    ) * np.var(right, ddof=1)
    return float(np.sqrt(numerator / (left.size + right.size - 2)))


def standardized_mean_difference(left: np.ndarray, right: np.ndarray) -> float:
    scale = pooled_standard_deviation(left, right)
    mean_difference = float(np.mean(right) - np.mean(left))
    if scale == 0:
        return 0.0 if mean_difference == 0 else float(np.sign(mean_difference) * np.inf)
    return mean_difference / scale


def variable_balance(
    values: np.ndarray,
    *,
    negative_indices: np.ndarray,
    positive_indices: np.ndarray,
    match: MatchResult,
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
        "primary_match_variables": list(match.match_variables),
        "caliper_pooled_sd": match.caliper,
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
            label: result.pair_count
            for label, result in exploration.sensitivity.items()
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

    fig.suptitle(
        "One-to-one optimal matching on peak temperature anomaly "
        f"(caliper = {match.caliper:.2f} pooled SD)"
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

    variables = list(BALANCE_VARIABLES)
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
    balance_ax.set_yticklabels([VARIABLE_LABELS[name] for name in variables])
    plot_style.use_default_numeric_formatter(balance_ax.yaxis)
    balance_ax.invert_yaxis()
    balance_ax.set_xlabel("Standardized mean difference\n(positive minus negative)")
    balance_ax.set_title("Balance audit")
    balance_ax.legend(**plot_style.legend_kwargs(loc="upper left"))
    balance_limit = max(0.65, float(np.max(np.abs(np.concatenate([before, after])))))
    balance_ax.set_xlim(-balance_limit - 0.10, balance_limit + 0.10)
    plot_style.style_axis(balance_ax, grid=False)
    balance_ax.grid(True, axis="x", color=plot_style.COLORS["grid"], linewidth=0.6)

    labels = list(exploration.sensitivity)
    counts = np.array(
        [exploration.sensitivity[label].pair_count for label in labels]
    )
    bar_positions = np.arange(len(labels))
    bars = sensitivity_ax.barh(
        bar_positions,
        counts,
        color=[MATCHED_COLOR, NEGATIVE_COLOR, POSITIVE_COLOR, "#777777"],
        alpha=0.88,
    )
    sensitivity_ax.set_yticks(bar_positions)
    sensitivity_ax.set_yticklabels(labels)
    plot_style.use_default_numeric_formatter(sensitivity_ax.yaxis)
    sensitivity_ax.invert_yaxis()
    sensitivity_ax.set_xlabel("Matched pairs")
    sensitivity_ax.set_title(
        f"Retention sensitivity\n{exploration.primary_match.caliper:.2f} SD per-variable caliper"
    )
    sensitivity_ax.set_xlim(0, max(exploration.negative_indices.size, counts.max()) * 1.12)
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


def plot_ecdf(ax, values: np.ndarray, *, color: str, label: str) -> None:
    ordered = np.sort(np.asarray(values, dtype=float))
    cumulative = np.arange(1, ordered.size + 1, dtype=float) / ordered.size
    ax.step(ordered, cumulative, where="post", color=color, label=label)


def main() -> int:
    args = parse_args()
    validate_args(args)
    features = open_event_features(args.input_path)
    exploration = prepare_exploration(features, caliper=args.caliper)
    written = write_figures(exploration, args.output_dir)

    print(json.dumps(metrics_summary(exploration), indent=2, sort_keys=True))
    print("Wrote exploratory figures:")
    for path in written.values():
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
