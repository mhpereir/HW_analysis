r"""Explore daily anticyclonic LWA changes versus heat-budget tendencies.

Example (on Venus, through a tracked PBS allocation)::

    python scripts/plot_hw_lwa_analysis.py \
        --input-path /path/to/tas_selected_stage1.nc \
        --output-dir /path/to/artifacts/plots_hw_lwa_analysis/new_run

LWA is daily-native, repeated on the hourly Stage-1 axis. Recover one value
per day and average each heating rate over all 24 hours. Pair consecutive days
within the same selected heatwave or with both days outside heatwaves, always
within one calendar year, then calculate
``(LWA_a[next] - LWA_a[current]) / 1 day``. The plotted heating rate is the
midpoint average of the two daily means, retaining K hr-1 units. This is a
daily finite-difference diagnostic, not an instantaneous hourly derivative.

Default selection: complete JJA TAS events and JJA non-heatwave days. Plot the
two populations in separate rows with matching axes. Each row uses the same
finite pairs for all three terms. Correlations describe equally weighted pairs,
not independent events; no significance or causal interpretation is implied.
All new analysis code is contained here; existing readers, season selection
and style are reused.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, plot_style, season_selection, selectors

TERMS = ("advection", "adiabatic", "diabatic")
POPULATIONS = {"heatwave": "Heatwaves", "baseline": "Non-heatwave baseline"}
LWA = "lwa_a_region"
PAIR_COLUMNS = {
    "rate": "d_lwa_a_dt",
    "change": "delta_lwa_a",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="TAS-selected hourly Stage-1 NetCDF containing daily LWA_a.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Fresh directory for PNG, paired-sample CSV and summary JSON.",
    )
    parser.add_argument(
        "--x-axis",
        choices=tuple(PAIR_COLUMNS),
        default="rate",
        help="LWA_a difference per day (default) or raw daily change.",
    )
    parser.add_argument(
        "--lwa-units", help="Explicit units for historical inputs lacking LWA metadata."
    )
    parser.add_argument(
        "--lwa-units-evidence",
        help="Source/formula evidence for an explicit units override.",
    )
    season_selection.add_season_arguments(parser, default_full_event=True)
    return season_selection.parse_args(parser)


def _validate_input(
    ds: xr.Dataset,
    *,
    lwa_units: str | None = None,
    lwa_units_evidence: str | None = None,
) -> tuple[pd.DatetimeIndex, str]:
    if ds.attrs.get("pipeline_stage") != analysis_io.EXPECTED_PIPELINE_STAGE:
        raise ValueError("Expected a Stage-1 harmonized regional time-series product.")
    if (
        ds.attrs.get("event_id_source") != "hw_event_id"
        or ds.attrs.get("threshold_variable", "tas") != "tas"
    ):
        raise ValueError(
            "Use a TAS-selected product with event_id_source='hw_event_id'."
        )
    if ds.attrs.get("analysis_time_resolution") != "hourly":
        raise ValueError("Stage-1 analysis_time_resolution must be 'hourly'.")
    for name in (LWA, "hw_event_id", *TERMS):
        if name not in ds or ds[name].dims != ("time",):
            raise ValueError(f"{name!r} must be a one-dimensional time variable.")
    if "time" not in ds.coords or not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError("time must be a datetime64 coordinate.")
    time = pd.DatetimeIndex(ds.time.values)
    if (
        len(time) < 2
        or time.hasnans
        or not time.is_unique
        or not time.is_monotonic_increasing
    ):
        raise ValueError(
            "time must have at least two unique, increasing, non-missing timestamps."
        )
    if not time.equals(time.floor("h")):
        raise ValueError("Expected timestamps on exact UTC hours.")
    if (
        ds[LWA].attrs.get("native_time_resolution") != "daily"
        or ds[LWA].attrs.get("projection_method") != "calendar_day_lookup"
    ):
        raise ValueError(
            "LWA_a must declare daily native resolution and calendar_day_lookup projection."
        )
    units = str(ds[LWA].attrs.get("units", "")).strip()
    if lwa_units is not None:
        override = lwa_units.strip()
        if not override or not lwa_units_evidence or not lwa_units_evidence.strip():
            raise ValueError(
                "An explicit --lwa-units requires nonempty --lwa-units-evidence."
            )
        if units and units != override:
            raise ValueError(
                "Explicit LWA units contradict the input's declared units."
            )
        units = override
    elif lwa_units_evidence is not None:
        raise ValueError("--lwa-units-evidence requires --lwa-units.")
    if not units:
        raise ValueError(
            "lwa_a_region must declare units, or supply --lwa-units with source evidence."
        )
    for term in TERMS:
        if ds[term].attrs.get("units") not in {"K hr-1", "K h-1", "K hour-1"}:
            raise ValueError(f"{term} must have hourly rate units 'K hr-1'.")
    for name, dimension in (("hw_event_id", "time"), ("event_id", "event")):
        if name not in ds or ds[name].dims != (dimension,):
            raise ValueError(f"{name!r} must have dimension {dimension!r}.")
        ids = np.asarray(ds[name].values, dtype=float)
        if not np.all(np.isfinite(ids) & (ids >= 0) & (ids == np.floor(ids))):
            raise ValueError(f"{name!r} must contain finite nonnegative integer IDs.")
    event_ids = ds.event_id.values
    if np.any(event_ids == 0) or np.unique(event_ids).size != event_ids.size:
        raise ValueError("The event table must contain unique positive event IDs.")
    return time, units


def prepare_pairs(
    ds: xr.Dataset,
    *,
    season_months: list[int] | None = None,
    all_seasons: bool = False,
    require_full_event: bool | None = None,
    lwa_units: str | None = None,
    lwa_units_evidence: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Build complete adjacent daily pairs without mutating or trimming Stage 1."""
    time, units = _validate_input(
        ds, lwa_units=lwa_units, lwa_units_evidence=lwa_units_evidence
    )
    months = season_selection.resolve_months(season_months, all_seasons=all_seasons)
    full_event = not all_seasons if require_full_event is None else require_full_event
    selected = season_selection.select_event_population(
        ds,
        season_months=season_months,
        all_seasons=all_seasons,
        require_full_event=full_event,
    )
    selected_ids = selectors.selected_event_ids(selected)
    hourly = pd.DataFrame(
        {
            name: np.asarray(ds[name].values, dtype=float)
            for name in (LWA, *TERMS, "hw_event_id")
        },
        index=time,
    ).replace([np.inf, -np.inf], np.nan)
    grouped = hourly.groupby(time.floor("D"), sort=True)
    minima, maxima = grouped.min(), grouped.max()
    if np.any((maxima[LWA] - minima[LWA]).fillna(0).to_numpy() != 0):
        raise ValueError(
            "Daily-native LWA_a varies within a calendar day; hourly differences are unsafe."
        )

    daily = grouped.mean()
    daily[LWA] = grouped[LWA].first()
    complete = grouped.size() == 24
    finite = (grouped[[LWA, *TERMS]].count() == 24).all(axis=1)
    stable_id = minima.hw_event_id.eq(maxima.hw_event_id)
    in_season = (
        np.ones(len(daily), dtype=bool)
        if months is None
        else daily.index.month.isin(months)
    )
    day_masks = {
        "heatwave": stable_id & daily.hw_event_id.isin(selected_ids),
        "baseline": stable_id & daily.hw_event_id.eq(0) & in_season,
    }
    selected_day = day_masks["heatwave"] | day_masks["baseline"]
    valid = daily.loc[selected_day & complete & finite]
    dates = valid.index
    # Guard the calendar BEFORE subtracting LWA, including summer-to-summer gaps.
    consecutive = (dates[1:] - dates[:-1]) == pd.Timedelta(days=1)
    same_year = dates[1:].year == dates[:-1].year
    same_event = valid.hw_event_id.to_numpy()[1:] == valid.hw_event_id.to_numpy()[:-1]
    left_indices = np.flatnonzero(consecutive & same_year & same_event)
    if left_indices.size == 0:
        raise ValueError(
            "No complete consecutive daily pairs in either the heatwave or baseline population."
        )
    left, right = valid.iloc[left_indices], valid.iloc[left_indices + 1]
    elapsed = (right.index - left.index) / pd.Timedelta(days=1)
    delta = right[LWA].to_numpy() - left[LWA].to_numpy()
    pairs = pd.DataFrame(
        {
            "day_start": left.index,
            "day_end": right.index,
            "event_id": left.hw_event_id.to_numpy(dtype=np.int64),
            "population": np.where(left.hw_event_id.eq(0), "baseline", "heatwave"),
            "elapsed_days": np.asarray(elapsed, dtype=float),
            "lwa_a_start": left[LWA].to_numpy(),
            "lwa_a_end": right[LWA].to_numpy(),
            "delta_lwa_a": delta,
            "d_lwa_a_dt": delta / elapsed,
        }
    )
    for term in TERMS:
        pairs[f"{term}_daily_start"] = left[term].to_numpy()
        pairs[f"{term}_daily_end"] = right[term].to_numpy()
        pairs[f"{term}_pair_mean"] = (
            pairs[f"{term}_daily_start"] + pairs[f"{term}_daily_end"]
        ) / 2

    details = {
        "lwa_units": units,
        "lwa_units_source": "explicit override"
        if lwa_units is not None
        else "Stage-1 metadata",
        "lwa_units_evidence": lwa_units_evidence,
        "tendency_units": "K hr-1",
        "season_months": None if months is None else list(months),
        "require_full_event": bool(full_event),
        "selected_event_ids": selected_ids.tolist(),
        "n_selected_events": int(selected_ids.size),
        "n_represented_events": int(
            pairs.loc[pairs.event_id > 0, "event_id"].nunique()
        ),
        "n_source_hours": len(time),
        "n_source_days": len(daily),
        "n_selected_event_days": int(day_masks["heatwave"].sum()),
        "n_days_excluded_incomplete_hours": int((selected_day & ~complete).sum()),
        "n_days_excluded_nonfinite": int((selected_day & complete & ~finite).sum()),
        "n_valid_days": len(valid),
        "n_pairs": len(pairs),
        "populations": {
            population: {
                "n_selected_days": int(mask.sum()),
                "n_days_excluded_incomplete_hours": int((mask & ~complete).sum()),
                "n_days_excluded_nonfinite": int((mask & complete & ~finite).sum()),
                "n_valid_days": int((mask & complete & finite).sum()),
                "n_pairs": int(pairs.population.eq(population).sum()),
            }
            for population, mask in day_masks.items()
        },
        "n_candidate_pairs_excluded_gap": int((~consecutive).sum()),
        "n_candidate_pairs_excluded_year": int((consecutive & ~same_year).sum()),
        "n_candidate_pairs_excluded_event": int(
            (consecutive & same_year & ~same_event).sum()
        ),
        "daily_reduction": "24 finite native UTC hourly samples per day, no gap filling",
        "difference": "LWA_a[end] - LWA_a[start]; exactly one day, same year and saved event ID (zero for baseline)",
        "baseline_definition": (
            "Both dates in the selected season and hw_event_id=0 at all 24 hours "
            "of each day. No additional event-adjacency buffer. Excluded heatwaves "
            "are not baseline days."
        ),
        "tendency_alignment": "(daily_mean[start] + daily_mean[end]) / 2, for every term",
        "interpretation": (
            "Descriptive equal-weight daily pairs; adjacent pairs share days and longer "
            "events contribute more pairs. No p-values or causal inference. Diabatic is "
            "the EHB closure residual, not separately diagnosed physical heating."
        ),
    }
    return pairs, details


def validate_saved_pairs(ds: xr.Dataset, pairs: pd.DataFrame, details: dict) -> dict:
    """Independently reconstruct CSV rows using direct hourly slices and sums."""
    months = details["season_months"]
    eligible_ids = set()
    for eid, start, end, peak in zip(
        ds.event_id.values,
        ds.start_time.values,
        ds.end_time.values,
        ds.peak_time.values,
        strict=True,
    ):
        dates = pd.date_range(
            pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize()
        )
        eligible = months is None or (
            bool(np.isin(dates.month, months).all())
            if details["require_full_event"]
            else pd.Timestamp(peak).month in months
        )
        if eligible:
            eligible_ids.add(int(eid))
    if eligible_ids != set(details["selected_event_ids"]):
        raise ValueError("Independent validation: selected event IDs differ.")

    times = np.asarray(ds.time.values, dtype="datetime64[ns]")
    days, starts, counts = np.unique(
        times.astype("datetime64[D]"), return_index=True, return_counts=True
    )
    fields = np.column_stack(
        [np.asarray(ds[name].values, dtype=float) for name in (LWA, *TERMS)]
    )
    event_ids = np.asarray(ds.hw_event_id.values, dtype=np.int64)
    valid_days = []
    for day, start, count in zip(days, starts, counts, strict=True):
        section = slice(start, start + count)
        expected_hours = day + np.arange(24).astype("timedelta64[h]")
        if count != 24 or not np.array_equal(times[section], expected_hours):
            continue
        values, ids = fields[section], event_ids[section]
        if not np.isfinite(values).all() or not np.all(ids == ids[0]):
            continue
        if not np.all(values[:, 0] == values[0, 0]):
            raise ValueError("Independent validation: LWA varies within one day.")
        if ids[0] in eligible_ids:
            population = "heatwave"
        elif ids[0] == 0 and (months is None or pd.Timestamp(day).month in months):
            population = "baseline"
        else:
            continue
        rates = np.array([math.fsum(values[:, k]) / 24 for k in range(1, 4)])
        valid_days.append((day, int(ids[0]), population, float(values[0, 0]), rates))

    rows = []
    for left, right in pairwise(valid_days):
        day0, eid, population, a0, q0 = left
        day1, eid1, population1, a1, q1 = right
        if (
            day1 - day0 != np.timedelta64(1, "D")
            or eid != eid1
            or population != population1
            or pd.Timestamp(day0).year != pd.Timestamp(day1).year
        ):
            continue
        row = {
            "day_start": pd.Timestamp(day0),
            "day_end": pd.Timestamp(day1),
            "event_id": eid,
            "population": population,
            "elapsed_days": 1.0,
            "lwa_a_start": a0,
            "lwa_a_end": a1,
            "delta_lwa_a": a1 - a0,
            "d_lwa_a_dt": a1 - a0,
        }
        for k, term in enumerate(TERMS):
            row[f"{term}_daily_start"] = q0[k]
            row[f"{term}_daily_end"] = q1[k]
            row[f"{term}_pair_mean"] = (q0[k] + q1[k]) / 2
        rows.append(row)
    reference = pd.DataFrame(rows)
    if (
        len(reference) != len(pairs)
        or len(pairs) != details["n_pairs"]
        or not len(pairs)
    ):
        raise ValueError("Independent validation: daily pair coverage/count differs.")
    for column in ("day_start", "day_end", "event_id", "population"):
        actual = pairs[column]
        if column.startswith("day_"):
            actual = pd.to_datetime(actual)
        np.testing.assert_array_equal(
            actual, reference[column], err_msg=f"Saved {column} differs"
        )
    errors = {}
    for column in reference.select_dtypes(include="number").columns:
        actual, expected = pairs[column].to_numpy(), reference[column].to_numpy()
        np.testing.assert_allclose(
            actual,
            expected,
            rtol=1e-12,
            atol=1e-12,
            err_msg=f"Saved {column} differs from direct source reduction",
        )
        errors[column] = float(np.max(np.abs(actual - expected)))
    population_counts = reference.population.value_counts().to_dict()
    for population in POPULATIONS:
        expected_count = population_counts.get(population, 0)
        expected_days = sum(day[2] == population for day in valid_days)
        if (
            details["populations"][population]["n_pairs"] != expected_count
            or details["populations"][population]["n_valid_days"] != expected_days
        ):
            raise ValueError(f"Independent validation: {population} counts differ.")
    return {
        "state": "passed",
        "method": "Direct hourly slices and math.fsum, independent event selection",
        "n_checked_pairs": len(reference),
        "population_counts": population_counts,
        "max_absolute_errors": errors,
        "rtol": 1e-12,
        "atol": 1e-12,
        "source_time_start": str(times[0]),
        "source_time_end": str(times[-1]),
    }


def correlations(pairs: pd.DataFrame, x_axis: str = "rate") -> dict:
    """Correlate each population separately; empty/constant samples are undefined."""
    output = {}
    for population in POPULATIONS:
        group = pairs.loc[pairs.population.eq(population)]
        x = group[PAIR_COLUMNS[x_axis]].to_numpy()
        output[population] = {}
        for term in TERMS:
            y = group[f"{term}_pair_mean"].to_numpy()
            if len(x) < 2 or np.ptp(x) == 0 or np.ptp(y) == 0:
                pearson = spearman = None
            else:
                pearson = float(np.corrcoef(x, y)[0, 1])
                spearman = float(
                    np.corrcoef(pd.Series(x).rank(), pd.Series(y).rank())[0, 1]
                )
            output[population][term] = {
                "n_pairs": len(x),
                "pearson_r": pearson,
                "spearman_rho": spearman,
            }
    return output


def plot_pairs(
    pairs: pd.DataFrame, details: dict, *, x_axis: str = "rate", context: str = ""
):
    """Render heatwaves and baseline separately with identical limits in all panels."""
    if pairs.empty:
        raise ValueError("No daily pairs to plot.")
    plot_style.apply_theme()
    fig, axes = plt.subplots(
        2,
        3,
        sharex=True,
        sharey=True,
        figsize=plot_style.publication_figsize(width="full", aspect=0.82),
    )
    x = pairs[PAIR_COLUMNS[x_axis]].to_numpy()
    units_label = details["lwa_units"]
    if units_label in {"m2", "m^2", "m**2", "m²"}:
        units_label = r"m$^2$"
    x_label = rf"$\Delta\mathrm{{LWA}}_a$ ({units_label})"
    if x_axis == "rate":
        x_label = rf"$\Delta\mathrm{{LWA}}_a/\Delta t$ ({units_label} day$^{{-1}}$)"
    stats = correlations(pairs, x_axis)
    y_all = pairs[[f"{term}_pair_mean" for term in TERMS]].to_numpy().ravel()
    for row, (population, label) in enumerate(POPULATIONS.items()):
        group = pairs.loc[pairs.population.eq(population)]
        for column, (ax, term) in enumerate(zip(axes[row], TERMS, strict=True)):
            ax.scatter(
                group[PAIR_COLUMNS[x_axis]],
                group[f"{term}_pair_mean"],
                color=plot_style.VARIABLE_COLORS[term],
                s=plot_style.SCATTER_SIZE_PT2,
                alpha=plot_style.SCATTER_ALPHA,
                edgecolors="none",
                rasterized=True,
            )
            if row == 0:
                title = plot_style.VARIABLE_NAME_MAPPING[term]
                ax.set_title(f"{title} residual" if term == "diabatic" else title)
            ax.set_xlabel(x_label)
            ax.xaxis.label.set_visible(row == 1)
            ax.yaxis.label.set_visible(column == 0)
            plot_style.style_axis(ax)
            for draw in (ax.axhline, ax.axvline):
                draw(
                    0,
                    color="0.4",
                    linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
                    zorder=0,
                )
            ax.set_xlim(plot_style.padded_data_limits(x, required_values=(0,)))
            ax.set_ylim(plot_style.padded_data_limits(y_all, required_values=(0,)))
            r = stats[population][term]["pearson_r"]
            rho = stats[population][term]["spearman_rho"]
            annotation = (
                "r, rho: undefined" if r is None else f"r = {r:.2f}\nrho = {rho:.2f}"
            )
            if group.empty:
                annotation = "No eligible daily pairs"
            ax.text(
                0.04,
                0.96,
                annotation,
                transform=ax.transAxes,
                va="top",
                fontsize=plot_style.LEGEND_FONT_SIZE_PT,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
            )
        axes[row, 0].set_ylabel(
            f"{label} ({len(group):,} pairs)\n" + r"Mean tendency (K hr$^{-1}$)"
        )
    months = details["season_months"]
    population = (
        "all seasons" if months is None else "months " + ",".join(map(str, months))
    )
    if months is not None:
        population += (
            ", full HW events"
            if details["require_full_event"]
            else ", peak-selected HW events"
        )
    prefix = f"{context}\n" if context else ""
    fig.suptitle(
        f"{prefix}Anticyclonic LWA changes and heat-budget tendencies\n"
        f"{details['n_represented_events']:,} heatwaves | {population}; baseline dates in season",
        y=0.98,
    )
    fig.text(
        0.5,
        0.035,
        "Heating: midpoint average of two complete daily means; positive values warm.\n"
        "Daily pairs share days; correlations are descriptive, without significance tests.",
        ha="center",
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
    )
    fig.subplots_adjust(
        left=0.105, right=0.99, bottom=0.14, top=0.84, wspace=0.12, hspace=0.15
    )
    return fig


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _git_output(*args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> int:
    args = parse_args()
    input_path = args.input_path.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists; choose a fresh path: {output_dir}"
        )
    with analysis_io.open_harmonized_timeseries(input_path) as ds:
        pairs, details = prepare_pairs(
            ds,
            **season_selection.season_kwargs(args),
            lwa_units=args.lwa_units,
            lwa_units_evidence=args.lwa_units_evidence,
        )
        source_attrs = dict(ds.attrs)
        variable_attrs = {name: dict(ds[name].attrs) for name in (LWA, *TERMS)}

    context = str(source_attrs.get("region", input_path.stem))
    bottom = source_attrs.get("heat_budget_bottom_boundary")
    top = source_attrs.get("heat_budget_top_boundary")
    if bottom is not None and top is not None:
        bounds = [
            f"{value} hPa" if str(value).isdigit() else str(value)
            for value in (bottom, top)
        ]
        context += f" | {' to '.join(bounds)}"
    context += f" | {pairs.day_start.dt.year.min()}-{pairs.day_end.dt.year.max()}"
    if "quantile" in source_attrs:
        context += f" | TAS q{source_attrs['quantile']}"
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "source_attrs": source_attrs,
        "source_variable_attrs": variable_attrs,
        "checkout": str(REPO_ROOT),
        "branch": _git_output("branch", "--show-current"),
        "commit": _git_output("rev-parse", "HEAD"),
        "working_tree_status": _git_output("status", "--porcelain"),
        "script_sha256": _sha256(Path(__file__)),
        "command": sys.argv,
        "python": sys.version,
        "interpreter": sys.executable,
        "host": socket.gethostname(),
        "pbs_job_id": os.environ.get("PBS_JOBID"),
        "x_axis": args.x_axis,
        "x_column": PAIR_COLUMNS[args.x_axis],
        "column_units": {
            "elapsed_days": "day",
            "lwa_a_start": details["lwa_units"],
            "lwa_a_end": details["lwa_units"],
            "delta_lwa_a": details["lwa_units"],
            "d_lwa_a_dt": f"{details['lwa_units']} day-1",
            **{
                f"{term}_{suffix}": "K hr-1"
                for term in TERMS
                for suffix in ("daily_start", "daily_end", "pair_mean")
            },
        },
        "analysis": details,
        "correlations": correlations(pairs, args.x_axis),
    }
    # Reserve a new namespace atomically. Failed attempts remain visible for diagnosis.
    output_dir.mkdir(parents=True, exist_ok=False)
    figure_path = output_dir / "hw_lwa_analysis.png"
    csv_path = output_dir / "daily_pairs.csv"
    fig = plot_pairs(pairs, details, x_axis=args.x_axis, context=context)
    try:
        plot_style.save_figure(fig, figure_path)
    finally:
        plt.close(fig)
    pairs.to_csv(csv_path, index=False, mode="x", float_format="%.17g")
    with analysis_io.open_harmonized_timeseries(input_path) as ds:
        summary["numerical_validation"] = validate_saved_pairs(
            ds, pd.read_csv(csv_path, float_precision="round_trip"), details
        )
    summary["outputs"] = {
        path.name: {"path": str(path), "sha256": _sha256(path)}
        for path in (figure_path, csv_path)
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, default=str, allow_nan=False)
        stream.write("\n")
    print(
        f"Wrote {details['populations']['heatwave']['n_pairs']} heatwave pairs and "
        f"{details['populations']['baseline']['n_pairs']} baseline pairs to {output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
