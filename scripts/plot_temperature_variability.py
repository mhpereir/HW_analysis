"""Compare prepared regional T2m with EHB temperature, hourly and daily.

Stage-1 tas_region is daily-native. Supply --hourly-t2m-path or use an embedded
field with --stage1-hourly-t2m, or request --daily-only. See the temperature
variability section of docs/workflows/plotting.md for the input contract.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import MaxNLocator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import analysis_io, plot_style

T2M_LABEL = r"$T_{2m}$"
TMEAN_LABEL = plot_style.VARIABLE_NAME_MAPPING["T_mean"]
RATIO_LABEL = r"$\sigma_{\langle T\rangle}/\sigma_{T_{2m}}$"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-path", type=Path, required=True, help="Stage-1 NetCDF."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--hourly-t2m-path", type=Path, help="Prepared regional hourly NetCDF."
    )
    mode.add_argument(
        "--daily-only", action="store_true", help="No hourly T2m comparison."
    )
    mode.add_argument(
        "--stage1-hourly-t2m",
        action="store_true",
        help="Use the optional prepared tas_region_hourly field embedded in Stage 1.",
    )
    parser.add_argument(
        "--hourly-t2m-variable",
        default="tas_region",
        help="Regional temperature variable in the hourly file (default: tas_region).",
    )
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="New output directory."
    )
    parser.add_argument(
        "--figure-layout",
        choices=("diagnostic", "paper"),
        default="diagnostic",
        help="Paper renders only the daily scatter; statistical exports are unchanged.",
    )
    parser.add_argument(
        "--plot-kind",
        choices=("scatter", "hexbin"),
        default="scatter",
        help="Show individual pairs or hexagonal sample counts.",
    )
    parser.add_argument(
        "--hexbin-gridsize",
        type=int,
        default=plot_style.HEXBIN_GRIDSIZE,
        help="Number of hexagons across the x range (default: 35, minimum: 2).",
    )
    season = parser.add_mutually_exclusive_group()
    season.add_argument("--season-months", type=int, nargs="+", choices=range(1, 13))
    season.add_argument("--all-seasons", action="store_true")
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    args = parser.parse_args(argv)
    if args.hexbin_gridsize < 2:
        parser.error("--hexbin-gridsize must be at least 2.")
    if args.start_year and args.end_year and args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year.")
    args.season_months = (
        list(range(1, 13))
        if args.all_seasons
        else sorted(set(args.season_months or [6, 7, 8]))
    )
    return args


def temperature_series(ds: xr.Dataset, variable: str) -> pd.Series:
    """Read one hourly-indexed series in K without changing the input."""
    if variable not in ds or ds[variable].dims != ("time",):
        raise ValueError(f"{variable!r} must have exactly the dimension ('time',).")
    if "time" not in ds.coords or not np.issubdtype(ds.time.dtype, np.datetime64):
        raise ValueError("A Gregorian datetime64 time coordinate in UTC is required.")
    index = pd.DatetimeIndex(ds.time.values)
    if (
        index.empty
        or index.hasnans
        or index.has_duplicates
        or not index.is_monotonic_increasing
    ):
        raise ValueError("Time must be nonempty, unique, ordered and free of NaT.")
    if not index.equals(index.floor("h")):
        raise ValueError("Timestamps must fall on exact UTC hours.")
    da = ds[variable]
    units = str(da.attrs.get("units", "")).lower().replace(" ", "").replace("_", "")
    if units in {"k", "kelvin", "degk", "degreekelvin", "degreeskelvin"}:
        offset = 0.0
    elif units in {"c", "degc", "celsius", "degreecelsius", "degreescelsius", "°c"}:
        offset = 273.15
    else:
        raise ValueError(
            f"{variable!r} needs explicit K or Celsius units, got {units!r}."
        )
    values = np.asarray(da.values, dtype=np.float64) + offset
    values[~np.isfinite(values)] = np.nan
    return pd.Series(values, index=index, name=variable)


def validate_hourly_source(
    stage1: xr.Dataset, hourly: xr.Dataset, variable: str
) -> None:
    """Reject daily projections and incompatible regional metadata."""
    region = stage1.attrs.get("region")
    if not region or hourly.attrs.get("region") != region:
        raise ValueError("Hourly T2m and Stage 1 must declare the same region.")
    if variable not in hourly:
        raise ValueError(f"Hourly companion is missing {variable!r}.")
    attrs = hourly[variable].attrs
    if str(attrs.get("native_time_resolution", "")).lower() != "hourly":
        raise ValueError("Hourly T2m must declare native_time_resolution='hourly'.")
    projected = str(attrs.get("projected_from_daily", False)).lower()
    if (
        projected not in {"false", "0"}
        or attrs.get("projection_method") == "calendar_day_lookup"
    ):
        raise ValueError("Daily-projected T2m cannot be used as native hourly data.")
    for key in ("lat_bounds", "lon_bounds"):
        left = stage1.tas_region.attrs.get(key)
        right = attrs.get(key, hourly.attrs.get(key))
        if left is None or right is None:
            continue
        left = np.asarray(
            ast.literal_eval(left) if isinstance(left, str) else left, dtype=float
        )
        right = np.asarray(
            ast.literal_eval(right) if isinstance(right, str) else right, dtype=float
        )
        if (
            left.shape != (2,)
            or right.shape != (2,)
            or not np.allclose(left, right, atol=1e-8, rtol=0)
        ):
            raise ValueError(f"Hourly T2m and Stage 1 disagree on {key}.")


def prepare_pairs(
    stage1: xr.Dataset,
    hourly: xr.Dataset | None = None,
    *,
    hourly_variable: str = "tas_region",
    months: tuple[int, ...] | list[int] = (6, 7, 8),
    start_year: int | None = None,
    end_year: int | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict]:
    """Use exact timestamp matches and common, complete UTC days."""
    if not months or any(month not in range(1, 13) for month in months):
        raise ValueError("Select at least one calendar month from 1 to 12.")
    if start_year is not None and end_year is not None and start_year > end_year:
        raise ValueError("start_year must not exceed end_year.")
    if not stage1.attrs.get("region"):
        raise ValueError("Stage 1 must declare its region.")
    frame = pd.DataFrame(
        {
            "T_mean": temperature_series(stage1, "T_mean"),
            "tas_region": temperature_series(stage1, "tas_region"),
        }
    )
    total_hours = len(frame)
    keep = frame.index.month.isin(months)
    if start_year is not None:
        keep &= frame.index.year >= start_year
    if end_year is not None:
        keep &= frame.index.year <= end_year
    frame = frame.loc[keep].copy()
    if frame.empty:
        raise ValueError("The year/month selection contains no Stage-1 samples.")
    if hourly is not None:
        validate_hourly_source(stage1, hourly, hourly_variable)
        source = temperature_series(hourly, hourly_variable)
        frame["tas_region_hourly"] = source.reindex(frame.index)

    groups = frame.groupby(frame.index.normalize())
    daily_range = groups["tas_region"].max() - groups["tas_region"].min()
    if (daily_range > 1e-6).any():
        raise ValueError("Stage-1 tas_region must be constant within each UTC day.")
    counts = groups.count()
    complete = groups.size().eq(24) & counts.eq(24).all(axis=1)
    days = complete.index[complete]
    if len(days) < 2:
        raise ValueError(
            "At least two complete UTC days with 24 finite paired hours are required."
        )
    retained = frame.loc[frame.index.normalize().isin(days)]
    daily = retained.groupby(retained.index.normalize()).mean()
    daily.index.name = "date_utc"
    pairs = {"daily": daily[["tas_region", "T_mean"]]}
    if hourly is not None:
        pairs["hourly"] = retained[["tas_region_hourly", "T_mean"]].rename(
            columns={"tas_region_hourly": "tas_region"}
        )
        pairs["daily_from_hourly"] = daily[["tas_region_hourly", "T_mean"]].rename(
            columns={"tas_region_hourly": "tas_region"}
        )
    coverage = {
        "stage1_hours": total_hours,
        "selected_hours": len(frame),
        "selected_days_present": len(complete),
        "missing_or_nonfinite_hours_by_field": (len(frame) - frame.count()).to_dict(),
        "incomplete_or_nonfinite_days_dropped": int((~complete).sum()),
        "retained_days": len(days),
        "retained_hours": len(retained),
        "first_hour_utc": retained.index.min().isoformat(),
        "last_hour_utc": retained.index.max().isoformat(),
        "note": "Absent days and off-season gaps are not counted as observed days.",
    }
    return pairs, daily, coverage


def pair_statistics(frame: pd.DataFrame) -> dict:
    """Descriptive paired statistics; undefined values remain None."""
    values = frame[["tas_region", "T_mean"]].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    n = len(values)
    result = {"n": n}
    for column, name in enumerate(("tas_region", "T_mean")):
        data = values[:, column]
        constant = bool(n and np.ptp(data) == 0)
        result[f"{name}_mean_K"] = float(data.mean()) if n else None
        result[f"{name}_sd_K"] = (
            (0.0 if constant else float(data.std(ddof=1))) if n > 1 else None
        )
        result[f"{name}_variance_K2"] = (
            (0.0 if constant else float(data.var(ddof=1))) if n > 1 else None
        )
        result[f"{name}_iqr_K"] = (
            float(np.diff(np.quantile(data, [0.25, 0.75]))[0]) if n else None
        )
    for name in (
        "pearson_r",
        "spearman_rho",
        "slope_K_per_K",
        "intercept_K",
        "r_squared",
        "sd_ratio_T_mean_over_tas_region",
        "variance_ratio_T_mean_over_tas_region",
    ):
        result[name] = None
    if n < 2:
        return result
    x, y = values.T
    dx, dy = x - x.mean(), y - y.mean()
    xx = 0.0 if np.ptp(x) == 0 else float(dx @ dx)
    yy = 0.0 if np.ptp(y) == 0 else float(dy @ dy)
    xy = float(dx @ dy) if xx > 0 and yy > 0 else 0.0
    if xx > 0:
        result.update(
            slope_K_per_K=xy / xx,
            intercept_K=float(y.mean() - (xy / xx) * x.mean()),
            sd_ratio_T_mean_over_tas_region=float(np.sqrt(yy / xx)),
            variance_ratio_T_mean_over_tas_region=yy / xx,
        )
    if xx > 0 and yy > 0:
        r = float(np.clip(xy / np.sqrt(xx * yy), -1, 1))
        ranks = pd.DataFrame(values).rank(method="average").to_numpy()
        result.update(
            pearson_r=r,
            spearman_rho=float(np.corrcoef(ranks.T)[0, 1]),
            r_squared=r * r,
        )
    return result


def statistics_table(pairs: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for scale, frame in pairs.items():
        for period, data in [("all", frame), *frame.groupby(frame.index.year)]:
            rows.append(
                {
                    "timescale": scale,
                    "period": str(period),
                    "first_sample_utc": data.index.min().isoformat(),
                    "last_sample_utc": data.index.max().isoformat(),
                    **pair_statistics(data),
                }
            )
    return rows


def source_agreement(daily: pd.DataFrame) -> dict | None:
    if "tas_region_hourly" not in daily:
        return None
    delta = daily["tas_region_hourly"] - daily["tas_region"]
    return {
        "definition": "daily mean of hourly companion minus Stage-1 daily event T2m",
        "n_days": len(delta),
        "mean_difference_K": float(delta.mean()),
        "rmse_K": float(np.sqrt(np.mean(delta**2))),
        "max_absolute_difference_K": float(delta.abs().max()),
    }


def format_stat(value: float | None) -> str:
    return "undefined" if value is None else f"{value:.3f}"


def draw_scatter(
    ax,
    data: pd.DataFrame,
    summary: dict,
    *,
    compact: bool = False,
    kind: str = "scatter",
    gridsize: int = plot_style.HEXBIN_GRIDSIZE,
):
    """Render every retained pair and the fit computed from that same sample."""
    if kind == "hexbin":
        counts = ax.hexbin(
            data.tas_region,
            data.T_mean,
            gridsize=gridsize,
            mincnt=1,
            cmap=plot_style.COUNT_COLOR_MAP,
            linewidths=0,
            rasterized=True,
        )
        if counts.get_array().sum() != len(data):
            raise ValueError("Hexbin counts must include every retained pair.")
        counts.set_norm(LogNorm(vmin=1, vmax=max(2, counts.get_array().max())))
        colorbar = ax.figure.colorbar(counts, ax=ax, pad=0.02)
        unit = "Hours" if summary["timescale"] == "hourly" else "Days"
        colorbar.set_label(
            f"{unit} per hexagon (log)", fontsize=plot_style.legend_font_size()
        )
        colorbar.ax.tick_params(labelsize=plot_style.legend_font_size())
    else:
        ax.scatter(
            data.tas_region,
            data.T_mean,
            s=plot_style.SCATTER_SIZE_PT2,
            alpha=plot_style.SCATTER_ALPHA,
            rasterized=True,
        )
    if summary["slope_K_per_K"] is not None:
        xx = np.array([data.tas_region.min(), data.tas_region.max()])
        ax.plot(
            xx,
            summary["slope_K_per_K"] * xx + summary["intercept_K"],
            color=plot_style.VARIABLE_COLORS["T_mean"],
            label="OLS fit",
        )
        ax.legend(**plot_style.legend_kwargs(loc="lower right"))
    annotation = f"N = {summary['n']:,}\nr = {format_stat(summary['pearson_r'])}\n"
    if not compact:
        annotation += f"rho = {format_stat(summary['spearman_rho'])}\n"
    annotation += (
        f"{RATIO_LABEL if compact else 'SD ratio'} = "
        f"{format_stat(summary['sd_ratio_T_mean_over_tas_region'])}"
    )
    ax.text(
        0.03,
        0.97,
        annotation,
        transform=ax.transAxes,
        va="top",
        fontsize=plot_style.legend_font_size(),
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none"},
    )
    ax.set(xlabel=f"{T2M_LABEL} [K]", ylabel=f"{TMEAN_LABEL} [K]")
    if compact:
        plot_style.format_integer_axis(ax.xaxis)
        plot_style.format_integer_axis(ax.yaxis)
    plot_style.limit_major_ticks(ax.xaxis, max_intervals=4, integer=compact)
    plot_style.limit_major_ticks(ax.yaxis, max_intervals=5, integer=compact)
    plot_style.style_axis(ax)


def plot_comparison(
    pairs: dict[str, pd.DataFrame],
    rows: list[dict],
    title: str,
    *,
    layout="diagnostic",
    kind="scatter",
    gridsize=plot_style.HEXBIN_GRIDSIZE,
):
    if layout not in {"diagnostic", "paper"}:
        raise ValueError(f"Unknown figure layout: {layout!r}")
    if kind not in {"scatter", "hexbin"}:
        raise ValueError(f"Unknown plot kind: {kind!r}")
    if isinstance(gridsize, bool) or not isinstance(gridsize, int) or gridsize < 2:
        raise ValueError("Hexbin gridsize must be an integer of at least 2.")
    plot_style.apply_theme()
    if layout == "paper":
        with plot_style.font_context("single"):
            fig, ax = plt.subplots(
                figsize=plot_style.publication_figsize("single"),
                constrained_layout=True,
            )
            summary = next(
                row
                for row in rows
                if row["timescale"] == "daily" and row["period"] == "all"
            )
            draw_scatter(
                ax, pairs["daily"], summary, compact=True, kind=kind, gridsize=gridsize
            )
            return fig
    fig, axes = plt.subplots(
        2,
        2,
        figsize=plot_style.publication_figsize("full", 0.8),
        constrained_layout=True,
    )
    for column, scale in enumerate(("hourly", "daily")):
        ax, ratio_ax = axes[:, column]
        if scale not in pairs:
            for panel in (ax, ratio_ax):
                panel.text(
                    0.5,
                    0.5,
                    "Hourly T2m not supplied\nDaily-only analysis",
                    ha="center",
                    va="center",
                    transform=panel.transAxes,
                )
                panel.set_axis_off()
            continue
        data = pairs[scale]
        summary = next(
            row for row in rows if row["timescale"] == scale and row["period"] == "all"
        )
        draw_scatter(ax, data, summary, kind=kind, gridsize=gridsize)
        ax.set_title(
            "Hourly native temperatures"
            if scale == "hourly"
            else "Daily means (event-selection T2m)"
        )
        yearly = [
            row for row in rows if row["timescale"] == scale and row["period"] != "all"
        ]
        ratios = [row["sd_ratio_T_mean_over_tas_region"] for row in yearly]
        ratio_ax.plot(
            [int(row["period"]) for row in yearly],
            [np.nan if value is None else value for value in ratios],
            marker="o",
            markersize=3,
            color=plot_style.VARIABLE_COLORS["T_mean"],
        )
        ratio_ax.axhline(
            1, color="0.4", linestyle="--", linewidth=plot_style.REFERENCE_LINE_WIDTH_PT
        )
        ratio_ax.set(
            xlabel="Year", ylabel=RATIO_LABEL, title=f"Yearly {scale} variability ratio"
        )
        ratio_ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        plot_style.use_default_numeric_formatter(ratio_ax.xaxis)
        ratio_ax.ticklabel_format(axis="x", style="plain", useOffset=False)
        plot_style.style_axis(ratio_ax)
    fig.suptitle(title)
    return fig


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def git_provenance() -> dict:
    def git(*args):
        try:
            return subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), *args],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = git("status", "--porcelain")
    return {
        "checkout": str(REPO_ROOT),
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": None if status is None else bool(status),
        "script_sha256": sha256(Path(__file__).resolve()),
    }


def input_provenance(path: Path, variables: tuple[str, ...]) -> dict:
    with xr.open_dataset(path, engine="h5netcdf", decode_timedelta=True) as ds:
        return {
            "path": str(path),
            "sha256": sha256(path),
            "attributes": dict(ds.attrs),
            "variables": {name: dict(ds[name].attrs) for name in variables},
        }


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def main(argv: list[str] | None = None) -> Path:
    args = parse_args(argv)
    input_path = args.input_path.expanduser().resolve()
    hourly_path = (
        args.hourly_t2m_path.expanduser().resolve() if args.hourly_t2m_path else None
    )
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {output_dir}")
    stage1_variables = ["T_mean", "tas_region"]
    if args.stage1_hourly_t2m:
        stage1_variables.append("tas_region_hourly")
    with analysis_io.open_harmonized_timeseries(input_path) as ds:
        if args.stage1_hourly_t2m and "tas_region_hourly" not in ds:
            raise ValueError(
                "Stage 1 lacks tas_region_hourly; supply --hourly-t2m-path or --daily-only."
            )
        stage1 = ds[stage1_variables].load()
    hourly = None
    hourly_variable = args.hourly_t2m_variable
    if hourly_path is not None:
        with xr.open_dataset(
            hourly_path, engine="h5netcdf", decode_timedelta=True
        ) as ds:
            if args.hourly_t2m_variable not in ds:
                raise ValueError(
                    f"Hourly companion is missing {args.hourly_t2m_variable!r}."
                )
            hourly = ds[[args.hourly_t2m_variable]].load()
    elif args.stage1_hourly_t2m:
        hourly = stage1[["tas_region_hourly"]].rename(tas_region_hourly="tas_region")
        hourly_variable = "tas_region"
    pairs, daily, coverage = prepare_pairs(
        stage1,
        hourly,
        hourly_variable=hourly_variable,
        months=args.season_months,
        start_year=args.start_year,
        end_year=args.end_year,
    )
    rows = statistics_table(pairs)
    inputs = {"stage1": input_provenance(input_path, tuple(stage1_variables))}
    if hourly_path is not None:
        inputs["hourly_t2m"] = input_provenance(
            hourly_path, (args.hourly_t2m_variable,)
        )
    elif args.stage1_hourly_t2m:
        inputs["hourly_t2m"] = {
            "source": "embedded Stage-1 tas_region_hourly",
            "path": str(input_path),
        }
    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "execution": git_provenance(),
        "argv": sys.argv[1:] if argv is None else argv,
        "python": sys.version,
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xarray": xr.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "inputs": inputs,
        "coverage": coverage,
        "selection": {
            "months_utc": args.season_months,
            "start_year": args.start_year,
            "end_year": args.end_year,
            "population": "all complete days, including non-events",
        },
        "definitions": {
            "hourly": "Native hourly T2m and T_mean; exact UTC matches on retained complete days.",
            "daily": "Stage-1 daily event T2m versus arithmetic 24-hour mean T_mean.",
            "daily_from_hourly": "Arithmetic 24-hour means of native hourly T2m and T_mean.",
            "variance": "Sample variance and SD, ddof=1; ratios are T_mean divided by T2m.",
            "regression": "OLS T_mean on T2m with intercept; undefined statistics are null.",
            "interpretation": "Raw temperatures with seasonality/trends retained; descriptive only, no p-values.",
        },
        "hourly_available": hourly is not None,
        "daily_source_agreement": source_agreement(daily),
        "statistics": rows,
    }
    # Validate serializability before reserving the no-overwrite output namespace.
    json.dumps(metadata, default=json_default, allow_nan=False)
    output_dir.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(rows).to_csv(output_dir / "statistics.csv", index=False)
    daily.to_csv(output_dir / "daily_pairs.csv")
    region = stage1.attrs["region"]
    bottom = stage1.attrs.get("heat_budget_bottom_boundary", "unspecified")
    top = stage1.attrs.get("heat_budget_top_boundary", "unspecified")
    season = (
        "JJA"
        if args.season_months == [6, 7, 8]
        else "months " + ",".join(map(str, args.season_months))
    )
    title = (
        f"{region}: {bottom} to {top} | {season}, {daily.index.year.min()}-{daily.index.year.max()}\n"
        f"Raw temperatures | {len(daily):,} complete UTC days | SD ratio > 1: greater atmospheric variability"
    )
    fig = plot_comparison(
        pairs,
        rows,
        title,
        layout=args.figure_layout,
        kind=args.plot_kind,
        gridsize=args.hexbin_gridsize,
    )
    paper = args.figure_layout == "paper"
    basename = (
        f"temperature_daily_{args.plot_kind}" if paper else "temperature_variability"
    )
    if not paper and args.plot_kind == "hexbin":
        basename += "_hexbin"
    metadata["figure"] = {
        "layout": args.figure_layout,
        "plot_kind": args.plot_kind,
        "temperature_tick_decimals": 0 if paper else plot_style.AXIS_TICK_DECIMALS,
        "hexbin": {
            "gridsize": args.hexbin_gridsize,
            "min_count": 1,
            "normalization": "logarithmic",
            "quantity": "retained pairs per hexagon (days for daily, hours for hourly)",
            "colormap": plot_style.COUNT_COLOR_MAP,
        }
        if args.plot_kind == "hexbin"
        else None,
        "basename": basename,
        "size_inches": fig.get_size_inches().tolist(),
        "font_size_pt": (
            plot_style.SINGLE_COLUMN_FONT_SIZE_PT
            if paper
            else plot_style.PAPER_FONT_SIZE_PT
        ),
        "legend_font_size_pt": (
            plot_style.SINGLE_COLUMN_LEGEND_FONT_SIZE_PT
            if paper
            else plot_style.LEGEND_FONT_SIZE_PT
        ),
        "crop": not paper,
        "style_sha256": sha256(Path(plot_style.__file__)),
    }
    try:
        for suffix in ("png", "pdf"):
            plot_style.save_figure(
                fig, output_dir / f"{basename}.{suffix}", crop=not paper
            )
    finally:
        plt.close(fig)
    metadata["output_sha256"] = {
        path.name: sha256(path) for path in sorted(output_dir.iterdir())
    }
    with (output_dir / "statistics.json").open("x") as handle:
        json.dump(metadata, handle, indent=2, default=json_default, allow_nan=False)
        handle.write("\n")
    for row in rows:
        if row["period"] == "all":
            print(
                f"{row['timescale']}: n={row['n']}, r={row['pearson_r']}, "
                f"SD(T_mean)/SD(T2m)={row['sd_ratio_T_mean_over_tas_region']}"
            )
    print(f"Saved temperature diagnostic to {output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
