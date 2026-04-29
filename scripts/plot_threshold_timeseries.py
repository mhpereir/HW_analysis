"""Plot annual threshold/event diagnostics for current ERA5 regional products.

This script is self-contained: it loads the required ERA5 inputs, rebuilds the
daily HW and LWA_a event-ID products, and writes one threshold-timeseries figure
per selected year. It does not depend on stage-1 output files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import typing
from matplotlib.axes import Axes

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import data_io, events, preprocess

DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "plots_diagnostics"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for threshold-timeseries diagnostics."""
    parser = argparse.ArgumentParser(
        description="Plot annual ERA5 threshold/event diagnostic time series."
    )
    parser.add_argument(
        "--region",
        default="pnw_bartusek",
        help="Region key used by threshold products.",
    )
    parser.add_argument(
        "--quantile",
        default="95",
        help="Threshold quantile token, for example 95 or 97p5.",
    )
    parser.add_argument(
        "--zg-level",
        type=int,
        default=500,
        help="Pressure level used for the LWA products.",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=None,
        metavar="YEAR",
        help=(
            "Optional year selection. Pass one year for a single-year run, or two "
            "years as START END to plot the inclusive range. If omitted, all "
            "loaded years are plotted."
        ),
    )
    parser.add_argument(
        "--min-duration",
        type=int,
        default=1,
        help="Minimum contiguous exceedance duration retained as an event.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where diagnostic PNG files will be written.",
    )
    args = parser.parse_args()
    if args.years is not None:
        if len(args.years) == 1:
            args.years = [args.years[0]]
        elif len(args.years) == 2:
            start_year, end_year = args.years
            if start_year > end_year:
                parser.error("--years START END requires START to be less than or equal to END.")
            args.years = list(range(start_year, end_year + 1))
        else:
            parser.error("--years accepts either one year or two years: START END.")

    if args.output_dir == DEFAULT_OUTPUT_DIR:
        args.output_dir = DEFAULT_OUTPUT_DIR / f"q{args.quantile}"

    return args


def load_plot_inputs(args: argparse.Namespace) -> dict[str, xr.Dataset]:
    """Open only the ERA5 inputs required by the plot workflow."""
    return {
        "tas": data_io.open_era5_tas(years=args.years),
        "lwa": data_io.open_era5_lwa(zg_level=args.zg_level),
        "lwa_threshold": data_io.open_era5_lwa_threshold(
            region=args.region,
            quantile=args.quantile,
            zg_level=args.zg_level,
        ),
        "hw_threshold": data_io.open_era5_hw_threshold(
            region=args.region,
            quantile=args.quantile,
            method="evolving",
        ),
    }


def build_hw_plot_products(
    tas: xr.DataArray,
    hw_threshold: xr.DataArray,
    hw_climatology: xr.DataArray,
    *,
    region: str,
    min_duration: int,
) -> dict[str, xr.DataArray]:
    """Build regional HW plot inputs from tas and HW threshold products."""
    products = events.build_hw_event_ids(
        tas,
        hw_threshold,
        hw_climatology,
        region=region,
        min_duration=min_duration,
    )
    return {
        "series": products["tas_region"],
        "climatology": products["tas_climatology"],
        "threshold": products["hw_threshold"],
        "mask": products["hw_exceedance_mask"],
        "event_id": products["hw_event_id"],
    }


def build_lwa_a_plot_products(
    lwa_a: xr.DataArray,
    lwa_a_threshold: xr.DataArray,
    *,
    region: str,
    years: list[int] | None,
    min_duration: int,
) -> dict[str, xr.DataArray]:
    """Build regional LWA_a plot inputs from LWA_a and LWA threshold products."""
    products = events.build_lwa_event_ids(
        lwa_a,
        lwa_a_threshold,
        region=region,
        variable="LWA_a",
        years=years,
        min_duration=min_duration,
    )
    climatology_time = _compute_dayofyear_climatology(
        products["lwa_a_region"],
        name="lwa_a_climatology",
    )
    return {
        "series": products["lwa_a_region"],
        "climatology": climatology_time,
        "threshold": products["lwa_a_threshold"],
        "mask": products["lwa_a_exceedance_mask"],
        "event_id": products["lwa_a_event_id"],
    }


def transform_lwa_a_for_plot(product: dict[str, xr.DataArray]) -> dict[str, xr.DataArray]:
    """Return LWA_a plot products with magnitude variables transformed to sqrt scale."""
    return {
        **product,
        "series": _sqrt_nonnegative(product["series"], name="sqrt_lwa_a"),
        "climatology": _sqrt_nonnegative(
            product["climatology"],
            name="sqrt_lwa_a_climatology",
        ),
        "threshold": _sqrt_nonnegative(
            product["threshold"],
            name="sqrt_lwa_a_threshold",
        ),
    }


def _sqrt_nonnegative(
    da: xr.DataArray,
    *,
    name: str,
    tolerance: float = 0,
) -> xr.DataArray:
    """Apply sqrt while preserving xarray metadata and rejecting invalid negatives."""
    minimum = da.min(skipna=True)
    if bool((minimum < -tolerance).compute()):
        raise ValueError(
            f"{da.name or 'DataArray'} contains negative values; cannot plot sqrt scale."
        )

    transformed = da ** 0.5 #square root of non-negative values
    transformed.name = name
    attrs = da.attrs.copy()
    if "units" in attrs:
        attrs["units"] = f"sqrt({attrs['units']})"
    transformed.attrs = attrs
    return transformed


def write_threshold_timeseries_plots(
    hw: dict[str, xr.DataArray],
    lwa_a: dict[str, xr.DataArray],
    *,
    region: str,
    quantile: str,
    output_dir: Path,
) -> list[Path]:
    """Write one two-panel threshold/event diagnostic figure per year."""
    output_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for year in _years_in_time(hw["series"]):
        fig, axes = plt.subplots(
            nrows=2,
            ncols=1,
            figsize=(13, 8),
            sharex=False,
            constrained_layout=True,
        )
        _plot_event_panel(
            axes[0],
            year=year,
            title="HW events from regional tas",
            product=hw,
            ylabel="tas",
        )
        _plot_event_panel(
            axes[1],
            year=year,
            title="LWA_a events from regional LWA_a",
            product=lwa_a,
            ylabel=r"$\sqrt{LWA\ [m\ hPa]}$",
        )
        fig.suptitle(f"{region} q{quantile} event diagnostics, {year}")
        path = output_dir / f"{region}_q{quantile}_{year}_event_diagnostics.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

    return written


def _compute_dayofyear_climatology(
    series: xr.DataArray,
    *,
    name: str,
) -> xr.DataArray:
    """Compute and project a selected-period day-of-year climatology."""
    climatology = series.groupby("time.dayofyear").mean("time")
    return preprocess.threshold_to_time(
        climatology,
        series["time"],
        name=name,
    )


def _years_in_time(da: xr.DataArray) -> list[int]:
    """Return sorted year values present in a time-indexed DataArray."""
    years = np.unique(da["time"].dt.year.values)
    return [int(year) for year in years]


def _plot_event_panel(
    ax: Axes, 
    *,
    year: int,
    title: str,
    product: dict[str, xr.DataArray],
    ylabel: str,
) -> None:
    """Plot one annual event-diagnostic panel."""
    series = product["series"]
    year_mask = series["time"].dt.year == year
    series_y = series.where(year_mask, drop=True).compute()
    climatology_y = product["climatology"].where(year_mask, drop=True).compute()
    threshold_y = product["threshold"].where(year_mask, drop=True).compute()
    mask_y = product["mask"].where(year_mask, drop=True).compute()
    event_id_y = product["event_id"].where(year_mask, drop=True).compute()

    times = series_y["time"].values
    mask_values = mask_y.values.astype(bool).ravel().tolist()
    event_values = event_id_y.values != 0

    for idx, (start, stop) in enumerate(_true_runs(times, event_values)):
        label = "event_id != 0" if idx == 0 else None
        ax.axvspan(
            start, #type: ignore
            stop,  #type: ignore
            color="tab:blue",
            alpha=0.12,
            linewidth=0,
            label=label,
            zorder=0,
        )

    ax.fill_between(
        times,
        threshold_y.values,
        series_y.values,
        where=mask_values,
        color="tab:orange",
        alpha=0.30,
        interpolate=False,
        label="series > threshold",
        zorder=1,
    )
    ax.plot(
        times,
        climatology_y.values,
        color="0.45",
        linewidth=1.2,
        label="climatology",
        zorder=2,
    )
    ax.plot(
        times,
        threshold_y.values,
        color="tab:red",
        linewidth=1.4,
        label="threshold",
        zorder=3,
    )
    ax.plot(times, series_y.values, color="black", linewidth=1.1, label="series", zorder=4)

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", ncols=5, fontsize=8)


def _true_runs(times: np.ndarray, values: np.ndarray) -> list[tuple[np.datetime64, np.datetime64]]:
    """Return inclusive/exclusive plot spans for contiguous true runs."""
    runs: list[tuple[np.datetime64, np.datetime64]] = []
    idx = 0
    while idx < values.size:
        if not values[idx]:
            idx += 1
            continue

        start = idx
        while idx < values.size and values[idx]:
            idx += 1
        stop = idx

        start_time = times[start]
        if stop < times.size:
            stop_time = times[stop]
        else:
            stop_time = times[stop - 1] + np.timedelta64(1, "D")
        runs.append((start_time, stop_time))

    return runs


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    """Load data, build event products, and write diagnostic plots."""
    args = parse_args()
    datasets = load_plot_inputs(args)

    hw = build_hw_plot_products(
        datasets["tas"]["tas"],
        datasets["hw_threshold"]["threshold"],
        datasets["hw_threshold"]["climatology"],
        region=args.region,
        min_duration=args.min_duration,
    )
    lwa_a = build_lwa_a_plot_products(
        datasets["lwa"]["LWA_a"],
        datasets["lwa_threshold"]["LWA_a"],
        region=args.region,
        years=args.years,
        min_duration=args.min_duration,
    )
    lwa_a = transform_lwa_a_for_plot(lwa_a)

    written = write_threshold_timeseries_plots(
        hw,
        lwa_a,
        region=args.region,
        quantile=args.quantile,
        output_dir=args.output_dir,
    )

    print("Threshold timeseries plots:")
    for path in written:
        print(f"  {_display_path(path)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
