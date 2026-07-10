"""Plot annual threshold/event diagnostics from a Stage-1 harmonized dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from matplotlib.axes import Axes

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, plot_paths, plot_style, preprocess

PLOT_NAME = "threshold_timeseries"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / f"plots_{PLOT_NAME}"


def parse_args() -> argparse.Namespace:
    """Parse command-line options for threshold-timeseries diagnostics."""
    parser = argparse.ArgumentParser(
        description="Plot annual Stage-1 threshold/event diagnostic time series."
    )
    plot_paths.add_stage1_path_arguments(parser)
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
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where diagnostic PNG files will be written.",
    )
    args = parser.parse_args()
    args = plot_paths.finalize_stage1_plot_paths(
        args,
        parser,
        plot_name=PLOT_NAME,
    )
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

    return args


def build_hw_plot_products(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Build HW plot inputs from a harmonized Stage-1 dataset."""
    return {
        "series": ds["tas_region"],
        "climatology": ds["tas_climatology"],
        "threshold": ds["hw_threshold"],
        "mask": ds["hw_flag"].astype(bool),
        "event_id": ds["hw_event_id"],
    }


def build_lwa_a_plot_products(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Build LWA_a plot inputs from a harmonized Stage-1 dataset."""
    climatology_time = _compute_dayofyear_climatology(
        ds["lwa_a_region"],
        name="lwa_a_climatology",
    )
    return {
        "series": ds["lwa_a_region"],
        "climatology": climatology_time,
        "threshold": ds["lwa_a_threshold"],
        "mask": ds["lwa_a_flag"].astype(bool),
        "event_id": ds["lwa_a_event_id"],
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
            figsize=plot_style.publication_figsize("full", aspect=0.68),
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
        plot_style.save_figure(fig, path)
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


def _filter_plot_products_years(
    product: dict[str, xr.DataArray],
    years: list[int],
) -> dict[str, xr.DataArray]:
    """Restrict every time-indexed plot product to selected calendar years."""
    return {
        name: da.where(da["time"].dt.year.isin(years), drop=True)
        for name, da in product.items()
    }


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
            color=plot_style.COLORS["volume"],
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
        color=plot_style.COLORS["temperature"],
        alpha=0.30,
        interpolate=False,
        label="series > threshold",
        zorder=1,
    )
    ax.plot(
        times,
        climatology_y.values,
        color=plot_style.COLORS["benchmark"],
        linewidth=plot_style.LINE_WIDTH_PT,
        label="climatology",
        zorder=2,
    )
    ax.plot(
        times,
        threshold_y.values,
        color=plot_style.COLORS["diabatic"],
        linewidth=plot_style.LINE_WIDTH_PT,
        label="threshold",
        zorder=3,
    )
    ax.plot(
        times,
        series_y.values,
        color=plot_style.COLORS["calculated"],
        linewidth=plot_style.LINE_WIDTH_PT,
        label="series",
        zorder=4,
    )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    plot_style.format_time_axis(ax)
    plot_style.style_axis(ax)
    ax.legend(loc="upper left", ncols=5, **plot_style.legend_kwargs())


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
    """Open Stage-1 data, build plot products, and write diagnostic plots."""
    args = parse_args()
    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        hw = build_hw_plot_products(ds)
        lwa_a = transform_lwa_a_for_plot(build_lwa_a_plot_products(ds))
        if args.years is not None:
            hw = _filter_plot_products_years(hw, args.years)
            lwa_a = _filter_plot_products_years(lwa_a, args.years)

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
    finally:
        ds.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
