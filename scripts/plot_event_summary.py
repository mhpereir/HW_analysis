"""Plot histograms of event summary table variables."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from src import analysis_io, plot_paths, plot_style, selectors


PLOT_NAME = "event_summary"
DEFAULT_OUTPUT_FILENAME = "event_summary_histograms.png"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / f"plots_{PLOT_NAME}" / DEFAULT_OUTPUT_FILENAME
DEFAULT_BINS = 30
DEFAULT_EVENT_DIM = "event"
DEFAULT_EXCLUDED_VARIABLES: frozenset[str] = frozenset(
    {
        "event_id",
        "start_time",
        "end_time",
        "peak_time",
    }
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for event-summary histogram plotting."""
    parser = argparse.ArgumentParser(
        description="Plot histograms of variables in the saved event summary table."
    )
    plot_paths.add_stage1_path_arguments(parser)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Path where the histogram PNG will be written.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Event-summary variables to plot. Defaults to all numeric event-level "
            "variables except event_id."
        ),
    )
    parser.add_argument(
        "--bins",
        type=int,
        default=DEFAULT_BINS,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Optional calendar months to retain before plotting, e.g. 6 7 8.",
    )
    parser.add_argument(
        "--require-full-event",
        action="store_true",
        help="Require the full event interval to fall within --season-months.",
    )
    args = parser.parse_args()
    return plot_paths.finalize_stage1_plot_paths(
        args,
        parser,
        plot_name=PLOT_NAME,
        default_output_filename=DEFAULT_OUTPUT_FILENAME,
    )


def validate_args(args: argparse.Namespace) -> None:
    """Validate histogram plotting CLI arguments."""
    if args.bins < 1:
        raise ValueError("--bins must be >= 1.")
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if args.season_months is not None:
        _validate_season_months(args.season_months)


def main() -> int:
    """Open the harmonized dataset and write event summary histograms."""
    args = parse_args()
    validate_args(args)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        event_table = _event_summary_table(ds)
        if args.season_months is not None:
            event_table = selectors.select_events_by_season(
                event_table,
                args.season_months,
                require_full_event=args.require_full_event,
            )
            if event_table.sizes.get(DEFAULT_EVENT_DIM, 0) == 0:
                months = " ".join(str(month) for month in args.season_months)
                raise ValueError(f"No events remain after filtering to season months: {months}.")

        variables = _selected_variables(event_table, args.variables)
        written = write_event_summary_histograms(
            event_table,
            args.output_path,
            variables=variables,
            bins=args.bins,
        )
        print("Wrote event summary histogram figure:")
        print(f"  {_display_path(written)}")
    finally:
        ds.close()
    return 0


def write_event_summary_histograms(
    event_table: xr.Dataset,
    output_path: Path,
    *,
    variables: list[str],
    bins: int,
) -> Path:
    """Write a multi-panel histogram figure for event-level variables."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_event_summary_histograms(event_table, variables=variables, bins=bins)
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def plot_event_summary_histograms(
    event_table: xr.Dataset,
    *,
    variables: list[str],
    bins: int,
) -> plt.Figure: # type: ignore
    """Return a figure containing one histogram per requested variable."""
    n_variables = len(variables)
    ncols = 3 if n_variables > 2 else n_variables
    nrows = math.ceil(n_variables / ncols)
    fig_width = (
        plot_style.FULL_TWO_COLUMN_WIDTH_IN
        if ncols > 1
        else plot_style.SINGLE_COLUMN_WIDTH_IN
    )
    fig, axes_array = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, 3.2 * nrows),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes_array).ravel()

    for ax, name in zip(axes, variables):
        _plot_variable_histogram(ax, event_table[name], bins=bins)

    for ax in axes[n_variables:]:
        ax.set_visible(False)

    n_events = int(event_table.sizes.get(DEFAULT_EVENT_DIM, 0))
    title_parts = [f"Event summary histograms (n={n_events})"]
    if "selection_type" in event_table.attrs:
        title_parts.append(str(event_table.attrs["selection_type"]))
    fig.suptitle(" - ".join(title_parts))
    return fig


def _plot_variable_histogram(ax: Axes, da: xr.DataArray, *, bins: int) -> None:
    """Plot one event-summary variable histogram."""
    values = _finite_values(da)
    ax.set_title(_display_name(da.name)) # type: ignore
    ax.set_ylabel("Events")

    if values.size == 0:
        ax.text(
            0.5,
            0.5,
            "No finite values",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        plot_style.style_axis(ax)
        return

    if np.issubdtype(da.dtype, np.datetime64):
        dates = mdates.date2num(values.astype("datetime64[ms]").astype(object))
        ax.hist(
            dates,
            bins=bins,
            color=plot_style.COLORS["volume"],
            edgecolor="white",
            linewidth=0.6,
        )
        ax.xaxis_date()
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    else:
        hist_bins = _histogram_bins(values, da, bins=bins)
        ax.hist(
            values,
            bins=hist_bins,
            color=plot_style.COLORS["volume"],
            edgecolor="white",
            linewidth=0.6,
        )
        if _uses_integer_day_bins(da, values):
            ax.set_xticks(np.arange(int(values.min()), int(values.max()) + 1))

    xlabel = _axis_label(da)
    if xlabel:
        ax.set_xlabel(xlabel)
    plot_style.style_axis(ax)


def _event_summary_table(ds: xr.Dataset) -> xr.Dataset:
    """Return the event-level portion of a harmonized dataset."""
    if DEFAULT_EVENT_DIM not in ds.dims:
        raise ValueError(
            "Input dataset does not contain an event summary table; "
            f"missing dimension {DEFAULT_EVENT_DIM!r}."
        )

    names = [
        name
        for name, da in ds.data_vars.items()
        if DEFAULT_EVENT_DIM in da.dims and set(da.dims).issubset({DEFAULT_EVENT_DIM})
    ]
    if not names:
        raise ValueError("Input dataset contains no 1D event-summary variables.")
    return ds[names]


def _selected_variables(event_table: xr.Dataset, requested: list[str] | None) -> list[str]:
    """Return variables that should be included in the histogram figure."""
    if requested is not None:
        missing = [name for name in requested if name not in event_table]
        if missing:
            valid = ", ".join(sorted(str(name) for name in event_table.data_vars))
            raise ValueError(
                "Requested event-summary variables are missing: "
                f"{', '.join(missing)}. Valid variables: {valid}."
            )
        invalid = [name for name in requested if not _is_plottable(event_table[name])]
        if invalid:
            raise ValueError(
                "Requested variables are not numeric or datetime-like event-level "
                f"variables: {', '.join(invalid)}."
            )
        return requested

    variables = [
        str(name)
        for name, da in event_table.data_vars.items()
        if name not in DEFAULT_EXCLUDED_VARIABLES and _is_plottable(da)
    ]
    if not variables:
        raise ValueError("No numeric event-summary variables were found to plot.")
    return variables 


def _is_plottable(da: xr.DataArray) -> bool:
    """Return True for event-level numeric, datetime, or timedelta variables."""
    return (
        np.issubdtype(da.dtype, np.number)
        or np.issubdtype(da.dtype, np.datetime64)
        or np.issubdtype(da.dtype, np.timedelta64)
    )


def _finite_values(da: xr.DataArray) -> np.ndarray:
    """Return finite non-missing values from a 1D event-summary variable."""
    values = np.asarray(da.values)
    if np.issubdtype(da.dtype, np.datetime64):
        values = values[~np.isnat(values)]
        return values
    if np.issubdtype(da.dtype, np.timedelta64):
        values = values[~np.isnat(values)]
        return values / np.timedelta64(1, "D")

    values = values.astype(float, copy=False)
    return values[np.isfinite(values)]


def _histogram_bins(
    values: np.ndarray,
    da: xr.DataArray,
    *,
    bins: int,
) -> int | list[float]:
    """Return explicit bin edges for discrete day counts, otherwise the CLI bin count."""
    if not _uses_integer_day_bins(da, values):
        return bins

    min_day = int(np.nanmin(values))
    max_day = int(np.nanmax(values))
    return np.arange(min_day - 0.5, max_day + 1.5, 1.0).tolist()


def _uses_integer_day_bins(da: xr.DataArray, values: np.ndarray) -> bool:
    """Return True when values should be shown as one bin per integer day."""
    return bool(
        str(da.name) == "duration"
        and values.size > 0
        and bool(np.all(np.isclose(values, np.round(values))))
    )

def _display_name(name: str | None) -> str:
    """Return a readable title for a variable name."""
    if name is None:
        return ""
    return name.replace("_", " ")


def _axis_label(da: xr.DataArray) -> str:
    """Return an x-axis label using variable name and units when available."""
    name = _display_name(str(da.name))
    if np.issubdtype(da.dtype, np.timedelta64):
        return f"{name} [days]"
    units = da.attrs.get("units")
    if units:
        return f"{name} [{units}]"
    return name


def _validate_season_months(months: list[int]) -> None:
    """Validate CLI season-month values."""
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        values = ", ".join(str(month) for month in invalid)
        raise ValueError(f"--season-months values must be between 1 and 12; got {values}.")


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
