"""Plot peak-aligned composites from the saved Stage-1 harmonized dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, composites, plotting, selectors


DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "plots_composites_split" / "hw_events_composite.png"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_SMOOTHING_WINDOW = 24
SPLIT_BIN_DIM = "split_bin"
COMPOSITE_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
)
EXTENDED_COMPOSITE_VARIABLES: tuple[str, ...] = COMPOSITE_VARIABLES + (
    "pbl_p_mean",
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)
SMOOTHED_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
)
EXTENDED_SMOOTHED_VARIABLES: tuple[str, ...] = SMOOTHED_VARIABLES + (
    "pbl_p_mean",
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for composite time-series plotting."""
    parser = argparse.ArgumentParser(
        description="Plot peak-aligned composite time series for all HW events."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
        help="Path to the saved harmonized Stage-1 regional dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path where the composite PNG will be written.",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=DEFAULT_WINDOW_DAYS,
        help="Number of days to include on each side of event peak time.",
    )
    parser.add_argument(
        "--smoothing-window",
        type=int,
        default=DEFAULT_SMOOTHING_WINDOW,
        help="Hourly running-mean window for the smoothed composite figure.",
    )
    parser.add_argument(
        "--split-variable",
        type=str,
        required=True,
        help="Variable to split the composite by. Examples: duration, peak_time, peak_value, tas_peak, tas_anom_peak, tas_excess_peak, tas_excess_integral, lwa_a_peak, lwa_c_peak",
    )
    parser.add_argument(
        "--split-quantiles",
        type=float,
        nargs="+",
        help="Quantiles to split the composite by, e.g. 0.25 0.5 0.75.",
    )
    parser.add_argument(
        "--split-years",
        type=int,
        nargs="+",
        help=(
            "Calendar years where peak-time bins start. Used only with "
            "--split-variable peak_time, e.g. 1980 2000."
        ),
    )
    parser.add_argument(
        "--season-months",
        type=int,
        nargs="+",
        default=None,
        metavar="MONTH",
        help="Optional calendar months to retain before compositing, e.g. 6 7 8.",
    )
    parser.add_argument(
        "--require-full-event",
        action="store_true",
        help="Require the full event interval to fall within --season-months.",
    )
    parser.add_argument(
        "--plot-extended-variables",
        action="store_true",
        help="Plot optional extended diagnostics when present in the input dataset.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate composite plotting CLI arguments."""
    if args.window_days < 0:
        raise ValueError("--window-days must be >= 0.")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1.")
    if args.split_variable == "peak_time":
        _validate_split_years(args.split_years)
        if args.split_quantiles is not None:
            raise ValueError("--split-variable peak_time uses --split-years, not --split-quantiles.")
    else:
        _validate_split_quantiles(args.split_quantiles)
        if args.split_years is not None:
            raise ValueError("--split-years is only supported with --split-variable peak_time.")
    if args.require_full_event and args.season_months is None:
        raise ValueError("--require-full-event requires --season-months.")
    if args.season_months is not None:
        _validate_season_months(args.season_months)


def main() -> int:
    """Open the harmonized dataset and write the split-bin composite figure."""
    args = parse_args()
    validate_args(args)

    ds = analysis_io.open_harmonized_timeseries(args.input_path)
    try:
        variables = _composite_variables(args.plot_extended_variables)
        smoothed_variables = _smoothed_variables(args.plot_extended_variables)
        composite_kwargs: dict[str, object] = {
            "variables": variables,                 # Sequence[str]
            "pre_days": args.window_days,           # int
            "post_days": args.window_days,          # int
            "event_percentiles": (0.25, 0.5, 0.75), # Sequence[float]
        }
        event_table = ds
        if args.season_months is not None:
            event_table = selectors.select_events_by_season(
                ds,
                args.season_months,
                require_full_event=args.require_full_event,
            )
            if event_table.sizes.get("event", 0) == 0:
                months = " ".join(str(month) for month in args.season_months)
                raise ValueError(f"No events remain after filtering to season months: {months}.")

        if args.split_variable == "peak_time":
            composite = build_split_year_composite(
                ds,
                event_table=event_table,
                split_years=args.split_years,
                composite_kwargs=composite_kwargs,
            )
        else:
            composite = build_split_quantile_composite(
                ds,
                event_table=event_table,
                split_variable=args.split_variable,
                split_quantiles=args.split_quantiles,
                composite_kwargs=composite_kwargs,
            )
        output_path = _split_output_path(args.output_path, args.split_variable)
        written = plotting.write_split_composite_timeseries_outputs(
            composite,
            output_path,
            smoothed_output_path=_smoothed_output_path(output_path),
            smoothing_window=args.smoothing_window,
            smoothed_variables=smoothed_variables,
            plot_extended_variables=args.plot_extended_variables,
        )
        print("Wrote HW split-bin composite figures:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        ds.close()
    return 0


def build_split_quantile_composite(
    ds: xr.Dataset,
    *,
    event_table: xr.Dataset,
    split_variable: str,
    split_quantiles: list[float] | tuple[float, ...],
    composite_kwargs: dict[str, object],
) -> xr.Dataset:
    """Build one composite per split-variable quantile bin."""
    _validate_split_variable(event_table, split_variable)
    quantiles = _validate_split_quantiles(split_quantiles)
    bounds = (0.0, *quantiles, 1.0)

    bin_composites: list[xr.Dataset] = []
    labels: list[str] = []
    qmins: list[float] = []
    qmaxs: list[float] = []
    lower_values: list[float] = []
    upper_values: list[float] = []
    n_events: list[int] = []

    for index, (qmin, qmax) in enumerate(zip(bounds[:-1], bounds[1:], strict=True)):
        inclusive = "both" if index == 0 else "right"
        selected = selectors.select_event_quantile_bin(
            event_table,
            split_variable,
            qmin=qmin,
            qmax=qmax,
            inclusive=inclusive,
        )
        selected_count = int(selected.sizes.get("event", 0))
        if selected_count == 0:
            raise ValueError(
                f"Split bin {qmin:g}-{qmax:g} for {split_variable!r} contains no events."
            )

        composite = composites.all_event_peak_aligned_composite(
            ds,
            event_table=selected,
            **composite_kwargs, # type: ignore
        )
        label = _split_bin_label(qmin, qmax, selected_count)
        bin_composites.append(composite)
        labels.append(label)
        qmins.append(float(qmin))
        qmaxs.append(float(qmax))
        lower_values.append(float(selected.attrs["selection_lower_value"]))
        upper_values.append(float(selected.attrs["selection_upper_value"]))
        n_events.append(selected_count)

    split = xr.concat(
        bin_composites,
        dim=xr.IndexVariable(SPLIT_BIN_DIM, labels),
    )
    split = split.assign_coords(
        split_qmin=(SPLIT_BIN_DIM, np.asarray(qmins, dtype=float)),
        split_qmax=(SPLIT_BIN_DIM, np.asarray(qmaxs, dtype=float)),
        split_lower_value=(SPLIT_BIN_DIM, np.asarray(lower_values, dtype=float)),
        split_upper_value=(SPLIT_BIN_DIM, np.asarray(upper_values, dtype=float)),
        split_n_events=(SPLIT_BIN_DIM, np.asarray(n_events, dtype=np.int64)),
    )
    split.attrs.update(
        {
            "composite_reduction": "mean over split HW event quantile bins",
            "split_variable": split_variable,
            "split_quantiles": ",".join(str(value) for value in quantiles),
            "split_bin_labels": ",".join(labels),
            "n_events": int(np.sum(n_events)),
        }
    )
    return split


def build_split_year_composite(
    ds: xr.Dataset,
    *,
    event_table: xr.Dataset,
    split_years: list[int] | tuple[int, ...],
    composite_kwargs: dict[str, object],
) -> xr.Dataset:
    """Build one composite per peak-time calendar-year bin."""
    _validate_peak_time_variable(event_table, "peak_time")
    years = _event_peak_years(event_table["peak_time"])
    finite_years = years[~np.isnan(years)]
    if finite_years.size == 0:
        raise ValueError("Split variable 'peak_time' contains no finite timestamps.")

    first_year = int(np.min(finite_years))
    last_year = int(np.max(finite_years))
    cut_years = _validate_split_years(split_years)
    invalid = [year for year in cut_years if year <= first_year or year > last_year]
    if invalid:
        text = ", ".join(str(year) for year in invalid)
        raise ValueError(
            "--split-years must be within the filtered peak-time year range "
            f"({first_year + 1}-{last_year}); got {text}."
        )

    starts = (first_year, *cut_years)
    ends = (*(year - 1 for year in cut_years), last_year)

    bin_composites: list[xr.Dataset] = []
    labels: list[str] = []
    start_values: list[int] = []
    end_values: list[int] = []
    n_events: list[int] = []

    for start_year, end_year in zip(starts, ends, strict=True):
        selected_mask = (years >= start_year) & (years <= end_year)
        selected_indices = np.flatnonzero(selected_mask)
        selected_count = int(selected_indices.size)
        if selected_count == 0:
            raise ValueError(
                f"Split year bin {start_year}-{end_year} for 'peak_time' contains no events."
            )

        selected = event_table.isel(event=selected_indices)
        composite = composites.all_event_peak_aligned_composite(
            ds,
            event_table=selected,
            **composite_kwargs, # type: ignore
        )
        label = _split_year_bin_label(start_year, end_year, selected_count)
        bin_composites.append(composite)
        labels.append(label)
        start_values.append(int(start_year))
        end_values.append(int(end_year))
        n_events.append(selected_count)

    split = xr.concat(
        bin_composites,
        dim=xr.IndexVariable(SPLIT_BIN_DIM, labels),
    )
    split = split.assign_coords(
        split_start_year=(SPLIT_BIN_DIM, np.asarray(start_values, dtype=np.int64)),
        split_end_year=(SPLIT_BIN_DIM, np.asarray(end_values, dtype=np.int64)),
        split_n_events=(SPLIT_BIN_DIM, np.asarray(n_events, dtype=np.int64)),
    )
    split.attrs.update(
        {
            "composite_reduction": "mean over split HW event peak-time year bins",
            "split_variable": "peak_time",
            "split_type": "year_bin",
            "split_years": ",".join(str(value) for value in cut_years),
            "split_bin_labels": ",".join(labels),
            "n_events": int(np.sum(n_events)),
        }
    )
    return split


def _validate_split_quantiles(
    quantiles: list[float] | tuple[float, ...] | None,
) -> tuple[float, ...]:
    """Return sorted, validated split quantiles."""
    if quantiles is None:
        raise ValueError("--split-quantiles must provide at least one quantile.")
    values = tuple(sorted(float(value) for value in quantiles))
    if not values:
        raise ValueError("--split-quantiles must provide at least one quantile.")
    invalid = [value for value in values if not 0.0 < value < 1.0]
    if invalid:
        text = ", ".join(str(value) for value in invalid)
        raise ValueError(
            f"--split-quantiles must be strictly between 0 and 1; got {text}."
        )
    unique_values = set(values)
    if len(unique_values) != len(values):
        raise ValueError("--split-quantiles must not contain duplicate values.")
    return values


def _validate_split_years(
    years: list[int] | tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Return sorted, validated peak-time split years."""
    if years is None:
        raise ValueError("--split-years must provide at least one year.")
    values = tuple(sorted(int(year) for year in years))
    if not values:
        raise ValueError("--split-years must provide at least one year.")
    invalid = [year for year in values if year < 1 or year > 9999]
    if invalid:
        text = ", ".join(str(year) for year in invalid)
        raise ValueError(f"--split-years values must be between 1 and 9999; got {text}.")
    unique_values = set(values)
    if len(unique_values) != len(values):
        raise ValueError("--split-years must not contain duplicate values.")
    return values


def _validate_split_variable(event_table: xr.Dataset, split_variable: str) -> None:
    """Validate that the requested split variable is a numeric event-summary variable."""
    if split_variable not in event_table:
        raise ValueError(f"event table is missing split variable {split_variable!r}.")
    da = event_table[split_variable]
    if da.dims != ("event",):
        raise ValueError(
            f"Split variable {split_variable!r} must be 1D with dims ('event',); "
            f"got {da.dims!r}."
        )
    if not np.issubdtype(da.dtype, np.number):
        raise TypeError(f"Split variable {split_variable!r} must be numeric.")


def _validate_peak_time_variable(event_table: xr.Dataset, split_variable: str) -> None:
    """Validate that the requested split variable is a datetime event-summary variable."""
    if split_variable not in event_table:
        raise ValueError(f"event table is missing split variable {split_variable!r}.")
    da = event_table[split_variable]
    if da.dims != ("event",):
        raise ValueError(
            f"Split variable {split_variable!r} must be 1D with dims ('event',); "
            f"got {da.dims!r}."
        )
    if not np.issubdtype(da.dtype, np.datetime64):
        raise TypeError(f"Split variable {split_variable!r} must be datetime64.")


def _event_peak_years(peak_time: xr.DataArray) -> np.ndarray:
    """Return event peak years as floats, using NaN for missing timestamps."""
    peak_values = np.asarray(peak_time.values).astype("datetime64[ns]")
    years = np.full(peak_values.shape, np.nan, dtype=float)
    finite = ~np.isnat(peak_values)
    years[finite] = peak_values[finite].astype("datetime64[Y]").astype(int) + 1970
    return years


def _split_bin_label(qmin: float, qmax: float, n_events: int) -> str:
    """Return a compact split-bin label for legends and coordinates."""
    return f"q{qmin:g}-{qmax:g} (n={n_events})"


def _split_year_bin_label(start_year: int, end_year: int, n_events: int) -> str:
    """Return a compact year-bin label for legends and coordinates."""
    return f"{start_year}-{end_year} (n={n_events})"


def _validate_season_months(months: list[int]) -> None:
    """Validate CLI season-month values."""
    invalid = [month for month in months if month < 1 or month > 12]
    if invalid:
        values = ", ".join(str(month) for month in invalid)
        raise ValueError(f"--season-months values must be between 1 and 12; got {values}.")


def _composite_variables(plot_extended_variables: bool) -> tuple[str, ...]:
    """Return variables required for the selected split composite layout."""
    if plot_extended_variables:
        return EXTENDED_COMPOSITE_VARIABLES
    return COMPOSITE_VARIABLES


def _smoothed_variables(plot_extended_variables: bool) -> tuple[str, ...]:
    """Return display-smoothed variables for the selected split layout."""
    if plot_extended_variables:
        return EXTENDED_SMOOTHED_VARIABLES
    return SMOOTHED_VARIABLES


def _split_output_path(output_path: Path, split_variable: str) -> Path:
    """Return output path with the split-variable token added to the stem."""
    token = _filename_token(split_variable)
    if output_path.stem.endswith(f"_{token}"):
        return output_path
    return output_path.with_name(f"{output_path.stem}_{token}{output_path.suffix}")


def _smoothed_output_path(output_path: Path) -> Path:
    """Return sibling path for the smoothed composite figure."""
    if output_path.name == DEFAULT_OUTPUT_PATH.name:
        return output_path.with_name("hw_all_events_composite_smoothed.png")
    return output_path.with_name(f"{output_path.stem}_smoothed{output_path.suffix}")


def _filename_token(value: str) -> str:
    """Return a conservative filename token."""
    token = "".join(
        char.lower() if char.isascii() and char.isalnum() else "_"
        for char in value.strip()
    )
    token = "_".join(part for part in token.split("_") if part)
    if not token:
        raise ValueError("split variable must contain at least one filename-safe character.")
    return token


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
