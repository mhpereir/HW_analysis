"""Plot top-event diagnostics from the saved Stage-1 harmonized dataset.

This script consumes the harmonized regional time-series product written by
``build_regional_timeseries.py``. Event ranking and plotting will be layered on
top of this Stage-2 entrypoint.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, composites, plotting, selectors


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "plots_top_events"
DEFAULT_TOP_N = 10
DEFAULT_WINDOW_DAYS = 7
DEFAULT_RANK_METRIC = "tas_peak"
DEFAULT_SMOOTHING_WINDOW = 24
TOP_EVENT_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
)
EXTENDED_TOP_EVENT_VARIABLES: tuple[str, ...] = TOP_EVENT_VARIABLES + (
    "pbl_p_mean",
    "pbl_p_p05",
    "pbl_p_p95",
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)
SMOOTHED_TOP_EVENT_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
)
EXTENDED_SMOOTHED_TOP_EVENT_VARIABLES: tuple[str, ...] = SMOOTHED_TOP_EVENT_VARIABLES + (
    "pbl_p_mean",
    "pbl_p_p05",
    "pbl_p_p95",
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)
REFERENCE_EVENT_PERCENTILES: tuple[float, ...] = (0.25, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    """Parse command-line options for top-event plotting."""
    parser = argparse.ArgumentParser(
        description="Plot top-event diagnostics from a harmonized Stage-1 dataset."
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        default=analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
        help="Path to the saved harmonized Stage-1 regional dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where top-event figures will be written.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=DEFAULT_TOP_N,
        help="Number of peak-tas events to plot.",
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
        help="Hourly running-mean window for the smoothed top-event figures.",
    )
    parser.add_argument(
        "--plot-extended-variables",
        action="store_true",
        help="Plot optional extended diagnostics when present in the input dataset.",
    )
    return parser.parse_args()


def open_harmonized_dataset(
    path: str | Path = analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
    *,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open the harmonized Stage-1 dataset used by top-event plots."""
    return analysis_io.open_harmonized_timeseries(path, chunks=chunks)


def load_plot_inputs(args: argparse.Namespace) -> xr.Dataset:
    """Open the saved harmonized dataset requested by CLI args."""
    return open_harmonized_dataset(args.input_path)


def describe_harmonized_dataset(ds: xr.Dataset) -> None:
    """Print a compact summary of the loaded harmonized dataset."""
    dims_str = ", ".join(f"{dim}={size}" for dim, size in ds.sizes.items())
    vars_str = ", ".join(ds.data_vars) # type: ignore
    print("Loaded harmonized Stage-1 regional dataset:")
    print(f"  dims: {dims_str}")
    print(f"  vars: {vars_str}")


def select_top_tas_events(ds: xr.Dataset, *, n: int = DEFAULT_TOP_N) -> xr.Dataset:
    """Select top heatwave events by peak regional tas."""
    return selectors.select_top_n_events(
        ds,
        DEFAULT_RANK_METRIC,
        n,
        largest=True,
        keep_order="ranked",
    )


def write_top_event_plots(
    ds: xr.Dataset,
    selected_events: xr.Dataset,
    *,
    output_dir: Path,
    window_days: int = DEFAULT_WINDOW_DAYS,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
    plot_extended_variables: bool = False,
) -> list[Path]:
    """Write raw and display-smoothed time-series figures per selected event."""
    if window_days < 0:
        raise ValueError("window_days must be >= 0.")
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if selected_events.sizes.get("event", 0) == 0:
        return written

    variables = _top_event_variables(plot_extended_variables)
    smoothed_variables = _smoothed_top_event_variables(plot_extended_variables)
    reference_composite = composites.all_event_peak_aligned_composite(
        ds,
        variables=variables,
        pre_days=window_days,
        post_days=window_days,
        event_percentiles=REFERENCE_EVENT_PERCENTILES,
    )
    smoothed_reference_composite = plotting.smooth_composite_for_display(
        reference_composite,
        variables=smoothed_variables,
        smoothing_window=smoothing_window,
    )
    for event_index in range(selected_events.sizes.get("event", 0)):
        event = selected_events.isel(event=event_index)
        fig = plot_one_top_event(
            ds,
            event,
            window_days=window_days,
            reference_composite=reference_composite,
            plot_extended_variables=plot_extended_variables,
        )
        path = output_dir / _event_figure_filename(event)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)

        fig = plot_one_top_event(
            ds,
            event,
            window_days=window_days,
            reference_composite=smoothed_reference_composite,
            smoothing_window=smoothing_window,
            plot_extended_variables=plot_extended_variables,
        )
        path = output_dir / _smoothed_event_figure_filename(event)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def plot_one_top_event(
    ds: xr.Dataset,
    event: xr.Dataset,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    reference_composite: xr.Dataset | None = None,
    smoothing_window: int | None = None,
    plot_extended_variables: bool = False,
) -> plt.Figure: # type: ignore
    """Return a figure for one selected event."""
    peak_time = _event_time_value(event, "peak_time")
    window = np.timedelta64(window_days, "D")
    ds_window = ds.sel(time=slice(peak_time - window, peak_time + window))
    if smoothing_window is not None:
        ds_window = plotting.smooth_composite_for_display(
            ds_window,
            variables=_smoothed_top_event_variables(plot_extended_variables),
            smoothing_window=smoothing_window,
            lag_dim="time",
        )

    return plotting.plot_top_event_timeseries(
        ds_window,
        event,
        reference_composite=reference_composite,
        plot_extended_variables=plot_extended_variables,
    )


def _event_time_value(event: xr.Dataset, name: str) -> np.datetime64:
    """Return an event timestamp scalar as datetime64[ns]."""
    return np.asarray(event[name].values).astype("datetime64[ns]")[()] #type: ignore


def _event_figure_filename(event: xr.Dataset) -> str:
    """Return a stable filename for one selected event figure."""
    event_id = int(event["event_id"].item())
    rank = int(event["selection_rank"].item()) if "selection_rank" in event else event_id
    peak_time = _event_time_value(event, "peak_time")
    peak_day = np.datetime_as_string(peak_time, unit="D")
    return f"top_event_rank_{rank:02d}_event_{event_id:04d}_{peak_day}.png"


def _smoothed_event_figure_filename(event: xr.Dataset) -> str:
    """Return a stable filename for one display-smoothed selected event figure."""
    raw_name = _event_figure_filename(event)
    stem = Path(raw_name).stem
    return f"{stem}_smoothed.png"


def main() -> int:
    """Open the harmonized dataset and write top-event plots."""
    args = parse_args()
    ds = load_plot_inputs(args)
    try:
        describe_harmonized_dataset(ds)
        selected_events = select_top_tas_events(ds, n=args.top_n)
        written = write_top_event_plots(
            ds,
            selected_events,
            output_dir=args.output_dir,
            window_days=args.window_days,
            smoothing_window=args.smoothing_window,
            plot_extended_variables=args.plot_extended_variables,
        )
        print(f"Wrote {len(written)} top-event figures:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        ds.close()
    return 0


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _top_event_variables(plot_extended_variables: bool) -> tuple[str, ...]:
    """Return variables required for the selected top-event plot layout."""
    if plot_extended_variables:
        return EXTENDED_TOP_EVENT_VARIABLES
    return TOP_EVENT_VARIABLES


def _smoothed_top_event_variables(plot_extended_variables: bool) -> tuple[str, ...]:
    """Return display-smoothed variables for the selected top-event layout."""
    if plot_extended_variables:
        return EXTENDED_SMOOTHED_TOP_EVENT_VARIABLES
    return SMOOTHED_TOP_EVENT_VARIABLES


if __name__ == "__main__":
    raise SystemExit(main())
