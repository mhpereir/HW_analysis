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
from matplotlib.axes import Axes


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, selectors


DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "plots_top_events"
DEFAULT_TOP_N = 10
DEFAULT_WINDOW_DAYS = 7
DEFAULT_RANK_METRIC = "tas_peak"


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
) -> list[Path]:
    """Write one multi-panel time-series figure per selected event."""
    if window_days < 0:
        raise ValueError("window_days must be >= 0.")

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for event_index in range(selected_events.sizes.get("event", 0)):
        event = selected_events.isel(event=event_index)
        fig = plot_one_top_event(ds, event, window_days=window_days)
        path = output_dir / _event_figure_filename(event)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def plot_one_top_event(
    ds: xr.Dataset,
    event: xr.Dataset,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> plt.Figure: # type: ignore
    """Return a four-panel figure for one selected event."""
    peak_time = _event_time_value(event, "peak_time")
    start_time = _event_time_value(event, "start_time")
    end_time = _event_time_value(event, "end_time")
    window = np.timedelta64(window_days, "D")
    ds_window = ds.sel(time=slice(peak_time - window, peak_time + window))

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    axes_arr = np.asarray(axes, dtype=object)

    _plot_temperature_volume_panel(axes_arr[0], ds_window)
    _plot_single_variable_panel(axes_arr[1], ds_window, "dTdt", ylabel="dTdt")
    _plot_tendency_panel(axes_arr[2], ds_window)
    _plot_lwa_panel(axes_arr[3], ds_window)

    for ax in axes_arr:
        _shade_event(ax, start_time, end_time)
        ax.axvline(peak_time, color="0.2", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(True, linewidth=0.5, alpha=0.35)

    event_id = int(event["event_id"].item())
    rank = int(event["selection_rank"].item()) if "selection_rank" in event else event_id
    peak_value = float(event["tas_peak"].item()) if "tas_peak" in event else np.nan
    fig.suptitle(
        f"Rank {rank} HW event {event_id}: peak tas={peak_value:.2f}",
        fontsize=13,
    )
    axes_arr[-1].set_xlabel("Time")
    return fig


def _plot_temperature_volume_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot T_mean and volume with separate y axes."""
    ax.plot(ds["time"].values, ds["T_mean"].values, color="tab:red", label="T_mean")
    ax.set_ylabel("T_mean")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax_volume = ax.twinx()
    ax_volume.plot(ds["time"].values, ds["volume"].values, color="tab:blue", label="volume")
    ax_volume.set_ylabel("volume")
    ax_volume.tick_params(axis="y", labelcolor="tab:blue")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_volume.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")


def _plot_single_variable_panel(ax: Axes, ds: xr.Dataset, name: str, *, ylabel: str) -> None:
    """Plot one time-series variable."""
    ax.plot(ds["time"].values, ds[name].values, label=name, color="tab:purple")
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")


def _plot_tendency_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot heat-budget tendency terms on one axis."""
    for name in ("advection", "adiabatic", "diabatic"):
        ax.plot(ds["time"].values, ds[name].values, label=name)
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel("heat-budget terms")
    ax.legend(loc="upper left")


def _plot_lwa_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot LWA_a and LWA_c regional time series."""
    ax.plot(ds["time"].values, ds["lwa_a_region"].values, label="LWA_a_region")
    ax.plot(ds["time"].values, ds["lwa_c_region"].values, label="LWA_c_region")
    ax.set_ylabel("LWA")
    ax.legend(loc="upper left")


def _shade_event(ax: Axes, start_time: np.datetime64, end_time: np.datetime64) -> None:
    """Shade the selected event interval on one axis."""
    ax.axvspan(start_time, end_time, color="tab:orange", alpha=0.2)  #type: ignore


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


if __name__ == "__main__":
    raise SystemExit(main())
