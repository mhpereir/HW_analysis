"""Plot peak-aligned composites from the saved Stage-1 harmonized dataset."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path

import matplotlib
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from src import analysis_io, composites


DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / "plots_composites" / "hw_all_events_composite.png"
DEFAULT_WINDOW_DAYS = 7
DEFAULT_SMOOTHING_WINDOW = 24
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
SMOOTHED_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
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
    return parser.parse_args()


def open_harmonized_dataset(
    path: str | Path = analysis_io.DEFAULT_HARMONIZED_TIMESERIES_PATH,
    *,
    chunks: Mapping[str, int] | None = None,
) -> xr.Dataset:
    """Open the harmonized Stage-1 dataset used by composite plots."""
    return analysis_io.open_harmonized_timeseries(path, chunks=chunks)


def build_all_hw_event_stack(
    ds: xr.Dataset,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
) -> xr.Dataset:
    """Stack all heatwave event windows centered on event peak time."""
    return composites.stack_events_centered_on_peak(
        ds,
        ds,
        variables=COMPOSITE_VARIABLES,
        pre_days=window_days,
        post_days=window_days,
    )


def build_event_mean_composite(stacked: xr.Dataset) -> xr.Dataset:
    """Average a stacked event-window dataset over events."""
    composite = stacked.mean("event", skipna=True)
    composite.attrs.update(stacked.attrs)
    composite.attrs["composite_reduction"] = "mean over all HW events"
    return composite


def smooth_composite(
    composite: xr.Dataset,
    *,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> xr.Dataset:
    """Return a plotting copy with selected hourly variables smoothed."""
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1.")

    out = composite.copy(deep=False)
    for name in SMOOTHED_VARIABLES:
        out[name] = composite[name].rolling(
            lag_hour=smoothing_window,
            center=True,
            min_periods=1,
        ).mean()
    out.attrs.update(composite.attrs)
    out.attrs["smoothing_window"] = int(smoothing_window)
    out.attrs["smoothing_applied_to"] = ", ".join(SMOOTHED_VARIABLES)
    return out


def write_composite_plot(composite: xr.Dataset, output_path: Path) -> Path:
    """Write a four-panel composite time-series figure."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_composite(composite)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_composite_outputs(
    composite: xr.Dataset,
    output_path: Path,
    *,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> list[Path]:
    """Write raw and smoothed composite figures."""
    written = [write_composite_plot(composite, output_path)]
    smoothed = smooth_composite(composite, smoothing_window=smoothing_window)
    written.append(write_composite_plot(smoothed, _smoothed_output_path(output_path)))
    return written


def plot_composite(composite: xr.Dataset) -> plt.Figure: # type: ignore
    """Return a four-panel figure for an event-mean composite."""
    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2, ax3 = axes

    _plot_temperature_volume_panel(ax0, composite)
    _plot_single_variable_panel(ax1, composite, "dTdt", ylabel="dTdt")
    _plot_tendency_panel(ax2, composite)
    _plot_lwa_panel(ax3, composite)

    for ax in axes:
        ax.axvline(0, color="0.2", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(True, linewidth=0.5, alpha=0.35)

    n_events = int(composite.attrs.get("n_events", 0))
    pre_days = int(composite.attrs.get("pre_days", DEFAULT_WINDOW_DAYS))
    post_days = int(composite.attrs.get("post_days", DEFAULT_WINDOW_DAYS))
    smoothing_window = composite.attrs.get("smoothing_window")
    smoothing_label = (
        f", {int(smoothing_window)}-hour running mean"
        if smoothing_window is not None
        else ""
    )
    fig.suptitle(
        f"HW event-mean composite centered on peak tas "
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})",
        fontsize=13,
    )
    ax3.set_xlabel("Lag from event peak (hours)")
    return fig


def _plot_temperature_volume_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot T_mean and volume with separate y axes."""
    lag = ds["lag_hour"].values
    ax.plot(lag, ds["T_mean"].values, color="tab:red", label="T_mean")
    ax.set_ylabel("T_mean")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax_volume = ax.twinx()
    ax_volume.plot(lag, ds["volume"].values, color="tab:blue", label="volume")
    ax_volume.set_ylabel("volume")
    ax_volume.tick_params(axis="y", labelcolor="tab:blue")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_volume.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")


def _plot_single_variable_panel(ax: Axes, ds: xr.Dataset, name: str, *, ylabel: str) -> None:
    """Plot one composite variable."""
    ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color="tab:purple")
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")


def _plot_tendency_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot heat-budget tendency terms on one axis."""
    for name in ("advection", "adiabatic", "diabatic"):
        ax.plot(ds["lag_hour"].values, ds[name].values, label=name)
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel("heat-budget terms")
    ax.legend(loc="upper left")


def _plot_lwa_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot LWA_a and LWA_c regional composite time series."""
    ax.plot(ds["lag_hour"].values, ds["lwa_a_region"].values, label="LWA_a_region")
    ax.plot(ds["lag_hour"].values, ds["lwa_c_region"].values, label="LWA_c_region")
    ax.set_ylabel("LWA")
    ax.legend(loc="upper left")


def main() -> int:
    """Open the harmonized dataset and write the all-event composite figure."""
    args = parse_args()
    if args.window_days < 0:
        raise ValueError("--window-days must be >= 0.")
    if args.smoothing_window < 1:
        raise ValueError("--smoothing-window must be >= 1.")

    ds = open_harmonized_dataset(args.input_path)
    try:
        stacked = build_all_hw_event_stack(ds, window_days=args.window_days)
        composite = build_event_mean_composite(stacked)
        written = write_composite_outputs(
            composite,
            args.output_path,
            smoothing_window=args.smoothing_window,
        )
        print("Wrote HW all-event composite figures:")
        for path in written:
            print(f"  {_display_path(path)}")
    finally:
        ds.close()
    return 0


def _smoothed_output_path(output_path: Path) -> Path:
    """Return sibling path for the smoothed composite figure."""
    if output_path.name == DEFAULT_OUTPUT_PATH.name:
        return output_path.with_name("hw_all_events_composite_smoothed.png")
    return output_path.with_name(f"{output_path.stem}_smoothed{output_path.suffix}")


def _display_path(path: Path) -> str:
    """Return a compact path for repo-local outputs and absolute path otherwise."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    raise SystemExit(main())
