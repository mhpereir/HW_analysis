"""Plotting functions for the heatwave analysis pipeline.

Pipeline role:
- Workflow layer: plotting and export within top-level Stage 2.

Responsibilities:
- Accept prepared datasets, composites, or tables.
- Render composite time series panels.
- Render top-event individual traces.
- Support future map-based composite visualizations.

Out of scope:
- Raw data loading.
- Mask generation.
- Event definition.
- Core analysis logic embedded inside plotting functions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "figure.titlesize": 20,
})

DEFAULT_COMPOSITE_WINDOW_DAYS = 7
EVENT_PERCENTILE_PREFIX = "event_percentile_"
LOWER_EVENT_PERCENTILE = 0.25
UPPER_EVENT_PERCENTILE = 0.75
SPLIT_BIN_DIM = "split_bin"
SPLIT_LINE_STYLES = ("-", "--", ":", "-.")
VARIABLE_COLORS = {
    "T_mean": "tab:red",
    "volume": "tab:blue",
    "dTdt": "tab:purple",
    "advection": "tab:orange",
    "adiabatic": "tab:green",
    "diabatic": "tab:brown",
    "lwa_a_region": "tab:olive",
    "lwa_c_region": "tab:cyan",
}


def plot_composite_timeseries(composite: xr.Dataset) -> Figure:
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
    _plot_single_variable_panel(ax1, composite, "dTdt", ylabel="[K hr-1]")
    _plot_tendency_panel(ax2, composite)
    _plot_lwa_panel(ax3, composite)

    for ax in axes:
        ax.axvline(0, color="0.2", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(True, linewidth=0.5, alpha=0.35)

    n_events = int(composite.attrs.get("n_events", 0))
    pre_days = int(composite.attrs.get("pre_days", DEFAULT_COMPOSITE_WINDOW_DAYS))
    post_days = int(composite.attrs.get("post_days", DEFAULT_COMPOSITE_WINDOW_DAYS))
    smoothing_window = composite.attrs.get("smoothing_window")
    smoothing_label = (
        f", {int(smoothing_window)}-hour running mean"
        if smoothing_window is not None
        else ""
    )
    fig.suptitle(
        f"HW event-mean composite centered on peak tas "
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})",
        fontsize=18,
    )
    ax3.set_xlabel("Lag from event peak (hours)")
    return fig


def plot_split_composite_timeseries(composite: xr.Dataset) -> Figure:
    """Return a four-panel figure for split-bin event-mean composites."""
    if SPLIT_BIN_DIM not in composite.dims:
        raise ValueError(f"split composite is missing dimension {SPLIT_BIN_DIM!r}.")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2, ax3 = axes

    _plot_split_temperature_volume_panel(ax0, composite)
    _plot_split_single_variable_panel(ax1, composite, "dTdt", ylabel="[K hr-1]")
    _plot_split_tendency_panel(ax2, composite)
    _plot_split_lwa_panel(ax3, composite)

    for ax in axes:
        ax.axvline(0, color="0.2", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(True, linewidth=0.5, alpha=0.35)

    n_events = _split_total_events(composite)
    pre_days = int(composite.attrs.get("pre_days", DEFAULT_COMPOSITE_WINDOW_DAYS))
    post_days = int(composite.attrs.get("post_days", DEFAULT_COMPOSITE_WINDOW_DAYS))
    smoothing_window = composite.attrs.get("smoothing_window")
    smoothing_label = (
        f", {int(smoothing_window)}-hour running mean"
        if smoothing_window is not None
        else ""
    )
    split_variable = composite.attrs.get("split_variable", "event metric")
    fig.suptitle(
        f"HW event-mean composites split by {split_variable} "
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})",
        fontsize=18,
    )
    ax3.set_xlabel("Lag from event peak (hours)")
    return fig


def plot_top_event_timeseries(
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None = None,
) -> Figure:
    """Return a four-panel absolute-time figure for one selected event."""
    peak_time = _event_time_value(event, "peak_time")
    start_time = _event_time_value(event, "start_time")
    end_time = _event_time_value(event, "end_time")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(12, 10),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2, ax3 = axes

    _plot_top_event_temperature_volume_panel(
        ax0,
        event_window,
        event,
        reference_composite=reference_composite,
    )
    _plot_top_event_single_variable_panel(
        ax1,
        event_window,
        event,
        "dTdt",
        ylabel="[K hr-1]",
        reference_composite=reference_composite,
    )
    _plot_top_event_tendency_panel(
        ax2,
        event_window,
        event,
        reference_composite=reference_composite,
    )
    _plot_top_event_lwa_panel(
        ax3,
        event_window,
        event,
        reference_composite=reference_composite,
    )

    for ax in axes:
        ax.axvline(start_time, color="tab:orange", linewidth=1.2, linestyle=":", alpha=0.9)
        ax.axvline(end_time, color="tab:orange", linewidth=1.2, linestyle=":", alpha=0.9)
        ax.axvline(peak_time, color="0.2", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(True, linewidth=0.5, alpha=0.35)

    event_id = int(event["event_id"].item())
    rank = int(event["selection_rank"].item()) if "selection_rank" in event else event_id
    peak_value = float(event["tas_peak"].item()) if "tas_peak" in event else np.nan
    smoothing_window = event_window.attrs.get("smoothing_window")
    smoothing_label = (
        f", {int(smoothing_window)}-hour running mean"
        if smoothing_window is not None
        else ""
    )
    fig.suptitle(
        f"Rank {rank} HW event {event_id}: peak tas={peak_value:.2f}{smoothing_label}",
        fontsize=13,
    )
    ax3.set_xlabel("Time")
    return fig


def _plot_line(
    ax: Axes,
    x: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
    label: str | None = None,
    linestyle: str = "-",
    linewidth: float | None = None,
) -> None:
    """Plot one named variable from a dataset against an explicit x coordinate."""
    ax.plot(
        x,
        ds[name].values,
        color=color,
        label=label or name,
        linestyle=linestyle,
        linewidth=linewidth,
    )


def smooth_composite_for_display(
    composite: xr.Dataset,
    *,
    variables: Sequence[str],
    smoothing_window: int,
    lag_dim: str = "lag_hour",
) -> xr.Dataset:
    """Return a display-only rolling-mean copy of selected composite variables."""
    if smoothing_window < 1:
        raise ValueError("smoothing_window must be >= 1.")

    out = composite.copy(deep=False)
    variable_names = _display_smoothing_variable_names(composite, variables)
    for name in variable_names:
        out[name] = composite[name].rolling(
            {lag_dim: smoothing_window},
            center=True,
            min_periods=smoothing_window,
        ).mean()
    out.attrs.update(composite.attrs)
    out.attrs["smoothing_window"] = int(smoothing_window)
    out.attrs["smoothing_applied_to"] = ", ".join(variable_names)
    return out


def write_composite_timeseries_plot(
    composite: xr.Dataset,
    output_path: Path,
) -> Path:
    """Write a four-panel composite time-series figure."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_composite_timeseries(composite)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_split_composite_timeseries_plot(
    composite: xr.Dataset,
    output_path: Path,
) -> Path:
    """Write a four-panel split composite time-series figure."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_split_composite_timeseries(composite)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def write_composite_timeseries_outputs(
    composite: xr.Dataset,
    output_path: Path,
    *,
    smoothed_output_path: Path,
    smoothing_window: int,
    smoothed_variables: Sequence[str],
) -> list[Path]:
    """Write raw and display-smoothed composite time-series figures."""
    written = [write_composite_timeseries_plot(composite, output_path)]
    smoothed = smooth_composite_for_display(
        composite,
        variables=smoothed_variables,
        smoothing_window=smoothing_window,
    )
    written.append(write_composite_timeseries_plot(smoothed, smoothed_output_path))
    return written


def write_split_composite_timeseries_outputs(
    composite: xr.Dataset,
    output_path: Path,
    *,
    smoothed_output_path: Path,
    smoothing_window: int,
    smoothed_variables: Sequence[str],
) -> list[Path]:
    """Write raw and display-smoothed split composite time-series figures."""
    written = [write_split_composite_timeseries_plot(composite, output_path)]
    smoothed = smooth_composite_for_display(
        composite,
        variables=smoothed_variables,
        smoothing_window=smoothing_window,
    )
    written.append(write_split_composite_timeseries_plot(smoothed, smoothed_output_path))
    return written


def _plot_temperature_volume_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot T_mean and volume with separate y axes."""
    lag = ds["lag_hour"].values
    ax.plot(lag, ds["T_mean"].values, color="tab:red", label="T_mean")
    _plot_event_percentile_band(ax, lag, ds, "T_mean", color="tab:red")
    ax.set_ylabel("T_mean [K]", color="tab:red")
    ax.tick_params(axis="y", labelcolor="tab:red")

    ax_volume = ax.twinx()
    ax_volume.plot(lag, ds["volume"].values, color="tab:blue", label="volume")
    _plot_event_percentile_band(ax_volume, lag, ds, "volume", color="tab:blue")
    ax_volume.set_ylabel("volume [m2 Pa]", color="tab:blue")
    ax_volume.tick_params(axis="y", labelcolor="tab:blue")

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_volume.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    _add_iqr_legend(ax_volume)


def _plot_top_event_temperature_volume_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot top-event T_mean and volume with optional all-event reference."""
    time = event_window["time"].values
    _plot_line(
        ax,
        time,
        event_window,
        "T_mean",
        color=VARIABLE_COLORS["T_mean"],
        label="T_mean",
        linestyle="--",
    )
    if reference_composite is not None:
        _plot_top_event_reference(
            ax,
            event,
            reference_composite,
            "T_mean",
            color=VARIABLE_COLORS["T_mean"],
        )
    ax.set_ylabel("T_mean [K]", color=VARIABLE_COLORS["T_mean"])
    ax.tick_params(axis="y", labelcolor=VARIABLE_COLORS["T_mean"])

    ax_volume = ax.twinx()
    _plot_line(
        ax_volume,
        time,
        event_window,
        "volume",
        color=VARIABLE_COLORS["volume"],
        label="volume",
        linestyle="--",
    )
    if reference_composite is not None:
        _plot_top_event_reference(
            ax_volume,
            event,
            reference_composite,
            "volume",
            color=VARIABLE_COLORS["volume"],
        )
    ax_volume.set_ylabel("volume [m2 Pa]", color=VARIABLE_COLORS["volume"])
    ax_volume.tick_params(axis="y", labelcolor=VARIABLE_COLORS["volume"])

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_volume.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")
    if reference_composite is not None:
        _add_top_event_reference_legend(ax_volume)


def _plot_split_temperature_volume_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot split-bin T_mean and volume with separate y axes."""
    lag = ds["lag_hour"].values
    _plot_split_lines(
        ax,
        lag,
        ds,
        "T_mean",
        color=VARIABLE_COLORS["T_mean"],
    )
    ax.set_ylabel("T_mean [K]", color=VARIABLE_COLORS["T_mean"])
    ax.tick_params(axis="y", labelcolor=VARIABLE_COLORS["T_mean"])

    ax_volume = ax.twinx()
    _plot_split_lines(
        ax_volume,
        lag,
        ds,
        "volume",
        color=VARIABLE_COLORS["volume"],
    )
    ax_volume.set_ylabel("volume [m2 Pa]", color=VARIABLE_COLORS["volume"])
    ax_volume.tick_params(axis="y", labelcolor=VARIABLE_COLORS["volume"])

    ax.legend(
        handles=[
            _variable_legend_handle("T_mean"),
            _variable_legend_handle("volume"),
        ],
        loc="upper left",
    )
    _add_split_style_legend(ax_volume, _split_bin_labels(ds))


def _plot_single_variable_panel(
    ax: Axes,
    ds: xr.Dataset,
    name: str,
    *,
    ylabel: str,
) -> None:
    """Plot one composite variable."""
    ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color="tab:purple")
    _plot_event_percentile_band(
        ax,
        ds["lag_hour"].values,
        ds,
        name,
        color="tab:purple",
    )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")


def _plot_top_event_single_variable_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    name: str,
    *,
    ylabel: str,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot one top-event variable with optional all-event reference."""
    color = VARIABLE_COLORS[name]
    _plot_line(
        ax,
        event_window["time"].values,
        event_window,
        name,
        color=color,
        linestyle="--",
    )
    if reference_composite is not None:
        _plot_top_event_reference(
            ax,
            event,
            reference_composite,
            name,
            color=color,
        )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")


def _plot_split_single_variable_panel(
    ax: Axes,
    ds: xr.Dataset,
    name: str,
    *,
    ylabel: str,
) -> None:
    """Plot one split-bin composite variable."""
    _plot_split_lines(
        ax,
        ds["lag_hour"].values,
        ds,
        name,
        color=VARIABLE_COLORS[name],
    )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel(ylabel)


def _plot_tendency_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot heat-budget tendency terms on one axis."""
    for name in ("advection", "adiabatic", "diabatic"):
        color = VARIABLE_COLORS[name]
        ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color=color)
        _plot_event_percentile_band(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=color,
        )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel("[K hr-1]")
    _expand_yaxis(ax, factor=1.5)
    ax.legend(loc="upper left", ncol=3)


def _plot_top_event_tendency_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot top-event heat-budget terms with optional all-event reference."""
    for name in ("advection", "adiabatic", "diabatic"):
        color = VARIABLE_COLORS[name]
        _plot_line(
            ax,
            event_window["time"].values,
            event_window,
            name,
            color=color,
            linestyle="--",
        )
        if reference_composite is not None:
            _plot_top_event_reference(
                ax,
                event,
                reference_composite,
                name,
                color=color,
            )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel("[K hr-1]")
    _expand_yaxis(ax, factor=1.5)
    ax.legend(loc="upper left", ncol=3)


def _plot_split_tendency_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot split-bin heat-budget tendency terms on one axis."""
    for name in ("advection", "adiabatic", "diabatic"):
        _plot_split_lines(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=VARIABLE_COLORS[name],
        )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel("[K hr-1]")
    _expand_yaxis(ax, factor=1.5)
    variable_legend = ax.legend(
        handles=[
            _variable_legend_handle(name)
            for name in ("advection", "adiabatic", "diabatic")
        ],
        loc="upper left",
        ncol=3,
    )
    ax.add_artist(variable_legend)


def _plot_lwa_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot LWA_a and LWA_c regional composite time series."""
    for name in ("lwa_a_region", "lwa_c_region"):
        color = VARIABLE_COLORS[name]
        ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color=color)
        _plot_event_percentile_band(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=color,
        )
    ax.set_ylabel("LWA [m hPa]")
    ax.legend(loc="upper left")


def _plot_top_event_lwa_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot top-event LWA_a and LWA_c with optional all-event reference."""
    for name in ("lwa_a_region", "lwa_c_region"):
        color = VARIABLE_COLORS[name]
        _plot_line(
            ax,
            event_window["time"].values,
            event_window,
            name,
            color=color,
            linestyle="--",
        )
        if reference_composite is not None:
            _plot_top_event_reference(
                ax,
                event,
                reference_composite,
                name,
                color=color,
            )
    ax.set_ylabel("LWA [m hPa]")
    ax.legend(loc="upper left")


def _plot_split_lwa_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot split-bin LWA_a and LWA_c regional composite time series."""
    for name in ("lwa_a_region", "lwa_c_region"):
        _plot_split_lines(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=VARIABLE_COLORS[name],
        )
    ax.set_ylabel("LWA [m hPa]")
    variable_legend = ax.legend(
        handles=[
            _variable_legend_handle(name)
            for name in ("lwa_a_region", "lwa_c_region")
        ],
        loc="upper left",
    )
    ax.add_artist(variable_legend)


def _plot_split_lines(
    ax: Axes,
    lag: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
) -> None:
    """Plot split-bin mean traces and low-alpha event percentile bounds."""
    for index in range(ds.sizes[SPLIT_BIN_DIM]):
        style = _split_line_style(index)
        subset = ds.isel({SPLIT_BIN_DIM: index})
        ax.plot(
            lag,
            subset[name].values,
            color=color,
            linestyle=style,
            linewidth=1.8,
        )
        _plot_event_percentile_bound_lines(
            ax,
            lag,
            subset,
            name,
            color=color,
            linestyle=style,
        )


def _plot_event_percentile_band(
    ax: Axes,
    lag: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
) -> None:
    """Shade the event percentile envelope for one variable."""
    bounds = _event_percentile_bounds(ds, name)
    if bounds is None:
        return

    lower, upper = bounds
    ax.fill_between(
        lag,
        lower.values,
        upper.values,
        color=color,
        alpha=0.18,
        linewidth=0,
    )


def _plot_event_percentile_bound_lines(
    ax: Axes,
    lag: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
    linestyle: str,
) -> None:
    """Draw event percentile bounds as faint lines for split overlays."""
    bounds = _event_percentile_bounds(ds, name)
    if bounds is None:
        return

    lower, upper = bounds
    for bound in (lower, upper):
        ax.plot(
            lag,
            bound.values,
            color=color,
            linestyle=linestyle,
            alpha=0.28,
            linewidth=1.0,
        )


def _expand_yaxis(ax: Axes, *, factor: float) -> None:
    """Expand an axis y-range around its current center by a scale factor."""
    if factor <= 0:
        raise ValueError("factor must be positive.")

    lower, upper = ax.get_ylim()
    center = 0.5 * (lower + upper)
    half_range = 0.5 * (upper - lower) * factor
    ax.set_ylim(center - half_range, center + half_range)


def _plot_top_event_reference(
    ax: Axes,
    event: xr.Dataset,
    reference_composite: xr.Dataset,
    name: str,
    *,
    color: str,
) -> None:
    """Plot an all-event composite reference aligned to one event peak."""
    time = _reference_composite_time(event, reference_composite)
    _plot_line(
        ax,
        time,
        reference_composite,
        name,
        color=color,
        linewidth=1.8,
        label="_all_event_average",
    )
    _plot_event_percentile_band(ax, time, reference_composite, name, color=color)


def _add_iqr_legend(ax: Axes) -> None:
    """Add a first-panel legend entry describing percentile shading."""
    handle = Patch(facecolor="0.5", edgecolor="none", alpha=0.18, label="IQR")
    ax.legend(handles=[handle], loc="upper right")


def _add_top_event_reference_legend(ax: Axes) -> None:
    """Add a single legend for top-event reference line and IQR shading."""
    handles = [
        Line2D([0], [0], color="0.2", linestyle="-", label="all-event average"),
        Patch(facecolor="0.5", edgecolor="none", alpha=0.18, label="IQR"),
    ]
    ax.legend(handles=handles, loc="upper right")


def _add_split_style_legend(ax: Axes, labels: Sequence[str]) -> None:
    """Add a split-bin linestyle legend plus IQR-bound hint."""
    handles = [
        Line2D([0], [0], color="0.2", linestyle=_split_line_style(index), label=label)
        for index, label in enumerate(labels)
    ]
    handles.append(
        Line2D([0], [0], color="0.2", linestyle="-", alpha=0.28, label="IQR bounds")
    )
    ax.legend(handles=handles, loc="upper right")


def _event_percentile_bounds(
    ds: xr.Dataset,
    name: str,
) -> tuple[xr.DataArray, xr.DataArray] | None:
    """Return lower and upper event percentile traces for one variable."""
    envelope_name = f"{_event_percentile_prefix(ds)}{name}"
    if envelope_name not in ds:
        return None

    envelope = ds[envelope_name]
    if "quantile" not in envelope.dims:
        return None

    lower = _select_quantile(envelope, LOWER_EVENT_PERCENTILE)
    upper = _select_quantile(envelope, UPPER_EVENT_PERCENTILE)
    if lower is None or upper is None:
        return None
    return lower, upper


def _reference_composite_time(
    event: xr.Dataset,
    reference_composite: xr.Dataset,
) -> np.ndarray:
    """Return reference-composite lags as absolute datetimes for one event."""
    peak_time = _event_time_value(event, "peak_time")
    lag_hours = np.asarray(reference_composite["lag_hour"].values, dtype=np.int64)
    return peak_time + lag_hours.astype("timedelta64[h]")


def _event_time_value(event: xr.Dataset, name: str) -> np.datetime64:
    """Return an event timestamp scalar as datetime64[ns]."""
    return np.datetime64(np.asarray(event[name].values).item(), "ns")


def _select_quantile(
    da: xr.DataArray,
    quantile: float,
) -> xr.DataArray | None:
    """Return a quantile slice when that quantile is present."""
    quantiles = np.asarray(da["quantile"].values, dtype=float)
    matches = np.flatnonzero(np.isclose(quantiles, quantile))
    if matches.size == 0:
        return None
    return da.isel(quantile=int(matches[0]))


def _split_bin_labels(ds: xr.Dataset) -> list[str]:
    """Return split-bin labels from coordinates."""
    if SPLIT_BIN_DIM not in ds.coords:
        return [f"bin {index + 1}" for index in range(ds.sizes[SPLIT_BIN_DIM])]
    return [str(value) for value in ds[SPLIT_BIN_DIM].values]


def _split_line_style(index: int) -> str:
    """Return the linestyle for one split-bin index."""
    return SPLIT_LINE_STYLES[index % len(SPLIT_LINE_STYLES)]


def _variable_legend_handle(name: str) -> Line2D:
    """Return a solid-line variable legend handle."""
    return Line2D([0], [0], color=VARIABLE_COLORS[name], linestyle="-", label=name)


def _split_total_events(ds: xr.Dataset) -> int:
    """Return total events represented by split bins."""
    if "split_n_events" in ds.coords:
        return int(np.asarray(ds["split_n_events"].values, dtype=np.int64).sum())
    return int(ds.attrs.get("n_events", 0))


def _display_smoothing_variable_names(
    composite: xr.Dataset,
    variables: Sequence[str],
) -> list[str]:
    """Return requested variables plus matching percentile envelope variables."""
    percentile_prefix = _event_percentile_prefix(composite)
    names: list[str] = []
    for name in variables:
        names.append(str(name))
        envelope_name = f"{percentile_prefix}{name}"
        if envelope_name in composite:
            names.append(envelope_name)
    return names


def _event_percentile_prefix(ds: xr.Dataset) -> str:
    """Return the percentile-variable prefix used by a composite dataset."""
    return str(ds.attrs.get("event_percentile_prefix", EVENT_PERCENTILE_PREFIX))
