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

from . import plot_style

DEFAULT_COMPOSITE_WINDOW_DAYS = 7
EVENT_PERCENTILE_PREFIX = "event_percentile_"
LOWER_EVENT_PERCENTILE = 0.25
UPPER_EVENT_PERCENTILE = 0.75
SPLIT_BIN_DIM = "split_bin"
SPLIT_LINE_STYLES = ("-", "--", ":", "-.")
VARIABLE_COLORS = {
    "T_mean": plot_style.COLORS["temperature"],
    "volume": plot_style.COLORS["volume"],
    "dTdt": plot_style.COLORS["storage"],
    "advection": plot_style.COLORS["advection"],
    "adiabatic": plot_style.COLORS["adiabatic"],
    "diabatic": plot_style.COLORS["diabatic"],
    "lwa_a_region": plot_style.FACE_COLORS["east"],
    "lwa_c_region": plot_style.FACE_COLORS["west"],
    "pbl_p_mean": plot_style.COLORS["mass"],
    "pbl_p_p05": plot_style.COLORS["mass"],
    "pbl_p_p95": plot_style.COLORS["mass"],
    "nslr_heating_rate_approx": plot_style.COLORS["benchmark"],
    "nssr_heating_rate_approx": plot_style.FACE_COLORS["top"],
    "sshf_heating_rate_approx": plot_style.COLORS["heat_flux"],
    "slhf_heating_rate_approx": plot_style.FACE_COLORS["south"],
    "soil_moisture": plot_style.COLORS["adiabatic"],
    "cloud_cover": plot_style.FACE_COLORS["south"],
}
EXTENDED_PLOT_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
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
SPLIT_EXTENDED_PLOT_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
    "pbl_p_mean",
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
)
PBL_PRESSURE_TO_HPA = 0.01


def plot_composite_timeseries(
    composite: xr.Dataset,
    *,
    plot_extended_variables: bool = False,
) -> Figure:
    """Return a composite time-series figure."""
    if plot_extended_variables:
        return _plot_extended_composite_timeseries(composite)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=plot_style.publication_figsize("full", aspect=0.85),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2, ax3 = axes

    _plot_temperature_volume_panel(ax0, composite)
    _plot_single_variable_panel(ax1, composite, "dTdt", ylabel="[K hr-1]")
    _plot_tendency_panel(ax2, composite)
    _plot_lwa_panel(ax3, composite)

    for ax in axes:
        ax.axvline(
            0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle="--",
            alpha=0.8,
        )

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
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})"
    )
    ax3.set_xlabel("Lag from event peak (hours)")
    _style_axes(axes)
    return fig


def plot_split_composite_timeseries(
    composite: xr.Dataset,
    *,
    plot_extended_variables: bool = False,
) -> Figure:
    """Return a figure for split-bin event-mean composites."""
    if SPLIT_BIN_DIM not in composite.dims:
        raise ValueError(f"split composite is missing dimension {SPLIT_BIN_DIM!r}.")
    if plot_extended_variables:
        return _plot_extended_split_composite_timeseries(composite)

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=plot_style.publication_figsize("full", aspect=0.85),
        sharex=True,
        constrained_layout=True,
    )
    ax0, ax1, ax2, ax3 = axes

    _plot_split_temperature_volume_panel(ax0, composite)
    _plot_split_single_variable_panel(ax1, composite, "dTdt", ylabel="[K hr-1]")
    _plot_split_tendency_panel(ax2, composite)
    _plot_split_lwa_panel(ax3, composite)

    for ax in axes:
        ax.axvline(
            0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle="--",
            alpha=0.8,
        )

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
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})"
    )
    ax3.set_xlabel("Lag from event peak (hours)")
    _style_axes(axes)
    return fig


def plot_top_event_timeseries(
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None = None,
    plot_extended_variables: bool = False,
) -> Figure:
    """Return an absolute-time figure for one selected event."""
    if plot_extended_variables:
        return _plot_extended_top_event_timeseries(
            event_window,
            event,
            reference_composite=reference_composite,
        )

    peak_time = _event_time_value(event, "peak_time")
    start_time = _event_time_value(event, "start_time")
    end_time = _event_time_value(event, "end_time")

    fig, axes = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=plot_style.publication_figsize("full", aspect=0.85),
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
        ax.axvline(
            start_time,
            color=plot_style.FACE_COLORS["top"],
            linewidth=plot_style.LINE_WIDTH_PT,
            linestyle=":",
            alpha=0.9,
        )
        ax.axvline(
            end_time,
            color=plot_style.FACE_COLORS["top"],
            linewidth=plot_style.LINE_WIDTH_PT,
            linestyle=":",
            alpha=0.9,
        )
        ax.axvline(
            peak_time,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle="--",
            alpha=0.8,
        )

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
        f"Rank {rank} HW event {event_id}: peak tas={peak_value:.2f}{smoothing_label}"
    )
    ax3.set_xlabel("Time")

    _format_datetime_xaxis(axes)
    _style_axes(axes)

    return fig


def _plot_extended_composite_timeseries(composite: xr.Dataset) -> Figure:
    """Return a 5x2 extended diagnostic figure for an event-mean composite."""
    _require_dataset_variables(
        composite,
        EXTENDED_PLOT_VARIABLES,
        dataset_label="composite",
    )

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=plot_style.publication_figsize("full", aspect=1.08),
        sharex=True,
        constrained_layout=True,
    )
    left = axes[:, 0]
    right = axes[:, 1]

    _plot_temperature_volume_panel(left[0], composite)
    _plot_single_variable_panel(left[1], composite, "dTdt", ylabel="[K hr-1]")
    _plot_composite_variable_panel(left[2], composite, "advection", ylabel="[K hr-1]")
    _plot_composite_variable_panel(left[3], composite, "adiabatic", ylabel="[K hr-1]")
    _plot_composite_variable_panel(left[4], composite, "diabatic", ylabel="[K hr-1]")

    _plot_lwa_panel(right[0], composite)
    _plot_pbl_pressure_panel(right[1], composite)
    _plot_composite_multi_variable_panel(
        right[2],
        composite,
        ("nslr_heating_rate_approx", "nssr_heating_rate_approx"),
        ylabel="[K hr-1]",
    )
    _plot_composite_multi_variable_panel(
        right[3],
        composite,
        ("sshf_heating_rate_approx", "slhf_heating_rate_approx"),
        ylabel="[K hr-1]",
    )
    _plot_soil_moisture_cloud_panel(right[4], composite)

    for ax in axes.ravel():
        ax.axvline(
            0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle="--",
            alpha=0.8,
        )

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
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})"
    )
    left[-1].set_xlabel("Lag from event peak (hours)")
    right[-1].set_xlabel("Lag from event peak (hours)")
    _style_axes(axes.ravel())
    return fig


def _plot_extended_split_composite_timeseries(composite: xr.Dataset) -> Figure:
    """Return a 5x2 extended diagnostic figure for split-bin composites."""
    _require_dataset_variables(
        composite,
        SPLIT_EXTENDED_PLOT_VARIABLES,
        dataset_label="split composite",
    )

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=plot_style.publication_figsize("full", aspect=1.08),
        sharex=True,
        constrained_layout=True,
    )
    left = axes[:, 0]
    right = axes[:, 1]

    _plot_split_temperature_volume_panel(left[0], composite)
    _plot_split_single_variable_panel(left[1], composite, "dTdt", ylabel="[K hr-1]")
    _plot_split_single_variable_panel(left[2], composite, "advection", ylabel="[K hr-1]")
    _plot_split_single_variable_panel(left[3], composite, "adiabatic", ylabel="[K hr-1]")
    _plot_split_single_variable_panel(left[4], composite, "diabatic", ylabel="[K hr-1]")

    _plot_split_lwa_panel(right[0], composite)
    _plot_split_pbl_pressure_panel(right[1], composite)
    _plot_split_multi_variable_panel(
        right[2],
        composite,
        ("nslr_heating_rate_approx", "nssr_heating_rate_approx"),
        ylabel="[K hr-1]",
    )
    _plot_split_multi_variable_panel(
        right[3],
        composite,
        ("sshf_heating_rate_approx", "slhf_heating_rate_approx"),
        ylabel="[K hr-1]",
    )
    _plot_split_soil_moisture_cloud_panel(right[4], composite)

    for ax in axes.ravel():
        ax.axvline(
            0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle="--",
            alpha=0.8,
        )

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
        f"(n={n_events}, -{pre_days}/+{post_days} days{smoothing_label})"
    )
    left[-1].set_xlabel("Lag from event peak (hours)")
    right[-1].set_xlabel("Lag from event peak (hours)")
    _style_axes(axes.ravel())
    return fig


def _plot_extended_top_event_timeseries(
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None,
) -> Figure:
    """Return a 5x2 extended diagnostic figure for one selected event."""
    _require_dataset_variables(
        event_window,
        EXTENDED_PLOT_VARIABLES,
        dataset_label="event window",
    )
    if reference_composite is not None:
        _require_dataset_variables(
            reference_composite,
            EXTENDED_PLOT_VARIABLES,
            dataset_label="reference composite",
        )

    peak_time = _event_time_value(event, "peak_time")
    start_time = _event_time_value(event, "start_time")
    end_time = _event_time_value(event, "end_time")

    fig, axes = plt.subplots(
        nrows=5,
        ncols=2,
        figsize=plot_style.publication_figsize("full", aspect=1.08),
        sharex=True,
        constrained_layout=True,
    )
    left = axes[:, 0]
    right = axes[:, 1]

    _plot_top_event_temperature_volume_panel(
        left[0],
        event_window,
        event,
        reference_composite=reference_composite,
    )
    _plot_top_event_single_variable_panel(
        left[1],
        event_window,
        event,
        "dTdt",
        ylabel="[K hr-1]",
        reference_composite=reference_composite,
    )
    for ax, name in zip(left[2:], ("advection", "adiabatic", "diabatic"), strict=True):
        _plot_top_event_single_variable_panel(
            ax,
            event_window,
            event,
            name,
            ylabel="[K hr-1]",
            reference_composite=reference_composite,
        )

    _plot_top_event_lwa_panel(
        right[0],
        event_window,
        event,
        reference_composite=reference_composite,
    )
    _plot_top_event_pbl_pressure_panel(
        right[1],
        event_window,
        event,
        reference_composite=reference_composite,
    )
    _plot_top_event_multi_variable_panel(
        right[2],
        event_window,
        event,
        ("nslr_heating_rate_approx", "nssr_heating_rate_approx"),
        ylabel="[K hr-1]",
        reference_composite=reference_composite,
    )
    _plot_top_event_multi_variable_panel(
        right[3],
        event_window,
        event,
        ("sshf_heating_rate_approx", "slhf_heating_rate_approx"),
        ylabel="[K hr-1]",
        reference_composite=reference_composite,
    )
    _plot_top_event_soil_moisture_cloud_panel(
        right[4],
        event_window,
        event,
        reference_composite=reference_composite,
    )

    for ax in axes.ravel():
        ax.axvline(
            start_time,
            color=plot_style.FACE_COLORS["top"],
            linewidth=plot_style.LINE_WIDTH_PT,
            linestyle=":",
            alpha=0.9,
        )
        ax.axvline(
            end_time,
            color=plot_style.FACE_COLORS["top"],
            linewidth=plot_style.LINE_WIDTH_PT,
            linestyle=":",
            alpha=0.9,
        )
        ax.axvline(
            peak_time,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle="--",
            alpha=0.8,
        )

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
        f"Rank {rank} HW event {event_id}: peak tas={peak_value:.2f}{smoothing_label}"
    )
    left[-1].set_xlabel("Time")
    right[-1].set_xlabel("Time")
    _format_datetime_xaxis(axes.ravel())
    _style_axes(axes.ravel())
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
    scale: float = 1.0,
) -> None:
    """Plot one named variable from a dataset against an explicit x coordinate."""
    ax.plot(
        x,
        ds[name].values * scale,
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


def _format_datetime_xaxis(axes: Sequence[Axes]) -> None:
    for ax in axes:
        plot_style.format_time_axis(ax)


def _style_axes(axes: Sequence[Axes]) -> None:
    plot_style.style_axes(np.ravel(axes))


def write_composite_timeseries_plot(
    composite: xr.Dataset,
    output_path: Path,
    *,
    plot_extended_variables: bool = False,
) -> Path:
    """Write a composite time-series figure."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_composite_timeseries(
        composite,
        plot_extended_variables=plot_extended_variables,
    )
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def write_split_composite_timeseries_plot(
    composite: xr.Dataset,
    output_path: Path,
    *,
    plot_extended_variables: bool = False,
) -> Path:
    """Write a split composite time-series figure."""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_split_composite_timeseries(
        composite,
        plot_extended_variables=plot_extended_variables,
    )
    plot_style.save_figure(fig, output_path)
    plt.close(fig)
    return output_path


def write_composite_timeseries_outputs(
    composite: xr.Dataset,
    output_path: Path,
    *,
    smoothed_output_path: Path,
    smoothing_window: int,
    smoothed_variables: Sequence[str],
    plot_extended_variables: bool = False,
) -> list[Path]:
    """Write raw and display-smoothed composite time-series figures."""
    written = [
        write_composite_timeseries_plot(
            composite,
            output_path,
            plot_extended_variables=plot_extended_variables,
        )
    ]
    smoothed = smooth_composite_for_display(
        composite,
        variables=smoothed_variables,
        smoothing_window=smoothing_window,
    )
    written.append(
        write_composite_timeseries_plot(
            smoothed,
            smoothed_output_path,
            plot_extended_variables=plot_extended_variables,
        )
    )
    return written


def write_split_composite_timeseries_outputs(
    composite: xr.Dataset,
    output_path: Path,
    *,
    smoothed_output_path: Path,
    smoothing_window: int,
    smoothed_variables: Sequence[str],
    plot_extended_variables: bool = False,
) -> list[Path]:
    """Write raw and display-smoothed split composite time-series figures."""
    written = [
        write_split_composite_timeseries_plot(
            composite,
            output_path,
            plot_extended_variables=plot_extended_variables,
        )
    ]
    smoothed = smooth_composite_for_display(
        composite,
        variables=smoothed_variables,
        smoothing_window=smoothing_window,
    )
    written.append(
        write_split_composite_timeseries_plot(
            smoothed,
            smoothed_output_path,
            plot_extended_variables=plot_extended_variables,
        )
    )
    return written


def _plot_temperature_volume_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot T_mean and volume with separate y axes."""
    lag = ds["lag_hour"].values
    temperature_color = VARIABLE_COLORS["T_mean"]
    volume_color = VARIABLE_COLORS["volume"]
    ax.plot(lag, ds["T_mean"].values, color=temperature_color, label="T_mean")
    _plot_event_percentile_band(ax, lag, ds, "T_mean", color=temperature_color)
    ax.set_ylabel("T_mean [K]", color=temperature_color)
    ax.tick_params(axis="y", labelcolor=temperature_color)

    ax_volume = ax.twinx()
    ax_volume.plot(lag, ds["volume"].values, color=volume_color, label="volume")
    _plot_event_percentile_band(ax_volume, lag, ds, "volume", color=volume_color)
    ax_volume.set_ylabel("volume [m2 Pa]", color=volume_color)
    ax_volume.tick_params(axis="y", labelcolor=volume_color)

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
    color = VARIABLE_COLORS[name]
    ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color=color)
    _plot_event_percentile_band(
        ax,
        ds["lag_hour"].values,
        ds,
        name,
        color=color,
    )
    plot_style.zero_line(ax)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")


def _plot_composite_variable_panel(
    ax: Axes,
    ds: xr.Dataset,
    name: str,
    *,
    ylabel: str,
) -> None:
    """Plot one composite variable using its configured color."""
    color = VARIABLE_COLORS[name]
    ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color=color)
    _plot_event_percentile_band(
        ax,
        ds["lag_hour"].values,
        ds,
        name,
        color=color,
    )
    plot_style.zero_line(ax)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[_variable_legend_handle(name)], loc="upper left")
    ax.legend(loc="upper left")


def _plot_composite_multi_variable_panel(
    ax: Axes,
    ds: xr.Dataset,
    names: Sequence[str],
    *,
    ylabel: str,
) -> None:
    """Plot multiple composite variables on one axis."""
    for name in names:
        color = VARIABLE_COLORS[name]
        ax.plot(ds["lag_hour"].values, ds[name].values, label=name, color=color)
        _plot_event_percentile_band(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=color,
        )
    plot_style.zero_line(ax)
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
    plot_style.zero_line(ax)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left")


def _plot_top_event_multi_variable_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    names: Sequence[str],
    *,
    ylabel: str,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot top-event variables with optional all-event reference."""
    for name in names:
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
    plot_style.zero_line(ax)
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
    plot_style.zero_line(ax)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[_variable_legend_handle(name)], loc="upper left")


def _plot_split_multi_variable_panel(
    ax: Axes,
    ds: xr.Dataset,
    names: Sequence[str],
    *,
    ylabel: str,
) -> None:
    """Plot multiple split-bin composite variables on one axis."""
    for name in names:
        _plot_split_lines(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=VARIABLE_COLORS[name],
        )
    plot_style.zero_line(ax)
    ax.set_ylabel(ylabel)
    variable_legend = ax.legend(
        handles=[_variable_legend_handle(name) for name in names],
        loc="upper left",
    )
    ax.add_artist(variable_legend)


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
    plot_style.zero_line(ax)
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
    plot_style.zero_line(ax)
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
    plot_style.zero_line(ax)
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


def _plot_pbl_pressure_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot PBL top pressure mean and spatial bounds in hPa."""
    lag = ds["lag_hour"].values
    color = VARIABLE_COLORS["pbl_p_mean"]
    ax.plot(
        lag,
        ds["pbl_p_mean"].values * PBL_PRESSURE_TO_HPA,
        label="pbl_p_mean",
        color=color,
    )
    _plot_event_percentile_band(
        ax,
        lag,
        ds,
        "pbl_p_mean",
        color=color,
        scale=PBL_PRESSURE_TO_HPA,
    )
    ax.fill_between(
        lag,
        ds["pbl_p_p05"].values * PBL_PRESSURE_TO_HPA,
        ds["pbl_p_p95"].values * PBL_PRESSURE_TO_HPA,
        color=color,
        alpha=0.14,
        linewidth=0,
        label="pbl_p_p05-p95",
    )
    ax.set_ylabel("PBL top pressure [hPa]")
    ax.invert_yaxis()
    ax.legend(loc="upper left")


def _plot_split_pbl_pressure_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot split-bin PBL top pressure mean in hPa with event-IQR bounds."""
    _plot_split_lines(
        ax,
        ds["lag_hour"].values,
        ds,
        "pbl_p_mean",
        color=VARIABLE_COLORS["pbl_p_mean"],
        scale=PBL_PRESSURE_TO_HPA,
    )
    ax.set_ylabel("PBL top pressure [hPa]")
    ax.invert_yaxis()
    variable_legend = ax.legend(
        handles=[_variable_legend_handle("pbl_p_mean")],
        loc="upper left",
    )
    ax.add_artist(variable_legend)


def _plot_top_event_pbl_pressure_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot top-event PBL top pressure mean and spatial bounds in hPa."""
    time = event_window["time"].values
    color = VARIABLE_COLORS["pbl_p_mean"]
    _plot_line(
        ax,
        time,
        event_window,
        "pbl_p_mean",
        color=color,
        linestyle="--",
        scale=PBL_PRESSURE_TO_HPA,
    )
    for name in ("pbl_p_p05", "pbl_p_p95"):
        _plot_line(
            ax,
            time,
            event_window,
            name,
            color=color,
            label=f"_{name}",
            linestyle=":",
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            scale=PBL_PRESSURE_TO_HPA,
        )
    if reference_composite is not None:
        reference_time = _reference_composite_time(event, reference_composite)
        _plot_line(
            ax,
            reference_time,
            reference_composite,
            "pbl_p_mean",
            color=color,
            label="_all_event_average",
            linewidth=plot_style.LINE_WIDTH_PT,
            scale=PBL_PRESSURE_TO_HPA,
        )
        _plot_event_percentile_band(
            ax,
            reference_time,
            reference_composite,
            "pbl_p_mean",
            color=color,
            scale=PBL_PRESSURE_TO_HPA,
        )
        ax.fill_between(
            reference_time,
            reference_composite["pbl_p_p05"].values * PBL_PRESSURE_TO_HPA,
            reference_composite["pbl_p_p95"].values * PBL_PRESSURE_TO_HPA,
            color=color,
            alpha=0.14,
            linewidth=0,
        )
    ax.set_ylabel("PBL top pressure [hPa]")
    ax.invert_yaxis()
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


def _plot_soil_moisture_cloud_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot soil moisture and cloud cover with separate y axes."""
    lag = ds["lag_hour"].values
    soil_color = VARIABLE_COLORS["soil_moisture"]
    cloud_color = VARIABLE_COLORS["cloud_cover"]
    ax.plot(lag, ds["soil_moisture"].values, color=soil_color, label="soil_moisture")
    _plot_event_percentile_band(ax, lag, ds, "soil_moisture", color=soil_color)
    ax.set_ylabel("soil moisture [m3 m-3]", color=soil_color)
    ax.tick_params(axis="y", labelcolor=soil_color)

    ax_cloud = ax.twinx()
    ax_cloud.plot(lag, ds["cloud_cover"].values, color=cloud_color, label="cloud_cover")
    _plot_event_percentile_band(ax_cloud, lag, ds, "cloud_cover", color=cloud_color)
    ax_cloud.set_ylabel("cloud cover fraction", color=cloud_color)
    ax_cloud.tick_params(axis="y", labelcolor=cloud_color)
    ax_cloud.set_ylim(0, 1)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_cloud.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")


def _plot_split_soil_moisture_cloud_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot split-bin soil moisture and cloud cover with separate y axes."""
    lag = ds["lag_hour"].values
    soil_color = VARIABLE_COLORS["soil_moisture"]
    cloud_color = VARIABLE_COLORS["cloud_cover"]
    _plot_split_lines(
        ax,
        lag,
        ds,
        "soil_moisture",
        color=soil_color,
    )
    ax.set_ylabel("soil moisture [m3 m-3]", color=soil_color)
    ax.tick_params(axis="y", labelcolor=soil_color)

    ax_cloud = ax.twinx()
    _plot_split_lines(
        ax_cloud,
        lag,
        ds,
        "cloud_cover",
        color=cloud_color,
    )
    ax_cloud.set_ylabel("cloud cover fraction", color=cloud_color)
    ax_cloud.tick_params(axis="y", labelcolor=cloud_color)
    ax_cloud.set_ylim(0, 1)

    ax.legend(
        handles=[
            _variable_legend_handle("soil_moisture"),
            _variable_legend_handle("cloud_cover"),
        ],
        loc="upper left",
    )


def _plot_top_event_soil_moisture_cloud_panel(
    ax: Axes,
    event_window: xr.Dataset,
    event: xr.Dataset,
    *,
    reference_composite: xr.Dataset | None,
) -> None:
    """Plot top-event soil moisture and cloud cover with separate y axes."""
    time = event_window["time"].values
    soil_color = VARIABLE_COLORS["soil_moisture"]
    cloud_color = VARIABLE_COLORS["cloud_cover"]
    _plot_line(
        ax,
        time,
        event_window,
        "soil_moisture",
        color=soil_color,
        linestyle="--",
    )
    if reference_composite is not None:
        _plot_top_event_reference(
            ax,
            event,
            reference_composite,
            "soil_moisture",
            color=soil_color,
        )
    ax.set_ylabel("soil moisture [m3 m-3]", color=soil_color)
    ax.tick_params(axis="y", labelcolor=soil_color)

    ax_cloud = ax.twinx()
    _plot_line(
        ax_cloud,
        time,
        event_window,
        "cloud_cover",
        color=cloud_color,
        linestyle="--",
    )
    if reference_composite is not None:
        _plot_top_event_reference(
            ax_cloud,
            event,
            reference_composite,
            "cloud_cover",
            color=cloud_color,
        )
    ax_cloud.set_ylabel("cloud cover fraction", color=cloud_color)
    ax_cloud.tick_params(axis="y", labelcolor=cloud_color)
    ax_cloud.set_ylim(0, 1)

    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_cloud.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")


def _plot_split_lines(
    ax: Axes,
    lag: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
    scale: float = 1.0,
) -> None:
    """Plot split-bin mean traces and low-alpha event percentile bounds."""
    for index in range(ds.sizes[SPLIT_BIN_DIM]):
        style = _split_line_style(index)
        subset = ds.isel({SPLIT_BIN_DIM: index})
        ax.plot(
            lag,
            subset[name].values * scale,
            color=color,
            linestyle=style,
            linewidth=plot_style.LINE_WIDTH_PT,
        )
        _plot_event_percentile_bound_lines(
            ax,
            lag,
            subset,
            name,
            color=color,
            linestyle=style,
            scale=scale,
        )


def _plot_event_percentile_band(
    ax: Axes,
    lag: np.ndarray,
    ds: xr.Dataset,
    name: str,
    *,
    color: str,
    scale: float = 1.0,
) -> None:
    """Shade the event percentile envelope for one variable."""
    bounds = _event_percentile_bounds(ds, name)
    if bounds is None:
        return

    lower, upper = bounds
    ax.fill_between(
        lag,
        lower.values * scale,
        upper.values * scale,
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
    scale: float = 1.0,
) -> None:
    """Draw event percentile bounds as faint lines for split overlays."""
    bounds = _event_percentile_bounds(ds, name)
    if bounds is None:
        return

    lower, upper = bounds
    for bound in (lower, upper):
        ax.plot(
            lag,
            bound.values * scale,
            color=color,
            linestyle=linestyle,
            alpha=0.28,
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
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
        linewidth=plot_style.LINE_WIDTH_PT,
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
        Line2D(
            [0],
            [0],
            color=plot_style.COLORS["zero"],
            linestyle="-",
            label="all-event average",
        ),
        Patch(facecolor="0.5", edgecolor="none", alpha=0.18, label="IQR"),
    ]
    ax.legend(handles=handles, loc="upper right")


def _add_split_style_legend(ax: Axes, labels: Sequence[str]) -> None:
    """Add a split-bin linestyle legend plus IQR-bound hint."""
    handles = [
        Line2D(
            [0],
            [0],
            color=plot_style.COLORS["zero"],
            linestyle=_split_line_style(index),
            label=label,
        )
        for index, label in enumerate(labels)
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=plot_style.COLORS["zero"],
            linestyle="-",
            alpha=0.28,
            label="IQR bounds",
        )
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


def _require_dataset_variables(
    ds: xr.Dataset,
    names: Sequence[str],
    *,
    dataset_label: str,
) -> None:
    """Raise a clear error when an extended plot input is missing variables."""
    missing = [name for name in names if name not in ds]
    if missing:
        values = ", ".join(missing)
        raise ValueError(
            f"Extended plot requires missing variables in {dataset_label}: {values}."
        )


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
