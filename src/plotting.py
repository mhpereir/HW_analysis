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


def _plot_tendency_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot heat-budget tendency terms on one axis."""
    for name in ("advection", "adiabatic", "diabatic"):
        (line,) = ax.plot(ds["lag_hour"].values, ds[name].values, label=name)
        _plot_event_percentile_band(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=line.get_color(), #type: ignore
        )
    ax.axhline(0, color="0.2", linewidth=1.0)
    ax.set_ylabel("[K hr-1]")
    ax.legend(loc="upper left", ncol=3)


def _plot_lwa_panel(ax: Axes, ds: xr.Dataset) -> None:
    """Plot LWA_a and LWA_c regional composite time series."""
    for name in ("lwa_a_region", "lwa_c_region"):
        (line,) = ax.plot(ds["lag_hour"].values, ds[name].values, label=name)
        _plot_event_percentile_band(
            ax,
            ds["lag_hour"].values,
            ds,
            name,
            color=line.get_color(), #type: ignore
        )
    ax.set_ylabel("LWA [m hPa]")
    ax.legend(loc="upper left")


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


def _add_iqr_legend(ax: Axes) -> None:
    """Add a first-panel legend entry describing percentile shading."""
    handle = Patch(facecolor="0.5", edgecolor="none", alpha=0.18, label="IQR")
    ax.legend(handles=[handle], loc="upper right")


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
