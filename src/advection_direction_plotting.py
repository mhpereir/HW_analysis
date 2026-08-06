"""Standalone rendering for the face-resolved advection prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from . import advection_direction, plot_style


GROUP_COLORS = {
    "advection_zonal": "#6A5ACD",
    "advection_meridional": "#009E73",
    "advection_horizontal": "#CC79A7",
    "advection_vertical": "#E69F00",
    "advection_face_total": "#111111",
}
GROUP_LABELS = {
    "advection_zonal": "Zonal (west + east)",
    "advection_meridional": "Meridional (south + north)",
    "advection_horizontal": "Horizontal",
    "advection_vertical": "Vertical",
    "advection_face_total": "All faces",
}
LEGEND_HEADROOM_FRACTION = 0.30


def plot_advection_direction_exploration(
    composite: xr.Dataset,
) -> Figure:
    """Return the standalone two-panel face-component figure."""
    required = (
        "advection",
        *(
            advection_direction.stage1_face_variable(face)
            for face in advection_direction.REQUIRED_FACES
        ),
    )
    _require_variables(composite, required)
    diagnostics = xr.merge(
        [composite, advection_direction.grouped_advection_components(composite)],
        compat="override",
    )
    diagnostics.attrs.update(composite.attrs)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=plot_style.publication_figsize(
            "full",
            aspect=plot_style.TWO_PANEL_STACK_ASPECT,
        ),
        sharex=True,
        constrained_layout=True,
    )
    face_ax, grouped_ax = axes

    lag_days = np.asarray(diagnostics["lag_hour"].values, dtype=float) / 24.0
    _plot_face_timeseries(face_ax, diagnostics, lag_days)
    _plot_grouped_timeseries(grouped_ax, diagnostics, lag_days)

    for ax in axes:
        ax.axvline(
            0,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle=":",
            zorder=1,
        )
        plot_style.style_axis(ax)
        plot_style.format_integer_axis(ax.xaxis, spacing=1)
        ax.set_xlim(float(lag_days.min()), float(lag_days.max()))
    grouped_ax.set_xlabel("Days relative to event peak")

    title = _figure_title(composite)
    fig.suptitle(title)
    return fig


def write_advection_direction_exploration_plot(
    composite: xr.Dataset,
    output_path: str | Path,
) -> Path:
    """Write the standalone two-panel face-component figure."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_advection_direction_exploration(composite)
    try:
        plot_style.save_figure(fig, path)
    finally:
        plt.close(fig)
    return path


def _plot_face_timeseries(
    ax,
    ds: xr.Dataset,
    lag_days: np.ndarray,
) -> None:
    for face in advection_direction.available_stage1_faces(ds):
        name = advection_direction.stage1_face_variable(face)
        ax.plot(
            lag_days,
            ds[name].values,
            color=plot_style.FACE_COLORS[face],
            label=face.capitalize(),
        )
    plot_style.zero_line(ax)
    ax.set_ylabel("K hr-1")
    ax.set_title("Signed face contributions")
    _add_upper_axis_headroom(ax)
    ax.legend(ncol=5, loc="upper center", **plot_style.legend_kwargs())


def _plot_grouped_timeseries(
    ax,
    ds: xr.Dataset,
    lag_days: np.ndarray,
) -> None:
    for name in (
        "advection_zonal",
        "advection_meridional",
        "advection_horizontal",
        "advection_vertical",
        "advection_face_total",
    ):
        ax.plot(
            lag_days,
            ds[name].values,
            color=GROUP_COLORS[name],
            label=GROUP_LABELS[name],
            linestyle="--" if name == "advection_face_total" else "-",
        )
    plot_style.zero_line(ax)
    ax.set_ylabel("K hr-1")
    ax.set_title("Grouped advective contributions")
    _add_upper_axis_headroom(ax)
    ax.legend(ncol=5, loc="upper center", **plot_style.legend_kwargs())


def _add_upper_axis_headroom(
    ax,
    *,
    fraction: float = LEGEND_HEADROOM_FRACTION,
) -> None:
    """Expand only the upper y-limit by a fraction of the autoscaled range."""
    if fraction < 0:
        raise ValueError("fraction must be >= 0.")
    lower, upper = ax.get_ylim()
    span = upper - lower
    if not np.isfinite(span) or span <= 0:
        raise ValueError("axis must have a finite positive y-range.")
    ax.set_ylim(lower, upper + fraction * span)


def _require_variables(ds: xr.Dataset, names: tuple[str, ...]) -> None:
    missing = sorted(name for name in names if name not in ds)
    if missing:
        raise ValueError(
            "Advection-direction plot is missing required variables: "
            f"{', '.join(missing)}"
        )


def _figure_title(ds: xr.Dataset) -> str:
    region = ds.attrs.get("region", "unknown region")
    n_events = ds.attrs.get("n_events", "unknown")
    pre_days = ds.attrs.get("pre_days", "?")
    post_days = ds.attrs.get("post_days", "?")
    representation = (
        " climatological-anomaly"
        if ds.attrs.get("data_representation") == "climatological_anomaly"
        else ""
    )
    return (
        f"Face-resolved advection{representation} composite for {region} "
        f"(n={n_events}, -{pre_days} to +{post_days} days)"
    )
