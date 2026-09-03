"""Standalone rendering for the face-resolved advection prototype."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from . import advection_direction, plot_style, plotting

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
GROUP_NAMES = tuple(GROUP_LABELS)
MATCHED_SIGN_LINESTYLES = {
    "positive": "-",
    "negative": "--",
}
LEGEND_HEADROOM_FRACTION = 0.30
DEFAULT_SMOOTHING_WINDOW = 24


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


def write_advection_direction_exploration_outputs(
    composite: xr.Dataset,
    output_path: str | Path,
    *,
    smoothed_output_path: str | Path,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> list[Path]:
    """Write unsmoothed and display-smoothed face-advection figures."""
    smoothed = _smooth_advection_composite_for_display(
        composite,
        smoothing_window=smoothing_window,
    )
    written = [
        write_advection_direction_exploration_plot(
            composite,
            output_path,
        )
    ]
    written.append(
        write_advection_direction_exploration_plot(
            smoothed,
            smoothed_output_path,
        )
    )
    return written


def plot_matched_advection_direction_exploration(
    negative_composite: xr.Dataset,
    positive_composite: xr.Dataset,
) -> Figure:
    """Return a two-panel overlay for matched negative and positive events."""
    negative = _advection_direction_diagnostics(negative_composite)
    positive = _advection_direction_diagnostics(positive_composite)
    _validate_matched_composites(negative, positive)

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
    lag_days = np.asarray(negative["lag_hour"].values, dtype=float) / 24.0

    _plot_matched_face_timeseries(
        face_ax,
        negative=negative,
        positive=positive,
        lag_days=lag_days,
    )
    _plot_matched_grouped_timeseries(
        grouped_ax,
        negative=negative,
        positive=positive,
        lag_days=lag_days,
    )

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
    fig.suptitle(_matched_figure_title(negative, positive))
    return fig


def write_matched_advection_direction_exploration_plot(
    negative_composite: xr.Dataset,
    positive_composite: xr.Dataset,
    output_path: str | Path,
) -> Path:
    """Write the matched negative/positive face-component overlay."""
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_matched_advection_direction_exploration(
        negative_composite,
        positive_composite,
    )
    try:
        plot_style.save_figure(fig, path)
    finally:
        plt.close(fig)
    return path


def write_matched_advection_direction_exploration_outputs(
    negative_composite: xr.Dataset,
    positive_composite: xr.Dataset,
    output_path: str | Path,
    *,
    smoothed_output_path: str | Path,
    smoothing_window: int = DEFAULT_SMOOTHING_WINDOW,
) -> list[Path]:
    """Write unsmoothed and display-smoothed matched-population figures."""
    negative_smoothed = _smooth_advection_composite_for_display(
        negative_composite,
        smoothing_window=smoothing_window,
    )
    positive_smoothed = _smooth_advection_composite_for_display(
        positive_composite,
        smoothing_window=smoothing_window,
    )
    written = [
        write_matched_advection_direction_exploration_plot(
            negative_composite,
            positive_composite,
            output_path,
        )
    ]
    written.append(
        write_matched_advection_direction_exploration_plot(
            negative_smoothed,
            positive_smoothed,
            smoothed_output_path,
        )
    )
    return written


def smoothed_output_path(output_path: str | Path) -> Path:
    """Return the sibling `_smoothed` PNG path for an advection figure."""
    path = Path(output_path).expanduser().resolve()
    return path.with_name(f"{path.stem}_smoothed{path.suffix}")


def _smooth_advection_composite_for_display(
    composite: xr.Dataset,
    *,
    smoothing_window: int,
) -> xr.Dataset:
    """Smooth face tendencies before grouped diagnostics are derived."""
    variables = (
        "advection",
        *(
            advection_direction.stage1_face_variable(face)
            for face in advection_direction.available_stage1_faces(composite)
        ),
    )
    return plotting.smooth_composite_for_display(
        composite,
        variables=variables,
        smoothing_window=smoothing_window,
    )


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


def _plot_matched_face_timeseries(
    ax,
    *,
    negative: xr.Dataset,
    positive: xr.Dataset,
    lag_days: np.ndarray,
) -> None:
    for sign, ds in (("positive", positive), ("negative", negative)):
        for face in advection_direction.available_stage1_faces(ds):
            name = advection_direction.stage1_face_variable(face)
            ax.plot(
                lag_days,
                ds[name].values,
                color=plot_style.FACE_COLORS[face],
                linestyle=MATCHED_SIGN_LINESTYLES[sign],
                label=face.capitalize() if sign == "positive" else "_nolegend_",
                gid=f"matched_{sign}_{name}",
            )
    plot_style.zero_line(ax)
    ax.set_ylabel("Δ [K hr-1]")
    ax.set_title("Signed face contributions")
    _add_upper_axis_headroom(ax)
    component_handles = [
        Line2D(
            [0],
            [0],
            color=plot_style.FACE_COLORS[face],
            linestyle="-",
            label=face.capitalize(),
        )
        for face in advection_direction.available_stage1_faces(positive)
    ]
    ax.legend(
        handles=[*component_handles, *_matched_sign_legend_handles()],
        ncol=4,
        loc="upper center",
        **plot_style.legend_kwargs(),
    )


def _plot_matched_grouped_timeseries(
    ax,
    *,
    negative: xr.Dataset,
    positive: xr.Dataset,
    lag_days: np.ndarray,
) -> None:
    for sign, ds in (("positive", positive), ("negative", negative)):
        for name in GROUP_NAMES:
            ax.plot(
                lag_days,
                ds[name].values,
                color=GROUP_COLORS[name],
                linestyle=MATCHED_SIGN_LINESTYLES[sign],
                label=GROUP_LABELS[name] if sign == "positive" else "_nolegend_",
                gid=f"matched_{sign}_{name}",
            )
    plot_style.zero_line(ax)
    ax.set_ylabel("Δ [K hr-1]")
    ax.set_title("Grouped advective contributions")
    _add_upper_axis_headroom(ax)
    component_handles = [
        Line2D(
            [0],
            [0],
            color=GROUP_COLORS[name],
            linestyle="-",
            label=GROUP_LABELS[name],
        )
        for name in GROUP_NAMES
    ]
    ax.legend(
        handles=[*component_handles, *_matched_sign_legend_handles()],
        ncol=4,
        loc="upper center",
        **plot_style.legend_kwargs(),
    )


def _matched_sign_legend_handles() -> list[Line2D]:
    """Return population line-style handles independent of component color."""
    return [
        Line2D(
            [0],
            [0],
            color=plot_style.COLORS["zero"],
            linestyle=MATCHED_SIGN_LINESTYLES[sign],
            label=rf"{sign.capitalize()} $I_{{\mathrm{{dyn,pre}}}}$",
        )
        for sign in ("positive", "negative")
    ]


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


def _advection_direction_diagnostics(composite: xr.Dataset) -> xr.Dataset:
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
    return diagnostics


def _validate_matched_composites(
    negative: xr.Dataset,
    positive: xr.Dataset,
) -> None:
    if not negative["lag_hour"].equals(positive["lag_hour"]):
        raise ValueError("Matched composites must use identical lag_hour coordinates.")
    negative_count = int(negative.attrs.get("n_events", 0))
    positive_count = int(positive.attrs.get("n_events", 0))
    if negative_count < 1 or negative_count != positive_count:
        raise ValueError(
            "Matched composites must contain the same positive event count as "
            "negative event count."
        )
    if negative.attrs.get("matched_sign") != "negative":
        raise ValueError("Negative composite must declare matched_sign='negative'.")
    if positive.attrs.get("matched_sign") != "positive":
        raise ValueError("Positive composite must declare matched_sign='positive'.")
    for name in (
        "data_representation",
        "matching_specification",
        "matching_label",
        "matching_variables",
        "matching_caliper_sd",
        "smoothing_window",
        "smoothing_applied_to",
    ):
        if negative.attrs.get(name) != positive.attrs.get(name):
            raise ValueError(f"Matched composites disagree on attribute {name!r}.")
    if negative.attrs.get("data_representation") != "climatological_anomaly":
        raise ValueError(
            "Matched advection composites must use climatological anomalies."
        )


def _matched_figure_title(
    negative: xr.Dataset,
    positive: xr.Dataset,
) -> str:
    region = negative.attrs.get("region", "unknown region")
    n_pairs = int(negative.attrs["n_events"])
    pre_days = negative.attrs.get("pre_days", "?")
    post_days = negative.attrs.get("post_days", "?")
    matching_label = negative.attrs.get(
        "matching_label",
        negative.attrs.get("matching_variables", "unknown variable"),
    )
    caliper = negative.attrs.get("matching_caliper_sd", "?")
    try:
        caliper_label = f"{float(caliper):.2f}"
    except (TypeError, ValueError):
        caliper_label = str(caliper)
    smoothing_label = _smoothing_title_label(negative)
    return (
        "Matched face-resolved advection climatological-anomaly composites "
        f"for {region} (n={n_pairs} pairs, {matching_label}, "
        f"{caliper_label} pooled SD, "
        f"-{pre_days} to +{post_days} days{smoothing_label})"
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
    smoothing_label = _smoothing_title_label(ds)
    return (
        f"Face-resolved advection{representation} composite for {region} "
        f"(n={n_events}, -{pre_days} to +{post_days} days{smoothing_label})"
    )


def _smoothing_title_label(ds: xr.Dataset) -> str:
    smoothing_window = ds.attrs.get("smoothing_window")
    if smoothing_window is None:
        return ""
    return f", {int(smoothing_window)}-hour running mean"
