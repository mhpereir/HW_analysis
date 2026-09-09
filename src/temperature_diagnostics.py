"""Render Stage-2 temperature diagnostics without loading or deriving sources."""

from __future__ import annotations

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from . import plot_style

X_VARIABLES = (
    "tas_anom_antecedent_mean",
    "tas_anom_at_budget_start",
    "T_mean_anom_antecedent_mean",
    "T_mean_anom_at_budget_start",
)
Y_VARIABLE = "tas_anom_at_anchor"
COLOR_VARIABLE = "I_dTdt_pre"


def add_temperature_reference_lines(ax, *, slope: int, spacing: float = 2.0):
    """Draw and label visible y=slope*x+c lines without altering data limits."""
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    corners = np.array([y - slope * x for x in xlim for y in ylim])
    levels = (
        np.arange(
            np.ceil(corners.min() / spacing), np.floor(corners.max() / spacing) + 1
        )
        * spacing
    )
    for level in levels:
        # Clip analytically to the panel, so labels sit on visible segments.
        xs = sorted(((ylim[0] - level) / slope, (ylim[1] - level) / slope))
        lo, hi = max(xlim[0], xs[0]), min(xlim[1], xs[1])
        if hi - lo < 0.1 * (xlim[1] - xlim[0]):
            continue
        xx = np.array([lo, hi])
        ax.plot(
            xx,
            slope * xx + level,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            linestyle=":",
            alpha=0.55,
            zorder=0,
            label=f"reference {level:g} K",
        )
        xpos = lo + 0.82 * (hi - lo)
        ax.text(
            xpos,
            slope * xpos + level,
            f"{level:g} K",
            fontsize=plot_style.LEGEND_FONT_SIZE_PT - 1,
            color="0.35",
            ha="center",
            va="center",
            clip_on=True,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75, "pad": 0.3},
        )
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def temperature_plot_data(features: xr.Dataset):
    """Validate the new contract and return a single finite plotted population."""
    required = (*X_VARIABLES, Y_VARIABLE, COLOR_VARIABLE, "tas_anom_peak", "peak_time")
    missing = [name for name in required if name not in features]
    metadata = (
        "budget_start_lag_hours",
        "antecedent_temperature_window_hours",
        "antecedent_temperature_endpoint_inclusion",
        "temperature_anchor_variable",
    )
    missing += [name for name in metadata if name not in features.attrs]
    if missing or features.attrs.get("antecedent_temperature_contract_version") != 1:
        raise ValueError(
            "Rebuild Stage-2 with a climatology companion for this plot. "
            f"Missing fields/metadata: {', '.join(missing)}."
        )
    if features.attrs.get("pipeline_stage") != "stage_2_event_features":
        raise ValueError(
            "The antecedent-temperature plot requires Stage-2 event features."
        )
    if any(features[name].dims != ("event",) for name in required):
        raise ValueError(
            "Every antecedent-temperature plot field must have dims ('event',)."
        )
    if features.attrs["temperature_anchor_variable"] != "peak_time":
        raise ValueError("Event temperature anchor must be peak_time.")
    try:
        start, end = (
            int(v)
            for v in features.attrs["antecedent_temperature_window_hours"].split(",")
        )
        budget = int(features.attrs["budget_start_lag_hours"])
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Invalid temperature window metadata; rebuild Stage-2."
        ) from error
    if (
        start >= end
        or end != budget
        or budget >= 0
        or features.attrs["antecedent_temperature_endpoint_inclusion"]
        != "left_closed_right_open"
    ):
        raise ValueError("Inconsistent temperature window metadata; rebuild Stage-2.")
    values = {
        name: np.asarray(features[name].values, dtype=float)
        for name in (*X_VARIABLES, Y_VARIABLE, COLOR_VARIABLE)
    }
    keep = np.logical_and.reduce([np.isfinite(value) for value in values.values()])
    if not keep.any():
        raise ValueError(
            "No events have complete finite antecedent-temperature plot values."
        )
    severity = np.asarray(features["tas_anom_peak"].values, dtype=float)
    highlight = (
        int(np.nanargmax(np.where(np.isfinite(severity), severity, np.nan)))
        if np.isfinite(severity).any()
        else None
    )
    return values, keep, highlight, (start, end)


def plot_antecedent_temperature(features: xr.Dataset):
    """Compare surface anchor anomaly with preceding surface/volume warmth."""
    values, keep, highlight, (start, end) = temperature_plot_data(features)
    plot_style.apply_theme()
    fig, axes = plt.subplots(
        2,
        2,
        figsize=plot_style.publication_figsize("full", aspect=0.82),
        sharey=True,
        constrained_layout=True,
    )
    norm = plot_style.finite_range_color_norm(values[COLOR_VARIABLE][keep])
    interval = f"[{start / 24:g}, {end / 24:g}) days"
    start_label = f"{end / 24:g} days"
    titles = (
        "Surface antecedent mean",
        "Surface budget initial state",
        "Atmospheric antecedent mean",
        "Atmospheric budget initial state",
    )
    labels = (
        f"Mean TAS anomaly, {interval} (K)",
        f"TAS anomaly at {start_label} (K)",
        f"Mean $\\langle T\\rangle$ anomaly, {interval} (K)",
        f"$\\langle T\\rangle$ anomaly at {start_label} (K)",
    )
    ylim = plot_style.padded_data_limits(values[Y_VARIABLE][keep])
    for i, (ax, name) in enumerate(zip(axes.flat, X_VARIABLES, strict=True)):
        points = ax.scatter(
            values[name][keep],
            values[Y_VARIABLE][keep],
            c=values[COLOR_VARIABLE][keep],
            cmap=plot_style.INTEGRATED_WARMING_COLOR_MAP,
            norm=norm,
            s=32,
            alpha=0.9,
            edgecolors="white",
            linewidths=0.35,
            zorder=3,
        )
        ax.set_title(titles[i])
        ax.set_xlabel(labels[i])
        ax.set_ylabel("TAS anomaly at event anchor (K)")
        ax.set_xlim(plot_style.padded_data_limits(values[name][keep]))
        ax.set_ylim(ylim)
        plot_style.style_axis(ax)
        if i < 2:
            add_temperature_reference_lines(ax, slope=1)
        if highlight is not None and keep[highlight]:
            x, y = values[name][highlight], values[Y_VARIABLE][highlight]
            ax.scatter(
                [x],
                [y],
                s=130,
                facecolors="none",
                edgecolors="black",
                linewidths=1.3,
                zorder=4,
            )
            date = np.datetime_as_string(
                features["peak_time"].values[highlight], unit="D"
            )
            ax.annotate(
                date,
                (x, y),
                xytext=(-6, -15),
                textcoords="offset points",
                ha="right",
                fontsize=plot_style.LEGEND_FONT_SIZE_PT,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
                zorder=5,
            )
    fig.colorbar(points, ax=axes, shrink=0.9, label=plot_style.INTEGRATED_WARMING_LABEL)
    n, total = int(keep.sum()), keep.size
    region = str(features.attrs.get("region", "")).replace("_", " ")
    excluded_highlight = (
        "; highest-severity event excluded"
        if highlight is not None and not keep[highlight]
        else ""
    )
    fig.suptitle(
        f"{region}: antecedent warmth and event-anchor TAS anomaly\n"
        f"Events: {n} retained, {total - n} excluded{excluded_highlight}"
    )
    fig.supxlabel(
        "Top-row lines: TAS anomaly difference (anchor minus x), not atmospheric budget warming",
        fontsize=plot_style.LEGEND_FONT_SIZE_PT,
    )
    return fig
