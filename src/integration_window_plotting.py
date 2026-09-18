"""Render target-event rank curves from saved integration-window products."""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src import plot_style
from src.integration_window_analysis import validate_heating_comparison


def plot_heating_ranks(tables: Sequence[xr.Dataset]):
    """Return one rank panel per region, with a common rank scale."""
    if not tables:
        raise ValueError("At least one comparison table is required.")
    for table in tables:
        validate_heating_comparison(table)
        np.testing.assert_array_equal(
            table.integration_days, tables[0].integration_days
        )
        for name in (
            "climatology_start_year",
            "climatology_end_year",
            "target_peak_time",
            "heat_budget_bottom_boundary",
            "heat_budget_top_boundary",
            "threshold_variable",
            "quantile",
        ):
            if table.attrs[name] != tables[0].attrs[name]:
                raise ValueError(f"Plot inputs disagree on {name}.")
    if len({table.attrs["region"] for table in tables}) != len(tables):
        raise ValueError("Each plot panel must describe a distinct region.")
    plot_style.apply_theme()
    fig, axes = plt.subplots(
        1,
        len(tables),
        squeeze=False,
        sharey=True,
        figsize=plot_style.publication_figsize(
            "full" if len(tables) > 1 else "single",
            aspect=0.48 if len(tables) > 1 else 0.8,
        ),
    )
    maximum = 1
    for ax, table in zip(axes.flat, tables, strict=True):
        target = table.isel(
            event=int(
                np.flatnonzero(table.event_id.values == table.attrs["target_event_id"])[
                    0
                ]
            )
        )
        for kind, style in plot_style.HEATING_RANK_STYLES.items():
            ranks = target[f"rank_{kind}_common"].values
            maximum = max(maximum, int(np.max(ranks)))
            ax.plot(
                table.integration_days,
                ranks,
                **style,
                markersize=4,
                markerfacecolor="none",
                linewidth=plot_style.LINE_WIDTH_PT,
            )
        plot_style.style_axis(ax)
        plot_style.format_integer_axis(ax.xaxis, spacing=3)
        ax.set_xticks(
            np.unique(
                np.r_[
                    table.integration_days.values[0],
                    np.arange(7, table.integration_days.values[-1] + 1, 3),
                    table.integration_days.values[-1],
                ]
            )
        )
        ax.set_xlim(
            float(table.integration_days.values[0]) - 0.5,
            float(table.integration_days.values[-1]) + 0.5,
        )
        ax.set_xlabel("Integration window (days)")
        region = plot_style.REGION_NAME_MAPPING.get(
            table.attrs["region"], table.attrs["region"]
        )
        ax.set_title(
            f"{region}\nFixed cohort: {table.attrs['common_population_size']} events",
            fontsize=plot_style.PAPER_FONT_SIZE_PT,
        )
        ax.axhline(
            1,
            color=plot_style.COLORS["zero"],
            linewidth=plot_style.REFERENCE_LINE_WIDTH_PT,
            alpha=0.5,
        )
    spacing = max(1, int(np.ceil(maximum / 5 / 5) * 5)) if maximum > 5 else 1
    for ax in axes.flat:
        plot_style.format_integer_axis(ax.yaxis, spacing=spacing)
        ax.set_yticks(
            np.unique(np.r_[1, np.arange(spacing, maximum + spacing, spacing)])
        )
        ax.set_ylim(maximum + max(1, spacing * 0.3), 0)
    axes[0, 0].set_ylabel("Heating rank (1 = largest)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=2,
        **plot_style.legend_kwargs(),
    )
    date = tables[0].attrs["target_peak_time"][:10]
    fig.suptitle(f"PNW heatwave peaking {date}: heating-rank sensitivity", y=1.01)
    start, end = (
        tables[0].attrs[name]
        for name in ("climatology_start_year", "climatology_end_year")
    )
    fig.text(
        0.5,
        0.045,
        r"Corrected heating = raw integral $-\,[\overline{\langle T\rangle}(t_p)-\overline{\langle T\rangle}(t_p-\Delta t)]$",
        ha="center",
        fontsize=11,
    )
    attrs = tables[0].attrs
    bottom = attrs["heat_budget_bottom_boundary"]
    top = attrs["heat_budget_top_boundary"].replace("hPa", " hPa")
    threshold = f"{attrs['threshold_variable'].upper()} q{attrs['quantile']}"
    fig.text(
        0.5,
        0.005,
        f"{start}-{end} hourly climatology; {threshold}, JJA; {bottom} to {top}. Same events at every window.",
        ha="center",
        fontsize=10,
    )
    fig.subplots_adjust(
        left=0.08 if len(tables) > 1 else 0.15,
        right=0.98,
        bottom=0.2,
        top=0.75,
        wspace=0.12,
    )
    return fig
