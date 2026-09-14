"""Presentation additions must preserve full-layout and scientific contracts."""

from itertools import combinations

import numpy as np
import pytest
import xarray as xr
from HW_analysis.scripts.event_features import (
    plot_adiabatic_advection_comparison as event_plot,
)
from HW_analysis.scripts.event_features import (
    plot_adiabatic_advection_comparison_baseline as baseline_plot,
)

BUDGET_FIELDS = (
    "I_adiabatic_pre",
    "I_advection_pre",
    "I_dyn_pre",
    "I_dTdt_pre",
    "I_diabatic_pre",
)
RENDERERS = ("events", "baseline")


def tables():
    dynamical = np.linspace(-8.0, 8.0, 8)
    residual = np.array([4.0, 5.0, 6.0, 7.0, 1.0, 2.0, 3.0, 4.0])
    events = xr.Dataset(
        {
            "I_adiabatic_pre": ("event", np.linspace(-20.0, 20.0, 8)),
            "I_advection_pre": ("event", np.linspace(12.0, -12.0, 8)),
            "I_dyn_pre": ("event", dynamical),
            "I_dTdt_pre": ("event", dynamical + residual),
            "I_diabatic_pre": ("event", residual),
            "tas_anom_peak": ("event", np.arange(2.0, 10.0)),
        },
        coords={"event": np.arange(8)},
    )
    baseline = (
        events.drop_vars("tas_anom_peak").rename(event="baseline_day").copy(deep=True)
    )
    baseline["event_adjacent"] = ("baseline_day", [0, 0, 0, 0, 0, 0, 0, 1])
    return events, baseline


def render(kind, events, baseline, **kwargs):
    if kind == "events":
        return event_plot.plot_tendency_scatter(events, **kwargs)
    return baseline_plot.plot_tendency_scatter(baseline, events, **kwargs)


@pytest.mark.parametrize("kind", RENDERERS)
def test_presentation_shared_population_norm_counts_and_immutability(kind):
    events, baseline = tables()
    events["I_adiabatic_pre"][0] = np.nan
    events["I_diabatic_pre"][7] = np.nan
    events["I_dTdt_pre"][2] = np.inf
    events["tas_anom_peak"][5] = np.nan
    baseline["I_adiabatic_pre"][0] = np.nan
    baseline["I_diabatic_pre"][6] = np.nan
    baseline["I_dTdt_pre"][4] = np.inf
    before = events.copy(deep=True), baseline.copy(deep=True)
    fig = render(kind, events, baseline, layout="presentation")
    try:
        event_indices = [1, 3, 4, 6]
        baseline_indices = [1, 2, 3, 5]
        coordinates = (
            ("I_adiabatic_pre", "I_advection_pre"),
            ("I_dyn_pre", "I_diabatic_pre"),
        )
        norms = []
        for ax, (x, y) in zip(fig.axes[:2], coordinates, strict=True):
            points = ax.collections[-1]
            np.testing.assert_allclose(
                points.get_offsets(),
                np.column_stack((events[x][event_indices], events[y][event_indices])),
            )
            np.testing.assert_allclose(
                points.get_array(), events.tas_anom_peak[event_indices]
            )
            norms.append(points.norm)
            if kind == "baseline":
                np.testing.assert_allclose(
                    ax.collections[0].get_offsets(),
                    np.column_stack(
                        (baseline[x][baseline_indices], baseline[y][baseline_indices])
                    ),
                )
        assert norms[0] is norms[1]
        assert (norms[0].vmin, norms[0].vmax) == (3.0, 8.0)
        assert "events n = 4" in fig._suptitle.get_text().lower()
        if kind == "baseline":
            assert "Clean baseline n = 4" in fig._suptitle.get_text()
        assert not fig.axes[0].texts
        assert all("n =" not in text.get_text() for text in fig.axes[1].texts)
        assert fig.axes[1].get_title() == r"Diabatic Residual vs $I_{dyn,net}$"
        assert (
            fig._supxlabel.get_text() == event_plot.plot_style.BUDGET_REFERENCE_CAPTION
        )
        xr.testing.assert_identical(events, before[0])
        xr.testing.assert_identical(baseline, before[1])
    finally:
        event_plot.plt.close(fig)


@pytest.mark.parametrize("kind", RENDERERS)
@pytest.mark.parametrize("field", (*BUDGET_FIELDS, "tas_anom_peak"))
def test_presentation_excludes_nonfinite_values_in_every_required_field(kind, field):
    events, baseline = tables()
    events[field][7] = np.nan
    if field in BUDGET_FIELDS:
        baseline[field][6] = np.nan
    fig = render(kind, events, baseline, layout="presentation")
    try:
        for ax in fig.axes[:2]:
            assert len(ax.collections[-1].get_offsets()) == 7
            assert ax.collections[-1].norm.vmax == 8.0
            if kind == "baseline":
                assert len(ax.collections[0].get_offsets()) == (
                    6 if field in BUDGET_FIELDS else 7
                )
    finally:
        event_plot.plt.close(fig)


@pytest.mark.parametrize("kind", RENDERERS)
def test_presentation_rejects_empty_common_event_population_without_leaking_figures(
    kind,
):
    events, baseline = tables()
    events["I_dTdt_pre"][:] = np.nan
    before = event_plot.plt.get_fignums()
    with pytest.raises(ValueError, match="common finite"):
        render(kind, events, baseline, layout="presentation")
    assert event_plot.plt.get_fignums() == before


def test_presentation_rejects_empty_common_clean_baseline_population():
    events, baseline = tables()
    baseline["I_diabatic_pre"][:] = np.inf
    before = event_plot.plt.get_fignums()
    with pytest.raises(ValueError, match="common finite"):
        render("baseline", events, baseline, layout="presentation")
    assert event_plot.plt.get_fignums() == before


def test_event_presentation_supports_uncoloured_events():
    events, baseline = tables()
    events = events.drop_vars("tas_anom_peak")
    events["I_diabatic_pre"][0] = np.nan
    fig = render("events", events, baseline, layout="presentation", color_variable=None)
    try:
        assert len(fig.axes) == 2
        assert all(len(ax.collections[0].get_offsets()) == 7 for ax in fig.axes)
        assert "Events n = 7" in fig._suptitle.get_text()
    finally:
        event_plot.plt.close(fig)


@pytest.mark.parametrize("kind", RENDERERS)
@pytest.mark.parametrize("override", [False, True])
def test_presentation_event_style_defaults_and_explicit_overrides(kind, override):
    events, baseline = tables()
    kwargs = {}
    if override:
        kwargs = (
            {"point_size": 17.0, "alpha": 0.4}
            if kind == "events"
            else {"event_point_size": 17.0, "event_alpha": 0.4}
        )
    fig = render(kind, events, baseline, layout="presentation", **kwargs)
    try:
        for ax in fig.axes[:2]:
            assert ax.collections[-1].get_sizes()[0] == (17.0 if override else 40.0)
            assert ax.collections[-1].get_alpha() == (0.4 if override else 0.9)
            if kind == "baseline":
                assert ax.collections[0].get_sizes()[0] == 24.0
                assert ax.collections[0].get_alpha() == 0.2
    finally:
        event_plot.plt.close(fig)


@pytest.mark.parametrize("kind", RENDERERS)
def test_full_layout_remains_without_new_guides_counts_or_style_changes(kind):
    events, baseline = tables()
    fig = render(kind, events, baseline)
    try:
        assert len(fig.axes) == 5
        assert fig._supxlabel is None
        assert fig.axes[3].get_title() == r"Diabatic Heating vs $I_{dyn,net}$"
        for ax in fig.axes[:4]:
            assert len(ax.texts) == 1 and "n =" in ax.texts[0].get_text()
            assert not any(
                line.get_label().startswith("reference ") for line in ax.lines
            )
            assert ax.collections[-1].get_sizes()[0] == (
                24.0 if kind == "events" else 40.0
            )
            assert ax.collections[-1].get_alpha() == (0.75 if kind == "events" else 0.9)
    finally:
        event_plot.plt.close(fig)


@pytest.mark.parametrize("kind", RENDERERS)
@pytest.mark.parametrize("broad_range", [False, True])
def test_presentation_export_guides_labels_and_footer_are_legible(
    kind, broad_range, tmp_path
):
    events, baseline = tables()
    if broad_range:
        for table in (events, baseline):
            table["I_adiabatic_pre"][:] = np.linspace(-54.0, 28.0, 8)
            table["I_advection_pre"][:] = np.linspace(33.0, -42.0, 8)
            table["I_dyn_pre"][:] = np.linspace(-26.0, 9.0, 8)
            table["I_diabatic_pre"][:] = np.array([19, 5, 12, 8, 9, 6, 0, -4])
            table["I_dTdt_pre"][:] = table["I_dyn_pre"] + table["I_diabatic_pre"]
    fig = render(kind, events, baseline, layout="presentation")
    try:
        output = tmp_path / f"{kind}-presentation.png"
        event_plot.plot_style.save_figure(fig, output)
        assert output.stat().st_size > 0
        fig.set_dpi(event_plot.plot_style.DPI)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax = fig.axes[1]
        guides = [
            line for line in ax.lines if line.get_label().startswith("reference ")
        ]
        assert guides
        for line in guides:
            x, y = line.get_data()
            np.testing.assert_allclose(x + y, (x + y)[0])
            assert (x + y)[0] / 5 == pytest.approx(round((x + y)[0] / 5))
        labels = [text.get_window_extent(renderer) for text in ax.texts]
        assert all(not left.overlaps(right) for left, right in combinations(labels, 2))
        for box in labels:
            assert ax.bbox.x0 <= box.x0 < box.x1 <= ax.bbox.x1
            assert ax.bbox.y0 <= box.y0 < box.y1 <= ax.bbox.y1
        for panel in fig.axes[:2]:
            points = panel.collections[-1]
            centers = panel.transData.transform(points.get_offsets())
            radius = np.sqrt(points.get_sizes().max()) * fig.dpi / 144.0
            assert np.all(centers[:, 0] - radius > panel.bbox.x0)
            assert np.all(centers[:, 0] + radius < panel.bbox.x1)
        assert (
            fig._supxlabel.get_window_extent(renderer).y1
            < ax.xaxis.label.get_window_extent(renderer).y0
        )
        assert (
            fig._suptitle.get_window_extent(renderer).y0
            > fig.axes[0].title.get_window_extent(renderer).y1
        )
        np.testing.assert_allclose(fig.get_size_inches(), [7.5, 9.0])
    finally:
        event_plot.plt.close(fig)
