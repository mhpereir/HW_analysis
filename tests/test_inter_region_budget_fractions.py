from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from HW_analysis.scripts.event_features import (
    plot_inter_region_budget_fractions as plot_diag,
)
from HW_analysis.src import diagnostics


def test_derive_signed_budget_fractions_preserves_signs_and_stored_dyn():
    features = _make_event_table()

    out = diagnostics.derive_signed_budget_fractions(features, row_dim="event")

    np.testing.assert_allclose(
        out["f_adiabatic"].values[:4],
        [0.6, 0.4, -0.2, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out["f_advection"].values[:4],
        [-0.2, 0.1, 0.2, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out["f_dyn"].values[:4],
        [0.4, 0.5, 0.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out["f_diabatic"].values[:4],
        [0.2, 0.5, 0.6, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        out[diagnostics.GROSS_BUDGET_ACTIVITY].values[:4],
        [10.0, 10.0, 10.0, np.nan],
        equal_nan=True,
    )
    for variable in diagnostics.BUDGET_FRACTION_SOURCES:
        finite = out[variable].values[np.isfinite(out[variable].values)]
        assert np.all(np.abs(finite) <= 1.0)
    assert np.isnan(out["f_adiabatic"].values[3])
    assert np.isnan(out["f_adiabatic"].values[5])
    assert out["f_dyn"].attrs["source_variable"] == "I_dyn_pre"


def test_derive_signed_budget_fractions_rejects_invalid_contracts():
    features = _make_event_table()
    features["I_dyn_pre"] = ("event", np.zeros(features.sizes["event"]))
    with pytest.raises(ValueError, match="Stored I_dyn_pre"):
        diagnostics.derive_signed_budget_fractions(features, row_dim="event")

    features = _make_event_table()
    features["I_diabatic_pre"].attrs["units"] = "K hr-1"
    with pytest.raises(ValueError, match="consistent units"):
        diagnostics.derive_signed_budget_fractions(features, row_dim="event")

    features = _make_event_table().drop_vars("I_advection_pre")
    with pytest.raises(ValueError, match="I_advection_pre"):
        diagnostics.derive_signed_budget_fractions(features, row_dim="event")


def test_summarize_budget_fraction_distributions_uses_finite_rows():
    fractions = diagnostics.derive_signed_budget_fractions(
        _make_event_table(),
        row_dim="event",
    )

    summary = diagnostics.summarize_budget_fraction_distributions(
        fractions,
        row_dim="event",
    )

    np.testing.assert_array_equal(
        summary["quantile"].values,
        diagnostics.BUDGET_SUMMARY_QUANTILES,
    )
    assert summary["f_adiabatic"].attrs["n_valid"] == 4
    assert summary[diagnostics.GROSS_BUDGET_ACTIVITY].attrs["n_valid"] == 4
    assert summary["f_adiabatic"].sel(quantile=0.5).item() == pytest.approx(0.325)

    with pytest.raises(ValueError, match="strictly increasing"):
        diagnostics.summarize_budget_fraction_distributions(
            fractions,
            row_dim="event",
            quantiles=(0.5, 0.5),
        )


def test_prepare_regional_summary_selects_only_clean_baseline_rows():
    baseline = _make_baseline_table()
    baseline["event_adjacent"] = (
        "baseline_day",
        np.array([0, 1, 0, 0, 1, 1], dtype=np.int8),
    )

    summary = plot_diag.prepare_regional_budget_summary(
        "alaska",
        _make_event_table(),
        baseline,
    )

    assert summary.events["f_adiabatic"].attrs["n_valid"] == 4
    assert summary.baseline["f_adiabatic"].attrs["n_valid"] == 2
    assert dict(summary.compatibility_signature) == _compatibility_attrs()


def test_prepare_regional_summary_rejects_incompatible_event_and_baseline():
    baseline = _make_baseline_table()
    baseline.attrs["heat_budget_pre_window_hours"] = "-48,0"

    with pytest.raises(ValueError, match="metadata differ"):
        plot_diag.prepare_regional_budget_summary(
            "central_china",
            _make_event_table(),
            baseline,
        )


def test_plot_has_common_fraction_axes_and_all_region_labels():
    summaries = (
        plot_diag.prepare_regional_budget_summary(
            "alaska",
            _make_event_table(),
            _make_baseline_table(),
        ),
        plot_diag.prepare_regional_budget_summary(
            "central_china",
            _make_event_table(scale=1.4),
            _make_baseline_table(scale=1.4),
        ),
    )

    figure = plot_diag.plot_inter_region_budget_fractions(summaries)
    try:
        assert len(figure.axes) == 5
        assert [axis.get_title() for axis in figure.axes] == [
            "Adiabatic",
            "Advection",
            "Net dynamical",
            "Diabatic",
            "Gross activity",
        ]
        for axis in figure.axes[:4]:
            np.testing.assert_allclose(axis.get_xlim(), [-1.05, 1.05])
            assert axis.get_xlabel() == "Signed fraction"
        labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        assert labels == ["Alaska", "Central China"]
        assert figure.axes[0].get_ylim()[0] > figure.axes[0].get_ylim()[1]
        assert figure.axes[4].get_xlim()[0] == 0.0
        assert figure.axes[4].get_xlim()[1] > 0.0
        assert [text.get_text() for text in figure.legends[0].get_texts()] == [
            "Heatwave events",
            "Clean baseline days",
        ]
        plot_diag.plot_style.format_numeric_axes(figure)
        assert [
            label.get_text() for label in figure.axes[0].get_yticklabels()
        ] == ["Alaska", "Central China"]
    finally:
        plt.close(figure)


def test_plot_sorts_regions_north_to_south_by_configured_mean_latitude():
    input_order = (
        "central_china",
        "gulf_usa",
        "eastern_canada",
        "western_eu",
        "pnw_bartusek",
        "pnw_hotz",
        "alaska",
    )
    summaries = tuple(
        plot_diag.prepare_regional_budget_summary(
            region,
            _make_event_table(),
            _make_baseline_table(),
        )
        for region in input_order
    )

    figure = plot_diag.plot_inter_region_budget_fractions(summaries)
    try:
        labels = [label.get_text() for label in figure.axes[0].get_yticklabels()]
        assert labels == [
            "Alaska",
            "Pacific Northwest (Hotz)",
            "Pacific Northwest (Bartusek)",
            "Western Europe",
            "Eastern Canada",
            "Gulf USA",
            "Central China",
        ]
    finally:
        plt.close(figure)


def test_region_input_normalization_preserves_order_and_rejects_duplicates(tmp_path):
    normalized = plot_diag.normalize_region_inputs(
        [
            ("pnw_hotz", str(tmp_path / "event-hotz.nc"), str(tmp_path / "base-hotz.nc")),
            ("eastern_canada", str(tmp_path / "event-ec.nc"), str(tmp_path / "base-ec.nc")),
        ]
    )
    assert [item.region for item in normalized] == ["pnw_hotz", "eastern_canada"]
    assert normalized[0].event_path.is_absolute()

    with pytest.raises(ValueError, match="more than once"):
        plot_diag.normalize_region_inputs(
            [
                ("alaska", "event-a.nc", "base-a.nc"),
                ("alaska", "event-b.nc", "base-b.nc"),
            ]
        )
    with pytest.raises(ValueError, match="Unknown region"):
        plot_diag.normalize_region_inputs(
            [("atlantis", "event.nc", "baseline.nc")]
        )


def test_cli_writes_nonempty_figure_and_refuses_overwrite(tmp_path):
    event_path = tmp_path / "events.nc"
    baseline_path = tmp_path / "baseline.nc"
    output_path = tmp_path / "inter-region.png"
    _make_event_table().to_netcdf(event_path, engine="h5netcdf")
    _make_baseline_table().to_netcdf(baseline_path, engine="h5netcdf")

    arguments = [
        "--region-input",
        "alaska",
        str(event_path),
        str(baseline_path),
        "--output-path",
        str(output_path),
    ]
    assert plot_diag.main(arguments) == 0
    assert output_path.stat().st_size > 0
    with pytest.raises(FileExistsError, match="already exists"):
        plot_diag.main(arguments)


def test_scheduler_is_commit_pinned_and_supplies_all_regions():
    scheduler = (
        Path(__file__).resolve().parents[1]
        / "schedulers"
        / "schedule_plot_inter_region_budget_fractions.sh"
    )
    text = scheduler.read_text()

    assert "EXPECTED_COMMIT" in text
    assert 'test "${actual_commit}" = "${EXPECTED_COMMIT}"' in text
    assert 'test -z "$(git -C "${PROJECT_ROOT}" status --porcelain' in text
    assert "plot_inter_region_budget_fractions.py" in text
    assert 'test ! -e "${OUTPUT_PATH}"' in text
    assert "PYTHONWARNINGS=error" in text
    for region in plot_diag.plot_style.REGION_NAME_MAPPING:
        assert f"  {region}\n" in text


def _make_event_table(*, scale: float = 1.0) -> xr.Dataset:
    adiabatic = scale * np.array([6.0, 4.0, -2.0, 0.0, 1.0, 0.0])
    advection = scale * np.array([-2.0, 1.0, 2.0, 0.0, -1.0, np.nan])
    diabatic = scale * np.array([2.0, 5.0, 6.0, 0.0, 2.0, 1.0])
    table = xr.Dataset(
        {
            "I_adiabatic_pre": ("event", adiabatic),
            "I_advection_pre": ("event", advection),
            "I_dyn_pre": ("event", adiabatic + advection),
            "I_diabatic_pre": ("event", diabatic),
        },
        attrs={
            "pipeline_stage": plot_diag.EVENT_PIPELINE_STAGE,
            **_compatibility_attrs(),
        },
    )
    _add_units(table)
    return table


def _make_baseline_table(*, scale: float = 1.0) -> xr.Dataset:
    event_table = _make_event_table(scale=scale).rename({"event": "baseline_day"})
    event_table.attrs["pipeline_stage"] = plot_diag.BASELINE_PIPELINE_STAGE
    event_table["event_adjacent"] = (
        "baseline_day",
        np.array([0, 0, 0, 0, 0, 1], dtype=np.int8),
    )
    return event_table


def _add_units(table: xr.Dataset) -> None:
    for variable in diagnostics.BUDGET_FRACTION_SOURCES.values():
        table[variable].attrs["units"] = "K"


def _compatibility_attrs() -> dict[str, str]:
    return {
        "integral_method": "hourly_sum_assuming_1h_spacing",
        "window_endpoint_inclusion": "inclusive",
        "heat_budget_pre_window_hours": "-96,0",
    }
