"""Plot contract tests with distinct anchor and event-maximum anomalies."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from src import plot_style
from src.temperature_diagnostics import X_VARIABLES, plot_antecedent_temperature


def table():
    out = xr.Dataset(
        {name: ("event", np.arange(4.0) + i / 2) for i, name in enumerate(X_VARIABLES)}
    )
    out["tas_anom_at_anchor"] = ("event", [3.0, 5.0, 7.0, 8.0])
    out["tas_anom_peak"] = ("event", [4.0, 9.0, 8.0, 10.0])
    out["I_dTdt_pre"] = ("event", [2.0, 4.0, 6.0, 8.0])
    out["peak_time"] = (
        "event",
        np.array(
            ["2010-06-01", "2011-06-01", "2012-06-01", "2021-06-29"],
            dtype="datetime64[ns]",
        ),
    )
    out.attrs.update(
        {
            "pipeline_stage": "stage_2_event_features",
            "antecedent_temperature_contract_version": 1,
            "temperature_anchor_variable": "peak_time",
            "antecedent_temperature_window_hours": "-168,-96",
            "budget_start_lag_hours": -96,
            "antecedent_temperature_endpoint_inclusion": "left_closed_right_open",
        }
    )
    return out


def test_common_mask_shared_color_norm_anchor_y_and_top_only_reference_lines(tmp_path):
    data = table()
    data["T_mean_anom_at_budget_start"][1] = np.nan
    before = data.copy(deep=True)
    fig = plot_antecedent_temperature(data)
    try:
        assert len(fig.axes) == 5
        norm = fig.axes[0].collections[0].norm
        for i, ax in enumerate(fig.axes[:4]):
            points = ax.collections[0]
            assert points.norm is norm
            np.testing.assert_allclose(points.get_offsets()[:, 1], [3.0, 7.0, 8.0])
            np.testing.assert_allclose(points.get_array(), [2.0, 6.0, 8.0])
            np.testing.assert_allclose(
                points.get_offsets()[:, 0], data[X_VARIABLES[i]][[0, 2, 3]]
            )
            assert any(text.get_text() == "2021-06-29" for text in ax.texts)
            lines = [
                line for line in ax.lines if line.get_label().startswith("reference ")
            ]
            assert bool(lines) == (i < 2)
            for line in lines:
                difference = line.get_ydata() - line.get_xdata()
                np.testing.assert_allclose(difference, difference[0])
                assert difference[0] / 2 == pytest.approx(round(difference[0] / 2))
        assert "3 retained, 1 excluded" in fig._suptitle.get_text()
        assert "[-7, -4)" in fig.axes[0].get_xlabel()
        assert fig.axes[-1].get_ylabel() == plot_style.INTEGRATED_WARMING_LABEL
        plot_style.save_figure(fig, tmp_path / "temperature.png")
        assert (tmp_path / "temperature.png").stat().st_size > 0
        xr.testing.assert_identical(data, before)
    finally:
        plt.close(fig)


def test_plot_reads_product_lags_and_handles_excluded_severity_highlight():
    data = table()
    data.attrs.update(
        antecedent_temperature_window_hours="-120,-48", budget_start_lag_hours=-48
    )
    data["I_dTdt_pre"][3] = np.nan
    fig = plot_antecedent_temperature(data)
    try:
        assert "[-5, -2)" in fig.axes[0].get_xlabel()
        assert "-2 days" in fig.axes[1].get_xlabel()
        assert "highest-severity event excluded" in fig._suptitle.get_text()
        assert all(len(ax.collections) == 1 for ax in fig.axes[:4])
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "missing",
    [
        "tas_anom_at_anchor",
        "T_mean_anom_antecedent_mean",
        "antecedent_temperature_window_hours",
    ],
)
def test_old_products_require_rebuild(missing):
    data = table()
    if missing in data:
        data = data.drop_vars(missing)
    else:
        del data.attrs[missing]
    with pytest.raises(ValueError, match="Rebuild Stage-2"):
        plot_antecedent_temperature(data)


def test_empty_common_population_and_inconsistent_window_fail():
    data = table()
    data["tas_anom_at_anchor"][:] = np.nan
    with pytest.raises(ValueError, match="No events"):
        plot_antecedent_temperature(data)
    data = table()
    data.attrs["budget_start_lag_hours"] = -48
    with pytest.raises(ValueError, match="Inconsistent"):
        plot_antecedent_temperature(data)
