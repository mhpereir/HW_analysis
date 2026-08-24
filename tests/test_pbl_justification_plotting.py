from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from HW_analysis.src import pbl_justification, pbl_justification_plotting


def test_plot_product_shows_reference_envelope_and_domain_rectangle():
    product = _make_complete_product()

    figure = pbl_justification_plotting.plot_product(product)

    time_ax, map_ax, legend_ax, colorbar_ax = figure.axes
    assert time_ax.yaxis_inverted()
    assert time_ax.yaxis._plot_style_use_default_numeric_formatter
    assert any(
        np.allclose(np.asarray(line.get_ydata(), dtype=float), 700.0)
        for line in time_ax.lines
    )
    assert len(time_ax.collections) >= 1
    rectangles = [
        patch
        for patch in map_ax.patches
        if patch.get_label() == "pnw_bartusek analysis domain"
    ]
    assert len(rectangles) == 1
    assert rectangles[0].get_edgecolor()[:3] == pytest.approx(
        (0.8392, 0.1529, 0.1569), abs=1e-4
    )
    extent = map_ax.get_extent(crs=map_ax.projection)
    assert extent == pytest.approx((-132.5, -107.5, 37.5, 62.5))
    assert time_ax.get_legend() is None
    assert not legend_ax.axison
    assert [text.get_text() for text in legend_ax.get_legend().get_texts()] == [
        "Spatial 5th-95th percentile",
        "Area-weighted mean",
        "700 hPa analysis top",
    ]
    assert colorbar_ax.xaxis._plot_style_use_default_numeric_formatter
    assert colorbar_ax.get_xlabel() == ("PBL-top pressure (hPa)\nLower = deeper PBL")
    plt.close(figure)


def test_write_figure_is_non_overwriting_and_nonempty(tmp_path):
    figure, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    path = tmp_path / "figure.png"

    written = pbl_justification_plotting.write_figure(figure, path)

    assert written == path
    assert path.stat().st_size > 0
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pbl_justification_plotting.write_figure(figure, path)
    plt.close(figure)


def _make_complete_product() -> xr.Dataset:
    lag = np.arange(-24, 25)
    lat = np.array([40.0, 50.0, 60.0])
    lon = np.array([-130.0, -120.0, -110.0])
    mean = 90_000.0 - 2_000.0 * np.exp(-((lag / 12.0) ** 2))
    product = xr.Dataset(
        data_vars={
            pbl_justification.AREA_MEAN_NAME: ("lag_hour", mean),
            pbl_justification.SPATIAL_P05_NAME: ("lag_hour", mean - 5_000.0),
            pbl_justification.SPATIAL_P95_NAME: ("lag_hour", mean + 5_000.0),
            pbl_justification.MAP_NAME: (
                ("lat", "lon"),
                82_000.0 + np.arange(9).reshape(3, 3) * 500.0,
            ),
            pbl_justification.EVENT_SAMPLE_COUNT_NAME: (
                "lag_hour",
                np.full(lag.size, 3, dtype=np.int32),
            ),
            pbl_justification.SELECTED_EVENT_COUNT_NAME: np.int32(3),
            pbl_justification.SELECTED_DAY_COUNT_NAME: np.int32(8),
        },
        coords={"lag_hour": lag, "lat": lat, "lon": lon},
        attrs={
            "pipeline_stage": pbl_justification.PRODUCT_STAGE,
            "product_contract_version": pbl_justification.PRODUCT_CONTRACT_VERSION,
            "validation_status": "complete",
            "region": "pnw_bartusek",
            "pre_days": 1,
            "post_days": 1,
            "upper_boundary_reference_hpa": 700.0,
        },
    )
    for name in (
        pbl_justification.AREA_MEAN_NAME,
        pbl_justification.SPATIAL_P05_NAME,
        pbl_justification.SPATIAL_P95_NAME,
        pbl_justification.MAP_NAME,
    ):
        product[name].attrs["units"] = "Pa"
    return product
