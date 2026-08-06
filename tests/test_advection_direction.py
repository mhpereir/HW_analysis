import numpy as np
import pytest
import xarray as xr

from HW_analysis.src import advection_direction


def test_add_face_advection_tendencies_normalizes_and_reconstructs_total():
    stage1, heat_budget = _make_inputs()

    out = advection_direction.add_face_advection_tendencies(stage1, heat_budget)

    np.testing.assert_allclose(out["advection_west"].values, [3600.0, 3600.0])
    np.testing.assert_allclose(out["advection_east"].values, [-1800.0, -1800.0])
    np.testing.assert_allclose(out["advection_south"].values, [900.0, 900.0])
    np.testing.assert_allclose(out["advection_north"].values, [-450.0, -450.0])
    np.testing.assert_allclose(out["advection_top"].values, [1350.0, 1350.0])
    np.testing.assert_allclose(
        sum(
            out[advection_direction.stage1_face_variable(face)]
            for face in advection_direction.REQUIRED_FACES
        ),
        out["advection"],
    )
    assert out["advection_west"].attrs["units"] == "K hr-1"
    assert out["advection_west"].attrs["source_variable"] == "flux_contribution_west"
    assert out.attrs["advection_face_contributions"] == 1


def test_add_face_advection_tendencies_supports_optional_bottom_face():
    stage1, heat_budget = _make_inputs()
    heat_budget["flux_contribution_bottom"] = ("time", [0.5, 1.0])
    heat_budget["advection_term"] = (
        heat_budget["advection_term"] + heat_budget["flux_contribution_bottom"]
    )
    stage1["advection"] = (
        heat_budget["advection_term"] / heat_budget["domain_volume"] * 3600.0
    )

    out = advection_direction.add_face_advection_tendencies(stage1, heat_budget)

    np.testing.assert_allclose(out["advection_bottom"].values, [900.0, 900.0])
    assert out.attrs["advection_face_contribution_faces"].endswith(",bottom")


def test_add_face_advection_tendencies_rejects_raw_reconstruction_failure():
    stage1, heat_budget = _make_inputs()
    heat_budget["advection_term"] = heat_budget["advection_term"] + 1.0

    with pytest.raises(ValueError, match="summed raw face heat contributions"):
        advection_direction.add_face_advection_tendencies(stage1, heat_budget)


def test_add_face_advection_tendencies_rejects_stage1_advection_mismatch():
    stage1, heat_budget = _make_inputs()
    stage1["advection"] = stage1["advection"] + 1.0

    with pytest.raises(ValueError, match="normalized face contributions"):
        advection_direction.add_face_advection_tendencies(stage1, heat_budget)


def test_add_face_advection_tendencies_rejects_time_mismatch():
    stage1, heat_budget = _make_inputs()
    heat_budget = heat_budget.assign_coords(
        time=heat_budget["time"] + np.timedelta64(1, "h")
    )

    with pytest.raises(ValueError, match="time coordinates must match exactly"):
        advection_direction.add_face_advection_tendencies(stage1, heat_budget)


def test_add_face_advection_tendencies_rejects_volume_mismatch():
    stage1, heat_budget = _make_inputs()
    heat_budget["domain_volume"] = heat_budget["domain_volume"] + 1.0

    with pytest.raises(ValueError, match="Stage-1 volume"):
        advection_direction.add_face_advection_tendencies(stage1, heat_budget)


def test_grouped_advection_components_preserves_face_sums():
    stage1, heat_budget = _make_inputs()
    enhanced = advection_direction.add_face_advection_tendencies(stage1, heat_budget)

    grouped = advection_direction.grouped_advection_components(enhanced)

    np.testing.assert_allclose(grouped["advection_zonal"], [1800.0, 1800.0])
    np.testing.assert_allclose(grouped["advection_meridional"], [450.0, 450.0])
    np.testing.assert_allclose(grouped["advection_horizontal"], [2250.0, 2250.0])
    np.testing.assert_allclose(grouped["advection_vertical"], [1350.0, 1350.0])
    np.testing.assert_allclose(grouped["advection_face_total"], [3600.0, 3600.0])


def test_grouped_component_ratios_use_requested_numerator_order():
    stage1, heat_budget = _make_inputs()
    enhanced = advection_direction.add_face_advection_tendencies(stage1, heat_budget)

    out = advection_direction.add_grouped_components_and_ratios(
        enhanced,
        ratio_epsilon=0.0,
    )

    np.testing.assert_allclose(
        out["advection_meridional_zonal_ratio"],
        [0.25, 0.25],
    )
    np.testing.assert_allclose(
        out["advection_horizontal_vertical_ratio"],
        [5.0 / 3.0, 5.0 / 3.0],
    )
    assert (
        out["advection_meridional_zonal_ratio"].attrs["numerator"]
        == "advection_meridional"
    )
    assert (
        out["advection_horizontal_vertical_ratio"].attrs["denominator"]
        == "advection_vertical"
    )


def test_masked_ratio_masks_small_denominators_and_preserves_sign():
    numerator = xr.DataArray([2.0, -3.0, 4.0], dims=("time",))
    denominator = xr.DataArray([1.0, -2.0, 0.01], dims=("time",))

    out = advection_direction.masked_ratio(
        numerator,
        denominator,
        epsilon=0.1,
    )

    np.testing.assert_allclose(out.values[:2], [2.0, 1.5])
    assert np.isnan(out.values[2])


def test_complete_daily_face_means_uses_complete_nonoverlapping_windows():
    lag = np.arange(-24, 25)
    composite = xr.Dataset(
        {
            advection_direction.stage1_face_variable(face): (
                "lag_hour",
                lag.astype(float) + index,
            )
            for index, face in enumerate(advection_direction.REQUIRED_FACES)
        },
        coords={"lag_hour": lag},
    )

    out = advection_direction.complete_daily_face_means(composite)

    assert out.sizes == {"daily_window": 2}
    np.testing.assert_allclose(out["lag_hour_center"], [-12.5, 11.5])
    np.testing.assert_allclose(out["lag_day_center"], [-12.5 / 24.0, 11.5 / 24.0])
    np.testing.assert_allclose(out["advection_west"], [-12.5, 11.5])
    np.testing.assert_array_equal(out["lag_hour_start"], [-24, 0])
    np.testing.assert_array_equal(out["lag_hour_end_exclusive"], [0, 24])


def test_complete_daily_face_means_rejects_incomplete_span():
    lag = np.arange(-24, 24)
    composite = xr.Dataset(
        {
            advection_direction.stage1_face_variable(face): (
                "lag_hour",
                np.ones(lag.size),
            )
            for face in advection_direction.REQUIRED_FACES
        },
        coords={"lag_hour": lag},
    )

    with pytest.raises(ValueError, match="positive multiple"):
        advection_direction.complete_daily_face_means(composite)


def _make_inputs() -> tuple[xr.Dataset, xr.Dataset]:
    time = np.array(
        ["2000-05-01T00:00", "2000-05-01T01:00"],
        dtype="datetime64[m]",
    )
    volume = np.array([2.0, 4.0])
    heat_budget = xr.Dataset(
        {
            "domain_volume": ("time", volume),
            "flux_contribution_west": ("time", [2.0, 4.0]),
            "flux_contribution_east": ("time", [-1.0, -2.0]),
            "flux_contribution_south": ("time", [0.5, 1.0]),
            "flux_contribution_north": ("time", [-0.25, -0.5]),
            "flux_contribution_top": ("time", [0.75, 1.5]),
        },
        coords={"time": time},
    )
    heat_budget["advection_term"] = sum(
        heat_budget[advection_direction.source_face_variable(face)]
        for face in advection_direction.REQUIRED_FACES
    )
    stage1 = xr.Dataset(
        {
            "volume": ("time", volume),
            "advection": (
                "time",
                (
                    heat_budget["advection_term"].values
                    / volume
                    * advection_direction.SECONDS_PER_HOUR
                ),
            ),
        },
        coords={"time": time},
    )
    return stage1, heat_budget
