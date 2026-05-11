import numpy as np
import pytest
import xarray as xr

from src import config, diagnostics


def test_approximate_surface_energy_heating_rate_applies_pressure_volume_formula():
    energy = xr.DataArray(
        [10.0, 20.0],
        dims=("time",),
        coords={"time": [0, 1]},
        name="nssr",
        attrs={"units": "J m-2"},
    )
    volume = xr.DataArray(
        [100.0, 200.0],
        dims=("time",),
        coords={"time": [0, 1]},
        name="volume",
    )

    out = diagnostics.approximate_surface_energy_heating_rate(
        energy,
        volume,
        region_area_m2=1000.0,
        name="nssr_heating_rate_approx",
    )

    expected = energy.values * 1000.0 * config.G_M_S2 / (config.CP_J_KG_K * volume.values)
    np.testing.assert_allclose(out.values, expected)
    assert out.name == "nssr_heating_rate_approx"
    assert out.attrs["units"] == "K hr-1"
    assert out.attrs["source_variable"] == "nssr"
    assert out.attrs["region_area_m2"] == 1000.0
    assert out.attrs["source_sign_convention"] == "source sign retained"


def test_approximate_surface_energy_heating_rate_requires_positive_area():
    energy = xr.DataArray([1.0], dims=("time",), name="nssr")
    volume = xr.DataArray([1.0], dims=("time",), name="volume")

    with pytest.raises(ValueError, match="positive"):
        diagnostics.approximate_surface_energy_heating_rate(
            energy,
            volume,
            region_area_m2=0.0,
        )
