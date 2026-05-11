"""Domain-specific derived diagnostics for the heatwave analysis pipeline.

Pipeline role:
- Compute scientific diagnostics that sit above generic preprocessing.

Responsibilities:
- Compute combined radiative metrics.
- Perform residual checks.
- Compute transformed or normalized diagnostics.
- Derive optional event metrics for ranking or labeling.

Out of scope:
- Generic preprocessing utilities.
- Raw data loading.
- Plotting.
"""

from __future__ import annotations

import xarray as xr

from . import config


def approximate_surface_energy_heating_rate(
    energy: xr.DataArray,
    domain_volume: xr.DataArray,
    *,
    region_area_m2: float,
    name: str | None = None,
    g_m_s2: float = config.G_M_S2,
    cp_j_kg_k: float = config.CP_J_KG_K,
) -> xr.DataArray:
    """Approximate a surface energy accumulation as a domain heating rate.

    The input energy is assumed to be an hourly accumulated regional mean in
    J m-2. Multiplying by the regional area gives a total energy, then
    normalizing by pressure-coordinate volume approximates the temperature
    tendency that would result if that energy were uniformly distributed
    through the control volume.
    """
    if not isinstance(energy, xr.DataArray):
        raise TypeError("energy must be an xarray.DataArray.")
    if not isinstance(domain_volume, xr.DataArray):
        raise TypeError("domain_volume must be an xarray.DataArray.")
    if region_area_m2 <= 0.0:
        raise ValueError("region_area_m2 must be positive.")

    out = energy * region_area_m2 * g_m_s2 / (cp_j_kg_k * domain_volume)
    out.name = name if name is not None else f"{energy.name}_heating_rate_approx"
    out.attrs.update(
        {
            "units": "K hr-1",
            "source_variable": energy.name,
            "source_units": energy.attrs.get("units", "J m-2"),
            "region_area_m2": float(region_area_m2),
            "g_m_s2": float(g_m_s2),
            "cp_j_kg_k": float(cp_j_kg_k),
            "normalized_by": domain_volume.name,
            "source_sign_convention": "source sign retained",
            "approximation": (
                "Assumes hourly accumulated surface energy is uniformly "
                "distributed through the pressure-coordinate control volume."
            ),
        }
    )
    return out
