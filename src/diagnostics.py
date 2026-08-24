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

from collections.abc import Sequence

import numpy as np
import xarray as xr

from . import config


BUDGET_FRACTION_SOURCES = {
    "f_adiabatic": "I_adiabatic_pre",
    "f_advection": "I_advection_pre",
    "f_dyn": "I_dyn_pre",
    "f_diabatic": "I_diabatic_pre",
}
GROSS_BUDGET_ACTIVITY = "gross_budget_activity"
BUDGET_SUMMARY_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)


def derive_signed_budget_fractions(
    features: xr.Dataset,
    *,
    row_dim: str,
) -> xr.Dataset:
    """Derive bounded signed fractions from existing Stage-2 budget features."""
    if not isinstance(features, xr.Dataset):
        raise TypeError("features must be an xarray.Dataset.")
    if not row_dim:
        raise ValueError("row_dim must be non-empty.")
    if row_dim not in features.sizes:
        raise ValueError(f"Feature table is missing row dimension {row_dim!r}.")

    source_names = tuple(BUDGET_FRACTION_SOURCES.values())
    missing = [name for name in source_names if name not in features]
    if missing:
        raise ValueError(
            "Feature table is missing required budget variables: "
            f"{', '.join(missing)}."
        )
    for name in source_names:
        if features[name].dims != (row_dim,):
            raise ValueError(
                f"{name} must have only the {row_dim!r} dimension; "
                f"found {features[name].dims}."
            )

    units = {
        str(features[name].attrs["units"])
        for name in source_names
        if features[name].attrs.get("units") not in {None, ""}
    }
    if len(units) > 1:
        raise ValueError(
            "Budget variables must use consistent units; found "
            f"{', '.join(sorted(units))}."
        )
    output_units = next(iter(units), "K")

    adiabatic = _float_feature_values(features, "I_adiabatic_pre")
    advection = _float_feature_values(features, "I_advection_pre")
    dynamical = _float_feature_values(features, "I_dyn_pre")
    diabatic = _float_feature_values(features, "I_diabatic_pre")
    expected_dynamical = adiabatic + advection
    if not np.array_equal(dynamical, expected_dynamical, equal_nan=True):
        raise ValueError(
            "Stored I_dyn_pre does not exactly equal "
            "I_adiabatic_pre + I_advection_pre."
        )

    gross_activity = np.abs(adiabatic) + np.abs(advection) + np.abs(diabatic)
    valid = (
        np.isfinite(adiabatic)
        & np.isfinite(advection)
        & np.isfinite(dynamical)
        & np.isfinite(diabatic)
        & np.isfinite(gross_activity)
        & (gross_activity > 0.0)
    )

    coordinates = {row_dim: features[row_dim]}
    out = xr.Dataset(coords=coordinates)
    source_values = {
        "f_adiabatic": adiabatic,
        "f_advection": advection,
        "f_dyn": dynamical,
        "f_diabatic": diabatic,
    }
    for output_name, values in source_values.items():
        fractions = np.full(values.shape, np.nan, dtype=float)
        fractions[valid] = values[valid] / gross_activity[valid]
        out[output_name] = (row_dim, fractions)
        out[output_name].attrs.update(
            {
                "long_name": (
                    "Signed gross-activity-normalized fraction of "
                    f"{BUDGET_FRACTION_SOURCES[output_name]}"
                ),
                "units": "1",
                "source_variable": BUDGET_FRACTION_SOURCES[output_name],
                "normalized_by": GROSS_BUDGET_ACTIVITY,
                "sign_convention": "positive is heating; negative is cooling",
            }
        )

    out[GROSS_BUDGET_ACTIVITY] = (
        row_dim,
        np.where(valid, gross_activity, np.nan),
    )
    out[GROSS_BUDGET_ACTIVITY].attrs.update(
        {
            "long_name": "Gross pre-peak heat-budget activity",
            "units": output_units,
            "source_variables": (
                "I_adiabatic_pre,I_advection_pre,I_diabatic_pre"
            ),
            "formula": (
                "abs(I_adiabatic_pre) + abs(I_advection_pre) + "
                "abs(I_diabatic_pre)"
            ),
        }
    )
    out.attrs.update(
        {
            "diagnostic": "signed_gross_activity_normalized_budget_fractions",
            "canonical_dynamical_variable": "I_dyn_pre",
            "invalid_row_definition": (
                "Any required non-finite value or gross_budget_activity <= 0"
            ),
        }
    )
    return out


def summarize_budget_fraction_distributions(
    fractions: xr.Dataset,
    *,
    row_dim: str,
    quantiles: Sequence[float] = BUDGET_SUMMARY_QUANTILES,
) -> xr.Dataset:
    """Summarize valid fraction and activity rows at requested quantiles."""
    if not isinstance(fractions, xr.Dataset):
        raise TypeError("fractions must be an xarray.Dataset.")
    if row_dim not in fractions.sizes:
        raise ValueError(f"Fraction table is missing row dimension {row_dim!r}.")

    required = (*BUDGET_FRACTION_SOURCES, GROSS_BUDGET_ACTIVITY)
    missing = [name for name in required if name not in fractions]
    if missing:
        raise ValueError(
            "Fraction table is missing required variables: "
            f"{', '.join(missing)}."
        )
    requested = np.asarray(tuple(quantiles), dtype=float)
    if (
        requested.ndim != 1
        or requested.size == 0
        or not np.all(np.isfinite(requested))
        or np.any((requested < 0.0) | (requested > 1.0))
        or np.any(np.diff(requested) <= 0.0)
    ):
        raise ValueError(
            "quantiles must be a non-empty strictly increasing sequence "
            "within [0, 1]."
        )

    summary = xr.Dataset(coords={"quantile": requested})
    for name in required:
        if fractions[name].dims != (row_dim,):
            raise ValueError(
                f"{name} must have only the {row_dim!r} dimension; "
                f"found {fractions[name].dims}."
            )
        values = np.asarray(fractions[name].values, dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            raise ValueError(f"{name} contains no finite rows to summarize.")
        summary[name] = ("quantile", np.quantile(finite, requested))
        summary[name].attrs.update(fractions[name].attrs)
        summary[name].attrs["n_valid"] = int(finite.size)

    summary.attrs.update(
        {
            "diagnostic": "regional_budget_fraction_distribution_summary",
            "summary_method": "finite_row_quantiles",
        }
    )
    return summary


def _float_feature_values(features: xr.Dataset, variable: str) -> np.ndarray:
    """Return one Stage-2 budget feature as a float array."""
    values = np.asarray(features[variable].values)
    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"{variable} must be numeric; found {values.dtype}.")
    return np.asarray(values, dtype=float)


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
