"""Validation for Stage-1 assembly from accepted regional references."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from . import advection_direction, config, events, harmonize

FACES = (*advection_direction.REQUIRED_FACES, "bottom")
REUSED_VARIABLES = (
    "tas_region",
    "tas_climatology",
    "hw_threshold",
    "hw_flag",
    "hw_event_id",
    *(
        f"{prefix}_{suffix}"
        for prefix in ("lwa", "lwa_a", "lwa_c")
        for suffix in ("region", "threshold", "flag", "event_id")
    ),
    "soil_moisture",
    "cloud_cover",
    *harmonize.SURFACE_ENERGY_VARIABLES,
)
SELECTIONS = {
    "tas": ("hw_event_id", "tas_region"),
    "lwa_a": ("lwa_a_event_id", "lwa_a_region"),
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def require_equal(actual, expected, description: str) -> None:
    if not np.array_equal(np.asarray(actual), np.asarray(expected)):
        raise ValueError(f"{description} differ.")


def require_close(actual, expected, description: str) -> float:
    a, b = np.asarray(actual), np.asarray(expected)
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError(f"{description} contain non-finite values.")
    if not np.allclose(a, b, rtol=1e-10, atol=1e-12):
        raise ValueError(f"{description} fail closure or source equality.")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def require_finite(ds: xr.Dataset) -> None:
    for name, da in ds.data_vars.items():
        if np.issubdtype(da.dtype, np.number) and not np.isfinite(da.values).all():
            raise ValueError(f"{name} contains non-finite values.")
        if np.issubdtype(da.dtype, np.datetime64) and np.isnat(da.values).any():
            raise ValueError(f"{name} contains missing dates.")


def validate_manifest(manifest: dict, *, region: str, bottom: int, top: int) -> None:
    if not 0 < top < bottom:
        raise ValueError("Require 0 < top pressure < bottom pressure.")
    lat, lon = config.REGIONS[region]
    request = manifest["request"]
    require_equal(
        request["bbox"], [lat.start, lat.stop, lon.start, lon.stop], "Manifest region"
    )
    if (
        request["zg_bottom"] != "pressure_level"
        or request["zg_bottom_pressure"] != bottom * 100
        or request["zg_top_pressure"] != top * 100
    ):
        raise ValueError("Manifest pressure bounds do not match the requested layer.")
    if manifest["git"]["dirty"]:
        raise ValueError("EHB source manifest records dirty source code.")
    if manifest["surface_behaviour"]["allow_bottom_overflow"]:
        raise ValueError("Fixed pressure-layer assembly requires no bottom overflow.")


def validate_annual_budget(ds: xr.Dataset, *, year: int, bottom: int, top: int) -> dict:
    """Check the May-October centered-hour EHB campaign contract."""
    expected = pd.date_range(f"{year}-05-01T01:00", f"{year}-10-31T22:00", freq="h")
    require_equal(ds.time.values, expected.values, f"{year} seasonal hourly coverage")
    required = set(harmonize.HEAT_BUDGET_VARIABLE_MAP)
    required.update(f"flux_contribution_{face}" for face in FACES)
    required.update(f"mass_flux_contribution_{face}" for face in FACES)
    required.add("net_mass_advection")
    for name in sorted(required):
        if name not in ds or ds[name].dims != ("time",):
            raise ValueError(f"Missing hourly source field {name}.")
    require_finite(ds)
    volume = ds.domain_volume.values
    if not np.all(volume > 0):
        raise ValueError("Source volume must be positive.")
    for face, pressure in (("bottom", bottom), ("top", top)):
        for prefix in ("flux_contribution", "mass_flux_contribution"):
            if ds[f"{prefix}_{face}"].attrs.get("face_pressure_pa") != pressure * 100:
                raise ValueError(f"Incorrect {face} face pressure metadata.")
    attrs = ds.domain_volume.attrs
    if (
        attrs.get("zg_bottom_mode") != "pressure_level"
        or attrs.get("zg_bottom_pressure_pa") != bottom * 100
        or attrs.get("zg_top_pressure_pa") != top * 100
    ):
        raise ValueError("Incorrect volume pressure metadata.")
    rates = {
        name: ds[name].values / volume * 3600
        for name in harmonize.HEAT_BUDGET_RATE_VARIABLE_SIGNS
    }
    face_sum = (
        sum(ds[f"flux_contribution_{face}"].values for face in FACES) / volume * 3600
    )
    face_error = require_close(face_sum, rates["advection_term"], "Six EHB heat faces")
    closure_error = require_close(
        rates["dT_dt"],
        rates["advection_term"] + rates["adiabatic_term"] + rates["diabatic_term"],
        "EHB thermal budget",
    )
    mass_sum = (
        sum(ds[f"mass_flux_contribution_{face}"].values for face in FACES) / volume
    )
    mass_error = require_close(
        mass_sum, ds.net_mass_advection.values / volume, "Six EHB mass faces"
    )
    return {
        "year": year,
        "hours": len(expected),
        "minimum_volume": float(volume.min()),
        "heat_faces_max_abs_error_k_hr": face_error,
        "budget_max_abs_error_k_hr": closure_error,
        "mass_faces_max_abs_error_s_inverse": mass_error,
    }


def select_reference(
    reference: xr.Dataset,
    *,
    years: list[int],
    region: str,
    threshold_variable: str,
    quantile: str,
) -> xr.Dataset:
    expected_attrs = {
        "stage1_contract_version": 2,
        "region": region,
        "threshold_variable": threshold_variable,
        "quantile": quantile,
        "analysis_time_resolution": "hourly",
        "add_full_diagnostics": 1,
        "cloud_cover_source_layout": "global-hourly-grid",
    }
    for name, value in expected_attrs.items():
        if reference.attrs.get(name) != value:
            raise ValueError(f"Reference {name} must equal {value!r}.")
    missing = set(REUSED_VARIABLES).difference(reference.data_vars)
    if missing:
        raise ValueError(f"Reference lacks regional fields: {sorted(missing)}")
    selected = reference.sel(time=reference.time.dt.year.isin(years))
    expected_time = np.concatenate(
        [
            pd.date_range(f"{year}-05-01T01:00", f"{year}-10-31T22:00", freq="h").values
            for year in years
        ]
    )
    require_equal(
        selected.time.values, expected_time, "Reference complete-year coverage"
    )
    selected = selected.sel(event=selected.start_time.dt.year.isin(years))
    selected = selected.assign_coords(event=np.arange(selected.sizes["event"]))
    require_finite(selected)
    return selected


def validate_product(
    product: xr.Dataset, reference: xr.Dataset, budget: xr.Dataset
) -> dict:
    """Check stored output independently against source arrays and event membership."""
    require_finite(product)
    require_equal(product.time.values, budget.time.values, "Product EHB timestamps")
    require_equal(
        product.time.values, reference.time.values, "Product reference timestamps"
    )
    for name in REUSED_VARIABLES:
        require_equal(product[name].values, reference[name].values, f"Reused {name}")
    for name, variable in reference.data_vars.items():
        if variable.dims == ("event",):
            require_equal(
                product[name].values, variable.values, f"Preserved event field {name}"
            )
    volume = budget.domain_volume.values
    for source, target in harmonize.HEAT_BUDGET_VARIABLE_MAP.items():
        expected = budget[source].values
        if source in harmonize.HEAT_BUDGET_RATE_VARIABLE_SIGNS:
            expected = expected / volume * 3600
        require_close(product[target].values, expected, f"Normalized {target}")
    for face in FACES:
        require_close(
            product[f"advection_{face}"].values,
            budget[f"flux_contribution_{face}"].values / volume * 3600,
            f"Normalized {face} face",
        )
    face_error = require_close(
        sum(product[f"advection_{f}"].values for f in FACES),
        product.advection.values,
        "Product six-face advection",
    )
    closure_error = require_close(
        product.dTdt.values,
        product.advection.values + product.adiabatic.values + product.diabatic.values,
        "Product thermal budget",
    )
    for name in harmonize.SURFACE_ENERGY_VARIABLES:
        rate_name = f"{name}_heating_rate_approx"
        attrs = reference[rate_name].attrs
        expected = (
            reference[name].values
            * attrs["region_area_m2"]
            * attrs["g_m_s2"]
            / (attrs["cp_j_kg_k"] * volume)
        )
        require_close(product[rate_name].values, expected, f"Layer-volume {rate_name}")
    require_close(
        product.surface_energy_heating_rate_approx.values,
        sum(
            product[f"{n}_heating_rate_approx"].values
            for n in harmonize.SURFACE_ENERGY_VARIABLES
        ),
        "Surface-energy equivalent total",
    )
    event_name, peak_name = SELECTIONS[product.attrs["threshold_variable"]]
    rebuilt = events.build_event_summary_table(
        product, event_name, peak_variable=peak_name
    )
    for name in rebuilt.data_vars:
        values = product[name].values
        if name == "duration" and np.issubdtype(values.dtype, np.timedelta64):
            values = values / np.timedelta64(1, "D")
        require_equal(values, rebuilt[name].values, f"Rebuilt event field {name}")
    return {
        "hours": product.sizes["time"],
        "events": product.sizes["event"],
        "reused_fields_exact": list(REUSED_VARIABLES),
        "event_table_exact": True,
        "faces": list(FACES),
        "heat_faces_max_abs_error_k_hr": face_error,
        "budget_max_abs_error_k_hr": closure_error,
    }
