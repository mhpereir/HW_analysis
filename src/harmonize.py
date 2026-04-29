"""Harmonization of heterogeneous source data into a common analysis dataset.

Pipeline role:
- Workflow layer: dataset harmonization within top-level Stage 1.

Responsibilities:
- Map variables from different sources to shared internal names.
- Align time coordinates.
- Align source-specific spatial domains.
- Apply regional averaging where required.
- Harmonize variables to a target analysis timestep.
- Assemble the common analysis-ready dataset.

Out of scope:
- Event definition.
- Composite generation.
- Plotting.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


HEAT_BUDGET_VARIABLE_MAP: dict[str, str] = {
    "T_domain_avg": "T_mean",
    "domain_volume": "volume",
    "dT_dt": "dTdt",
    "advection_term": "advection",
    "adiabatic_term": "adiabatic",
    "diabatic_term": "diabatic",
}


def build_regional_analysis_dataset(
    *,
    heat_budget: xr.Dataset,
    hw_event_products: Mapping[str, xr.DataArray],
    lwa_event_products: Sequence[Mapping[str, xr.DataArray]] = (),
    attrs: Mapping[str, object] | None = None,
    time_dim: str = "time",
) -> xr.Dataset:
    """Assemble the Stage-1 regional analysis dataset on an hourly time axis.

    Daily-native variables are projected onto the hourly heat-budget time axis
    by calendar-day lookup and keep attrs that record their original daily
    resolution.
    """
    if time_dim not in heat_budget.coords:
        raise ValueError(f"heat_budget is missing required time coordinate {time_dim!r}.")

    hourly_time = heat_budget[time_dim]
    data_vars: dict[str, xr.DataArray] = {}
    data_vars.update(_prepare_heat_budget_variables(heat_budget, time_dim=time_dim))
    data_vars.update(
        _project_daily_product_variables(
            hw_event_products,
            hourly_time,
            specs={
                "tas_region": ("tas_region", "continuous"),
                "hw_threshold": ("hw_threshold", "continuous"),
                "hw_exceedance_mask": ("hw_flag", "flag"),
                "hw_event_id": ("hw_event_id", "event_id"),
            },
            time_dim=time_dim,
        )
    )

    for products in lwa_event_products:
        prefix = _infer_lwa_product_prefix(products)
        data_vars.update(
            _project_daily_product_variables(
                products,
                hourly_time,
                specs={
                    f"{prefix}_region": (f"{prefix}_region", "continuous"),
                    f"{prefix}_threshold": (f"{prefix}_threshold", "continuous"),
                    f"{prefix}_exceedance_mask": (f"{prefix}_flag", "flag"),
                    f"{prefix}_event_id": (f"{prefix}_event_id", "event_id"),
                },
                time_dim=time_dim,
            )
        )

    ds = xr.Dataset(data_vars=data_vars, coords={time_dim: hourly_time})
    ds.attrs.update(
        {
            "pipeline_stage": "stage_1_harmonized_regional_timeseries",
            "analysis_time_resolution": "hourly",
            "time_axis": time_dim,
        }
    )
    if attrs is not None:
        ds.attrs.update(dict(attrs))

    return ds


def project_daily_to_hourly(
    daily: xr.DataArray,
    hourly_time: xr.DataArray,
    *,
    daily_time_dim: str = "time",
    hourly_time_dim: str = "hourly_time",
    name: str | None = None,
) -> xr.DataArray:
    """Replicate daily values onto hourly timestamps by exact calendar day."""
    _validate_daily_projection_inputs(
        daily,
        hourly_time,
        daily_time_dim=daily_time_dim,
    )

    daily_days = pd.DatetimeIndex(daily[daily_time_dim].values).floor("D")
    hourly_values = hourly_time.values
    hourly_days = pd.DatetimeIndex(hourly_values).floor("D")

    if daily_days.has_duplicates:
        raise ValueError(
            "Daily input has duplicate dates after flooring; projection would be ambiguous."
        )

    missing_days = pd.DatetimeIndex(hourly_days.unique()).difference(daily_days)
    if len(missing_days) > 0:
        preview = ", ".join(day.strftime("%Y-%m-%d") for day in missing_days[:5])
        suffix = "" if len(missing_days) <= 5 else f", ... ({len(missing_days)} total)"
        raise ValueError(
            "Daily input is missing dates required by the hourly target: "
            f"{preview}{suffix}."
        )

    projected = daily.assign_coords({daily_time_dim: daily_days}).reindex(
        {daily_time_dim: hourly_days}
    )
    values = projected.values

    out = xr.DataArray(
        values,
        dims=(hourly_time_dim,),
        coords={hourly_time_dim: hourly_values},
        name=name if name is not None else daily.name,
        attrs=dict(daily.attrs),
    )
    out.attrs.update(
        {
            "projected_from_daily": True,
            "daily_time_dim": daily_time_dim,
            "hourly_time_dim": hourly_time_dim,
        }
    )
    return out


def _validate_daily_projection_inputs(
    daily: xr.DataArray,
    hourly_time: xr.DataArray,
    *,
    daily_time_dim: str,
) -> None:
    """Validate inputs for date-based daily-to-hourly projection."""
    if not isinstance(daily, xr.DataArray):
        raise TypeError("project_daily_to_hourly expects daily to be an xarray.DataArray.")

    if daily_time_dim not in daily.coords:
        raise ValueError(
            f"Daily input is missing required time coordinate {daily_time_dim!r}."
        )

    if daily.dims != (daily_time_dim,):
        raise ValueError(
            "project_daily_to_hourly currently accepts only 1D daily inputs "
            f"with dims ({daily_time_dim!r},); got {daily.dims!r}."
        )

    if hourly_time.ndim != 1:
        raise ValueError("hourly_time must be a 1D coordinate DataArray.")


def _prepare_heat_budget_variables(
    heat_budget: xr.Dataset,
    *,
    time_dim: str,
) -> dict[str, xr.DataArray]:
    """Rename hourly heat-budget variables into internal analysis names."""
    missing = sorted(source for source in HEAT_BUDGET_VARIABLE_MAP if source not in heat_budget)
    if missing:
        raise ValueError(f"heat_budget is missing required variables: {', '.join(missing)}")

    out: dict[str, xr.DataArray] = {}
    for source_name, output_name in HEAT_BUDGET_VARIABLE_MAP.items():
        da = heat_budget[source_name].rename(output_name)
        if time_dim not in da.dims:
            raise ValueError(
                f"heat_budget variable {source_name!r} is missing dimension {time_dim!r}."
            )
        da = da.assign_coords({time_dim: heat_budget[time_dim]})
        da.attrs.update(
            {
                "source_variable": source_name,
                "native_time_resolution": "hourly",
                "analysis_time_resolution": "hourly",
            }
        )
        out[output_name] = da

    return out


def _project_daily_product_variables(
    products: Mapping[str, xr.DataArray],
    hourly_time: xr.DataArray,
    *,
    specs: Mapping[str, tuple[str, str]],
    time_dim: str,
) -> dict[str, xr.DataArray]:
    """Project selected daily product variables to the hourly analysis axis."""
    missing = sorted(source for source in specs if source not in products)
    if missing:
        raise ValueError(f"daily event products are missing variables: {', '.join(missing)}")

    return {
        output_name: _project_daily_analysis_variable(
            products[source_name],
            hourly_time,
            name=output_name,
            kind=kind,
            time_dim=time_dim,
        )
        for source_name, (output_name, kind) in specs.items()
    }


def _project_daily_analysis_variable(
    daily: xr.DataArray,
    hourly_time: xr.DataArray,
    *,
    name: str,
    kind: str,
    time_dim: str,
) -> xr.DataArray:
    """Project one daily-native analysis variable to hourly storage."""
    projected = project_daily_to_hourly(
        daily,
        hourly_time,
        hourly_time_dim=time_dim,
        name=name,
    )

    if kind == "event_id":
        projected = projected.fillna(0).astype(np.int64)
    elif kind == "flag":
        projected = projected.fillna(False).astype(np.int8)
    elif kind != "continuous":
        raise ValueError(f"Unsupported daily projection kind {kind!r}.")

    projected.attrs.update(
        {
            "native_time_resolution": "daily",
            "analysis_time_resolution": "hourly",
            "projection_method": "calendar_day_lookup",
        }
    )
    return projected


def _infer_lwa_product_prefix(products: Mapping[str, xr.DataArray]) -> str:
    """Infer the LWA product key prefix, such as ``lwa_a`` or ``lwa_c``."""
    prefixes = sorted(
        key.removesuffix("_event_id")
        for key in products
        if key.startswith("lwa") and key.endswith("_event_id")
    )
    if len(prefixes) != 1:
        raise ValueError(
            "Could not infer exactly one LWA product prefix from keys: "
            f"{', '.join(sorted(products))}"
        )
    return prefixes[0]
