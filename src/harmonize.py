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

import pandas as pd
import xarray as xr


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

