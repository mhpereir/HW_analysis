"""Low-level preprocessing utilities for the heatwave analysis pipeline.

Pipeline role:
- Provide reusable preprocessing operations used before or during harmonization.

Responsibilities:
- Floor or otherwise standardize time coordinates.
- Drop leap days.
- Convert units.
- Standardize coordinates.
- Compute area-weighted regional means.
- Compute day-of-year climatologies and anomalies.
- Provide resampling and interpolation utilities.

Out of scope:
- Event definition.
- Composite logic.
- Plotting code.
- Higher-level scientific diagnostics.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from . import config


def floor_daily_time(
    obj: xr.DataArray | xr.Dataset,
    *,
    time_dim: str = "time",
) -> xr.DataArray | xr.Dataset:
    """Floor a Dataset or DataArray time coordinate to calendar days."""
    if time_dim not in obj.coords:
        raise ValueError(f"Object is missing required time coordinate {time_dim!r}.")

    return obj.assign_coords({time_dim: obj[time_dim].dt.floor("D")})


def threshold_to_time(
    threshold: xr.DataArray,
    time: xr.DataArray,
    *,
    time_dim: str = "time",
    dayofyear_dim: str = "dayofyear",
    year_dim: str = "year",
    name: str | None = None,
) -> xr.DataArray:
    """Project day-of-year or year/day-of-year thresholds onto a time axis."""
    _validate_threshold_to_time_inputs(
        threshold,
        time,
        dayofyear_dim=dayofyear_dim,
        year_dim=year_dim,
    )

    time_values = time.values
    time_index = xr.DataArray(time_values, dims=(time_dim,), coords={time_dim: time_values})
    target_dayofyear = time_index.dt.dayofyear

    if threshold.dims == (dayofyear_dim,):
        expanded = threshold.reindex({dayofyear_dim: np.unique(target_dayofyear.values)})
        out = expanded.sel({dayofyear_dim: target_dayofyear})
    else:
        target_year = time_index.dt.year
        expanded = threshold.reindex(
            {
                year_dim: np.unique(target_year.values),
                dayofyear_dim: np.unique(target_dayofyear.values),
            }
        )
        out = expanded.sel({year_dim: target_year, dayofyear_dim: target_dayofyear})

    out = out.rename({out.dims[0]: time_dim}) if out.dims != (time_dim,) else out
    out = out.assign_coords({time_dim: time_values})
    out.name = name if name is not None else threshold.name
    out.attrs.update(
        {
            "projected_to_time": True,
            "source_threshold_dims": threshold.dims,
        }
    )
    return out


def exceedance_mask(
    series: xr.DataArray,
    threshold: xr.DataArray,
    *,
    mode: str = "above",
    name: str = "exceedance_mask",
) -> xr.DataArray:
    """Return a boolean mask where a time series is above or below threshold."""
    if series.dims != threshold.dims:
        raise ValueError(
            "series and threshold must have matching dimensions; "
            f"got {series.dims!r} and {threshold.dims!r}."
        )

    if mode == "above":
        mask = series > threshold
    elif mode == "below":
        mask = series < threshold
    else:
        raise ValueError("mode must be either 'above' or 'below'.")

    mask = mask.fillna(False).astype(bool)
    mask.name = name
    mask.attrs.update(
        {
            "mode": mode,
            "series_name": series.name,
            "threshold_name": threshold.name,
        }
    )
    return mask


def compute_region_mean(
    da: xr.DataArray,
    region: str,
    *,
    lat_dim: str = "lat",
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Compute a cosine-latitude weighted mean over a configured region."""
    if not isinstance(da, xr.DataArray):
        raise TypeError("compute_region_mean expects an xarray.DataArray.")

    if lat_dim not in da.dims:
        raise ValueError(f"DataArray is missing latitude dimension {lat_dim!r}.")
    if lon_dim not in da.dims:
        raise ValueError(f"DataArray is missing longitude dimension {lon_dim!r}.")

    try:
        lat_bounds, lon_bounds = config.REGIONS[region]
    except KeyError as exc:
        available = ", ".join(sorted(config.REGIONS))
        raise ValueError(f"Unknown region {region!r}. Available regions: {available}") from exc

    da = _ensure_minus180_to_180_longitudes(da, lon_dim=lon_dim)
    da_region = da.sel({lat_dim: lat_bounds, lon_dim: lon_bounds})
    if da_region.sizes[lat_dim] == 0 or da_region.sizes[lon_dim] == 0:
        raise ValueError(
            f"Region {region!r} selected no grid cells for dimensions "
            f"{lat_dim!r}/{lon_dim!r}."
        )

    weights_np = np.cos(np.deg2rad(da_region[lat_dim]))
    weights = xr.DataArray(
        weights_np,
        dims=(lat_dim,),
        coords={lat_dim: da_region[lat_dim]},
    )
    out = da_region.weighted(weights).mean(dim=[lat_dim, lon_dim])
    out.attrs.update(
        {
            "region": region,
            "spatial_mean": "cosine-latitude weighted mean",
            "lat_bounds": (lat_bounds.start, lat_bounds.stop),
            "lon_bounds": (lon_bounds.start, lon_bounds.stop),
        }
    )
    return out


def _ensure_minus180_to_180_longitudes(
    da: xr.DataArray,
    *,
    lon_dim: str = "lon",
) -> xr.DataArray:
    """Convert longitude coordinates from 0..360 to -180..180 when needed."""
    if float(da[lon_dim].max()) <= 180.0:
        return da

    lon = ((da[lon_dim] + 180.0) % 360.0) - 180.0
    return da.assign_coords({lon_dim: lon}).sortby(lon_dim)


def _validate_threshold_to_time_inputs(
    threshold: xr.DataArray,
    time: xr.DataArray,
    *,
    dayofyear_dim: str,
    year_dim: str,
) -> None:
    """Validate threshold projection inputs."""
    if not isinstance(threshold, xr.DataArray):
        raise TypeError("threshold_to_time expects threshold to be an xarray.DataArray.")

    if not isinstance(time, xr.DataArray):
        raise TypeError("threshold_to_time expects time to be an xarray.DataArray.")

    if time.ndim != 1:
        raise ValueError("time must be a 1D coordinate DataArray.")

    supported_dims = {(dayofyear_dim,), (year_dim, dayofyear_dim)}
    if threshold.dims not in supported_dims:
        raise ValueError(
            "threshold must have dimensions "
            f"({dayofyear_dim!r},) or ({year_dim!r}, {dayofyear_dim!r}); "
            f"got {threshold.dims!r}."
        )


__all__ = [
    "compute_region_mean",
    "exceedance_mask",
    "floor_daily_time",
    "threshold_to_time",
]
