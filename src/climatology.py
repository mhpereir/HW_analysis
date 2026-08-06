"""Regional calendar-hour climatology and anomaly calculations.

Pipeline role:
- Build the compact Stage-1 regional hourly climatology companion.
- Match climatology by calendar month, day, and UTC hour.
- Apply climatological anomalies before event stacking.

Out of scope:
- Stage-1 source loading.
- Event definition or selection.
- Plot rendering.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr


TIME_DIM = "time"
CLIMATOLOGY_DIM = "climatology_time"
REFERENCE_YEAR = 2000
PIPELINE_STAGE = "stage_1_regional_hourly_climatology"
DEFAULT_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
    "pbl_p_mean",
    "pbl_p_p05",
    "pbl_p_p95",
    "nslr_heating_rate_approx",
    "nssr_heating_rate_approx",
    "sshf_heating_rate_approx",
    "slhf_heating_rate_approx",
    "soil_moisture",
    "cloud_cover",
    "advection_west",
    "advection_east",
    "advection_south",
    "advection_north",
    "advection_top",
)


def build_regional_hourly_climatology(
    ds: xr.Dataset,
    *,
    variables: Sequence[str] = DEFAULT_VARIABLES,
    time_dim: str = TIME_DIM,
) -> xr.Dataset:
    """Return calendar-month/day/UTC-hour means, standard deviations, and counts."""
    names = _validate_source_dataset(ds, variables=variables, time_dim=time_dim)
    keys = calendar_hour_keys(ds[time_dim], time_dim=time_dim)
    _validate_unique_year_keys(ds[time_dim], keys)

    selected = ds[list(names)].assign_coords({CLIMATOLOGY_DIM: keys})
    grouped = selected.groupby(CLIMATOLOGY_DIM)
    means = grouped.mean(time_dim, skipna=True)
    standard_deviations = grouped.std(time_dim, skipna=True, ddof=1)
    counts = grouped.count(time_dim)

    out = xr.Dataset(coords={CLIMATOLOGY_DIM: means[CLIMATOLOGY_DIM]})
    for name in names:
        source_attrs = dict(ds[name].attrs)
        out[name] = means[name]
        out[name].attrs = source_attrs
        out[name].attrs.update(
            {
                "long_name": f"Calendar-hour climatological mean of {name}",
                "climatology_statistic": "mean",
            }
        )

        std_name = f"{name}_std"
        out[std_name] = standard_deviations[name]
        out[std_name].attrs = source_attrs
        out[std_name].attrs.update(
            {
                "long_name": f"Interannual sample standard deviation of {name}",
                "climatology_statistic": "sample standard deviation",
                "climatology_ddof": 1,
            }
        )

        count_name = f"{name}_count"
        out[count_name] = counts[name].astype(np.int64)
        out[count_name].attrs.update(
            {
                "long_name": f"Finite source-year count for {name}",
                "units": "1",
                "climatology_statistic": "finite count",
            }
        )

    climate_index = pd.DatetimeIndex(out[CLIMATOLOGY_DIM].values)
    out = out.assign_coords(
        month=(CLIMATOLOGY_DIM, climate_index.month.astype(np.int16)),
        day=(CLIMATOLOGY_DIM, climate_index.day.astype(np.int16)),
        hour_utc=(CLIMATOLOGY_DIM, climate_index.hour.astype(np.int16)),
    )
    source_index = pd.DatetimeIndex(ds[time_dim].values)
    source_attrs = ds.attrs
    out.attrs.update(
        {
            "pipeline_stage": PIPELINE_STAGE,
            "climatology_reference_year": REFERENCE_YEAR,
            "climatology_matching": "calendar month, day, and UTC hour",
            "climatology_method": "arithmetic mean over source years",
            "climatology_standard_deviation_ddof": 1,
            "climatology_event_exclusion": "none",
            "climatology_missing_value_policy": "skip non-finite values per variable",
            "climatology_start_year": int(source_index.year.min()),
            "climatology_end_year": int(source_index.year.max()),
            "source_time_start": source_index[0].isoformat(),
            "source_time_end": source_index[-1].isoformat(),
            "source_time_count": int(source_index.size),
            "source_stage1_contract_version": int(
                source_attrs.get("stage1_contract_version", 1)
            ),
            "region": str(source_attrs.get("region", "")),
            "heat_budget_bottom_boundary": str(
                source_attrs.get("heat_budget_bottom_boundary", "")
            ),
            "heat_budget_top_boundary": str(
                source_attrs.get("heat_budget_top_boundary", "")
            ),
            "climatology_variables": ",".join(names),
        }
    )
    validate_regional_hourly_climatology(out, required_variables=names)
    return out


def apply_regional_hourly_climatology(
    ds: xr.Dataset,
    climatology: xr.Dataset,
    *,
    variables: Sequence[str],
    baseline_variables: Mapping[str, str] | None = None,
    time_dim: str = TIME_DIM,
) -> xr.Dataset:
    """Return a shallow Stage-1 view with selected variables replaced by anomalies."""
    names = _validate_source_dataset(ds, variables=variables, time_dim=time_dim)
    baseline_names = {
        name: name if baseline_variables is None else baseline_variables.get(name, name)
        for name in names
    }
    unknown = sorted(set(baseline_names.values()).difference(climatology.data_vars))
    if unknown:
        raise ValueError(
            "Climatology is missing baseline variables: " + ", ".join(unknown)
        )
    validate_regional_hourly_climatology(
        climatology,
        required_variables=tuple(dict.fromkeys((*names, *baseline_names.values()))),
    )
    keys = calendar_hour_keys(ds[time_dim], time_dim=time_dim)
    available = set(np.asarray(climatology[CLIMATOLOGY_DIM].values))
    missing = sorted(set(np.asarray(keys.values)).difference(available))
    if missing:
        preview = ", ".join(str(value) for value in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise ValueError(f"Climatology is missing calendar-hour keys: {preview}{suffix}.")

    selector = xr.DataArray(
        keys.values,
        dims=(time_dim,),
        coords={time_dim: ds[time_dim]},
    )
    matched = climatology[list(dict.fromkeys(baseline_names.values()))].sel(
        {CLIMATOLOGY_DIM: selector}
    )
    out = ds.copy(deep=False)
    for name in names:
        baseline_name = baseline_names[name]
        anomaly = (ds[name] - matched[baseline_name]).rename(name)
        anomaly.attrs = dict(ds[name].attrs)
        anomaly.attrs.update(
            {
                "long_name": f"Climatological anomaly of {name}",
                "data_representation": "climatological_anomaly",
                "climatology_matching": climatology.attrs["climatology_matching"],
                "climatology_start_year": int(
                    climatology.attrs["climatology_start_year"]
                ),
                "climatology_end_year": int(
                    climatology.attrs["climatology_end_year"]
                ),
                "climatology_baseline_variable": baseline_name,
            }
        )
        out[name] = anomaly

    out.attrs = dict(ds.attrs)
    out.attrs.update(
        {
            "data_representation": "climatological_anomaly",
            "climatology_pipeline_stage": climatology.attrs["pipeline_stage"],
            "climatology_matching": climatology.attrs["climatology_matching"],
            "climatology_start_year": int(climatology.attrs["climatology_start_year"]),
            "climatology_end_year": int(climatology.attrs["climatology_end_year"]),
            "climatology_variables_applied": ",".join(names),
            "climatology_baseline_variables": ",".join(
                f"{name}:{baseline_names[name]}" for name in names
            ),
        }
    )
    return out


def calendar_hour_keys(
    time: xr.DataArray,
    *,
    time_dim: str = TIME_DIM,
) -> xr.DataArray:
    """Encode month, day, and hour as timestamps in the leap reference year 2000."""
    if not isinstance(time, xr.DataArray):
        raise TypeError("time must be an xarray.DataArray.")
    if time.dims != (time_dim,):
        raise ValueError(f"time must have dims ({time_dim!r},); got {time.dims!r}.")
    if not np.issubdtype(time.dtype, np.datetime64):
        raise TypeError("time must contain datetime64 values.")
    index = pd.DatetimeIndex(time.values)
    if index.hasnans:
        raise ValueError("time must not contain missing timestamps.")
    reference = pd.to_datetime(
        {
            "year": np.full(index.size, REFERENCE_YEAR, dtype=np.int16),
            "month": index.month,
            "day": index.day,
            "hour": index.hour,
        }
    ).to_numpy(dtype="datetime64[ns]")
    return xr.DataArray(
        reference,
        dims=(time_dim,),
        coords={time_dim: time},
        name=CLIMATOLOGY_DIM,
        attrs={
            "long_name": "Calendar month-day-hour key encoded in reference year 2000",
            "reference_year": REFERENCE_YEAR,
            "time_standard": "UTC",
        },
    )


def validate_regional_hourly_climatology(
    ds: xr.Dataset,
    *,
    required_variables: Sequence[str] = DEFAULT_VARIABLES,
) -> None:
    """Validate the regional hourly climatology companion contract."""
    if not isinstance(ds, xr.Dataset):
        raise TypeError("climatology must be an xarray.Dataset.")
    if ds.attrs.get("pipeline_stage") != PIPELINE_STAGE:
        raise ValueError(
            f"Expected pipeline_stage={PIPELINE_STAGE!r}; "
            f"got {ds.attrs.get('pipeline_stage')!r}."
        )
    if CLIMATOLOGY_DIM not in ds.coords:
        raise ValueError(f"Climatology is missing coordinate {CLIMATOLOGY_DIM!r}.")
    index = pd.DatetimeIndex(ds[CLIMATOLOGY_DIM].values)
    if not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("climatology_time must be unique and strictly increasing.")
    if np.any(index.year != REFERENCE_YEAR):
        raise ValueError(
            f"climatology_time must use reference year {REFERENCE_YEAR}."
        )
    missing: list[str] = []
    for name in tuple(required_variables):
        for required in (name, f"{name}_std", f"{name}_count"):
            if required not in ds:
                missing.append(required)
    if missing:
        raise ValueError(
            "Climatology is missing required variables: " + ", ".join(sorted(missing))
        )
    for name in tuple(required_variables):
        if ds[name].dims != (CLIMATOLOGY_DIM,):
            raise ValueError(
                f"Climatology variable {name!r} must have dims "
                f"({CLIMATOLOGY_DIM!r},)."
            )
        counts = np.asarray(ds[f"{name}_count"].values)
        if np.any(counts <= 0):
            raise ValueError(f"Climatology counts for {name!r} must be positive.")


def _validate_source_dataset(
    ds: xr.Dataset,
    *,
    variables: Sequence[str],
    time_dim: str,
) -> tuple[str, ...]:
    if not isinstance(ds, xr.Dataset):
        raise TypeError("source must be an xarray.Dataset.")
    if time_dim not in ds.coords:
        raise ValueError(f"source is missing time coordinate {time_dim!r}.")
    names = tuple(dict.fromkeys(str(name) for name in variables))
    if not names:
        raise ValueError("At least one climatology variable is required.")
    missing = sorted(name for name in names if name not in ds)
    if missing:
        raise ValueError("source is missing variables: " + ", ".join(missing))
    for name in names:
        if ds[name].dims != (time_dim,):
            raise ValueError(
                f"source variable {name!r} must have dims ({time_dim!r},); "
                f"got {ds[name].dims!r}."
            )
        if not np.issubdtype(ds[name].dtype, np.number):
            raise TypeError(f"source variable {name!r} must be numeric.")
    return names


def _validate_unique_year_keys(time: xr.DataArray, keys: xr.DataArray) -> None:
    index = pd.DatetimeIndex(time.values)
    pairs = pd.MultiIndex.from_arrays(
        [index.year, np.asarray(keys.values)],
        names=("year", CLIMATOLOGY_DIM),
    )
    if pairs.has_duplicates:
        raise ValueError(
            "Source contains more than one timestamp for a year and calendar-hour key."
        )
