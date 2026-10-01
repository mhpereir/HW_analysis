"""Hourly means and sample quartiles for native-calendar heatwave classes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

DEFAULT_SEASON_MONTHS: tuple[int, ...] = (6, 7, 8)
DEFAULT_LOCAL_UTC_OFFSET_HOURS = 0
DIURNAL_VARIABLES: tuple[str, ...] = (
    "T_mean",
    "volume",
    "dTdt",
    "advection",
    "adiabatic",
    "diabatic",
    "lwa_a_region",
    "lwa_c_region",
)
DIURNAL_QUANTILES: tuple[float, ...] = (0.25, 0.5, 0.75)
HW_CLASS_DIM = "hw_class"
HW_CLASS_LABELS: tuple[str, ...] = ("Heatwave days", "Non-heatwave days")
LOCAL_HOURS = np.arange(24, dtype=np.int64)
SAMPLE_PERCENTILE_PREFIX = "sample_percentile_"


def build_diurnal_composite(
    ds: xr.Dataset,
    *,
    season_months: Sequence[int] = DEFAULT_SEASON_MONTHS,
    local_utc_offset_hours: int = DEFAULT_LOCAL_UTC_OFFSET_HOURS,
    variables: Sequence[str] = DIURNAL_VARIABLES,
    quantiles: Sequence[float] = DIURNAL_QUANTILES,
    time_dim: str = "time",
    hw_event_id_name: str = "hw_event_id",
) -> xr.Dataset:
    """Group native-month samples by stored HW class and displayed hour."""
    months = _validate_season_months(season_months)
    offset = _validate_local_utc_offset_hours(local_utc_offset_hours)
    qs = _validate_quantiles(quantiles)
    _validate_diurnal_inputs(
        ds,
        variables=variables,
        time_dim=time_dim,
        hw_event_id_name=hw_event_id_name,
    )

    local_times = utc_to_local_time_values(ds[time_dim], offset)
    local_index = pd.DatetimeIndex(local_times)
    local_hour = xr.DataArray(
        np.asarray(local_index.hour, dtype=np.int64),
        dims=(time_dim,),
        coords={time_dim: ds[time_dim]},
        name="local_hour",
    )
    season_mask = ds[time_dim].dt.month.isin(months)
    hw_event_id = ds[hw_event_id_name].fillna(0)
    class_masks = (
        season_mask & (hw_event_id > 0),
        season_mask & (hw_event_id == 0),
    )

    source = ds[list(variables)].assign_coords(
        local_hour=local_hour,
        local_time=(time_dim, local_times),
    )

    class_composites: list[xr.Dataset] = []
    class_sample_counts: list[int] = []
    class_day_counts: list[int] = []
    for label, mask in zip(HW_CLASS_LABELS, class_masks, strict=True):
        selected = source.where(mask, drop=True)
        sample_count = int(selected.sizes.get(time_dim, 0))
        if sample_count == 0:
            raise ValueError(
                f"No {label.lower()} samples remain after native-month filtering."
            )
        class_composites.append(
            _build_one_class_diurnal_composite(
                selected,
                variables=variables,
                quantiles=qs,
                time_dim=time_dim,
            )
        )
        class_sample_counts.append(sample_count)
        class_day_counts.append(
            int(np.unique(selected[time_dim].values.astype("datetime64[D]")).size)
        )

    composite = xr.concat(
        class_composites,
        dim=xr.IndexVariable(HW_CLASS_DIM, list(HW_CLASS_LABELS)),
    )
    composite = composite.assign_coords(
        class_sample_count=(
            HW_CLASS_DIM,
            np.asarray(class_sample_counts, dtype=np.int64),
        ),
        class_day_count=(HW_CLASS_DIM, np.asarray(class_day_counts, dtype=np.int64)),
    )
    composite.attrs.update(
        {
            "composite_reduction": "mean by local hour and heatwave class",
            "season_months": " ".join(str(month) for month in months),
            "season_time_basis": "native Stage-1 timestamps before local-hour grouping",
            "hw_classification": f"stored {hw_event_id_name}; missing IDs treated as zero",
            "day_count_basis": "distinct contributing native GMT dates, including partial days",
            "local_utc_offset_hours": int(offset),
            "local_timezone_label": _utc_offset_label(offset),
            "sample_percentiles": ", ".join(str(float(q)) for q in qs),
            "sample_percentile_prefix": SAMPLE_PERCENTILE_PREFIX,
            "n_hw_samples": int(class_sample_counts[0]),
            "n_non_hw_samples": int(class_sample_counts[1]),
            "n_hw_days": int(class_day_counts[0]),
            "n_non_hw_days": int(class_day_counts[1]),
        }
    )
    for attr_name in (
        "region",
        "threshold_variable",
        "quantile",
        "start_year",
        "end_year",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "regional_reference_path",
        "regional_reference_sha256",
    ):
        if attr_name in ds.attrs:
            composite.attrs[attr_name] = ds.attrs[attr_name]
    composite["local_hour"].attrs["long_name"] = (
        "GMT hour (UTC+0)"
        if offset == 0
        else f"Hour at fixed {_utc_offset_label(offset)}"
    )
    return composite


def utc_to_local_time_values(
    time: xr.DataArray,
    local_utc_offset_hours: int = DEFAULT_LOCAL_UTC_OFFSET_HOURS,
) -> np.ndarray:
    """Return timestamp values shifted from UTC/GMT to fixed local time."""
    offset = _validate_local_utc_offset_hours(local_utc_offset_hours)
    values = np.asarray(time.values, dtype="datetime64[ns]")
    return values + np.timedelta64(offset, "h")


def _build_one_class_diurnal_composite(
    selected: xr.Dataset,
    *,
    variables: Sequence[str],
    quantiles: tuple[float, ...],
    time_dim: str,
) -> xr.Dataset:
    """Build one class's local-hour mean and percentile traces."""
    data_vars: dict[str, xr.DataArray] = {}
    data_vars["class_hour_sample_count"] = (
        xr.ones_like(selected[time_dim], dtype=np.int64)
        .groupby("local_hour")
        .sum(time_dim)
        .reindex(local_hour=LOCAL_HOURS, fill_value=0)
    )
    for name in variables:
        grouped = selected[name].groupby("local_hour")
        data_vars[name] = grouped.mean(time_dim, skipna=True).reindex(
            local_hour=LOCAL_HOURS
        )
        data_vars[f"{SAMPLE_PERCENTILE_PREFIX}{name}"] = (
            grouped.quantile(quantiles, dim=time_dim, skipna=True)
            .reindex(local_hour=LOCAL_HOURS)
            .transpose("quantile", "local_hour")
        )
        data_vars[f"sample_count_{name}"] = grouped.count(time_dim).reindex(
            local_hour=LOCAL_HOURS, fill_value=0
        )
        data_vars[f"sample_count_{name}"].attrs = {
            "long_name": f"Nonmissing hourly samples contributing to {name}",
            "units": "1",
        }
        if "units" in selected[name].attrs:
            for key in (name, f"{SAMPLE_PERCENTILE_PREFIX}{name}"):
                data_vars[key].attrs["units"] = selected[name].attrs["units"]
    return xr.Dataset(
        data_vars=data_vars,
        coords={
            "local_hour": LOCAL_HOURS,
            "quantile": np.asarray(quantiles, dtype=float),
        },
    )


def _validate_diurnal_inputs(
    ds: xr.Dataset,
    *,
    variables: Sequence[str],
    time_dim: str,
    hw_event_id_name: str,
) -> None:
    """Validate required dataset variables and dimensions."""
    if time_dim not in ds.coords:
        raise ValueError(f"Dataset is missing required time coordinate {time_dim!r}.")
    if hw_event_id_name not in ds:
        raise ValueError(f"Dataset is missing event-ID variable {hw_event_id_name!r}.")
    missing = sorted(name for name in variables if name not in ds)
    if missing:
        raise ValueError(
            f"Dataset is missing requested variables: {', '.join(missing)}."
        )
    invalid = sorted(name for name in variables if time_dim not in ds[name].dims)
    if invalid:
        raise ValueError(
            "Requested variables must contain the time dimension; invalid: "
            f"{', '.join(invalid)}."
        )
    if time_dim not in ds[hw_event_id_name].dims:
        raise ValueError(f"{hw_event_id_name!r} must contain dimension {time_dim!r}.")


def _validate_season_months(season_months: Sequence[int]) -> tuple[int, ...]:
    """Return validated month numbers with duplicates removed in input order."""
    if isinstance(season_months, (str, bytes)):
        raise TypeError("season_months must be a sequence of integer month numbers.")

    months: list[int] = []
    for month in season_months:
        if isinstance(month, (bool, np.bool_)) or not isinstance(
            month, (int, np.integer)
        ):
            raise TypeError("season_months must contain only integer month numbers.")
        month_int = int(month)
        if month_int < 1 or month_int > 12:
            raise ValueError("--season-months values must be between 1 and 12.")
        if month_int not in months:
            months.append(month_int)
    if not months:
        raise ValueError("season_months must contain at least one month.")
    return tuple(months)


def _validate_local_utc_offset_hours(value: int) -> int:
    """Return a validated fixed UTC offset in whole hours."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError("--local-utc-offset-hours must be an integer.")
    offset = int(value)
    if offset < -23 or offset > 23:
        raise ValueError("--local-utc-offset-hours must be between -23 and 23.")
    return offset


def _validate_quantiles(quantiles: Sequence[float]) -> tuple[float, ...]:
    """Return sorted unique quantile values in [0, 1]."""
    values = tuple(sorted({float(value) for value in quantiles}))
    if not values:
        raise ValueError("quantiles must contain at least one value.")
    invalid = [value for value in values if value < 0.0 or value > 1.0]
    if invalid:
        text = ", ".join(str(value) for value in invalid)
        raise ValueError(f"quantiles must be between 0 and 1; got {text}.")
    return values


def _utc_offset_label(offset: int) -> str:
    """Return display label for a fixed UTC offset."""
    if offset == 0:
        return "UTC+0"
    sign = "+" if offset > 0 else "-"
    return f"UTC{sign}{abs(offset)}"
