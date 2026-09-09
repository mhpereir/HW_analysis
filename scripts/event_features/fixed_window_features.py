"""Shared fixed-window feature calculations for Stage-2 products."""

from __future__ import annotations

import json
import warnings
from collections.abc import Callable, Mapping, Sequence
from numbers import Integral
from pathlib import Path

import numpy as np
import xarray as xr

from scripts.event_features import event_feature_config as config
from src.climatology import (
    SOURCE_COMPATIBILITY_ATTRS,
    apply_regional_hourly_climatology,
)

SURFACE_FLUX_FEATURES = frozenset({"I_sshf_pre", "I_slhf_pre"})
ANTECEDENT_WINDOW = "antecedent_temperature"
TEMPERATURE_FEATURES = {
    "tas_anom_antecedent_mean": ("tas_anom", ANTECEDENT_WINDOW),
    "tas_anom_at_budget_start": ("tas_anom", "budget_start"),
    "T_mean_anom_antecedent_mean": ("T_mean_anom", ANTECEDENT_WINDOW),
    "T_mean_anom_at_budget_start": ("T_mean_anom", "budget_start"),
    "tas_anom_at_anchor": ("tas_anom", "anchor"),
}


def temperature_window_lags() -> tuple[int, int]:
    """Resolve the half-open interval from the current budget configuration."""
    start, end = config.WINDOWS["heat_budget_pre"]
    duration = config.ANTECEDENT_TEMPERATURE_DURATION_HOURS
    if any(
        isinstance(value, bool) or not isinstance(value, Integral)
        for value in (start, end, duration)
    ):
        raise ValueError("Budget lags and antecedent duration must be integer hours.")
    if start >= 0 or end != 0:
        raise ValueError(
            "heat_budget_pre must start before the anchor and end at zero."
        )
    if duration <= 0:
        raise ValueError("Antecedent temperature duration must be positive.")
    return int(start - duration), int(start)


def window_lags(name: str) -> tuple[int, int]:
    """Resolve legacy windows and the new derived temperature windows."""
    if name == ANTECEDENT_WINDOW:
        return temperature_window_lags()
    if name == "budget_start":
        lag = temperature_window_lags()[1]
        return lag, lag
    if name == "anchor":
        return 0, 0
    return config.WINDOWS[name]


def window_endpoint_inclusion(name: str) -> str:
    if name == ANTECEDENT_WINDOW:
        return "left_closed_right_open"
    if name in {"budget_start", "anchor"}:
        return "exact_timestamp"
    return "inclusive"


def prepare_temperature_sources(ds: xr.Dataset, climatology: xr.Dataset) -> xr.Dataset:
    """Add anomaly sources without replacing any absolute budget variables."""
    if ds.attrs.get("data_representation") == "climatological_anomaly":
        raise ValueError("Stage-2 builders require absolute Stage-1 input.")
    for source_attr, climate_attr in SOURCE_COMPATIBILITY_ATTRS:
        if ds.attrs.get(source_attr) in (None, ""):
            raise ValueError(f"Stage-1 source is missing {source_attr} metadata.")
        if climatology.attrs.get(climate_attr) in (None, ""):
            raise ValueError(f"Climatology is missing {climate_attr} metadata.")
    out = ensure_tas_anom(ds).copy(deep=False)
    if "tas_anom" not in out:
        raise ValueError(
            "Stage-1 source requires tas_anom or tas_region and tas_climatology."
        )
    anomaly_view = apply_regional_hourly_climatology(
        ds,
        climatology,
        variables=["T_mean"],
    )
    out["T_mean_anom"] = anomaly_view["T_mean"].reset_coords(drop=True)
    return out


def ensure_tas_anom(ds: xr.Dataset) -> xr.Dataset:
    """Return a view with tas_anom available for feature extraction."""
    if "tas_anom" in ds:
        return ds
    if "tas_region" not in ds or "tas_climatology" not in ds:
        return ds
    out = ds.copy(deep=False)
    out["tas_anom"] = out["tas_region"] - out["tas_climatology"]
    out["tas_anom"].attrs.update(
        {
            "description": "Derived tas_region minus tas_climatology.",
            "source_variables": "tas_region,tas_climatology",
        }
    )
    if "units" in out["tas_region"].attrs:
        out["tas_anom"].attrs["units"] = out["tas_region"].attrs["units"]
    return out


def active_feature_spec(
    ds: xr.Dataset,
    *,
    use_extended_variables: bool,
    allow_missing_extended: bool,
) -> dict[str, dict[str, str]]:
    """Return active source-variable to window-name feature mappings."""
    spec = {
        "integral": dict(config.DEFAULT_INTEGRAL_FEATURES),
        "mean": dict(config.DEFAULT_MEAN_FEATURES),
        "change": {},
    }
    if not use_extended_variables:
        return spec

    extended = {
        "integral": dict(config.EXTENDED_INTEGRAL_FEATURES),
        "mean": dict(config.EXTENDED_MEAN_FEATURES),
        "change": dict(config.EXTENDED_CHANGE_FEATURES),
    }
    missing = sorted(
        {name for group in extended.values() for name in group if name not in ds}
    )
    if missing and not allow_missing_extended:
        raise ValueError(
            "Input dataset is missing required extended variables: "
            f"{', '.join(missing)}."
        )
    if missing:
        warnings.warn(
            "Skipping missing extended variables: " + ", ".join(missing),
            RuntimeWarning,
            stacklevel=2,
        )

    for operation, mapping in extended.items():
        for name, window_name in mapping.items():
            if name in ds:
                spec[operation][name] = window_name
    return spec


def validate_required_time_variables(
    ds: xr.Dataset,
    feature_spec: Mapping[str, Mapping[str, str]],
) -> None:
    """Fail clearly when required time-indexed feature sources are absent."""
    missing = sorted(
        name
        for operation in ("integral", "mean", "change")
        for name in feature_spec[operation]
        if name not in ds
    )
    if missing:
        raise ValueError(
            "Input dataset is missing required time-indexed variables: "
            f"{', '.join(missing)}."
        )


def active_window_names(
    feature_spec: Mapping[str, Mapping[str, str]],
) -> tuple[str, ...]:
    """Return active window names in configuration order."""
    active = {
        window_name
        for mapping in feature_spec.values()
        for window_name in mapping.values()
    }
    return (
        *tuple(name for name in config.WINDOWS if name in active),
        ANTECEDENT_WINDOW,
    )


def complete_anchor_mask(
    time_values: np.ndarray,
    anchor_times: np.ndarray,
    window_names: Sequence[str],
) -> np.ndarray:
    """Return anchors whose active windows stay within the dataset time range."""
    times = np.asarray(time_values, dtype="datetime64[ns]")
    anchors = np.asarray(anchor_times, dtype="datetime64[ns]")
    if times.size == 0:
        raise ValueError("Input dataset has an empty time coordinate.")

    keep = np.ones(anchors.shape, dtype=bool)
    for window_name in window_names:
        start_lag, end_lag = window_lags(window_name)
        starts = anchors + np.timedelta64(start_lag, "h")
        ends = anchors + np.timedelta64(end_lag, "h")
        keep &= (starts >= times[0]) & (ends <= times[-1])
    return keep


class WindowReducer:
    """Compute inclusive timestamp-window reductions from cached source values."""

    def __init__(self, ds: xr.Dataset, *, time_dim: str = config.TIME_DIM) -> None:
        if time_dim not in ds.coords:
            raise ValueError(f"Input dataset is missing time coordinate {time_dim!r}.")
        times = np.asarray(ds[time_dim].values, dtype="datetime64[ns]")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("Input dataset time coordinate must be non-empty and 1D.")
        if np.isnat(times).any():
            raise ValueError(
                "Input dataset time coordinate contains missing timestamps."
            )
        if np.any(times[1:] <= times[:-1]):
            raise ValueError(
                "Input dataset time coordinate must be strictly increasing."
            )

        self.ds = ds
        self.time_dim = time_dim
        self.time_values = times
        self._source_cache: dict[str, np.ndarray] = {}

    def complete_anchor_mask(
        self,
        anchor_times: np.ndarray,
        window_names: Sequence[str],
    ) -> np.ndarray:
        return complete_anchor_mask(self.time_values, anchor_times, window_names)

    def sample_counts(self, anchor_times: np.ndarray, window_name: str) -> np.ndarray:
        left, right = self._bounds_for_window(anchor_times, window_name)
        return (right - left).astype(np.int64)

    def sums(
        self,
        source_name: str,
        anchor_times: np.ndarray,
        window_name: str,
    ) -> np.ndarray:
        left, right = self._bounds_for_window(anchor_times, window_name)
        return self._reduce(source_name, left, right, operation="sum")

    def means(
        self,
        source_name: str,
        anchor_times: np.ndarray,
        window_name: str,
    ) -> np.ndarray:
        left, right = self._bounds_for_window(anchor_times, window_name)
        return self._reduce(source_name, left, right, operation="mean")

    def changes(
        self,
        source_name: str,
        anchor_times: np.ndarray,
        window_name: str,
    ) -> np.ndarray:
        """Return final inclusive 24-hour mean minus first inclusive 24-hour mean."""
        start_lag, end_lag = config.WINDOWS[window_name]
        first_left, first_right = self._bounds_for_lags(
            anchor_times,
            start_lag,
            start_lag + 24,
        )
        final_left, final_right = self._bounds_for_lags(
            anchor_times,
            end_lag - 24,
            end_lag,
        )
        first = self._reduce(source_name, first_left, first_right, operation="mean")
        final = self._reduce(source_name, final_left, final_right, operation="mean")
        return final - first

    def any_nonzero(
        self,
        source_name: str,
        anchor_times: np.ndarray,
        window_names: Sequence[str],
    ) -> np.ndarray:
        """Return whether any nonzero source value occurs in any active window."""
        values = self._source_values(source_name)
        flags = np.asarray(values != 0, dtype=np.int64)
        prefix = np.concatenate(([0], np.cumsum(flags, dtype=np.int64)))
        found = np.zeros(np.asarray(anchor_times).shape, dtype=bool)
        for window_name in window_names:
            left, right = self._bounds_for_window(anchor_times, window_name)
            found |= (prefix[right] - prefix[left]) > 0
        return found

    def _bounds_for_window(
        self,
        anchor_times: np.ndarray,
        window_name: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._bounds_for_lags(
            anchor_times,
            *window_lags(window_name),
            end_inclusive=window_name != ANTECEDENT_WINDOW,
        )

    def _bounds_for_lags(
        self,
        anchor_times: np.ndarray,
        start_lag: int,
        end_lag: int,
        *,
        end_inclusive: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        anchors = np.asarray(anchor_times, dtype="datetime64[ns]")
        starts = anchors + np.timedelta64(start_lag, "h")
        ends = anchors + np.timedelta64(end_lag, "h")
        left = np.searchsorted(self.time_values, starts, side="left")
        right = np.searchsorted(
            self.time_values,
            ends,
            side="right" if end_inclusive else "left",
        )
        return left, right

    def strict_temperature_values(
        self,
        source_name: str,
        anchor_times: np.ndarray,
        window_name: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return strict hourly mean/exact point, timestamp count, finite count."""
        start, end = window_lags(window_name)
        lags = np.arange(start, end if window_name == ANTECEDENT_WINDOW else end + 1)
        left, right = self._bounds_for_window(anchor_times, window_name)
        source = self._source_values(source_name)
        values = np.full(left.shape, np.nan, dtype=float)
        finite_counts = np.zeros(left.shape, dtype=np.int64)
        anchors = np.asarray(anchor_times, dtype="datetime64[ns]")
        for i, (lo, hi) in enumerate(zip(left.flat, right.flat, strict=True)):
            selected = source[lo:hi]
            finite_counts.flat[i] = np.isfinite(selected).sum()
            expected = anchors.flat[i] + lags.astype("timedelta64[h]")
            if (
                np.array_equal(self.time_values[lo:hi], expected)
                and finite_counts.flat[i] == lags.size
            ):
                values.flat[i] = np.mean(selected, dtype=np.float64)
        return values, (right - left).astype(np.int64), finite_counts

    def _reduce(
        self,
        source_name: str,
        left: np.ndarray,
        right: np.ndarray,
        *,
        operation: str,
    ) -> np.ndarray:
        if operation not in {"sum", "mean"}:
            raise ValueError(f"Unsupported reduction operation {operation!r}.")

        values = self._source_values(source_name)
        out = np.full(left.shape, np.nan, dtype=float)
        bounds = zip(left.flat, right.flat, strict=True)
        for index, (start, stop) in enumerate(bounds):
            # Full-record float prefix differences lose precision after long,
            # high-magnitude histories, so reduce each short window directly.
            window = values[start:stop]
            finite_values = window[np.isfinite(window)]
            if finite_values.size == 0:
                continue
            if operation == "sum":
                out.flat[index] = np.sum(finite_values, dtype=np.float64)
            else:
                out.flat[index] = np.mean(finite_values, dtype=np.float64)
        return out

    def _source_values(self, source_name: str) -> np.ndarray:
        if source_name not in self._source_cache:
            if source_name not in self.ds:
                raise ValueError(
                    f"Input dataset is missing source variable {source_name!r}."
                )
            source = self.ds[source_name]
            if source.dims != (self.time_dim,):
                raise ValueError(
                    f"Source variable {source_name!r} must have dims "
                    f"({self.time_dim!r},); got {source.dims!r}."
                )
            self._source_cache[source_name] = np.asarray(
                source.compute().values,
                dtype=float,
            )
        return self._source_cache[source_name]


def add_window_features(
    out: xr.Dataset,
    ds: xr.Dataset,
    reducer: WindowReducer,
    anchor_times: np.ndarray,
    *,
    row_dim: str,
    feature_spec: Mapping[str, Mapping[str, str]],
    feature_name_for_source: Callable[[str, str], str],
    sample_count_name_for_window: Callable[[str], str] | None = None,
) -> None:
    """Add configured sample counts and reductions to a feature table."""
    for window_name in active_window_names(feature_spec):
        feature_name = (
            f"n_samples_{window_name}"
            if sample_count_name_for_window is None
            else sample_count_name_for_window(window_name)
        )
        out[feature_name] = (
            row_dim,
            reducer.sample_counts(anchor_times, window_name),
        )
        add_feature_attrs(
            out[feature_name],
            source_variable=config.TIME_DIM,
            window_name=window_name,
            operation="count",
            units="samples",
        )

    for source_name, window_name in feature_spec["integral"].items():
        feature_name = feature_name_for_source(source_name, "integral")
        out[feature_name] = (
            row_dim,
            reducer.sums(source_name, anchor_times, window_name),
        )
        source_units = ds[source_name].attrs.get("units")
        if source_units == "K hr-1":
            out[feature_name].attrs["units"] = "K"
        elif source_units is not None:
            out[feature_name].attrs["units"] = f"{source_units} hr"
        add_feature_attrs(
            out[feature_name],
            source_variable=source_name,
            window_name=window_name,
            operation="sum",
        )
        out[feature_name].attrs["integral_method"] = config.INTEGRAL_METHOD
        if source_name in {"lwa_a_region", "lwa_c_region"}:
            out[feature_name].attrs["description"] = "LWA exposure over fixed window."
        if feature_name in SURFACE_FLUX_FEATURES:
            out[feature_name].attrs["sign_convention"] = (
                "native Stage-1/source signs retained"
            )

    for source_name, window_name in feature_spec["mean"].items():
        feature_name = feature_name_for_source(source_name, "mean")
        out[feature_name] = (
            row_dim,
            reducer.means(source_name, anchor_times, window_name),
        )
        if "units" in ds[source_name].attrs:
            out[feature_name].attrs["units"] = ds[source_name].attrs["units"]
        add_feature_attrs(
            out[feature_name],
            source_variable=source_name,
            window_name=window_name,
            operation="mean",
        )

    for source_name, window_name in feature_spec["change"].items():
        feature_name = feature_name_for_source(source_name, "change")
        out[feature_name] = (
            row_dim,
            reducer.changes(source_name, anchor_times, window_name),
        )
        if "units" in ds[source_name].attrs:
            out[feature_name].attrs["units"] = ds[source_name].attrs["units"]
        add_feature_attrs(
            out[feature_name],
            source_variable=source_name,
            window_name=window_name,
            operation="change",
        )
        out[feature_name].attrs["change_method"] = "final_24h_mean_minus_first_24h_mean"


def add_temperature_features(
    out: xr.Dataset,
    reducer: WindowReducer,
    anchor_times: np.ndarray,
    *,
    row_dim: str,
    anchor_variable: str,
    climatology: xr.Dataset,
    climatology_path: str | Path | None,
) -> None:
    """Write the shared strict temperature features and their provenance."""
    start, end = temperature_window_lags()
    for name, (source, window) in TEMPERATURE_FEATURES.items():
        values, counts, finite = reducer.strict_temperature_values(
            source, anchor_times, window
        )
        out[name] = (row_dim, values)
        add_feature_attrs(
            out[name],
            source_variable=source,
            window_name=window,
            operation="mean" if window == ANTECEDENT_WINDOW else "sample",
            units="K",
        )
        expected = end - start if window == ANTECEDENT_WINDOW else 1
        out[name].attrs.update(
            {
                "anchor_variable": anchor_variable,
                "expected_sample_count": expected,
                "missing_value_policy": "require complete hourly timestamps and all finite values",
                "climatology_source": (
                    "Stage-1 tas_anom or tas_region minus tas_climatology"
                    if source == "tas_anom"
                    else "regional_hourly_climatology:T_mean"
                ),
            }
        )
        if source == "T_mean_anom":
            out[name].attrs.update(
                {
                    "climatology_path": ""
                    if climatology_path is None
                    else str(climatology_path),
                    "climatology_matching": climatology.attrs["climatology_matching"],
                    "climatology_start_year": climatology.attrs[
                        "climatology_start_year"
                    ],
                    "climatology_end_year": climatology.attrs["climatology_end_year"],
                }
            )
        count_name = f"n_samples_{window}"
        out[count_name] = (row_dim, counts)
        add_feature_attrs(
            out[count_name],
            source_variable=config.TIME_DIM,
            window_name=window,
            operation="count",
            units="samples",
        )
        out[count_name].attrs["expected_sample_count"] = expected
        finite_name = f"n_finite_{name}"
        out[finite_name] = (row_dim, finite)
        add_feature_attrs(
            out[finite_name],
            source_variable=source,
            window_name=window,
            operation="finite_count",
            units="samples",
        )
        out[finite_name].attrs["expected_sample_count"] = expected
    out.attrs.update(
        {
            "antecedent_temperature_contract_version": 1,
            "temperature_anchor_variable": anchor_variable,
            "antecedent_temperature_window_hours": f"{start},{end}",
            "antecedent_temperature_duration_hours": end - start,
            "budget_start_lag_hours": end,
            "temperature_point_sampling": "exact_timestamp",
            "antecedent_temperature_endpoint_inclusion": "left_closed_right_open",
            "temperature_climatology_path": ""
            if climatology_path is None
            else str(climatology_path),
            "temperature_climatology_metadata": json.dumps(
                dict(climatology.attrs), default=str, sort_keys=True
            ),
            "tas_anomaly_source_metadata": json.dumps(
                dict(reducer.ds["tas_anom"].attrs), default=str, sort_keys=True
            ),
        }
    )
    for key in (
        "region",
        "heat_budget_bottom_boundary",
        "heat_budget_top_boundary",
        "stage1_contract_version",
        "threshold_variable",
        "quantile_threshold",
    ):
        if key in reducer.ds.attrs:
            out.attrs[key] = reducer.ds.attrs[key]


def add_integrated_dynamical_feature(out: xr.Dataset, *, row_dim: str) -> None:
    """Add canonical integrated dynamical heating from its Stage-2 components."""
    feature_name = config.DYNAMICAL_FEATURE_NAME
    component_names = config.DYNAMICAL_COMPONENT_FEATURES
    if feature_name in out:
        raise ValueError(f"Output already contains derived feature {feature_name!r}.")

    missing = [name for name in component_names if name not in out]
    if missing:
        raise ValueError(
            "Cannot derive I_dyn_pre; output is missing component features: "
            + ", ".join(missing)
            + "."
        )
    for name in component_names:
        if out[name].dims != (row_dim,):
            raise ValueError(
                f"Component feature {name!r} must have dims ({row_dim!r},); "
                f"got {out[name].dims!r}."
            )

    first, second = component_names
    out[feature_name] = out[first] + out[second]
    attrs = {
        "description": (
            "Integrated pre-peak dynamical tendency: adiabatic plus advection."
        ),
        "source_variables": ",".join(component_names),
        "formula": f"{first} + {second}",
        "operation": "sum",
        "window_name": "heat_budget_pre",
        "window_lag_hours": ",".join(
            str(value) for value in config.WINDOWS["heat_budget_pre"]
        ),
        "window_endpoint_inclusion": "inclusive",
        "integral_method": config.INTEGRAL_METHOD,
    }
    component_units = tuple(out[name].attrs.get("units") for name in component_names)
    if component_units[0] != component_units[1]:
        raise ValueError(
            "Cannot derive I_dyn_pre from component features with different units: "
            f"{first}={component_units[0]!r}, {second}={component_units[1]!r}."
        )
    units = component_units[0]
    if isinstance(units, str) and units:
        attrs["units"] = units
    out[feature_name].attrs = attrs


def add_days_from_solstice(
    out: xr.Dataset,
    anchor_times: np.ndarray,
    *,
    row_dim: str,
    source_variable: str,
) -> None:
    """Add calendar-day distance from June 21 for each anchor timestamp."""
    anchor_days = np.asarray(anchor_times, dtype="datetime64[D]")
    year_numbers = anchor_days.astype("datetime64[Y]").astype(np.int64) + 1970
    solstices = np.asarray(
        [
            np.datetime64(
                f"{year}-{config.SOLSTICE_MONTH:02d}-{config.SOLSTICE_DAY:02d}",
                "D",
            )
            for year in year_numbers
        ]
    )
    values = (anchor_days - solstices) / np.timedelta64(1, "D")
    out["days_from_solstice"] = (row_dim, np.asarray(values, dtype=float))
    out["days_from_solstice"].attrs.update(
        {
            "source_variable": source_variable,
            "operation": "calendar_day_difference",
            "reference_date": f"{config.SOLSTICE_MONTH:02d}-{config.SOLSTICE_DAY:02d}",
            "units": "days",
        }
    )


def add_feature_attrs(
    da: xr.DataArray,
    *,
    source_variable: str,
    window_name: str,
    operation: str,
    units: str | None = None,
) -> None:
    """Add common fixed-window feature metadata."""
    start_lag, end_lag = window_lags(window_name)
    attrs = {
        "source_variable": source_variable,
        "window_name": window_name,
        "window_lag_hours": f"{start_lag},{end_lag}",
        "operation": operation,
        "window_endpoint_inclusion": window_endpoint_inclusion(window_name),
    }
    if units is not None:
        attrs["units"] = units
    da.attrs.update(attrs)


def nansum_or_nan(values: np.ndarray) -> float:
    """Return nansum or NaN for empty/all-NaN values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nansum(arr))


def nanmean_or_nan(values: np.ndarray) -> float:
    """Return nanmean or NaN for empty/all-NaN values."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return np.nan
    return float(np.nanmean(arr))
