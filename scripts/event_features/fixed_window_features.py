"""Shared fixed-window feature calculations for Stage-2 products."""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import xarray as xr

from scripts.event_features import event_feature_config as config


SURFACE_FLUX_FEATURES = frozenset({"I_sshf_pre", "I_slhf_pre"})


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
        {
            name
            for group in extended.values()
            for name in group
            if name not in ds
        }
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
    return tuple(name for name in config.WINDOWS if name in active)


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
        start_lag, end_lag = config.WINDOWS[window_name]
        starts = anchors + np.timedelta64(start_lag, "h")
        ends = anchors + np.timedelta64(end_lag, "h")
        keep &= (starts >= times[0]) & (ends <= times[-1])
    return keep


class WindowReducer:
    """Compute inclusive timestamp-window reductions with cached prefix arrays."""

    def __init__(self, ds: xr.Dataset, *, time_dim: str = config.TIME_DIM) -> None:
        if time_dim not in ds.coords:
            raise ValueError(f"Input dataset is missing time coordinate {time_dim!r}.")
        times = np.asarray(ds[time_dim].values, dtype="datetime64[ns]")
        if times.ndim != 1 or times.size == 0:
            raise ValueError("Input dataset time coordinate must be non-empty and 1D.")
        if np.isnat(times).any():
            raise ValueError("Input dataset time coordinate contains missing timestamps.")
        if np.any(times[1:] <= times[:-1]):
            raise ValueError("Input dataset time coordinate must be strictly increasing.")

        self.ds = ds
        self.time_dim = time_dim
        self.time_values = times
        self._prefix_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

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
        return self._bounds_for_lags(anchor_times, *config.WINDOWS[window_name])

    def _bounds_for_lags(
        self,
        anchor_times: np.ndarray,
        start_lag: int,
        end_lag: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        anchors = np.asarray(anchor_times, dtype="datetime64[ns]")
        starts = anchors + np.timedelta64(start_lag, "h")
        ends = anchors + np.timedelta64(end_lag, "h")
        left = np.searchsorted(self.time_values, starts, side="left")
        right = np.searchsorted(self.time_values, ends, side="right")
        return left, right

    def _reduce(
        self,
        source_name: str,
        left: np.ndarray,
        right: np.ndarray,
        *,
        operation: str,
    ) -> np.ndarray:
        prefix_sum, prefix_count = self._prefix_arrays(source_name)
        sums = prefix_sum[right] - prefix_sum[left]
        counts = prefix_count[right] - prefix_count[left]
        out = np.full(sums.shape, np.nan, dtype=float)
        valid = counts > 0
        if operation == "sum":
            out[valid] = sums[valid]
        elif operation == "mean":
            out[valid] = sums[valid] / counts[valid]
        else:
            raise ValueError(f"Unsupported reduction operation {operation!r}.")
        return out

    def _prefix_arrays(self, source_name: str) -> tuple[np.ndarray, np.ndarray]:
        if source_name not in self._prefix_cache:
            values = self._source_values(source_name)
            finite = np.isfinite(values)
            sums = np.where(finite, values, 0.0)
            self._prefix_cache[source_name] = (
                np.concatenate(([0.0], np.cumsum(sums, dtype=float))),
                np.concatenate(([0], np.cumsum(finite, dtype=np.int64))),
            )
        return self._prefix_cache[source_name]

    def _source_values(self, source_name: str) -> np.ndarray:
        if source_name not in self.ds:
            raise ValueError(f"Input dataset is missing source variable {source_name!r}.")
        source = self.ds[source_name]
        if source.dims != (self.time_dim,):
            raise ValueError(
                f"Source variable {source_name!r} must have dims "
                f"({self.time_dim!r},); got {source.dims!r}."
            )
        return np.asarray(source.compute().values, dtype=float)


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
        out[feature_name].attrs["change_method"] = (
            "final_24h_mean_minus_first_24h_mean"
        )


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
    start_lag, end_lag = config.WINDOWS[window_name]
    attrs = {
        "source_variable": source_variable,
        "window_name": window_name,
        "window_lag_hours": f"{start_lag},{end_lag}",
        "operation": operation,
        "window_endpoint_inclusion": "inclusive",
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
