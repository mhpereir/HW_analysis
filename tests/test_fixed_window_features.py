"""Tests for shared Stage-2 fixed-window reductions."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from scripts.event_features import event_feature_config as config
from scripts.event_features.fixed_window_features import WindowReducer


@pytest.fixture(params=(48, 96))
def integration_hours(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    """Configure each required integration span for one regression test."""
    hours = int(request.param)
    monkeypatch.setitem(config.WINDOWS, "heat_budget_pre", (-hours, 0))
    return hours


def test_reductions_do_not_lose_short_window_after_large_history(
    integration_hours: int,
) -> None:
    """Local reductions must not subtract rounded full-record prefix totals."""
    history_size = 10_000
    window_size = integration_hours + 1
    values = np.concatenate(
        (np.full(history_size, 1.0e12), np.ones(window_size)),
    )
    times = np.datetime64("2000-01-01T00") + np.arange(values.size).astype(
        "timedelta64[h]"
    )
    ds = xr.Dataset({"lwa": ("time", values)}, coords={"time": times})
    anchor_times = np.asarray([times[-1]], dtype="datetime64[ns]")

    reducer = WindowReducer(ds)

    np.testing.assert_array_equal(
        reducer.sums("lwa", anchor_times, "heat_budget_pre"),
        [float(window_size)],
    )
    np.testing.assert_array_equal(
        reducer.means("lwa", anchor_times, "heat_budget_pre"),
        [1.0],
    )


def test_reductions_match_direct_finite_sample_operations(
    integration_hours: int,
) -> None:
    """Sums and means should match direct operations for every selected window."""
    history_size = 10_000
    window_size = integration_hours + 1
    local_values = np.resize(
        np.asarray([1.0e12, -1.0e12, 3.0, np.nan, -2.0]),
        window_size,
    )
    values = np.concatenate(
        (np.full(history_size, 1.0e12), local_values),
    )
    times = np.datetime64("2000-01-01T00") + np.arange(values.size).astype(
        "timedelta64[h]"
    )
    ds = xr.Dataset({"lwa": ("time", values)}, coords={"time": times})
    anchor_times = np.asarray([times[-1]], dtype="datetime64[ns]")
    finite = local_values[np.isfinite(local_values)]

    reducer = WindowReducer(ds)

    np.testing.assert_array_equal(
        reducer.sums("lwa", anchor_times, "heat_budget_pre"),
        [np.sum(finite, dtype=np.float64)],
    )
    np.testing.assert_array_equal(
        reducer.means("lwa", anchor_times, "heat_budget_pre"),
        [np.mean(finite, dtype=np.float64)],
    )


def test_reductions_return_nan_for_windows_without_finite_values(
    integration_hours: int,
) -> None:
    """All-nonfinite windows retain the established missing-result behavior."""
    history_size = 10_000
    window_size = integration_hours + 1
    values = np.concatenate(
        (np.full(history_size, 1.0e12), np.full(window_size, np.nan)),
    )
    times = np.datetime64("2000-01-01T00") + np.arange(values.size).astype(
        "timedelta64[h]"
    )
    ds = xr.Dataset({"lwa": ("time", values)}, coords={"time": times})
    anchor_times = np.asarray([times[-1]], dtype="datetime64[ns]")

    reducer = WindowReducer(ds)

    assert np.isnan(reducer.sums("lwa", anchor_times, "heat_budget_pre")).all()
    assert np.isnan(reducer.means("lwa", anchor_times, "heat_budget_pre")).all()
