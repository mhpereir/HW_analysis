"""Numerical safeguards for the independent production audit."""

import numpy as np
import pytest
import xarray as xr
from HW_analysis.scripts.validate_advection_face_plots import (
    source_windows,
    verify_arrays,
)


def test_calendar_subtraction_uses_leap_day_and_exact_peak_hours():
    time = np.arange(
        np.datetime64("2020-02-20T00"),
        np.datetime64("2020-03-10T00"),
        np.timedelta64(1, "h"),
    )
    source = xr.Dataset(
        {
            "advection": ("time", np.arange(len(time), dtype=float)),
            "event_id": ("event", [7]),
            "peak_time": ("event", [np.datetime64("2020-02-29T12")]),
        },
        coords={"time": time},
    )
    dates = np.arange(
        np.datetime64("2000-02-20T00"),
        np.datetime64("2000-03-10T00"),
        np.timedelta64(1, "h"),
    )
    clock = xr.DataArray(dates, dims="calendar_hour")
    climate = xr.Dataset(
        {
            "advection": ("calendar_hour", np.full(len(dates), 3.0)),
            "month": clock.dt.month.astype("int16"),
            "day": clock.dt.day.astype("int16"),
            "hour_utc": clock.dt.hour.astype("int16"),
        }
    )
    result = source_windows(source, climate, [7], ["advection"], anomalies=True)
    np.testing.assert_array_equal(
        result["advection"], np.arange(60, 397)[None, :] - 3.0
    )
    with pytest.raises(AssertionError, match="Missing source hours"):
        source_windows(
            source.isel(time=slice(100, None)),
            climate,
            [7],
            ["advection"],
            anomalies=True,
        )


def test_audit_detects_broken_face_closure_and_checks_smoothing(tmp_path):
    values = np.sin(np.arange(337) / 8.0)
    variables = {
        f"advection_{face}": values * coefficient
        for face, coefficient in (
            ("west", 2.0),
            ("east", -1.0),
            ("north", 0.5),
            ("south", -0.2),
            ("top", 0.1),
            ("bottom", -0.1),
        )
    }
    variables["advection"] = sum(variables.values())
    actual = xr.Dataset(
        {name: ("lag_hour", value) for name, value in variables.items()},
        coords={"lag_hour": np.arange(-168, 169)},
    )
    verify_arrays(actual, variables, tmp_path / "valid")
    broken = actual.copy(deep=True)
    broken["advection_bottom"] += 0.1
    expected = {name: broken[name].values for name in variables}
    with pytest.raises(AssertionError):
        verify_arrays(broken, expected, tmp_path / "broken")
