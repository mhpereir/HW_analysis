from __future__ import annotations

import numpy as np
import pytest
import xarray as xr
from HW_analysis.src import pbl_justification


def test_build_product_uses_peak_windows_and_daily_minimum_pressure():
    source = _make_pbl_source()
    events = _make_events()

    out = pbl_justification.build_product(
        source,
        events,
        region="pnw_bartusek",
        pre_days=1,
        post_days=1,
    ).load()

    np.testing.assert_array_equal(out["lag_hour"], np.arange(-24, 25))
    assert int(out[pbl_justification.SELECTED_EVENT_COUNT_NAME].item()) == 1
    assert int(out[pbl_justification.SELECTED_DAY_COUNT_NAME].item()) == 1
    np.testing.assert_array_equal(
        out[pbl_justification.EVENT_SAMPLE_COUNT_NAME],
        np.ones(49, dtype=np.int32),
    )
    expected_daily_minimum = source.sel(
        time=slice("2000-06-02T00", "2000-06-02T23")
    ).min("time")
    xr.testing.assert_allclose(
        out[pbl_justification.MAP_NAME],
        expected_daily_minimum.rename(pbl_justification.MAP_NAME),
    )
    assert np.all(
        out[pbl_justification.SPATIAL_P05_NAME].values
        <= out[pbl_justification.SPATIAL_P95_NAME].values
    )
    pbl_justification.validate_product(out, require_complete=False)


def test_build_product_rejects_missing_event_window_hour():
    source = _make_pbl_source().drop_sel(time=np.datetime64("2000-06-01T12"))

    with pytest.raises(ValueError, match="missing 1 event windows timestamps"):
        pbl_justification.build_product(
            source,
            _make_events(),
            region="pnw_bartusek",
            pre_days=1,
            post_days=1,
        )


def test_build_product_requires_exact_configured_domain():
    source = _make_pbl_source().isel(lat=slice(1, None))

    with pytest.raises(ValueError, match="coordinate 'lat' is invalid"):
        pbl_justification.build_product(
            source,
            _make_events(),
            region="pnw_bartusek",
            pre_days=1,
            post_days=1,
        )


def test_save_and_open_product_publish_complete_contract(tmp_path):
    product = pbl_justification.build_product(
        _make_pbl_source(),
        _make_events(),
        region="pnw_bartusek",
        pre_days=1,
        post_days=1,
    ).load()
    path = tmp_path / "pbl_justification.nc"

    written = pbl_justification.save_product(product, path)

    assert written == path
    with pbl_justification.open_product(path) as reopened:
        assert reopened.attrs["validation_status"] == "complete"
        pbl_justification.validate_product(reopened)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        pbl_justification.save_product(product, path)


def test_select_full_season_events_excludes_boundary_crossing_event():
    events = xr.Dataset(
        data_vars={
            "event_id": ("event", [1, 2]),
            "start_time": (
                "event",
                np.array(["2000-06-02", "2000-05-31"], dtype="datetime64[ns]"),
            ),
            "end_time": (
                "event",
                np.array(["2000-06-03", "2000-06-02"], dtype="datetime64[ns]"),
            ),
            "peak_time": (
                "event",
                np.array(
                    ["2000-06-02T12", "2000-06-01T12"],
                    dtype="datetime64[ns]",
                ),
            ),
        }
    )

    selected = pbl_justification.select_full_season_events(events)

    np.testing.assert_array_equal(selected["event_id"], [1])


def test_annual_source_paths_rejects_missing_year(tmp_path):
    region_root = tmp_path / "outputs/pnw_bartusek"
    region_root.mkdir(parents=True)
    existing = region_root / "ERA5_ARCO_pbl_p_1940.nc"
    existing.touch()

    with pytest.raises(FileNotFoundError, match="1941"):
        pbl_justification.annual_source_paths(
            tmp_path / "outputs",
            region="pnw_bartusek",
            years=[1940, 1941],
        )


def _make_pbl_source() -> xr.DataArray:
    time = np.arange(
        np.datetime64("2000-06-01T12"),
        np.datetime64("2000-06-03T13"),
        np.timedelta64(1, "h"),
    )
    lat = np.array([40.0, 60.0])
    lon = np.array([-130.0, -110.0])
    hour = np.arange(time.size, dtype=float)[:, np.newaxis, np.newaxis]
    spatial = np.array([[0.0, 200.0], [400.0, 600.0]])[np.newaxis, :, :]
    values = 92_000.0 - 10.0 * hour + spatial
    return xr.DataArray(
        values,
        dims=("time", "lat", "lon"),
        coords={"time": time, "lat": lat, "lon": lon},
        name="pbl_p",
        attrs={"units": "Pa"},
    )


def _make_events() -> xr.Dataset:
    return xr.Dataset(
        data_vars={
            "event_id": ("event", [17]),
            "start_time": (
                "event",
                np.array(["2000-06-02T00"], dtype="datetime64[ns]"),
            ),
            "end_time": (
                "event",
                np.array(["2000-06-02T23"], dtype="datetime64[ns]"),
            ),
            "peak_time": (
                "event",
                np.array(["2000-06-02T12"], dtype="datetime64[ns]"),
            ),
        }
    )
