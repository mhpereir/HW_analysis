from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from HW_analysis.scripts.spatial_composites import (
    build_dyn_net_spatial_composites as builder,
)


def test_prepare_events_derives_sign_and_records_zero_exclusion():
    events = _event_features(include_zero=True)

    selected = builder.prepare_events(events)

    np.testing.assert_allclose(selected["I_dyn_net"], [1.0, -1.0])
    np.testing.assert_array_equal(
        selected["event_dyn_sign"].values,
        ["positive", "negative"],
    )
    assert selected.attrs["zero_dyn_net_events_excluded"] == 1


def test_timestamp_weights_preserve_equal_event_weight_with_overlaps():
    dates = np.array(
        [
            ["2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04"],
            ["2000-06-02", "2000-06-03", "2000-06-04", "2000-06-05"],
            ["2000-06-01", "2000-06-02", "2000-06-03", "2000-06-04"],
        ],
        dtype="datetime64[ns]",
    )
    signs = np.array(["positive", "positive", "negative"])

    weights = builder.timestamp_weights(dates, signs, np.array([2, 1]))

    np.testing.assert_allclose(
        np.sum(np.stack(list(weights.values())), axis=0),
        np.ones((2, 4)),
    )
    np.testing.assert_allclose(
        weights[pd.Timestamp("2000-06-02")],
        [[0.5, 0.5, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
    )


def test_build_spatial_composites_uses_daily_data_and_calendar_climatology(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    climate_path = tmp_path / "climatology.nc"
    _write_spatial_file(
        daily_dir / "ERA5_daily_t2m_z500_2000.nc",
        year=2000,
        t2m_offset=2.0,
        z_height_offset=50.0,
    )
    _write_spatial_file(climate_path, year=2004)
    events = builder.prepare_events(_event_features())

    out = builder.build_spatial_composites(
        events,
        daily_dir=daily_dir,
        climatology_path=climate_path,
        lat_bounds=(10.0, 80.0),
        lon_bounds=(-170.0, -40.0),
    )

    np.testing.assert_array_equal(out["event_count"], [1, 1])
    np.testing.assert_allclose(out["t2m_anomaly"], 2.0, atol=1e-10)
    np.testing.assert_allclose(out["z500_anomaly"], 50.0, atol=1e-10)
    assert out["t2m_anomaly"].dims == (
        "dyn_sign",
        "lag",
        "latitude",
        "longitude",
    )
    np.testing.assert_array_equal(out["lag"], np.arange(-3, 4))
    np.testing.assert_array_equal(out["latitude"], [10.0, 20.0, 80.0])
    np.testing.assert_array_equal(out["longitude"], [-170.0, -160.0, -40.0])
    assert out.attrs["daily_lags"] == "-3,-2,-1,0,1,2,3"

    output_path = tmp_path / "full_composite.nc"
    builder.write_composite_product(out, output_path)
    with xr.open_dataset(output_path, engine="h5netcdf") as reopened:
        np.testing.assert_array_equal(
            reopened["event_dyn_sign"].values,
            ["positive", "negative"],
        )
        np.testing.assert_allclose(reopened["z500_anomaly"], 50.0, atol=1e-5)


def test_build_spatial_composites_rejects_missing_required_daily_date(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    daily_path = daily_dir / "ERA5_daily_t2m_z500_2000.nc"
    climate_path = tmp_path / "climatology.nc"
    _write_spatial_file(daily_path, year=2000)
    _write_spatial_file(climate_path, year=2004)
    with xr.open_dataset(daily_path, engine="h5netcdf") as source:
        reduced = source.sel(valid_time=slice("2000-06-08", None)).load()
    reduced.to_netcdf(daily_path, engine="h5netcdf", mode="w")

    with pytest.raises(ValueError, match="missing required daily timestamps"):
        builder.build_spatial_composites(
            builder.prepare_events(_event_features()),
            daily_dir=daily_dir,
            climatology_path=climate_path,
            lat_bounds=(10.0, 80.0),
            lon_bounds=(-170.0, -40.0),
        )


def test_standardize_spatial_dataset_rejects_wrong_pressure_level():
    ds = _spatial_dataset(2000).assign_coords(pressure_level=[700.0])

    with pytest.raises(ValueError, match="Expected only Z500"):
        builder.standardize_spatial_dataset(
            ds,
            lat_bounds=(10.0, 80.0),
            lon_bounds=(-170.0, -40.0),
        )


def test_write_composite_product_round_trips(tmp_path):
    ds = xr.Dataset(
        {"t2m_anomaly": (("dyn_sign", "latitude", "longitude"), np.ones((2, 2, 2)))},
        coords={
            "dyn_sign": ["positive", "negative"],
            "latitude": [10.0, 20.0],
            "longitude": [-100.0, -90.0],
        },
    )
    path = tmp_path / "composite.nc"

    written = builder.write_composite_product(ds, path)

    assert written == path.resolve()
    with xr.open_dataset(path, engine="h5netcdf") as reopened:
        np.testing.assert_allclose(reopened["t2m_anomaly"], 1.0)


def _event_features(*, include_zero: bool = False) -> xr.Dataset:
    peaks = [np.datetime64("2000-06-10"), np.datetime64("2000-06-20")]
    event_ids = [1, 2]
    i_dyn = [1.0, -1.0]
    if include_zero:
        peaks.append(np.datetime64("2000-06-25"))
        event_ids.append(3)
        i_dyn.append(0.0)
    return xr.Dataset(
        {
            "event_id": ("event", event_ids),
            "peak_time": ("event", np.asarray(peaks, dtype="datetime64[ns]")),
            "I_dyn_pre": ("event", i_dyn),
        },
        coords={"event": np.arange(len(peaks))},
    )


def _write_spatial_file(
    path: Path,
    *,
    year: int,
    t2m_offset: float = 0.0,
    z_height_offset: float = 0.0,
) -> None:
    ds = _spatial_dataset(
        year,
        t2m_offset=t2m_offset,
        z_height_offset=z_height_offset,
    )
    ds.to_netcdf(path, engine="h5netcdf")


def _spatial_dataset(
    year: int,
    *,
    t2m_offset: float = 0.0,
    z_height_offset: float = 0.0,
) -> xr.Dataset:
    time = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    latitude = np.array([90.0, 80.0, 20.0, 10.0, 0.0])
    longitude = np.array([0.0, 190.0, 200.0, 320.0, 330.0])
    doy = time.dayofyear.to_numpy(dtype=float)
    base_t2m = 250.0 + doy[:, None, None]
    base_height = 5000.0 + doy[:, None, None]
    shape = (time.size, latitude.size, longitude.size)
    t2m = np.broadcast_to(base_t2m + t2m_offset, shape).copy()
    z = np.broadcast_to(
        (base_height + z_height_offset) * builder.GEOPOTENTIAL_TO_HEIGHT_M_S2,
        shape,
    ).copy()
    return xr.Dataset(
        {
            "t2m": (("valid_time", "latitude", "longitude"), t2m),
            "z": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                z[:, None, :, :],
            ),
        },
        coords={
            "valid_time": time,
            "pressure_level": [500.0],
            "latitude": latitude,
            "longitude": longitude,
        },
    )
