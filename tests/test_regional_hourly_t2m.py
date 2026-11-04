"""Small complete-year fixtures for the native-grid temperature cache."""

import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from HW_analysis.scripts import prepare_regional_hourly_t2m as producer

from src import harmonize

regional = producer.regional_temperature
analysis_io = producer.analysis_io


@pytest.mark.parametrize("unit", ["s", "ms", "us", "ns"])
def test_hour_coverage_accepts_equivalent_datetime_encodings(unit):
    values = regional.expected_hours(2000, 2000).values.astype(f"datetime64[{unit}]")
    regional.validate_hours(xr.DataArray(values, dims="time"), 2000, 2000)


def test_hour_coverage_rejects_a_nanosecond_offset():
    values = regional.expected_hours(2000, 2000).values.astype("datetime64[ns]")
    values[0] += np.timedelta64(1, "ns")
    with pytest.raises(ValueError, match="Every UTC hour"):
        regional.validate_hours(xr.DataArray(values, dims="time"), 2000, 2000)


@pytest.fixture
def sources(monkeypatch):
    monkeypatch.setitem(
        producer.config.REGIONS, "test_region", (slice(49, 49.5), slice(-125, -124.5))
    )

    def make(year=2000):
        time = regional.expected_hours(year, year)
        lat = np.arange(49.75, 48.5, -0.25)
        lon = np.arange(234.75, 236, 0.25)
        values = (
            270
            + np.arange(len(time))[:, None, None] / 2400
            + np.arange(5)[None, :, None]
            + 3 * np.arange(5)[None, None, :]
        ).astype("float32")
        raw = xr.Dataset(
            {"t2m": (("valid_time", "latitude", "longitude"), values)},
            coords={
                "valid_time": time,
                "latitude": lat,
                "longitude": lon,
                "expver": ("valid_time", ["0001"] * len(time)),
            },
        )
        raw.t2m.attrs["units"] = "K"
        source = producer.data_io._standardize_common_structure(raw)
        source.t2m.encoding["chunksizes"] = (676, 5, 5)
        return source

    return make


def test_independent_weighted_means_float64_and_bounded_reads(sources, monkeypatch):
    source = sources()
    before = source.copy(deep=True)
    sizes = []
    original = xr.DataArray.load

    def record_load(self, **kwargs):
        if set(self.dims) == {"time", "lat", "lon"}:
            sizes.append(dict(self.sizes))
        return original(self, **kwargs)

    monkeypatch.setattr(xr.DataArray, "load", record_load)
    product, checks = regional.prepare_annual_region(
        source, region="test_region", year=2000
    )
    cells = source.t2m.values[:, 1:4, 1:4].astype(np.float64)
    weights = np.cos(np.deg2rad(source.lat.values[1:4]))
    expected = np.sum(cells * weights[None, :, None], axis=(1, 2)) / (weights.sum() * 3)
    np.testing.assert_allclose(product.tas_region, expected, rtol=0, atol=1e-12)
    assert product.tas_region.dtype == np.float64
    assert checks["hours"] == 8784
    assert checks["time_chunk"] == 676
    assert checks["source_expver"] == ["0001"]
    assert sizes and max(size["time"] for size in sizes) == 676
    assert all(size["lat"] == size["lon"] == 3 for size in sizes)
    expected_hash = hashlib.sha256(
        np.asarray(cells, dtype="<f8", order="C").tobytes()
    ).hexdigest()
    assert checks["decoded_source_float64_sha256"] == expected_hash
    xr.testing.assert_identical(source, before)


def test_grid_conventions_and_block_size_preserve_reduction(sources):
    source = sources(2001)
    first, first_checks = regional.prepare_annual_region(
        source, region="test_region", year=2001
    )
    smaller, small_checks = regional.prepare_annual_region(
        source, region="test_region", year=2001, time_chunk=168
    )
    np.testing.assert_allclose(first.tas_region, smaller.tas_region, rtol=0, atol=1e-12)
    assert (
        first_checks["decoded_source_float64_sha256"]
        == small_checks["decoded_source_float64_sha256"]
    )
    converted = source.assign_coords(lon=source.lon - 360).isel(
        lat=slice(None, None, -1)
    )
    reverse, _ = regional.prepare_annual_region(
        converted, region="test_region", year=2001
    )
    np.testing.assert_allclose(first.tas_region, reverse.tas_region, rtol=0, atol=1e-12)
    assert reverse.sizes["time"] == 8760


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing_hour", "Every UTC hour"),
        ("duplicate", "Every UTC hour"),
        ("shifted", "Every UTC hour"),
        ("unordered", "Every UTC hour"),
        ("missing_cell", "Missing/nonfinite"),
        ("sentinel", "Missing/nonfinite"),
        ("cropped", "complete test_region"),
        ("coarse", "0.25-degree"),
        ("units", "source units"),
        ("extra_dimension", "only time, lat and lon"),
    ],
)
def test_reject_invalid_source(sources, problem, match):
    source = sources()
    if problem == "missing_hour":
        source = source.isel(time=slice(1, None))
    elif problem == "duplicate":
        source = source.isel(time=[0, *range(source.sizes["time"] - 1)])
    elif problem == "shifted":
        source = source.assign_coords(time=source.time + np.timedelta64(1, "m"))
    elif problem == "unordered":
        source = source.isel(time=slice(None, None, -1))
    elif problem == "missing_cell":
        source.t2m.values[12, 2, 2] = np.nan
    elif problem == "sentinel":
        source.t2m.attrs["GRIB_missingValue"] = float(np.finfo(np.float32).max)
        source.t2m.values[12, 2, 2] = np.finfo(np.float32).max
    elif problem == "cropped":
        source = source.isel(lat=slice(2, None))
    elif problem == "coarse":
        source = source.isel(lat=slice(None, None, 2))
    elif problem == "units":
        source.t2m.attrs["units"] = "C"
    elif problem == "extra_dimension":
        source = source.expand_dims(member=[0])
    with pytest.raises(ValueError, match=match):
        regional.prepare_annual_region(source, region="test_region", year=2000)


def test_exact_stage1_join_keeps_daily_fields_and_rejects_wrong_region(sources):
    prepared, _ = regional.prepare_annual_region(
        sources(), region="test_region", year=2000
    )
    stage1 = xr.Dataset(
        {"tas_region": ("time", np.repeat([300.0, 301.0], 24))},
        coords={"time": pd.date_range("2000-06-01", periods=48, freq="h")},
        attrs={"region": "test_region"},
    )
    stage1.tas_region.attrs.update(lat_bounds=(49, 49.5), lon_bounds=(-125, -124.5))
    original = stage1.copy(deep=True)
    result = harmonize.add_regional_hourly_t2m(stage1, prepared)
    np.testing.assert_array_equal(
        result.tas_region_hourly, prepared.tas_region.sel(time=stage1.time)
    )
    xr.testing.assert_identical(result.tas_region, original.tas_region)
    xr.testing.assert_identical(stage1, original)
    with pytest.raises(ValueError, match="refusing replacement"):
        harmonize.add_regional_hourly_t2m(result, prepared)
    with pytest.raises(ValueError, match="incompatible region"):
        harmonize.add_regional_hourly_t2m(
            stage1.assign_attrs(region="alaska"), prepared
        )
    with pytest.raises(ValueError, match="missing required Stage-1 hours"):
        harmonize.add_regional_hourly_t2m(
            stage1.assign_coords(time=stage1.time + np.timedelta64(366, "D")), prepared
        )


def test_producer_roundtrip_manifest_and_exclusive_outputs(sources, tmp_path):
    source_root = tmp_path / "raw"
    source_root.mkdir()
    before = {}
    for year in (2000, 2001):
        source = sources(year).rename(
            time="valid_time", lat="latitude", lon="longitude"
        )
        path = source_root / f"2mT_hour_ERA5_{year}.nc"
        source.to_netcdf(path, engine="h5netcdf")
        before[path] = regional.sha256(path)
    output = tmp_path / "new-attempt"
    args = [
        "--region",
        "test_region",
        "--start-year",
        "2000",
        "--end-year",
        "2001",
        "--source-root",
        str(source_root),
        "--output-dir",
        str(output),
    ]
    path = producer.main(args)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["output"]["sha256"] == regional.sha256(path)
    assert manifest["output"]["variable"] == "tas_region"
    assert len(manifest["sources"]) == 2
    assert manifest["output"]["hours"] == 17544
    with analysis_io.open_regional_hourly_t2m(path, region="test_region") as reopened:
        regional.validate_product(reopened)
        assert set(reopened.data_vars) == {"tas_region"}
        assert reopened.tas_region.attrs["native_time_resolution"] == "hourly"
        assert reopened.tas_region.attrs["source_variable"] == "t2m"
        with pytest.raises(ValueError, match=r"requires tas_region\(time\)"):
            regional.validate_product(reopened.rename(tas_region="t2m"))
        assert reopened.time.values[-1] == np.datetime64("2001-12-31T23:00")
        with pytest.raises(FileExistsError):
            analysis_io.save_regional_hourly_t2m(reopened, path)
    with pytest.raises(FileExistsError, match="Refusing existing"):
        producer.main(args)
    assert before == {source: regional.sha256(source) for source in before}
    assert not list(output.glob(".*.partial"))


def test_missing_annual_file_fails_before_creating_output(sources, tmp_path):
    output = tmp_path / "attempt"
    with pytest.raises(FileNotFoundError, match="Missing annual ERA5"):
        producer.main(
            [
                "--region",
                "test_region",
                "--start-year",
                "2000",
                "--end-year",
                "2000",
                "--source-root",
                str(tmp_path),
                "--output-dir",
                str(output),
            ]
        )
    assert not output.exists()
