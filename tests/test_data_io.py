import numpy as np
import xarray as xr
import pytest

from src import data_io


def test_normalize_quantile_token_handles_numeric_and_string_inputs():
    assert data_io._normalize_quantile_token(95) == "95"
    assert data_io._normalize_quantile_token(95.0) == "95"
    assert data_io._normalize_quantile_token(97.5) == "97p5"
    assert data_io._normalize_quantile_token("95") == "95"
    assert data_io._normalize_quantile_token("q97p5") == "97p5"


def test_filter_yearly_files_filters_requested_years():
    paths = [
        "/tmp/heat_budget_1940.nc",
        "/tmp/heat_budget_1941.nc",
        "/tmp/heat_budget_1942.nc",
    ]

    filtered = data_io._filter_yearly_files(paths, [1940, 1942])

    assert filtered == ["/tmp/heat_budget_1940.nc", "/tmp/heat_budget_1942.nc"]


def test_filter_yearly_files_raises_when_no_year_matches():
    with pytest.raises(FileNotFoundError, match="requested years"):
        data_io._filter_yearly_files(["/tmp/heat_budget_1940.nc"], [1999])


def test_glob_required_raises_with_pattern():
    with pytest.raises(FileNotFoundError, match=r"No files matched pattern: /tmp/missing_\*"):
        data_io._glob_required("/tmp/missing_*")


def test_standardize_common_structure_renames_time_and_drops_bounds():
    ds = xr.Dataset(
        data_vars={
            "tas": (("valid_time", "latitude", "longitude"), [[[280.0]]]),
            "valid_time_bnds": (("valid_time", "bnds"), [[0.0, 1.0]]),
        },
        coords={
            "valid_time": [0],
            "latitude": [50.0],
            "longitude": [240.0],
            "bnds": [0, 1],
        },
    )

    out = data_io._standardize_common_structure(ds)

    assert "time" in out.coords
    assert "lat" in out.coords
    assert "lon" in out.coords
    assert "valid_time" not in out.coords
    assert "valid_time_bnds" not in out.variables
    assert "bnds" not in out.dims
    assert "tas" in out.data_vars


def test_open_era5_tas_constructs_pattern_and_standardizes(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={
            "tas": (("valid_time", "lat", "lon"), [[[280.0]], [[281.0]]]),
            "valid_time_bnds": (("valid_time", "bnds"), [[0.0, 1.0], [1.0, 2.0]]),
        },
        coords={"valid_time": [0, 1], "lat": [50.0], "lon": [240.0], "bnds": [0, 1]},
    )

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return [
            "/data/tas_daily_ERA_1940_2x2_bil.nc",
            "/data/tas_daily_ERA_1941_2x2_bil.nc",
        ]

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)

    def fake_open(paths, *, combine, chunks):
        captured["paths"] = list(paths)
        captured["combine"] = combine
        captured["chunks"] = dict(chunks)
        return ds

    monkeypatch.setattr(data_io, "_open_multiple_datasets", fake_open)

    out = data_io.open_era5_tas(years=[1941])

    assert captured["pattern"].endswith("/tas_daily_ERA5_*_2x2_bil.nc")
    assert captured["paths"] == ["/data/tas_daily_ERA_1941_2x2_bil.nc"]
    assert captured["combine"] == "by_coords"
    assert captured["chunks"] == data_io.DEFAULT_TAS_CHUNKS
    assert "time" in out.coords
    assert "valid_time_bnds" not in out.variables
    assert "tas" in out.data_vars


def test_open_era5_lwa_opens_expected_file(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"LWA": (("time", "lat", "lon"), [[[1.0]]])},
        coords={"time": [0], "lat": [50.0], "lon": [240.0]},
    )
    ds["LWA_a"] = ds["LWA"].copy()
    ds["LWA_c"] = ds["LWA"].copy()

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return ["/data/LWA_day_ERA5_2deg.500.nc"]

    def fake_open_single_dataset(path, *, chunks):
        captured["path"] = path
        return ds

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)
    monkeypatch.setattr(data_io, "_open_single_dataset", fake_open_single_dataset)

    out = data_io.open_era5_lwa()

    assert captured["pattern"].endswith("/z500/LWA_day_ERA5_2deg.500.nc")
    assert captured["path"] == "/data/LWA_day_ERA5_2deg.500.nc"
    assert {"LWA", "LWA_a", "LWA_c"} <= set(out.data_vars)


def test_open_era5_lwa_filters_requested_years(monkeypatch):
    ds = xr.Dataset(
        data_vars={"LWA": (("time", "lat", "lon"), np.ones((3, 1, 1)))},
        coords={
            "time": np.array(["1940-05-01", "1941-05-01", "1942-05-01"], dtype="datetime64[D]"),
            "lat": [50.0],
            "lon": [240.0],
        },
    )
    ds["LWA_a"] = ds["LWA"].copy()
    ds["LWA_c"] = ds["LWA"].copy()

    monkeypatch.setattr(data_io, "_glob_required", lambda pattern: ["/data/LWA_day_ERA5_2deg.500.nc"])
    monkeypatch.setattr(data_io, "_open_single_dataset", lambda path, *, chunks: ds)

    out = data_io.open_era5_lwa(years=[1941])

    assert out.sizes["time"] == 1
    assert int(out["time"].dt.year.item()) == 1941


def test_open_era5_lwa_threshold_builds_current_path(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"LWA": ("dayofyear", [1.0])},
        coords={"dayofyear": [1]},
    )
    ds["LWA_a"] = ds["LWA"].copy()
    ds["LWA_c"] = ds["LWA"].copy()

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return ["/data/lwa_thresh.nc"]

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)
    monkeypatch.setattr(data_io, "_open_single_dataset", lambda path, *, chunks: ds)

    out = data_io.open_era5_lwa_threshold(region="pnw_bartusek", quantile=95)

    assert "/pnw_bartusek/ERA5/q95/" in captured["pattern"]
    assert captured["pattern"].endswith("ERA5_LWAthresh_block_1970_2014_q95_pnw_bartusek.500.nc")
    assert "dayofyear" in out.coords


def test_open_era5_hw_threshold_rejects_unsupported_methods():
    with pytest.raises(ValueError, match="Only 'evolving' is implemented"):
        data_io.open_era5_hw_threshold(region="pnw_bartusek", quantile=95, method="block")


def test_open_era5_hw_threshold_builds_evolving_path(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={
            "threshold": (("year", "dayofyear"), [[1.0]]),
            "climatology": (("year", "dayofyear"), [[0.0]]),
        },
        coords={"year": [1950], "dayofyear": [1]},
    )

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return ["/data/hw_thresh.nc"]

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)
    monkeypatch.setattr(data_io, "_open_single_dataset", lambda path, *, chunks: ds)

    out = data_io.open_era5_hw_threshold(region="pnw_bartusek", quantile="97p5")

    assert "/pnw_bartusek/ERA5/evolving/q97p5/" in captured["pattern"]
    assert captured["pattern"].endswith(
        "ERA5_HWthresh_evolving_1940_2024_tas_q97p5_pnw_bartusek.nc"
    )
    assert {"threshold", "climatology"} <= set(out.data_vars)


def test_open_era5_hw_threshold_filters_requested_years(monkeypatch):
    ds = xr.Dataset(
        data_vars={
            "threshold": (("year", "dayofyear"), [[1.0], [2.0], [3.0]]),
            "climatology": (("year", "dayofyear"), [[0.0], [0.1], [0.2]]),
        },
        coords={"year": [1940, 1941, 1942], "dayofyear": [1]},
    )

    monkeypatch.setattr(data_io, "_glob_required", lambda pattern: ["/data/hw_thresh.nc"])
    monkeypatch.setattr(data_io, "_open_single_dataset", lambda path, *, chunks: ds)

    out = data_io.open_era5_hw_threshold(
        region="pnw_bartusek",
        quantile=95,
        years=[1941, 1942],
    )

    assert out.sizes["year"] == 2
    assert out["year"].values.tolist() == [1941, 1942]
    assert out["threshold"].values[:, 0].tolist() == [2.0, 3.0]


def test_open_era5_heat_budget_filters_requested_years(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"dT_dt": ("time", [1.0, 2.0])},
        coords={"time": [0, 1]},
    )
    ds["adiabatic_term"] = ds["dT_dt"].copy()
    ds["diabatic_term"] = ds["dT_dt"].copy()
    ds["T_domain_avg"] = ds["dT_dt"].copy()
    ds["domain_volume"] = ds["dT_dt"].copy()
    ds["advection_term"] = ds["dT_dt"].copy()

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return [
            "/data/heat_budget_1940.nc",
            "/data/heat_budget_1941.nc",
        ]

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)

    def fake_open(paths, *, combine, chunks):
        captured["paths"] = list(paths)
        captured["combine"] = combine
        captured["chunks"] = dict(chunks)
        return ds

    monkeypatch.setattr(data_io, "_open_multiple_datasets", fake_open)

    out = data_io.open_era5_heat_budget(years=[1940])

    assert captured["pattern"].endswith("/heat_budget_*.nc")
    assert captured["paths"] == ["/data/heat_budget_1940.nc"]
    assert captured["combine"] == "by_coords"
    assert captured["chunks"] == data_io.DEFAULT_HEAT_BUDGET_CHUNKS
    assert {"dT_dt", "adiabatic_term", "diabatic_term", "T_domain_avg", "domain_volume", "advection_term"} <= set(out.data_vars)
    assert "time" in out.coords
