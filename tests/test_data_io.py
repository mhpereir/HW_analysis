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


def test_filter_yearly_files_ignores_year_tokens_in_parent_directories():
    paths = [
        "/tmp/pnw_bartusek_surface_700hPa_1940_2025/annual/heat_budget_1940.nc",
        "/tmp/pnw_bartusek_surface_700hPa_1940_2025/annual/heat_budget_1941.nc",
        "/tmp/pnw_bartusek_surface_700hPa_1940_2025/annual/heat_budget_1942.nc",
    ]

    filtered = data_io._filter_yearly_files(paths, [1941, 1942])

    assert filtered == paths[1:]


def test_filter_yearly_files_raises_when_no_year_matches():
    with pytest.raises(FileNotFoundError, match="requested years"):
        data_io._filter_yearly_files(["/tmp/heat_budget_1940.nc"], [1999])


def test_filter_yearly_files_raises_when_one_requested_year_is_missing():
    paths = [
        "/tmp/heat_budget_1940.nc",
        "/tmp/heat_budget_1942.nc",
    ]

    with pytest.raises(FileNotFoundError, match="1941"):
        data_io._filter_yearly_files(paths, [1940, 1941, 1942])


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
        captured["chunks"] = chunks
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
        captured["chunks"] = chunks
        return ds

    monkeypatch.setattr(data_io, "_open_multiple_datasets", fake_open)

    out = data_io.open_era5_heat_budget(years=[1940])

    assert captured["pattern"].endswith(
        "/pnw_bartusek_surface_700hPa_1940_2025/annual/heat_budget_*.nc"
    )
    assert captured["paths"] == ["/data/heat_budget_1940.nc"]
    assert captured["combine"] == "by_coords"
    assert captured["chunks"] == data_io.DEFAULT_HEAT_BUDGET_CHUNKS
    assert {"dT_dt", "adiabatic_term", "diabatic_term", "T_domain_avg", "domain_volume", "advection_term"} <= set(out.data_vars)
    assert "time" in out.coords


def test_era5_heat_budget_annual_root_uses_saved_results_tokens():
    root = data_io.era5_heat_budget_annual_root(
        region="pnw_bartusek",
        bottom_boundary="surface",
        top_boundary=700,
        start_year_ehb=1940,
        end_year_ehb=2025,
    )

    assert root.name == "annual"
    assert root.parent.name == "pnw_bartusek_surface_700hPa_1940_2025"


def test_era5_heat_budget_annual_root_accepts_pressure_bottom_boundary():
    root = data_io.era5_heat_budget_annual_root(
        region="pnw_bartusek",
        bottom_boundary=700,
        top_boundary="500hPa",
        start_year_ehb=1940,
        end_year_ehb=2025,
    )

    assert root.parent.name == "pnw_bartusek_700hPa_500hPa_1940_2025"


def test_open_era5_heat_budget_accepts_explicit_root(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"dT_dt": ("time", [1.0])},
        coords={"time": [0]},
    )

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return ["/custom/annual/heat_budget_1940.nc"]

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)
    monkeypatch.setattr(
        data_io,
        "_open_multiple_datasets",
        lambda paths, *, combine, chunks: ds,
    )

    data_io.open_era5_heat_budget(
        years=[1940],
        heat_budget_root="/custom/annual",
    )

    assert captured["pattern"] == "/custom/annual/heat_budget_*.nc"


def test_open_era5_surface_diagnostic_builds_expected_path(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"ssr": (("valid_time", "latitude", "longitude"), [[[1.0]]])},
        coords={"valid_time": [0], "latitude": [50.0], "longitude": [240.0]},
    )

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return [
            "/data/nssr_hour_ERA5_1940.nc",
            "/data/nssr_hour_ERA5_1941.nc",
        ]

    def fake_open(paths, *, combine, chunks):
        captured["paths"] = list(paths)
        captured["combine"] = combine
        captured["chunks"] = chunks
        return ds

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)
    monkeypatch.setattr(data_io, "_open_multiple_datasets", fake_open)

    out = data_io.open_era5_surface_diagnostic("nssr", years=[1941])

    assert captured["pattern"].endswith("/nssr/nssr_hour_ERA5_*.nc")
    assert captured["paths"] == ["/data/nssr_hour_ERA5_1941.nc"]
    assert captured["combine"] == "by_coords"
    assert captured["chunks"] == data_io.DEFAULT_GLOBAL_HOURLY_CHUNKS
    assert "time" in out.coords
    assert "lat" in out.coords
    assert "lon" in out.coords
    assert "ssr" in out.data_vars


def test_open_era5_surface_diagnostic_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unsupported ERA5 surface diagnostic"):
        data_io.open_era5_surface_diagnostic("bad")


def test_open_era5_pbl_p_builds_expected_path(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"pbl_p": (("time", "lat", "lon"), [[[1.0]]])},
        coords={"time": [0], "lat": [50.0], "lon": [-120.0]},
    )

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return ["/data/ERA5_ARCO_pbl_p_1940.nc"]

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)

    def fake_open(paths, *, combine, chunks):
        captured["paths"] = list(paths)
        captured["chunks"] = chunks
        return ds

    monkeypatch.setattr(data_io, "_open_multiple_datasets", fake_open)

    out = data_io.open_era5_pbl_p(years=[1940])

    assert captured["pattern"].endswith("/ERA5_ARCO_pbl_p_*.nc")
    assert captured["paths"] == ["/data/ERA5_ARCO_pbl_p_1940.nc"]
    assert captured["chunks"] == data_io.DEFAULT_PBL_CHUNKS
    assert "pbl_p" in out.data_vars


def test_open_era5_total_cloud_cover_uses_region_in_pattern(monkeypatch):
    captured = {}
    ds = xr.Dataset(
        data_vars={"total_cloud_cover": ("time", [0.5])},
        coords={"time": [0]},
    )

    def fake_glob_required(pattern):
        captured["pattern"] = pattern
        return [
            "/data/ERA5_ARCO_total_cloud_cover_pnw_bartusek_1940.nc",
            "/data/ERA5_ARCO_total_cloud_cover_pnw_bartusek_1941.nc",
        ]

    def fake_open(paths, *, combine, chunks):
        captured["paths"] = list(paths)
        captured["chunks"] = chunks
        return ds

    monkeypatch.setattr(data_io, "_glob_required", fake_glob_required)
    monkeypatch.setattr(data_io, "_open_multiple_datasets", fake_open)

    out = data_io.open_era5_total_cloud_cover(
        region="pnw_bartusek",
        years=[1940],
    )

    assert "ERA5_ARCO_total_cloud_cover_pnw_bartusek_*.nc" in captured["pattern"]
    assert captured["paths"] == ["/data/ERA5_ARCO_total_cloud_cover_pnw_bartusek_1940.nc"]
    assert captured["chunks"] == data_io.DEFAULT_REGIONAL_HOURLY_CHUNKS
    assert "total_cloud_cover" in out.data_vars
