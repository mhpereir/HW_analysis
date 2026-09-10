"""Scientific and offline plotting checks for the single-event map workflow."""

import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from cartopy.mpl.contour import GeoContourSet
from HW_analysis.scripts.top_events_map import build_top_events_map as build_cli
from HW_analysis.scripts.top_events_map import plot_top_events_map as plot_cli
from HW_analysis.src import analysis_io, config, data_io, plot_style
from HW_analysis.src import top_events_map as maps
from HW_analysis.src import top_events_map_plotting as plotting
from matplotlib.colors import to_rgba

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "scripts/top_events_map/schedule_top_events_map.sh"


def _table(peaks=("2021-06-29T12:00", "1941-07-17", "2022-07-01")):
    count = len(peaks)
    return xr.Dataset(
        {
            "event_id": ("event", np.arange(1127, 1127 + count)),
            "peak_time": ("event", np.array(peaks, dtype="datetime64[ns]")),
            "tas_peak": ("event", np.arange(300.0, 300.0 - count, -1)),
            "tas_anom_peak": ("event", np.arange(3.0, 3.0 + count)),
        },
        coords={"event": np.arange(17, 17 + count)},
        attrs={"pipeline_stage": "stage_2_event_features"},
    )


def _spatial(times, *, climatology=False):
    dates = pd.DatetimeIndex(times)
    latitude = np.array([80.0, 65.0, 50.0, 35.0, 10.0])
    longitude = np.array([190.0, 215.0, 240.0, 265.0, 290.0, 320.0])
    lat, lon = np.meshgrid(latitude, longitude, indexing="ij")
    pattern = np.sin(np.deg2rad(lat)) * np.cos(np.deg2rad(lon))
    day = np.asarray(dates.dayofyear, dtype=float)[:, None, None]
    if climatology:
        temperature = 270 + 0.1 * day + 2 * pattern
        height = 5000 + 0.2 * day + 100 * pattern
    else:
        temperature = 285 + 0.05 * day + 4 * pattern
        height = 5100 + 0.5 * day + 300 * pattern
    ds = xr.Dataset(
        {
            "t2m": (
                ("valid_time", "latitude", "longitude"),
                temperature,
                {"units": "K"},
            ),
            "z": (
                ("valid_time", "pressure_level", "latitude", "longitude"),
                height[:, None] * config.G_M_S2,
                {"units": "m**2 s**-2"},
            ),
        },
        coords={
            "valid_time": dates,
            "latitude": latitude,
            "longitude": longitude,
            "pressure_level": [500.0],
        },
    )
    ds.pressure_level.attrs["units"] = "hPa"
    if climatology:
        ds.attrs.update(climatology_start_year=1940, climatology_end_year=2024)
    return ds


def _inputs(tmp_path, peaks=("2021-06-29T12:00",)):
    features = _table(peaks)
    table_path = tmp_path / "features.nc"
    features.to_netcdf(table_path, engine="h5netcdf")
    days = pd.DatetimeIndex(features.peak_time.values).normalize()
    samples = pd.DatetimeIndex(
        np.unique(days.values[:, None] + np.array([-1, 0, 1], dtype="timedelta64[D]"))
    )
    for year in set(samples.year):
        ds = _spatial(samples[samples.year == year])
        ds.to_netcdf(tmp_path / f"ERA5_daily_t2m_z500_{year}.nc", engine="h5netcdf")
    climate_path = tmp_path / "climate.nc"
    _spatial(pd.date_range("2024-01-01", "2024-12-31"), climatology=True).to_netcdf(
        climate_path,
        engine="h5netcdf",
    )
    kwargs = {
        "event_features_path": table_path,
        "region": "pnw_bartusek",
        "daily_dir": tmp_path,
        "climatology_path": climate_path,
        "climatology_years": (1940, 2024),
        "source_commit": "a" * 40,
    }
    return features, kwargs


def _edit_file(path, change):
    with xr.open_dataset(path, engine="h5netcdf") as source:
        ds = source.load()
    changed = change(ds)
    (ds if changed is None else changed).to_netcdf(path, engine="h5netcdf", mode="w")


def test_ranking_defaults_match_temporal_top_events_and_year_filters_before_rank():
    features = _table()
    selected = maps.select_events(features, region="pnw_hotz", top_n=2)
    assert selected.event_id.values.tolist() == [1127, 1128]
    assert selected.event.values.tolist() == [17, 18]
    assert selected.selection_rank.values.tolist() == [1, 2]
    by_anomaly = maps.select_events(
        features, region="pnw_hotz", rank_metric="tas_anom_peak"
    )
    assert by_anomaly.event_id.item() == 1129
    year = maps.select_events(features, region="pnw_hotz", peak_year=2022)
    assert year.event_id.item() == 1129
    assert year.selection_rank.item() == 1
    assert year.attrs["candidate_event_count"] == 1
    with pytest.raises(ValueError, match="No finite-ranked events"):
        maps.select_events(features, region="pnw_hotz", peak_year=1999)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda ds: ds.attrs.update(pipeline_stage="stage_1"), "pipeline_stage"),
        (lambda ds: ds.attrs.update(region="alaska"), "disagrees"),
        (lambda ds: ds.event_id.values.__setitem__(1, 1127), "unique integers"),
        (lambda ds: ds.peak_time.values.__setitem__(0, np.datetime64("NaT")), "finite"),
    ],
)
def test_bad_event_tables_fail(mutation, message):
    features = _table()
    mutation(features)
    with pytest.raises(ValueError, match=message):
        maps.select_events(features, region="pnw_hotz")


@pytest.mark.parametrize(
    "peak", ["2021-06-29T18:30", "2021-03-01", "2020-02-29", "2021-01-01"]
)
def test_three_day_means_match_independent_direct_sums(tmp_path, peak):
    features, kwargs = _inputs(tmp_path, (peak,))
    product = maps.build_top_event_maps(features, **kwargs)
    days = pd.date_range(
        pd.Timestamp(peak).normalize() - pd.Timedelta(days=1), periods=3
    )
    actual_t, actual_z, climate_t, climate_z = [], [], [], []
    with xr.open_dataset(kwargs["climatology_path"], engine="h5netcdf") as climate:
        for day in days:
            with xr.open_dataset(
                tmp_path / f"ERA5_daily_t2m_z500_{day.year}.nc", engine="h5netcdf"
            ) as source:
                row = source.sel(valid_time=day).sortby("latitude")
                actual_t.append(row.t2m.values)
                actual_z.append(row.z.squeeze("pressure_level").values / config.G_M_S2)
            row = climate.sel(valid_time=pd.Timestamp(2024, day.month, day.day)).sortby(
                "latitude"
            )
            climate_t.append(row.t2m.values)
            climate_z.append(row.z.squeeze("pressure_level").values / config.G_M_S2)
    for name, actual, baseline in (
        ("t2m", actual_t, climate_t),
        ("z500", actual_z, climate_z),
    ):
        expected = (actual[0] + actual[1] + actual[2]) / 3
        clim = (baseline[0] + baseline[1] + baseline[2]) / 3
        np.testing.assert_allclose(
            product[f"{name}_event_mean"][0], expected, rtol=0, atol=1e-11
        )
        np.testing.assert_allclose(
            product[f"{name}_climatology_mean"][0], clim, rtol=0, atol=1e-11
        )
        np.testing.assert_allclose(
            product[f"{name}_anomaly"][0], expected - clim, rtol=0, atol=1e-11
        )
    np.testing.assert_array_equal(product.sample_date.values[0], days.values)
    assert product.peak_time.values[0] == np.datetime64(peak)
    assert product.event.values.tolist() == [17]
    assert product.attrs["region_metadata_source"] == "explicit_argument"
    assert product.attrs["climatology_start_year"] == 1940
    assert product.attrs["event_features_sha256"] == maps.sha256_file(
        kwargs["event_features_path"]
    )


def test_overlapping_event_windows_keep_individual_fields(tmp_path):
    features, kwargs = _inputs(tmp_path, ("2021-06-29", "2021-06-30"))
    ds = maps.build_top_event_maps(features, top_n=2, **kwargs)
    assert ds.sizes["event"] == 2
    assert not np.array_equal(ds.t2m_anomaly[0], ds.t2m_anomaly[1])
    assert ds.sample_date.values[0, 1] == ds.sample_date.values[1, 0]


@pytest.mark.parametrize(
    "target, change, message",
    [
        (
            "daily",
            lambda ds: ds.isel(valid_time=slice(1, None)),
            "missing required daily",
        ),
        (
            "daily",
            lambda ds: ds.assign_coords(
                valid_time=ds.valid_time + np.timedelta64(12, "h")
            ),
            "midnight",
        ),
        ("daily", lambda ds: ds.isel(valid_time=[0, 0, 2]), "unique finite midnight"),
        (
            "daily",
            lambda ds: ds.t2m.values.__setitem__((0, 1, 1), np.nan),
            "non-finite",
        ),
        ("daily", lambda ds: ds.t2m.attrs.update(units="degC"), "t2m units"),
        ("daily", lambda ds: ds.z.attrs.update(units="m"), "z units"),
        (
            "daily",
            lambda ds: ds.assign_coords(
                pressure_level=xr.DataArray(
                    [700.0], dims="pressure_level", attrs={"units": "hPa"}
                )
            ),
            "500 hPa",
        ),
        ("daily", lambda ds: ds.pressure_level.attrs.clear(), "pressure units"),
        (
            "daily",
            lambda ds: ds.assign_coords(longitude=[190, 215, 240, 265, 290, 290]),
            "unique",
        ),
        ("climate", lambda ds: ds.isel(valid_time=slice(1, None)), "366 unique"),
        (
            "climate",
            lambda ds: ds.assign_coords(longitude=[190, 216, 240, 265, 290, 320]),
            "align",
        ),
        (
            "climate",
            lambda ds: ds.attrs.update(climatology_start_year=1991),
            "baseline",
        ),
    ],
)
def test_invalid_spatial_inputs_fail(tmp_path, target, change, message):
    features, kwargs = _inputs(tmp_path)
    path = (
        kwargs["climatology_path"]
        if target == "climate"
        else tmp_path / "ERA5_daily_t2m_z500_2021.nc"
    )
    _edit_file(path, change)
    with pytest.raises(ValueError, match=message):
        maps.build_top_event_maps(features, **kwargs)


def test_missing_adjacent_year_is_reported(tmp_path):
    features, kwargs = _inputs(tmp_path, ("2021-01-01",))
    (tmp_path / "ERA5_daily_t2m_z500_2020.nc").unlink()
    with pytest.raises(FileNotFoundError, match="ERA5_daily_t2m_z500_2020.nc"):
        maps.build_top_event_maps(features, **kwargs)


def test_loading_transposes_dimensions_and_accepts_pressure_in_pa(tmp_path):
    dates = pd.date_range("2021-06-28", periods=3)
    source = _spatial(dates).transpose(
        "longitude", "latitude", "pressure_level", "valid_time"
    )
    source = source.assign_coords(pressure_level=[50000.0])
    source.pressure_level.attrs["units"] = "Pa"
    path = tmp_path / "daily.nc"
    source.to_netcdf(path, engine="h5netcdf")
    loaded = data_io.load_daily_spatial_fields(
        path, dates, lat_bounds=(10, 80), lon_bounds=(-170, -40)
    )
    assert loaded.z500.dims == ("time", "latitude", "longitude")
    np.testing.assert_allclose(
        loaded.z500.values,
        _spatial(dates).z[:, 0].sortby("latitude").values / config.G_M_S2,
    )


def test_product_round_trip_and_no_overwrite(tmp_path):
    features, kwargs = _inputs(tmp_path)
    product = maps.build_top_event_maps(features, **kwargs)
    path = analysis_io.save_top_event_maps(product, tmp_path / "out/map.nc")
    with analysis_io.open_top_event_maps(path) as loaded:
        xr.testing.assert_allclose(product, loaded)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        analysis_io.save_top_event_maps(product, path)
    assert path.read_bytes() == original
    assert not list(path.parent.glob("*.partial"))
    product.t2m_anomaly.values[0, 0, 0] += 1
    with pytest.raises(ValueError, match="anomaly does not equal"):
        maps.validate_top_event_maps(product)


def test_single_panel_plot_shows_dates_units_contour_signs_and_both_regions(
    tmp_path, monkeypatch
):
    features, kwargs = _inputs(tmp_path)
    ds = maps.build_top_event_maps(features, **kwargs)
    # Give this figure both contour signs without changing the anomaly identity.
    ds.z500_event_mean.values -= 180
    ds.z500_anomaly.values -= 180
    monkeypatch.setattr(plotting, "_decorate_map", lambda *args: None)
    original = ds.copy(deep=True)
    fig = plotting.plot_top_event_map(ds)
    try:
        assert len(fig.axes) == 2  # one map plus its colorbar
        ax = fig.axes[0]
        assert "28-30 June 2021" in ax.get_title()
        assert "Bartusek" in ax.get_title()
        assert fig.axes[1].get_ylabel() == "2 m temperature anomaly [K]"
        assert len(ax.patches) == 2
        for patch, region in zip(ax.patches, plotting.PNW_REGIONS, strict=True):
            west, east, south, north = maps.region_bounds(region)
            assert patch.get_xy() == (west, south)
            assert patch.get_width() == east - west
            assert patch.get_height() == north - south
            assert patch.get_edgecolor() == to_rgba(plot_style.REGION_COLORS[region])
        contour = next(
            item for item in ax.collections if isinstance(item, GeoContourSet)
        )
        for level, (_, dashes) in zip(
            contour.levels, contour.get_linestyles(), strict=True
        ):
            assert (dashes is not None) == (level < 0)
        assert "1940-2024" in fig.texts[0].get_text()
        assert "Z500 anomaly contours [m]" in fig.texts[0].get_text()
        assert plt.rcParams["savefig.dpi"] == plot_style.DPI
        xr.testing.assert_identical(ds, original)
    finally:
        plt.close(fig)


@pytest.mark.parametrize(
    "options",
    [
        {"temperature_limit": np.nan},
        {"height_contour_interval": 0},
        {"outline_regions": ["alaska", "alaska"]},
        {"outline_regions": ["central_china"]},
    ],
)
def test_invalid_plot_options_fail(tmp_path, options):
    features, kwargs = _inputs(tmp_path)
    ds = maps.build_top_event_maps(features, **kwargs)
    with pytest.raises(ValueError):
        plotting.plot_top_event_map(ds, **options)


def test_build_and_plot_cli_round_trip_offline(tmp_path, monkeypatch):
    _, kwargs = _inputs(tmp_path)
    output = tmp_path / "map.nc"
    argv = [
        "--event-features-path",
        str(kwargs["event_features_path"]),
        "--region",
        "pnw_bartusek",
        "--daily-dir",
        str(tmp_path),
        "--climatology-path",
        str(kwargs["climatology_path"]),
        "--climatology-start-year",
        "1940",
        "--climatology-end-year",
        "2024",
        "--peak-year",
        "2021",
        "--output-path",
        str(output),
    ]
    assert build_cli.main(argv) == 0
    with pytest.raises(FileExistsError):
        build_cli.main(argv)
    monkeypatch.setattr(
        plot_cli.top_events_map_plotting, "_decorate_map", lambda *args: None
    )
    figure_dir = tmp_path / "figures"
    plot_argv = ["--input-path", str(output), "--output-dir", str(figure_dir)]
    assert plot_cli.main(plot_argv) == 0
    files = list(figure_dir.glob("*.png"))
    assert len(files) == 1
    assert files[0].name == "top_events_map_pnw_bartusek_rank01_event1127_20210629.png"
    assert files[0].stat().st_size > 10000
    assert not list(figure_dir.glob("*.partial.png"))
    with pytest.raises(FileExistsError):
        plot_cli.main(plot_argv)


def test_pbs_script_parses_and_rejects_stale_commit_without_creating_outputs(tmp_path):
    subprocess.run(["bash", "-n", str(SCHEDULER)], check=True)
    env = dict(
        os.environ,
        PROJECT_ROOT=str(REPO_ROOT),
        EXPECTED_COMMIT="0" * 40,
        EVENT_FEATURES_PATH=str(tmp_path / "not-read.nc"),
        REGION="pnw_bartusek",
        DAILY_DIR=str(tmp_path),
        CLIMATOLOGY_PATH=str(tmp_path / "not-read.nc"),
        RUN_DIR=str(tmp_path / "run"),
        LOG_DIR=str(tmp_path / "logs"),
        PBS_JOBID="test",
    )
    result = subprocess.run(
        ["bash", str(SCHEDULER)], env=env, text=True, capture_output=True, check=False
    )
    assert result.returncode != 0
    assert not (tmp_path / "run").exists()
    assert not (tmp_path / "logs").exists()
