from pathlib import Path

import h5netcdf
import matplotlib.pyplot as plt
import numpy as np
import pytest

from HW_analysis.scripts.region_vis import plot_stage1_regions as plotter
from HW_analysis.src import analysis_io, config, plot_style


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "scripts/region_vis/schedule_plot_stage1_regions.sh"


def test_discover_stage1_regions_deduplicates_products_by_region(tmp_path):
    alaska_bounds = plotter.configured_region_bounds("alaska")
    china_bounds = plotter.configured_region_bounds("central_china")
    _write_stage1_product(tmp_path / "harmonized_regional_timeseries_a.nc", "alaska")
    _write_stage1_product(tmp_path / "harmonized_regional_timeseries_b.nc", "alaska")
    _write_stage1_product(
        tmp_path / "harmonized_regional_timeseries_c.nc",
        "central_china",
    )

    domains = plotter.discover_stage1_regions(tmp_path)

    assert [domain.name for domain in domains] == ["alaska", "central_china"]
    assert len(domains[0].product_paths) == 2
    assert (
        domains[0].west,
        domains[0].east,
        domains[0].south,
        domains[0].north,
    ) == alaska_bounds
    assert (
        domains[1].west,
        domains[1].east,
        domains[1].south,
        domains[1].north,
    ) == china_bounds


def test_discover_stage1_regions_rejects_bounds_that_disagree_with_config(tmp_path):
    path = tmp_path / "harmonized_regional_timeseries_bad.nc"
    _write_stage1_product(path, "alaska", bounds=(-160.0, -150.0, 58.0, 69.5))

    with pytest.raises(ValueError, match="do not match src/config.py"):
        plotter.discover_stage1_regions(tmp_path)


def test_discover_stage1_regions_requires_stage1_marker(tmp_path):
    path = tmp_path / "harmonized_regional_timeseries_bad.nc"
    _write_stage1_product(path, "alaska", pipeline_stage="not_stage_1")

    with pytest.raises(ValueError, match="pipeline_stage"):
        plotter.discover_stage1_regions(tmp_path)


def test_plot_region_domains_draws_one_wireframe_per_configured_region():
    domains = tuple(_configured_domain(region) for region in sorted(config.REGIONS))

    fig = plotter.plot_region_domains(domains, run_label=plotter.DEFAULT_RUN_ID)
    try:
        ax = fig.axes[0]
        assert len(ax.patches) == len(config.REGIONS)
        assert ax.get_title() == (
            "Stage 1 Regional Domains\nRun bf232281_20260819"
        )
        assert len(fig.legends) == 1
        assert [text.get_text() for text in fig.legends[0].get_texts()] == [
            plot_style.REGION_NAME_MAPPING[region]
            for region in sorted(config.REGIONS)
        ]
        edge_colors = [patch.get_edgecolor() for patch in ax.patches]
        assert len(set(edge_colors)) == len(config.REGIONS)
        assert all(not patch.get_fill() for patch in ax.patches)
    finally:
        plt.close(fig)


def test_write_figure_creates_parent_directory(tmp_path):
    fig = plt.figure()
    output_path = tmp_path / "figures" / "regions.png"
    try:
        written = plotter.write_figure(fig, output_path)
    finally:
        plt.close(fig)

    assert written == output_path.resolve()
    assert written.stat().st_size > 0


def test_scheduler_is_commit_verified_and_runs_only_region_plotter():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=1:mem=4gb" in text
    assert "#PBS -l walltime=00:10:00" in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert "status --porcelain --untracked-files=normal" in text
    assert "scripts/region_vis/plot_stage1_regions.py" in text
    assert '--expected-region-count "${EXPECTED_REGION_COUNT}"' in text
    assert "build_stage1_harmonized_timeseries.py" not in text


def _configured_domain(region: str) -> plotter.RegionDomain:
    west, east, south, north = plotter.configured_region_bounds(region)
    return plotter.RegionDomain(
        name=region,
        west=west,
        east=east,
        south=south,
        north=north,
        product_paths=(Path(f"{region}.nc"),),
    )


def _write_stage1_product(
    path: Path,
    region: str,
    *,
    bounds: tuple[float, float, float, float] | None = None,
    pipeline_stage: str = analysis_io.EXPECTED_PIPELINE_STAGE,
) -> None:
    west, east, south, north = bounds or plotter.configured_region_bounds(region)
    with h5netcdf.File(path, "w") as dataset:
        dataset.attrs["pipeline_stage"] = pipeline_stage
        dataset.attrs["region"] = region
        dataset.dimensions = {"time": 1}
        tas_region = dataset.create_variable("tas_region", ("time",), dtype="f8")
        tas_region[:] = np.array([280.0])
        tas_region.attrs["lat_bounds"] = np.array([south, north])
        tas_region.attrs["lon_bounds"] = np.array([west, east])
