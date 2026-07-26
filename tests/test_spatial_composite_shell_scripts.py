import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DAILY_SCRIPT = REPO_ROOT / "scripts/spatial_composites/build_era5_daily_spatial_data.sh"
CLIMATE_SCRIPT = REPO_ROOT / "scripts/spatial_composites/build_era5_daily_doy_climatology.sh"
ARRAY_SUBMIT_SCRIPT = REPO_ROOT / "schedulers/submit_era5_daily_spatial_array.sh"
DAILY_SCHEDULER = REPO_ROOT / "schedulers/schedule_build_era5_daily_spatial_data.sh"


def test_daily_script_dry_run_builds_expected_cdo_commands(tmp_path):
    t2m_root = tmp_path / "t2m"
    z500_root = tmp_path / "z500"
    output_dir = tmp_path / "daily"
    t2m_root.mkdir()
    z500_root.mkdir()
    (t2m_root / "2mT_hour_ERA5_2000.nc").touch()
    (z500_root / "z500_hour_ERA5_2000.nc").touch()

    result = subprocess.run(
        [
            str(DAILY_SCRIPT),
            "--t2m-root",
            str(t2m_root),
            "--z500-root",
            str(z500_root),
            "--output-dir",
            str(output_dir),
            "--start-year",
            "2000",
            "--end-year",
            "2000",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "daymean" in result.stdout
    assert "sellevel\\,500" in result.stdout
    assert "merge" in result.stdout
    assert result.stdout.count("nccopy") == 2
    assert "valid_time/24\\,latitude/180\\,longitude/180" in result.stdout
    assert (
        "valid_time/24\\,pressure_level/1\\,latitude/180\\,longitude/180"
        in result.stdout
    )
    assert "t2m_hourly_rechunked_2000.nc" in result.stdout
    assert "z500_hourly_rechunked_2000.nc" in result.stdout
    assert result.stdout.count("rm -f") == 2
    assert not output_dir.exists()


def test_climatology_script_dry_run_uses_daily_ydaymean(tmp_path):
    daily_dir = tmp_path / "daily"
    daily_dir.mkdir()
    (daily_dir / "ERA5_daily_t2m_z500_2000.nc").touch()
    output_path = tmp_path / "climatology.nc"

    result = subprocess.run(
        [
            str(CLIMATE_SCRIPT),
            "--daily-dir",
            str(daily_dir),
            "--output-path",
            str(output_path),
            "--start-year",
            "2000",
            "--end-year",
            "2000",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "ydaymean" in result.stdout
    assert "mergetime" in result.stdout
    assert not output_path.exists()


def test_daily_script_refuses_existing_output_without_overwrite(tmp_path):
    t2m_root = tmp_path / "t2m"
    z500_root = tmp_path / "z500"
    output_dir = tmp_path / "daily"
    t2m_root.mkdir()
    z500_root.mkdir()
    output_dir.mkdir()
    (t2m_root / "2mT_hour_ERA5_2000.nc").touch()
    (z500_root / "z500_hour_ERA5_2000.nc").touch()
    (output_dir / "ERA5_daily_t2m_z500_2000.nc").touch()

    result = subprocess.run(
        [
            str(DAILY_SCRIPT),
            "--t2m-root",
            str(t2m_root),
            "--z500-root",
            str(z500_root),
            "--output-dir",
            str(output_dir),
            "--start-year",
            "2000",
            "--end-year",
            "2000",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Output exists" in result.stderr


def test_daily_script_can_skip_existing_output(tmp_path):
    t2m_root = tmp_path / "t2m"
    z500_root = tmp_path / "z500"
    output_dir = tmp_path / "daily"
    t2m_root.mkdir()
    z500_root.mkdir()
    output_dir.mkdir()
    (t2m_root / "2mT_hour_ERA5_2000.nc").touch()
    (z500_root / "z500_hour_ERA5_2000.nc").touch()
    output_path = output_dir / "ERA5_daily_t2m_z500_2000.nc"
    output_path.touch()

    result = subprocess.run(
        [
            str(DAILY_SCRIPT),
            "--t2m-root",
            str(t2m_root),
            "--z500-root",
            str(z500_root),
            "--output-dir",
            str(output_dir),
            "--start-year",
            "2000",
            "--end-year",
            "2000",
            "--skip-existing",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert f"Skipping existing output: {output_path}" in result.stdout
    assert "daymean" not in result.stdout


def test_array_submitter_builds_throttled_year_range():
    result = subprocess.run(
        [
            str(ARRAY_SUBMIT_SCRIPT),
            "--start-year",
            "2000",
            "--end-year",
            "2004",
            "--max-concurrent",
            "3",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "qsub -J 2000-2004%3" in result.stdout
    assert "schedule_build_era5_daily_spatial_data.sh" in result.stdout


def test_daily_scheduler_is_pinned_to_venus05():
    scheduler = DAILY_SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=1:mem=2gb:host=venus05" in scheduler
    assert "EXPECTED_HOST=venus05" in scheduler
    assert 'hostname -s' in scheduler
