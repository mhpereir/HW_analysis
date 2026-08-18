from pathlib import Path

from HW_analysis.src import config


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    REPO_ROOT / "schedulers" / "schedule_build_stage1_harmonized_timeseries.sh"
)


def test_eastern_canada_region_matches_ehb_domain():
    lat, lon = config.REGIONS["eastern_canada"]

    assert (lat.start, lat.stop) == (42, 52)
    assert (lon.start, lon.stop) == (-83, -73)


def test_stage1_scheduler_requires_commit_region_and_output_path():
    text = SCHEDULER.read_text()

    assert "PROJECT_ROOT must be supplied" in text
    assert "EXPECTED_COMMIT must be supplied" in text
    assert "REGION must be supplied" in text
    assert "OUTPUT_PATH must be supplied" in text
    assert "status --porcelain --untracked-files=normal" in text


def test_stage1_scheduler_requires_global_cloud_cover_and_full_diagnostics():
    text = SCHEDULER.read_text()

    assert '--add-full-diagnostics \\\n' in text
    assert '--cloud-cover-source-layout "global-hourly-grid"' in text
    assert "--cloud-cover-root" in text
    assert "stage1_contract_version=2" in text
    assert "face_advection_contributions=enabled" in text
    assert "pbl_diagnostics=excluded" in text
    assert "pbl" not in text.lower().replace("pbl_diagnostics=excluded", "")
    assert "legacy-regional" not in text


def test_stage1_scheduler_declares_production_resources():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=8:mem=85gb" in text
    assert "#PBS -l walltime=24:00:00" in text
