from pathlib import Path

import pytest
from HW_analysis.src import config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    REPO_ROOT / "schedulers" / "schedule_build_stage1_harmonized_timeseries.sh"
)


def test_eastern_canada_region_matches_ehb_domain():
    lat, lon = config.REGIONS["eastern_canada"]

    assert (lat.start, lat.stop) == (42, 52)
    assert (lon.start, lon.stop) == (-83, -73)


@pytest.mark.parametrize(
    ("region", "expected_lat", "expected_lon"),
    [
        ("alaska", (59.5, 69.5), (-160, -150)),
        ("central_usa", (36, 46), (-105, -95)),
        ("gulf_usa", (31, 41), (-90, -80)),
        ("western_eu", (43, 53), (-2, 8)),
        ("central_china", (25, 35), (105, 115)),
        ("pnw_hotz", (49, 59), (-125, -115)),
    ],
)
def test_new_stage1_regions_match_ehb_campaign_domains(
    region, expected_lat, expected_lon
):
    lat, lon = config.REGIONS[region]

    assert (lat.start, lat.stop) == expected_lat
    assert (lon.start, lon.stop) == expected_lon


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


def test_stage1_scheduler_supports_explicit_threshold_variable_selection():
    text = SCHEDULER.read_text()

    assert 'THRESHOLD_VARIABLE="${THRESHOLD_VARIABLE:-tas}"' in text
    assert 'tas|lwa|lwa_a|lwa_c)' in text
    assert 'threshold_variable=${THRESHOLD_VARIABLE}' in text
    assert '--threshold-variable "${THRESHOLD_VARIABLE}"' in text


def test_stage1_scheduler_supports_explicit_heat_budget_root():
    text = SCHEDULER.read_text()

    assert 'HEAT_BUDGET_ROOT="${HEAT_BUDGET_ROOT:-}"' in text
    assert 'HEAT_BUDGET_ARGS=(--heat-budget-root "${HEAT_BUDGET_ROOT}")' in text
    assert 'heat_budget_root=${HEAT_BUDGET_ROOT:-configured-saved-results-default}' in text
    assert '"${HEAT_BUDGET_ARGS[@]}"' in text


def test_stage1_scheduler_declares_production_resources():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=8:mem=85gb" in text
    assert "#PBS -l walltime=24:00:00" in text
