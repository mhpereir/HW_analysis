import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCHEDULER = (
    REPO_ROOT / "schedulers" / "schedule_plot_advection_direction_exploration.sh"
)


def test_plot_only_scheduler_reuses_enhanced_stage1_without_overwriting():
    text = PLOT_SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=2:mem=8gb" in text
    assert "#PBS -l walltime=00:30:00" in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert "harmonized_regional_timeseries_pnw_bartusek" in text
    assert "advection_face_contributions_two_panel.png" in text
    assert "refusing to overwrite existing output" in text
    assert "build_stage1" not in text
    subprocess.run(["bash", "-n", str(PLOT_SCHEDULER)], check=True)


def test_combined_advection_direction_pipeline_entrypoints_are_absent():
    assert not (
        REPO_ROOT
        / "schedulers"
        / "schedule_advection_direction_exploration.sh"
    ).exists()
    assert not (
        REPO_ROOT
        / "schedulers"
        / "schedule_advection_direction_exploration_smoke.sh"
    ).exists()
    assert not (
        REPO_ROOT / "scripts" / "run_advection_direction_pipeline.sh"
    ).exists()
