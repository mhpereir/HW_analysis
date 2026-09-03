from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    REPO_ROOT / "schedulers/schedule_matched_dyn_pre_spatial_composites.sh"
)


def test_matched_spatial_scheduler_is_commit_pinned_and_atomic():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=4:mem=32gb" in text
    assert "#PBS -l walltime=02:00:00" in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert 'PBS_O_WORKDIR:?PBS_O_WORKDIR is required' in text
    assert "PYTHONWARNINGS=error" in text
    assert "peak_anomaly_0p20" in text
    assert "build_matched_dyn_pre_spatial_composites.py" in text
    assert "plot_matched_dyn_pre_spatial_composites.py" in text
    assert "COMPOSITE_STAGING_DIR" in text
    assert "FIGURE_STAGING_DIR" in text
    assert "--overwrite" not in text
