from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "schedulers" / "schedule_explore_idyn_matching.sh"


def test_matching_scheduler_is_commit_pinned_and_warning_free():
    text = SCHEDULER.read_text()

    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert "status --porcelain --untracked-files=normal" in text
    assert 'test -s "${INPUT_PATH}"' in text
    assert 'test -s "${SETTINGS_PATH}"' in text
    assert "PYTHONWARNINGS=error" in text


def test_matching_scheduler_stages_all_outputs_before_publication():
    text = SCHEDULER.read_text()

    assert 'STAGED_OUTPUT_DIR="${OUTPUT_DIR}.tmp.${PBS_JOBID}"' in text
    assert '--output-dir "${STAGED_OUTPUT_DIR}"' in text
    for filename in (
        "idyn_population_overview.png",
        "tas_anom_matching_diagnostics.png",
        "covariate_balance_and_sensitivity.png",
        "matching_specification_tradeoff.png",
        "matching_summary.json",
    ):
        assert filename in text
    assert 'test -s "${STAGED_OUTPUT_DIR}/${filename}"' in text
    assert 'mv -f -- "${STAGED_OUTPUT_DIR}/${filename}"' in text


def test_matching_scheduler_uses_right_sized_resources_and_dev_env():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=1:mem=2gb" in text
    assert "#PBS -l walltime=00:10:00" in text
    assert "PBS_O_WORKDIR" in text
    assert 'mamba activate "${VENUS_MAMBA_ENV:-dev_env}"' in text
    assert "OMP_NUM_THREADS=1" in text
