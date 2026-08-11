from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = {
    "event": REPO_ROOT / "schedulers" / "schedule_build_stage2_event_features.sh",
    "baseline": (
        REPO_ROOT / "schedulers" / "schedule_build_stage2_baseline_features.sh"
    ),
}


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_stage2_scheduler_is_commit_pinned_and_publishes_atomically(kind):
    text = SCHEDULERS[kind].read_text()

    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert "status --porcelain --untracked-files=normal" in text
    assert 'test -s "${INPUT_PATH}"' in text
    assert 'STAGED_OUTPUT_PATH="${OUTPUT_PATH}.tmp.${PBS_JOBID}"' in text
    assert 'trap \'rm -f -- "${STAGED_OUTPUT_PATH}"\' EXIT' in text
    assert '--output-path "${STAGED_OUTPUT_PATH}"' in text
    assert 'mv -f -- "${STAGED_OUTPUT_PATH}" "${OUTPUT_PATH}"' in text
    assert "--overwrite" not in text


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_stage2_scheduler_declares_right_sized_resources_and_environment(kind):
    text = SCHEDULERS[kind].read_text()

    assert "#PBS -l select=1:ncpus=1:mem=2gb" in text
    assert "#PBS -l walltime=00:10:00" in text
    assert "PBS_O_WORKDIR" in text
    assert 'mamba activate "${VENUS_MAMBA_ENV:-dev_env}"' in text
    assert "OMP_NUM_THREADS=1" in text


@pytest.mark.parametrize("kind", SCHEDULERS)
def test_stage2_scheduler_builds_tas_and_lwa_a_array_variants(kind):
    text = SCHEDULERS[kind].read_text()

    assert "#PBS -J 0-1" in text
    assert "THRESHOLD_VARIABLES=(tas lwa_a)" in text
    assert "PBS_ARRAY_INDEX" in text
    assert '[[ ! "${ARRAY_INDEX}" =~ ^[0-9]+$ ]]' in text
    assert 'THRESHOLD_VARIABLE="${THRESHOLD_VARIABLES[ARRAY_INDEX]}"' in text


def test_event_scheduler_preserves_canonical_pnw_selection():
    text = SCHEDULERS["event"].read_text()

    assert "hw_event_features_fixed_windows_${REGION}" in text
    assert "--season-months 6 7 8" in text
    assert "--require-full-event" in text


def test_baseline_scheduler_preserves_canonical_pnw_selection():
    text = SCHEDULERS["baseline"].read_text()

    assert "non_event_day_features_fixed_windows_${REGION}" in text
    assert "--season-months 6 7 8" in text
