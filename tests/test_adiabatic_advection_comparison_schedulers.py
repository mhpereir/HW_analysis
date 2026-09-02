import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = {
    "event": REPO_ROOT
    / "schedulers"
    / "schedule_plot_adiabatic_advection_comparison.sh",
    "baseline_net": REPO_ROOT
    / "schedulers"
    / "schedule_plot_adiabatic_advection_comparison_baseline.sh",
    "baseline_diabatic": REPO_ROOT
    / "schedulers"
    / "schedule_plot_adiabatic_diabatic_advection_baseline.sh",
}
PRESENTATION_SCHEDULERS = (
    SCHEDULERS["event"],
    SCHEDULERS["baseline_net"],
)


@pytest.mark.parametrize("scheduler", SCHEDULERS.values(), ids=SCHEDULERS.keys())
def test_scheduler_is_commit_pinned_warning_free_and_syntax_valid(scheduler):
    text = scheduler.read_text()

    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert 'test "${actual_commit}" = "${EXPECTED_COMMIT}"' in text
    assert "status --porcelain --untracked-files=normal" in text
    assert "export PYTHONWARNINGS=error" in text
    assert 'mamba activate "${VENUS_MAMBA_ENV:-dev_env}"' in text
    assert 'cd "${PROJECT_ROOT}"' in text
    subprocess.run(["bash", "-n", str(scheduler)], check=True)


@pytest.mark.parametrize("scheduler", SCHEDULERS.values(), ids=SCHEDULERS.keys())
def test_scheduler_uses_bounded_serial_resources_and_isolated_logs(scheduler):
    text = scheduler.read_text()

    assert "#PBS -l select=1:ncpus=1:mem=4gb" in text
    assert "#PBS -l walltime=00:15:00" in text
    assert 'LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"' in text
    assert 'LOGFILE="${LOG_DIR}/${PBS_JOBID}_' in text
    assert "export OMP_NUM_THREADS=1" in text
    assert "export MKL_NUM_THREADS=1" in text
    assert "export OPENBLAS_NUM_THREADS=1" in text
    assert "export NUMEXPR_NUM_THREADS=1" in text


def test_event_scheduler_requires_explicit_non_overwriting_paths():
    text = SCHEDULERS["event"].read_text()

    assert (
        'EVENT_INPUT_PATH="${EVENT_INPUT_PATH:?EVENT_INPUT_PATH is required}"' in text
    )
    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
    assert 'test -s "${EVENT_INPUT_PATH}"' in text
    assert 'test ! -e "${OUTPUT_PATH}"' in text
    assert '--input-path "${EVENT_INPUT_PATH}"' in text
    assert '--output-path "${OUTPUT_PATH}"' in text
    assert 'test -s "${OUTPUT_PATH}"' in text


@pytest.mark.parametrize("scheduler", PRESENTATION_SCHEDULERS)
def test_four_panel_schedulers_forward_the_requested_layout(scheduler):
    text = scheduler.read_text()

    assert 'LAYOUT="${LAYOUT:-full}"' in text
    assert 'echo "[info] layout=${LAYOUT}"' in text
    assert '--layout "${LAYOUT}"' in text


@pytest.mark.parametrize(
    "scheduler",
    (SCHEDULERS["baseline_net"], SCHEDULERS["baseline_diabatic"]),
)
def test_baseline_scheduler_requires_matching_inputs_and_non_overwriting_output(
    scheduler,
):
    text = scheduler.read_text()

    assert (
        'BASELINE_INPUT_PATH="${BASELINE_INPUT_PATH:?BASELINE_INPUT_PATH is required}"'
        in text
    )
    assert (
        'EVENT_INPUT_PATH="${EVENT_INPUT_PATH:?EVENT_INPUT_PATH is required}"' in text
    )
    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
    assert 'test -s "${BASELINE_INPUT_PATH}"' in text
    assert 'test -s "${EVENT_INPUT_PATH}"' in text
    assert 'test ! -e "${OUTPUT_PATH}"' in text
    assert '--input-path "${BASELINE_INPUT_PATH}"' in text
    assert '--event-input-path "${EVENT_INPUT_PATH}"' in text
    assert '--output-path "${OUTPUT_PATH}"' in text
    assert 'COLOR_VARIABLE="${COLOR_VARIABLE:-tas_anom_peak}"' in text
    assert '--color-variable "${COLOR_VARIABLE}"' in text
    assert 'test -s "${OUTPUT_PATH}"' in text
