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


def test_presentation_smoke_is_pinned_and_runs_export_regressions():
    scheduler = REPO_ROOT / "schedulers" / "schedule_presentation_feature_smoke.sh"
    test_scheduler_is_commit_pinned_warning_free_and_syntax_valid(scheduler)
    text = scheduler.read_text()
    assert "#PBS -l select=1:ncpus=1:mem=4gb" in text
    assert "#PBS -l walltime=00:10:00" in text
    assert "-m pytest -q -W error" in text
    assert 'test ! -e "${LOGFILE}"' in text
    for name in (
        "test_plot_style.py",
        "test_presentation_budget_comparisons.py",
        "test_selectors.py",
        "test_plot_adiabatic_advection_comparison.py",
        "test_plot_adiabatic_advection_comparison_baseline.py",
    ):
        assert f"tests/{name}" in text


@pytest.mark.parametrize("scheduler", PRESENTATION_SCHEDULERS)
@pytest.mark.parametrize("layout", ["full", "presentation"])
@pytest.mark.parametrize("override", [False, True])
def test_scheduler_marker_defaults_and_overrides_match_layout(
    scheduler, layout, override, tmp_path
):
    # Evaluate only local variable declarations, stopping before Git, files,
    # environments or job execution. No scheduler or production input is used.
    declarations = scheduler.read_text().split("\nactual_commit=", 1)[0]
    env = {
        "PBS_O_WORKDIR": str(tmp_path),
        "PROJECT_ROOT": str(REPO_ROOT),
        "EXPECTED_COMMIT": "synthetic-not-a-deployment",
        "EVENT_INPUT_PATH": "events.nc",
        "BASELINE_INPUT_PATH": "baseline.nc",
        "OUTPUT_PATH": "plot.png",
        "LAYOUT": layout,
    }
    prefix = "" if scheduler == SCHEDULERS["event"] else "EVENT_"
    if override:
        env[f"{prefix}POINT_SIZE"] = "17.0"
        env[f"{prefix}ALPHA"] = "0.4"
    result = subprocess.run(
        [
            "bash",
            "-c",
            declarations
            + f'\nprintf "%s %s" "${{{prefix}POINT_SIZE}}" "${{{prefix}ALPHA}}"',
        ],
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    expected = (
        "17.0 0.4"
        if override
        else "40.0 0.9"
        if layout == "presentation"
        else "24.0 0.7"
    )
    assert result.stdout == expected


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
