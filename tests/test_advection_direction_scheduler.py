import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PLOT_SCHEDULER = (
    REPO_ROOT / "schedulers" / "schedule_plot_advection_direction_exploration.sh"
)


def test_plot_only_scheduler_reuses_explicit_stage1_without_overwriting():
    text = PLOT_SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=2:mem=8gb" in text
    assert "#PBS -l walltime=00:30:00" in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert 'REGION="${REGION:?REGION is required}"' in text
    assert 'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"' in text
    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
    expected_defaults = {
        "BOTTOM_BOUNDARY": "surface",
        "TOP_BOUNDARY": "700",
        "THRESHOLD_VARIABLE": "tas",
        "QUANTILE": "90",
        "TIME_START": "1940",
        "TIME_END": "2024",
        "WINDOW_DAYS": "7",
    }
    for variable, default in expected_defaults.items():
        assert f'{variable}="${{{variable}:-{default}}}"' in text
    assert 'SMOOTHING_WINDOW="${SMOOTHING_WINDOW:-24}"' in text
    assert (
        'SMOOTHED_OUTPUT_PATH="${OUTPUT_PATH%.*}_smoothed.${OUTPUT_PATH##*.}"' in text
    )
    assert 'LOG_DIR="${LOG_DIR:-${PROJECT_ROOT}/logs}"' in text
    assert '--region "${REGION}"' in text
    expected_arguments = {
        "bottom-boundary": "BOTTOM_BOUNDARY",
        "top-boundary": "TOP_BOUNDARY",
        "threshold-variable": "THRESHOLD_VARIABLE",
        "quantile": "QUANTILE",
        "start-year": "TIME_START",
        "end-year": "TIME_END",
        "window-days": "WINDOW_DAYS",
    }
    for argument, variable in expected_arguments.items():
        assert f'--{argument} "${{{variable}}}"' in text
    assert '--input-path "${INPUT_PATH}"' in text
    assert '--output-path "${STAGED_OUTPUT}"' in text
    assert '--smoothing-window "${SMOOTHING_WINDOW}"' in text
    assert "pnw_bartusek" not in text
    assert 'test ! -e "${OUTPUT_PATH}"' in text
    assert 'test ! -e "${SMOOTHED_OUTPUT_PATH}"' in text
    assert 'mv --no-clobber "${STAGED_OUTPUT}" "${OUTPUT_PATH}"' in text
    assert (
        'mv --no-clobber "${STAGED_SMOOTHED_OUTPUT}" "${SMOOTHED_OUTPUT_PATH}"' in text
    )
    assert 'echo "[info] output_sha256=' in text
    assert 'echo "[info] smoothed_output_sha256=' in text
    assert "build_stage1" not in text
    subprocess.run(["bash", "-n", str(PLOT_SCHEDULER)], check=True)


def test_combined_advection_direction_pipeline_entrypoints_are_absent():
    assert not (
        REPO_ROOT / "schedulers" / "schedule_advection_direction_exploration.sh"
    ).exists()
    assert not (
        REPO_ROOT / "schedulers" / "schedule_advection_direction_exploration_smoke.sh"
    ).exists()
    assert not (REPO_ROOT / "scripts" / "run_advection_direction_pipeline.sh").exists()
