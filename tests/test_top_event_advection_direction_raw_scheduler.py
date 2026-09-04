import stat
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    REPO_ROOT
    / "schedulers"
    / "schedule_plot_advection_direction_exploration_top_events.sh"
)


def test_raw_top_event_scheduler_is_commit_verified_and_atomic():
    text = SCHEDULER.read_text()

    assert SCHEDULER.stat().st_mode & stat.S_IXUSR
    assert "#PBS -l select=1:ncpus=1:mem=4gb" in text
    assert "#PBS -l walltime=00:10:00" in text
    for required in (
        'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"',
        'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"',
        'REGION="${REGION:?REGION is required}"',
        'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"',
        'OUTPUT_DIR="${OUTPUT_DIR:?OUTPUT_DIR is required}"',
    ):
        assert required in text
    assert "CLIMATOLOGY_PATH" not in text
    assert "status --porcelain --untracked-files=normal" in text
    assert 'mamba activate "${VENUS_MAMBA_ENV:-dev_env}"' in text
    assert "export PYTHONWARNINGS=error" in text
    assert "python_executable=$(command -v python)" in text
    assert '/usr/bin/time -v "${python_executable}"' in text
    assert "plot_advection_direction_exploration_top_events.py" in text
    assert "--season-months 6 7 8" in text
    assert "--require-full-event" in text
    assert 'test ! -e "${OUTPUT_DIR}"' in text
    assert 'STAGING_DIR="${OUTPUT_DIR}.staging.${PBS_JOBID}"' in text
    assert 'mv --no-clobber -T "${STAGING_DIR}" "${OUTPUT_DIR}"' in text
    assert 'test ! -e "${STAGING_DIR}"' in text
    assert "expected_png_count=$((2 * TOP_N))" in text
    assert 'test "${actual_png_count}" -eq "${expected_png_count}"' in text
    assert 'test "${empty_png_count}" -eq 0' in text
    subprocess.run(["bash", "-n", str(SCHEDULER)], check=True)


def test_raw_top_event_scheduler_accepts_runtime_plot_configuration():
    text = SCHEDULER.read_text()
    expected_defaults = {
        "BOTTOM_BOUNDARY": "surface",
        "TOP_BOUNDARY": "700",
        "THRESHOLD_VARIABLE": "tas",
        "QUANTILE": "90",
        "TIME_START": "1940",
        "TIME_END": "2024",
        "TOP_N": "10",
        "WINDOW_DAYS": "7",
        "SMOOTHING_WINDOW": "24",
    }
    for variable, default in expected_defaults.items():
        assert f'{variable}="${{{variable}:-{default}}}"' in text

    expected_arguments = {
        "region": "REGION",
        "bottom-boundary": "BOTTOM_BOUNDARY",
        "top-boundary": "TOP_BOUNDARY",
        "threshold-variable": "THRESHOLD_VARIABLE",
        "quantile": "QUANTILE",
        "start-year": "TIME_START",
        "end-year": "TIME_END",
        "top-n": "TOP_N",
        "window-days": "WINDOW_DAYS",
        "smoothing-window": "SMOOTHING_WINDOW",
    }
    for argument, variable in expected_arguments.items():
        assert f'--{argument} "${{{variable}}}"' in text
    assert '--input-path "${INPUT_PATH}"' in text
    assert '--output-dir "${STAGING_DIR}"' in text
    assert "pnw_bartusek" not in text
