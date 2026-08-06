import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = (
    REPO_ROOT / "schedulers" / "schedule_build_stage1_hourly_climatology.sh",
    REPO_ROOT / "schedulers" / "schedule_plot_composite_timeseries_all_clim_anom.sh",
    REPO_ROOT / "schedulers" / "schedule_plot_composite_timeseries_split_clim_anom.sh",
    REPO_ROOT
    / "schedulers"
    / "schedule_plot_advection_direction_exploration_clim_anom.sh",
)


def test_climatology_schedulers_are_commit_verified_and_syntax_valid():
    for scheduler in SCHEDULERS:
        text = scheduler.read_text()
        assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
        assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
        assert "status --porcelain --untracked-files=normal" in text
        assert "mamba activate \"${VENUS_MAMBA_ENV:-dev_env}\"" in text
        subprocess.run(["bash", "-n", str(scheduler)], check=True)


def test_climatology_builder_requires_explicit_products():
    text = SCHEDULERS[0].read_text()

    assert 'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"' in text
    assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
    assert "build_stage1_hourly_climatology.py" in text
    assert "refusing to overwrite" in text


def test_anomaly_plot_schedulers_require_stage1_climatology_and_output():
    for scheduler in SCHEDULERS[1:]:
        text = scheduler.read_text()
        assert 'INPUT_PATH="${INPUT_PATH:?INPUT_PATH is required}"' in text
        assert (
            'CLIMATOLOGY_PATH="${CLIMATOLOGY_PATH:?CLIMATOLOGY_PATH is required}"'
            in text
        )
        assert 'OUTPUT_PATH="${OUTPUT_PATH:?OUTPUT_PATH is required}"' in text
        assert "--input-path \"${INPUT_PATH}\"" in text
        assert "--climatology-path \"${CLIMATOLOGY_PATH}\"" in text
