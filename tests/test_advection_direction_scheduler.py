import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "schedulers" / "schedule_advection_direction_exploration.sh"
SMOKE_SCHEDULER = (
    REPO_ROOT / "schedulers" / "schedule_advection_direction_exploration_smoke.sh"
)
PIPELINE_RUNNER = REPO_ROOT / "scripts" / "run_advection_direction_pipeline.sh"


def test_advection_direction_scheduler_has_commit_and_resource_guards():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=8:mem=48gb" in text
    assert "#PBS -l walltime=08:00:00" in text
    assert "#PBS -V" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert 'cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"' in text
    assert "run_advection_direction_pipeline.sh" in text
    assert "base_stage1" in text
    assert "advection_direction_exploration" in text


def test_advection_direction_scheduler_passes_bash_syntax_check():
    subprocess.run(["bash", "-n", str(SCHEDULER)], check=True)


def test_advection_direction_smoke_scheduler_is_commit_guarded_and_isolated():
    text = SMOKE_SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=4:mem=24gb" in text
    assert "#PBS -l walltime=00:45:00" in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert "smoke_2024" in text
    assert "run_advection_direction_pipeline.sh" in text
    subprocess.run(["bash", "-n", str(SMOKE_SCHEDULER)], check=True)


def test_advection_direction_pipeline_runner_uses_explicit_legacy_provenance():
    text = PIPELINE_RUNNER.read_text()

    assert "build_stage1_harmonized_timeseries.py" in text
    assert "--add-full-diagnostics" in text
    assert "--cloud-cover-source-layout legacy-regional" in text
    assert '--cloud-cover-root "${LEGACY_CLOUD_ROOT}"' in text
    assert "build_stage1_advection_exploration.py" in text
    assert "plot_advection_direction_exploration.py" in text
    subprocess.run(["bash", "-n", str(PIPELINE_RUNNER)], check=True)
