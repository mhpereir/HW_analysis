import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = REPO_ROOT / "schedulers" / "schedule_advection_direction_exploration.sh"


def test_advection_direction_scheduler_has_commit_and_resource_guards():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=4:mem=32gb" in text
    assert "#PBS -l walltime=01:00:00" in text
    assert "#PBS -V" not in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert 'cd "${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"' in text
    assert "build_stage1_advection_exploration.py" in text
    assert "plot_advection_direction_exploration.py" in text
    assert "advection_direction_exploration" in text


def test_advection_direction_scheduler_passes_bash_syntax_check():
    subprocess.run(["bash", "-n", str(SCHEDULER)], check=True)
