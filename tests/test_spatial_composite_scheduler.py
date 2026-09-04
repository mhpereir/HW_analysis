from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULER = (
    REPO_ROOT / "schedulers/schedule_plot_dyn_net_spatial_composites.sh"
)


def test_plot_scheduler_is_commit_verified_and_plot_only():
    text = SCHEDULER.read_text()

    assert "#PBS -l select=1:ncpus=1:mem=8gb" in text
    assert "#PBS -l walltime=00:20:00" in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
    assert 'PBS_O_WORKDIR="${PBS_O_WORKDIR:?PBS_O_WORKDIR is required}"' in text
    assert 'cd "${PBS_O_WORKDIR}"' in text
    assert "plot_dyn_net_spatial_composites.py" in text
    assert "--plot-lags -2 0 2" in text
    assert "build_dyn_net_spatial_composites.py" not in text
