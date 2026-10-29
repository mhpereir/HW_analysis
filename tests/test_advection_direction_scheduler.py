import subprocess
from pathlib import Path

import pytest

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
    assert 'LOG_DIR="${LOG_DIR:-${HWA_LOG_ROOT}}"' in text
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
    assert 'ln -T -- "${STAGED_OUTPUT}" "${OUTPUT_PATH}"' in text
    assert 'ln -T -- "${STAGED_SMOOTHED_OUTPUT}" "${SMOOTHED_OUTPUT_PATH}"' in text
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


@pytest.mark.parametrize("collision", ["hourly", "smoothed"])
def test_publication_collision_preserves_other_attempt(tmp_path, collision):
    """Execute the scheduler with a destination appearing during publication."""
    import os
    import shutil

    repo = tmp_path / "repo"
    (repo / "config").mkdir(parents=True)
    shutil.copyfile(
        REPO_ROOT / "config/artifact_paths.sh", repo / "config/artifact_paths.sh"
    )
    for command in (
        ["init", "-q"],
        ["add", "config"],
        [
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.test",
            "commit",
            "-qm",
            "fixture",
        ],
    ):
        subprocess.run(["git", "-C", str(repo), *command], check=True)
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text("""#!/bin/bash
while [[ "$1" != --output-path ]]; do shift; done
output=$2
printf 'candidate' > "$output"
printf 'candidate' > "${output%.png}_smoothed.png"
""")
    link = fake_bin / "ln"
    link.write_text("""#!/bin/bash
target="${@: -1}"
if [[ "$target" == "$COLLISION_PATH" ]]; then printf 'other attempt' > "$target"; fi
exec /bin/ln "$@"
""")
    python.chmod(0o755)
    link.chmod(0o755)
    mamba = tmp_path / "mamba/etc/profile.d"
    mamba.mkdir(parents=True)
    (mamba / "mamba.sh").write_text(
        'mamba() { export CONDA_PREFIX="$MAMBA_ROOT_PREFIX"; }\n'
    )
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    source = artifacts / "source.nc"
    source.write_bytes(b"fixture")
    output = artifacts / "plot.png"
    smoothed = artifacts / "plot_smoothed.png"
    other = output if collision == "hourly" else smoothed
    owned = smoothed if collision == "hourly" else output
    env = {
        key: value
        for key, value in os.environ.items()
        if key in ("HOME", "LANG", "LC_ALL", "LD_LIBRARY_PATH")
    }
    env.update(
        PATH=f"{fake_bin}:/usr/bin:/bin",
        PBS_O_WORKDIR=str(repo),
        PBS_JOBID="synthetic",
        PROJECT_ROOT=str(repo),
        EXPECTED_COMMIT=commit,
        REGION="pnw_hotz",
        INPUT_PATH=str(source),
        OUTPUT_PATH=str(output),
        HWA_ARTIFACT_ROOT=str(artifacts),
        HWA_LOG_ROOT=str(tmp_path / "logs"),
        MAMBA_ROOT_PREFIX=str(tmp_path / "mamba"),
        COLLISION_PATH=str(other),
    )
    result = subprocess.run(
        ["bash", str(PLOT_SCHEDULER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert other.read_text() == "other attempt"
    assert not owned.exists()
    assert not list(artifacts.glob("*.staging.*"))
