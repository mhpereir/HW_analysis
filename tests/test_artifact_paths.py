"""Storage-contract tests using temporary paths, never production data or PBS."""

import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
from HW_analysis.src.artifact_paths import artifact_root, log_root

REPO_ROOT = Path(__file__).resolve().parents[1]
SHELL_CONFIG = REPO_ROOT / "config/artifact_paths.sh"
LEGACY_SCHEDULERS = {
    "schedule_build_stage3_event_feature_pca.sh",
    "schedule_build_stage4_event_feature_clusters.sh",
}
ACTIVE_SCHEDULERS = sorted(
    path
    for directory in (REPO_ROOT / "schedulers", REPO_ROOT / "scripts")
    for path in directory.rglob("*.sh")
    if "#PBS" in path.read_text() and path.name not in LEGACY_SCHEDULERS
)


def _env(tmp_path, **overrides):
    env = os.environ.copy()
    for name in ("PROJECT_ROOT", "REPO_ROOT", "LOG_DIR", "EXPECTED_COMMIT"):
        env.pop(name, None)
    env.update(
        HWA_ARTIFACT_ROOT=str(tmp_path / "artifacts"),
        HWA_LOG_ROOT=str(tmp_path / "logs"),
        MPLCONFIGDIR=str(tmp_path / "matplotlib"),
    )
    env.update(overrides)
    return env


def _shell(body, env, *, cwd=None):
    return subprocess.run(
        [
            "bash",
            "-c",
            'set -euo pipefail; source "$1"; ' + body,
            "test",
            str(SHELL_CONFIG),
        ],
        env=env,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


def test_default_roots_are_siblings_and_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.delenv("HWA_ARTIFACT_ROOT", raising=False)
    monkeypatch.delenv("HWA_LOG_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    expected_artifacts = (Path.home() / "HW-analysis/artifacts").resolve()
    expected_logs = (Path.home() / "HW-analysis/logs").resolve()
    assert artifact_root() == expected_artifacts
    assert log_root() == expected_logs
    env = _env(tmp_path)
    env.pop("HWA_ARTIFACT_ROOT")
    env.pop("HWA_LOG_ROOT")
    result = _shell('printf "%s\n" "$HWA_ARTIFACT_ROOT" "$HWA_LOG_ROOT"', env)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(expected_artifacts), str(expected_logs)]
    assert not list(tmp_path.iterdir())


def test_overrides_are_normalized_without_creating_directories(monkeypatch, tmp_path):
    artifacts = tmp_path / "shared products" / "child" / ".."
    logs = tmp_path / "separate logs"
    monkeypatch.setenv("HWA_ARTIFACT_ROOT", str(artifacts))
    monkeypatch.setenv("HWA_LOG_ROOT", str(logs))
    result = _shell(
        'printf "%s\n" "$HWA_ARTIFACT_ROOT" "$HWA_LOG_ROOT"',
        _env(tmp_path, HWA_ARTIFACT_ROOT=str(artifacts), HWA_LOG_ROOT=str(logs)),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [str(artifact_root()), str(log_root())]
    assert artifact_root() == artifacts.resolve()
    assert log_root() == logs
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "name,resolver", [("HWA_ARTIFACT_ROOT", artifact_root), ("HWA_LOG_ROOT", log_root)]
)
@pytest.mark.parametrize(
    "value", ["", "relative/path", "~/HW-analysis/artifacts", "/", str(Path.home())]
)
def test_invalid_roots_fail_in_python_and_bash(
    name, resolver, value, monkeypatch, tmp_path
):
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        resolver()
    result = _shell(":", _env(tmp_path, **{name: value}))
    assert result.returncode != 0
    assert name in result.stderr
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("name", ["HWA_ARTIFACT_ROOT", "HWA_LOG_ROOT"])
def test_scheduler_storage_root_cannot_be_inside_checkout(name, tmp_path):
    result = _shell(
        ":",
        _env(
            tmp_path, PROJECT_ROOT=str(REPO_ROOT), **{name: str(REPO_ROOT / "results")}
        ),
    )
    assert result.returncode != 0
    assert "outside the source checkout" in result.stderr


@pytest.mark.parametrize("name", ["HWA_ARTIFACT_ROOT", "HWA_LOG_ROOT"])
def test_scheduler_storage_root_cannot_contain_checkout(name, tmp_path):
    result = _shell(
        ":",
        _env(tmp_path, PROJECT_ROOT=str(REPO_ROOT), **{name: str(REPO_ROOT.parent)}),
    )
    assert result.returncode != 0
    assert "must not contain the source checkout" in result.stderr


@pytest.mark.parametrize(
    "case", ["relative", "root", "outside", "lookalike", "traversal", "symlink"]
)
def test_prepared_paths_cannot_escape_artifact_root(case, tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    paths = {
        "relative": "stage1/input.nc",
        "root": str(root),
        "outside": str(tmp_path / "input.nc"),
        "lookalike": str(tmp_path / "artifacts-other" / "input.nc"),
        "traversal": str(root / ".." / "input.nc"),
        "symlink": str(root / "escape" / "input.nc"),
    }
    result = _shell(
        "hwa_validate_artifact_paths INPUT_PATH", _env(tmp_path, INPUT_PATH=paths[case])
    )
    assert result.returncode != 0
    assert "INPUT_PATH must be" in result.stderr


def test_prepared_paths_allow_internal_symlinks_and_future_outputs(tmp_path):
    root = tmp_path / "artifacts"
    root.mkdir()
    (root / "alias").symlink_to(root / "stage1")
    result = _shell(
        'hwa_validate_artifact_paths INPUT_PATH; printf "%s" "$INPUT_PATH"',
        _env(tmp_path, INPUT_PATH=str(root / "alias" / "future.nc")),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout == str(root / "stage1" / "future.nc")
    assert not (root / "stage1").exists()


def test_job_log_is_external_exclusive_and_keeps_preflight_errors(tmp_path):
    env = _env(
        tmp_path, PBS_JOBID="123[4].venus", LOG_DIR=str(tmp_path / "logs" / "attempt")
    )
    result = _shell('hwa_start_log smoke; echo "preflight failure" >&2; exit 7', env)
    log = tmp_path / "logs" / "attempt" / "123[4].venus_smoke.log"
    assert result.returncode == 7
    saved = log.read_bytes()
    assert b"preflight failure" in saved
    assert f"artifact_root={tmp_path / 'artifacts'}".encode() in saved
    assert f"log_root={tmp_path / 'logs'}".encode() in saved
    repeated = _shell('hwa_start_log smoke; echo "must not append"', env)
    assert repeated.returncode != 0
    assert "Cannot create a fresh log" in repeated.stderr
    assert log.read_bytes() == saved
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("case", ["relative", "outside", "traversal", "symlink"])
def test_log_directory_cannot_escape_log_root(case, tmp_path):
    root = tmp_path / "logs"
    root.mkdir()
    (root / "escape").symlink_to(tmp_path)
    paths = {
        "relative": "logs/attempt",
        "outside": str(tmp_path / "artifacts"),
        "traversal": str(root / ".." / "elsewhere"),
        "symlink": str(root / "escape" / "elsewhere"),
    }
    result = _shell(
        "hwa_start_log smoke",
        _env(tmp_path, PBS_JOBID="123.venus", LOG_DIR=paths[case]),
    )
    assert result.returncode != 0
    assert "LOG_DIR must be" in result.stderr
    assert not list(root.rglob("*.log"))


@pytest.mark.parametrize("scheduler", ACTIVE_SCHEDULERS, ids=lambda path: path.name)
def test_every_active_scheduler_uses_shared_paths_and_commit_pin(scheduler):
    text = scheduler.read_text()
    assert 'source "${PROJECT_ROOT}/config/artifact_paths.sh"' in text
    assert 'PROJECT_ROOT="${PROJECT_ROOT:?' in text
    assert 'EXPECTED_COMMIT="${EXPECTED_COMMIT:?' in text
    assert "status --porcelain --untracked-files=normal" in text
    assert "hwa_start_log " in text
    assert "hwa_validate_artifact_paths" in text
    assert "${PROJECT_ROOT}/results" not in text
    assert "${PROJECT_ROOT}/logs" not in text
    assert "/home/mhpereir/HW_analysis" not in text
    assert "#PBS -o /dev/null" in text
    subprocess.run(["bash", "-n", str(scheduler)], check=True)


def _scheduler_env(scheduler, tmp_path):
    env = _env(
        tmp_path,
        PROJECT_ROOT=str(REPO_ROOT),
        EXPECTED_COMMIT="deliberately-wrong-commit-for-local-preflight-only",
        PBS_O_WORKDIR=str(tmp_path),
        PBS_JOBID="synthetic[0].venus",
        PBS_ARRAY_INDEX="2000" if "era5_daily_spatial_data" in scheduler.name else "0",
        REGION="pnw_hotz",
    )
    # Supply required prepared paths, but no data. The invalid commit must stop
    # the scheduler before any scientific environment or producer is invoked.
    for name in re.findall(
        r'^([A-Z_][A-Z0-9_]*)="\$\{\1:\?', scheduler.read_text(), re.MULTILINE
    ):
        if name not in env:
            env[name] = str(tmp_path / "artifacts" / name.lower())
    return env


@pytest.mark.parametrize("scheduler", ACTIVE_SCHEDULERS, ids=lambda path: path.name)
def test_each_scheduler_logs_preflight_outside_checkout_without_computation(
    scheduler, tmp_path
):
    result = subprocess.run(
        ["bash", str(scheduler)],
        env=_scheduler_env(scheduler, tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    logs = list((tmp_path / "logs").glob("*.log"))
    assert len(logs) == 1, result.stdout + result.stderr
    assert f"artifact_root={tmp_path / 'artifacts'}" in logs[0].read_text()
    assert not (tmp_path / "artifacts").exists()
    assert "mamba" not in result.stderr


def test_all_python_product_defaults_honor_root_before_startup(tmp_path):
    modules = []
    for directory in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            if "old" not in path.parts and "artifact_root()" in path.read_text():
                modules.append(
                    ".".join(path.relative_to(REPO_ROOT).with_suffix("").parts)
                )
    program = """
import importlib, json, pathlib, sys
paths = []
for name in json.loads(sys.argv[1]):
    module = importlib.import_module(name)
    for key, value in vars(module).items():
        if key.startswith("DEFAULT_") and isinstance(value, pathlib.Path):
            if any(token in key for token in ("INPUT", "OUTPUT", "DAILY", "RUN_DIR", "CLIMATOLOGY")):
                paths.append([name, key, str(value)])
print(json.dumps(paths))
"""
    result = subprocess.run(
        [sys.executable, "-c", program, json.dumps(modules)],
        env=_env(tmp_path, PYTHONPATH=str(REPO_ROOT)),
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    paths = json.loads(result.stdout)
    assert len(paths) >= 40
    for module, name, value in paths:
        assert Path(value).is_relative_to(tmp_path / "artifacts"), (module, name, value)
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "logs").exists()


def test_active_python_entrypoints_have_no_checkout_results_defaults():
    for directory in (REPO_ROOT / "src", REPO_ROOT / "scripts"):
        for path in directory.rglob("*.py"):
            if "old" not in path.parts:
                assert not re.search(
                    r'REPO_ROOT\s*/\s*["\']results', path.read_text()
                ), path


def test_array_dry_run_forwards_paths_and_commit_without_writes(tmp_path):
    script = REPO_ROOT / "schedulers/submit_era5_daily_spatial_array.sh"
    result = subprocess.run(
        [
            "bash",
            str(script),
            "--start-year",
            "2000",
            "--end-year",
            "2001",
            "--dry-run",
        ],
        env=_env(tmp_path),
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    for name in (
        "PROJECT_ROOT",
        "EXPECTED_COMMIT",
        "HWA_ARTIFACT_ROOT",
        "HWA_LOG_ROOT",
        "OUTPUT_DIR",
        "LOG_DIR",
    ):
        assert f"export {name}=" in result.stdout
    command = next(
        line.removeprefix("[dry-run] ")
        for line in result.stdout.splitlines()
        if "qsub -J" in line
    )
    assert shlex.split(command)[:5] == [
        "qsub",
        "-J",
        "2000-2001%8",
        "-v",
        "PROJECT_ROOT,EXPECTED_COMMIT,HWA_ARTIFACT_ROOT,HWA_LOG_ROOT,OUTPUT_DIR,LOG_DIR",
    ]
    assert str(tmp_path / "artifacts/spatial_composites/daily") in result.stdout
    assert not list(tmp_path.iterdir())


GUARDED_SCHEDULERS = [
    (path, match.group(1))
    for path in ACTIVE_SCHEDULERS
    if (
        match := re.search(
            r"^hwa_validate_artifact_paths ([A-Z_]+)", path.read_text(), re.MULTILINE
        )
    )
]


@pytest.mark.parametrize(
    "scheduler,name",
    GUARDED_SCHEDULERS,
    ids=[path.name for path, _ in GUARDED_SCHEDULERS],
)
def test_each_product_scheduler_rejects_relative_prepared_paths(
    scheduler, name, tmp_path
):
    env = _scheduler_env(scheduler, tmp_path)
    env[name] = "relative/product.nc"
    result = subprocess.run(
        ["bash", str(scheduler)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert (
        f"{name} must be an absolute prepared-product path"
        in result.stdout + result.stderr
    )
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("name", ["REFERENCE_PATH", "OUTPUT_DIR"])
def test_pressure_layer_scheduler_guards_reference_and_output(name, tmp_path):
    scheduler = REPO_ROOT / "schedulers/schedule_build_stage1_pressure_layer.sh"
    env = _scheduler_env(scheduler, tmp_path)
    env[name] = str(tmp_path / "outside-artifact-root" / "product.nc")
    result = subprocess.run(
        ["bash", str(scheduler)],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert f"{name} must be beneath HWA_ARTIFACT_ROOT" in result.stdout + result.stderr
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "outside-artifact-root").exists()


@pytest.mark.parametrize("dirty", [False, True])
def test_array_submitter_passes_exported_paths_only_from_clean_checkout(
    dirty, tmp_path
):
    # These stand-ins test the submission plumbing without invoking PBS or
    # changing Git state. The real checkout/SHA guards are tested above.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_git = bin_dir / "git"
    fake_git.write_text(
        "#!/bin/bash\n"
        'case "$3" in\n'
        '  rev-parse) printf "%s\\n" "0123456789012345678901234567890123456789" ;;\n'
        '  status) printf "%s" "${FAKE_GIT_STATUS}" ;;\n'
        "  *) exit 99 ;;\n"
        "esac\n"
    )
    fake_git.chmod(0o755)
    fake_qsub = bin_dir / "qsub"
    fake_qsub.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys\n"
        "names = sys.argv[sys.argv.index('-v') + 1].split(',')\n"
        "payload = {'argv': sys.argv[1:], 'cwd': os.getcwd(), 'env': {name: os.environ[name] for name in names}}\n"
        "pathlib.Path(os.environ['SUBMISSION_CAPTURE']).write_text(json.dumps(payload))\n"
        "print('synthetic-not-a-real-job')\n"
    )
    fake_qsub.chmod(0o755)
    capture = tmp_path / "submission.json"
    env = _env(
        tmp_path,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        HWA_ARTIFACT_ROOT=str(tmp_path / "shared products"),
        LOG_DIR=str(tmp_path / "logs" / "attempt"),
        FAKE_GIT_STATUS=" M file.py" if dirty else "",
        SUBMISSION_CAPTURE=str(capture),
    )
    result = subprocess.run(
        ["bash", str(REPO_ROOT / "schedulers/submit_era5_daily_spatial_array.sh")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if dirty:
        assert result.returncode != 0
        assert "dirty source checkout" in result.stderr
        assert not capture.exists()
    else:
        assert result.returncode == 0, result.stderr
        record = json.loads(capture.read_text())
        assert record["cwd"] == str(REPO_ROOT)
        assert record["env"]["PROJECT_ROOT"] == str(REPO_ROOT)
        assert (
            record["env"]["EXPECTED_COMMIT"]
            == "0123456789012345678901234567890123456789"
        )
        assert record["env"]["HWA_ARTIFACT_ROOT"] == str(tmp_path / "shared products")
        assert record["env"]["HWA_LOG_ROOT"] == str(tmp_path / "logs")
        assert record["env"]["LOG_DIR"] == str(tmp_path / "logs" / "attempt")
        assert record["env"]["OUTPUT_DIR"] == str(
            tmp_path / "shared products/spatial_composites/daily"
        )
        assert record["argv"][-1] == str(
            REPO_ROOT / "schedulers/schedule_build_era5_daily_spatial_data.sh"
        )
    assert not (tmp_path / "shared products").exists()
    assert not (tmp_path / "logs").exists()


@pytest.mark.parametrize(
    "name", ["HWA_ARTIFACT_ROOT", "HWA_LOG_ROOT", "OUTPUT_DIR", "LOG_DIR"]
)
def test_array_submission_rejects_relative_paths_before_qsub(name, tmp_path):
    result = subprocess.run(
        [
            "bash",
            str(REPO_ROOT / "schedulers/submit_era5_daily_spatial_array.sh"),
            "--dry-run",
        ],
        env=_env(tmp_path, **{name: "relative/path"}),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert name in result.stderr
    assert "qsub -J" not in result.stdout
    assert not list(tmp_path.iterdir())
