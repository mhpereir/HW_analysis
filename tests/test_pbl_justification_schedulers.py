from __future__ import annotations

import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEDULERS = (
    REPO_ROOT / "schedulers/schedule_build_pbl_700hpa_justification.sh",
    REPO_ROOT / "schedulers/schedule_plot_pbl_700hpa_justification.sh",
)


def test_pbl_schedulers_have_valid_bash_syntax():
    subprocess.run(["bash", "-n", *map(str, SCHEDULERS)], check=True)


def test_pbl_schedulers_require_commit_pinned_clean_checkout():
    for path in SCHEDULERS:
        text = path.read_text()
        assert (
            'EXPECTED_COMMIT="${EXPECTED_COMMIT:?EXPECTED_COMMIT is required}"' in text
        )
        assert "status --porcelain --untracked-files=normal" in text
        assert 'realpath "${PBS_O_WORKDIR}"' in text
        assert "refusing to overwrite existing output" in text


def test_build_scheduler_uses_current_outputs_path_via_explicit_root():
    text = SCHEDULERS[0].read_text()
    assert 'PBL_ROOT="${PBL_ROOT:?PBL_ROOT is required}"' in text
    assert '--pbl-root "${PBL_ROOT}"' in text
    assert "PBL_download/_old" not in text
