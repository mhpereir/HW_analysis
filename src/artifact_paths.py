"""Checkout-independent locations for prepared HW_analysis products and logs.

Set overrides before starting a CLI. Resolving a path never creates a directory.
The Bash counterpart is config/artifact_paths.sh; tests enforce parity.
"""

from __future__ import annotations

import os
from pathlib import Path


def _storage_root(variable: str, directory: str) -> Path:
    path = Path(os.environ.get(variable, str(Path.home() / "HW-analysis" / directory)))
    if not path.is_absolute():
        raise ValueError(f"{variable} must be an absolute path: {path}")
    resolved = path.resolve()
    if resolved in (Path("/"), Path.home().resolve()):
        raise ValueError(
            f"{variable} must be a dedicated storage directory: {resolved}"
        )
    return resolved


def artifact_root() -> Path:
    """Return the root for both prepared inputs and generated products."""
    return _storage_root("HWA_ARTIFACT_ROOT", "artifacts")


def log_root() -> Path:
    """Return the independent scheduler-log root, a sibling by default."""
    return _storage_root("HWA_LOG_ROOT", "logs")
