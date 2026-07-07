"""Pytest configuration for package-style test imports."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_PARENT = Path(__file__).resolve().parents[2]
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))
