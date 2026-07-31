"""Pytest configuration for package-style test imports."""

from __future__ import annotations

from importlib.machinery import ModuleSpec
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_NAME = "HW_analysis"

package = ModuleType(PACKAGE_NAME)
package.__path__ = [str(REPO_ROOT)]
package.__package__ = PACKAGE_NAME
package.__spec__ = ModuleSpec(PACKAGE_NAME, loader=None, is_package=True)
sys.modules[PACKAGE_NAME] = package
