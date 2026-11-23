"""Writable user locations and installed application resources."""

from __future__ import annotations

import os
import sys
from importlib import resources
from pathlib import Path
from typing import Optional


def app_home() -> Path:
    """An explicit override, Windows LocalAppData, or the existing Unix location."""
    override = os.environ.get("SPECTRAFIT_HOME")
    if override:
        return Path(override).expanduser().resolve()
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        return local / "SpectraFit"
    return Path.home() / ".spectrafit"


def chromophore_dirs() -> list[Path]:
    """Read old Windows libraries too; new/explicit locations take precedence."""
    current = app_home() / "chromophores"
    legacy = Path.home() / ".spectrafit" / "chromophores"
    if sys.platform == "win32" and not os.environ.get("SPECTRAFIT_HOME") and legacy != current:
        return [legacy, current]
    return [current]


def find_guide() -> Optional[Path]:
    """Prefer the guide shipped with this package, including frozen distributions."""
    packaged = Path(str(resources.files("spectrafit") / "data" / "README.html"))
    source = Path(__file__).resolve().parents[2] / "README.html"
    for path in (packaged, app_home() / "README.html", Path.home() / ".spectrafit" / "README.html", source):
        if path.is_file():
            return path
    return None
