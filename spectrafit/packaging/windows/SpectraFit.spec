"""One-folder Windows distribution with shared GUI/CLI payload and a separate probe."""

import os
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_entry_point, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parents[1]
os.environ["QT_API"] = "pyqt6"


def application_module(name):
    return not any(part in {"tests", "_tests", "testing"} for part in name.split("."))


datas = collect_data_files("spectrafit")
hidden = ["vispy.app.backends._pyqt6", "vispy.gloo.gl._gl2", "OpenGL.platform.win32",
          "magicgui.backends._qtpy", "PyQt6.QtOpenGL", "PyQt6.QtOpenGLWidgets"]
for package in ("napari", "napari_builtins", "napari_console", "napari_svg", "vispy", "npe2", "napari_resources"):
    datas += collect_data_files(package, excludes=["**/tests/**", "**/_tests/**"])
for package in ("napari", "napari_builtins", "napari_console", "napari_svg"):
    hidden += collect_submodules(package, filter=application_module)
plugin_data, plugin_imports = collect_entry_point("napari.manifest")
datas += plugin_data
hidden += plugin_imports
for distribution in ("spectrafit", "napari", "vispy", "PyQt6", "qtpy", "npe2", "app-model", "magicgui"):
    datas += copy_metadata(distribution, recursive=True)
datas += [(str(ROOT / "README.html"), "."), (str(ROOT / "LICENSE"), "."),
          (str(Path(SPECPATH) / "PORTABLE-README.txt"), ".")]

entries = {"gui": ROOT / "launchers" / "spectrafit_gui.pyw",
           "cli": ROOT / "launchers" / "spectrafit_cli.py",
           "probe": Path(SPECPATH) / "frozen_probe.py"}
a = Analysis([str(path) for path in entries.values()], pathex=[str(ROOT / "src")],
             binaries=[], datas=datas, hiddenimports=sorted(set(hidden)),
             runtime_hooks=[str(Path(SPECPATH) / "runtime_qt.py")],
             hooksconfig={"matplotlib": {"backends": ["QtAgg"]}},
             excludes=["PyQt5", "PySide2", "PySide6", "tkinter", "pytest", "_pytest"],
             noarchive=False)

# Every library must come from this Python installation, its environment or Windows itself.
# Anything else was found through PATH (for example a Java runtime's freetype.dll and
# ucrtbase.dll) and depends on whatever happens to be installed on the build computer.
allowed_roots = [Path(root).resolve() for root in {sys.base_prefix, sys.prefix}]
allowed_roots.append(Path(os.environ.get("SystemRoot", r"C:\Windows")).resolve())
foreign = sorted(f"{dest} <- {src}" for dest, src, _ in a.binaries
                 if not any(Path(src).resolve().is_relative_to(root) for root in allowed_roots))
if foreign:
    raise SystemExit("Libraries from outside the build environment were collected:\n  " + "\n  ".join(foreign)
                     + "\nRemove their folders from PATH, or build with build_windows.py, which uses a clean PATH.")

# Installer records such as direct_url.json contain the build computer's folder (and user name).
# They are not needed at run time.
BUILD_RECORDS = {"direct_url.json", "uv_cache.json", "uv_build.json"}
a.datas = [entry for entry in a.datas
           if not (Path(entry[0]).parent.name.endswith(".dist-info") and Path(entry[0]).name in BUILD_RECORDS)]
pyz = PYZ(a.pure)

# Each executable runs the shared runtime hooks and exactly its own entry script.
# The probe's test code is an entry script, not a production application feature.
entry_paths = {str(path.resolve()) for path in entries.values()}
runtime_scripts = [item for item in a.scripts if str(Path(item[1]).resolve()) not in entry_paths]


def executable(key, name, console):
    scripts = [item for item in a.scripts if Path(item[1]).resolve() == entries[key].resolve()]
    assert len(scripts) == 1, (key, scripts)
    return EXE(pyz, runtime_scripts + scripts, [("X utf8=1", None, "OPTION")], exclude_binaries=True, name=name,
               debug=False, strip=False, upx=False, console=console, disable_windowed_traceback=False)


gui = executable("gui", "SpectraFit", False)
cli = executable("cli", "SpectraFit-cli", True)
probe = executable("probe", "SpectraFit-verify", False)
COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name="SpectraFit")
# This directory is for verification only and is excluded from the delivered ZIP.
COLLECT(probe, a.binaries, a.datas, strip=False, upx=False, name="SpectraFit-verify")
