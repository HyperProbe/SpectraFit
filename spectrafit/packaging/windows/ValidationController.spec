"""Small standard-library-only, single-file validation runner."""

from pathlib import Path

root = Path(SPECPATH).resolve().parents[1]
a = Analysis([str(root / "packaging/windows/validation_controller.py")], pathex=[], binaries=[], datas=[],
             hiddenimports=[], excludes=["numpy", "scipy", "PyQt6", "napari", "matplotlib", "pytest", "tkinter"])
pyz = PYZ(a.pure)
EXE(pyz, a.scripts, a.binaries, a.datas, [("X utf8=1", None, "OPTION")], name="Validate-Windows",
    console=True, strip=False, upx=False)
