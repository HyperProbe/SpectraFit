"""README.html (the user guide in the release zip) must match README.md."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_readme_html_is_up_to_date():
    pytest.importorskip("markdown")
    spec = importlib.util.spec_from_file_location("build_guide", ROOT / "tools" / "build_guide.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    current = (ROOT / "README.html").read_text(encoding="utf-8")
    assert current == mod.build(), "README.html is out of date: run  python tools/build_guide.py"


def test_guide_is_found_from_the_source_folder():
    pytest.importorskip("qtpy")
    from spectrafit.napari_plugin.widget import find_guide

    path = find_guide()
    assert path is not None and path.name == "README.html"


def test_guide_button_opens_the_guide(qtbot, monkeypatch):
    napari = pytest.importorskip("napari")
    from qtpy.QtGui import QDesktopServices

    from spectrafit.napari_plugin.widget import SpectraFitWidget

    opened = []
    monkeypatch.setattr(QDesktopServices, "openUrl", lambda url: opened.append(url.toLocalFile()) or True)
    viewer = napari.Viewer(show=False)
    try:
        SpectraFitWidget(viewer)._open_guide()
    finally:
        viewer.close()
    assert opened and opened[0].endswith("README.html")
