"""Regression tests for desktop startup, Windows paths, resources and workers."""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from spectrafit import launcher, parallel, paths
from spectrafit.config import Config
from spectrafit.io import load_data, resolve_files
from spectrafit.synthetic import write_envi


@pytest.fixture
def desktop_home(monkeypatch, tmp_path):
    monkeypatch.setenv("SPECTRAFIT_HOME", str(tmp_path / "SpectraFit"))
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield tmp_path / "SpectraFit"
    for handler in list(root.handlers):
        if handler not in handlers:
            root.removeHandler(handler)
            handler.close()
    root.setLevel(level)


def test_gui_startup_without_console_logs_and_reports_failure(monkeypatch, desktop_home):
    from spectrafit import cli

    messages = []

    def fail(_):
        print("Output without a console")
        raise RuntimeError("Test startup failure")

    monkeypatch.setattr(cli, "main", fail)
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    with monkeypatch.context() as context:
        context.setattr(sys, "stdout", None)
        context.setattr(sys, "stderr", None)
        assert launcher.gui_main([]) == 1
    log = desktop_home / "logs" / "spectrafit.log"
    assert "Output without a console" in log.read_text(encoding="utf-8")
    assert "RuntimeError: Test startup failure" in log.read_text(encoding="utf-8")
    assert len(messages) == 1 and str(log) in messages[0]


def test_gui_handles_unsuccessful_startup_return(monkeypatch, desktop_home):
    from spectrafit import cli

    messages = []
    monkeypatch.setattr(cli, "main", lambda _: 1)
    monkeypatch.setattr(launcher, "_show_startup_error", messages.append)
    assert launcher.gui_main([]) == 1
    assert "could not start" in messages[0]


def test_entry_points_dispatch_workers_before_application(monkeypatch, desktop_home):
    from spectrafit import cli

    calls = []
    monkeypatch.setattr(launcher.multiprocessing, "freeze_support", lambda: calls.append("freeze"))
    monkeypatch.setattr(cli, "main", lambda args: calls.append(args) or 0)
    assert launcher.gui_main(["settings.yaml"]) == 0
    assert calls == ["freeze", ["gui", "settings.yaml"]]
    calls.clear()
    assert launcher.console_main(["chromophores"]) == 0
    assert calls == ["freeze", ["chromophores"]]


def test_logging_falls_back_when_home_is_not_writable(monkeypatch, desktop_home, tmp_path):
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("SPECTRAFIT_HOME", str(blocked))
    monkeypatch.setattr(launcher.tempfile, "tempdir", str(tmp_path))
    log = launcher._configure_logging()
    logging.getLogger("spectrafit").error("Fallback logging works")
    assert log.parent == tmp_path
    assert "Fallback logging works" in log.read_text(encoding="utf-8")


def test_windows_user_locations_and_legacy_library(monkeypatch, tmp_path):
    monkeypatch.delenv("SPECTRAFIT_HOME", raising=False)
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "Local"))
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    current = tmp_path / "Local" / "SpectraFit"
    assert paths.app_home() == current
    assert paths.chromophore_dirs() == [tmp_path / ".spectrafit" / "chromophores", current / "chromophores"]
    from spectrafit.chromophores import ChromophoreLibrary, read_chromophore_file, write_chromophore_file, builtin_dir

    for folder, label in zip(paths.chromophore_dirs(), ("Legacy custom spectrum", "Current custom spectrum")):
        folder.mkdir(parents=True)
        chrom = read_chromophore_file(builtin_dir() / "HbO2.csv")
        chrom.name, chrom.label = "custom", label
        write_chromophore_file(chrom, folder / "custom.csv")
    assert ChromophoreLibrary().get("custom").label == "Current custom spectrum"
    monkeypatch.setenv("SPECTRAFIT_HOME", str(tmp_path / "Explicit"))
    assert paths.chromophore_dirs() == [tmp_path / "Explicit" / "chromophores"]


def test_packaged_guide_is_used_before_user_copy(monkeypatch, desktop_home):
    desktop_home.mkdir(parents=True)
    (desktop_home / "README.html").write_text("old user guide", encoding="utf-8")
    guide = paths.find_guide()
    assert guide is not None and guide.parent.name == "data"
    assert "<!DOCTYPE html>" in guide.read_text(encoding="utf-8")


def test_settings_resolve_backslash_envi_references_from_other_directory(tmp_path, monkeypatch):
    folder = tmp_path / "Settings with spaces ü"
    calibration = folder / "calibration"
    calibration.mkdir(parents=True)
    wl = np.arange(500.0, 905.0, 5.0)
    write_envi(folder / "raw", np.full((8, 10, len(wl)), 1000, dtype="u2"), wl)
    write_envi(calibration / "whiteReference", np.full((1, 10, len(wl)), 2000, dtype="u2"), wl)
    write_envi(calibration / "darkReference", np.zeros((1, 10, len(wl)), dtype="u2"), wl)
    cfg = Config()
    cfg.data.file = "raw"
    cfg.data.is_reflectance = False
    cfg.data.white_reference = r"calibration\whiteReference"
    cfg.data.dark_reference = r"calibration\darkReference.hdr"
    cfg.save(folder / "settings.yaml")
    monkeypatch.chdir(tmp_path)
    restored = Config.load(folder / "settings.yaml")
    data = load_data(restored.data)
    np.testing.assert_allclose(data.cube, 0.6)


def test_relative_globs_wavelength_files_and_reference_variable_names(tmp_path, monkeypatch):
    folder = tmp_path / "settings"
    measurements = folder / "measurements"
    measurements.mkdir(parents=True)
    for name in ("frame2.mat", "frame10.mat"):
        (measurements / name).touch()
    cfg = Config()
    cfg.data.file = r"measurements\*.mat"
    cfg.data.wavelengths = r"calibration\wavelengths.TXT"
    cfg.data.white_reference = "white_in_same_mat"
    cfg.save(folder / "settings.yaml")
    monkeypatch.chdir(tmp_path)
    restored = Config.load(folder / "settings.yaml")
    assert [Path(p).name for p in resolve_files(restored.data.file)] == ["frame2.mat", "frame10.mat"]
    assert restored.data.wavelengths == str(folder / "calibration" / "wavelengths.TXT")
    assert restored.data.white_reference == "white_in_same_mat"


def test_windows_worker_limit_preserves_default_and_explicit_settings(monkeypatch):
    monkeypatch.setattr(parallel, "available_cores", lambda: 128)
    monkeypatch.setattr(parallel.sys, "platform", "win32")
    assert parallel.resolve_cores(None) == 61
    assert parallel.resolve_cores(128) == 61
    assert parallel.resolve_cores(2) == 2
    monkeypatch.setattr(parallel.sys, "platform", "linux")
    assert parallel.resolve_cores(None) == 127
    assert parallel.resolve_cores(128) == 128


@pytest.mark.skipif(sys.platform != "win32" or parallel.available_cores() < 2,
                    reason="Exercises native Windows process handles with two workers")
def test_workers_exit_after_parent_is_force_terminated(tmp_path):
    import ctypes
    from ctypes import wintypes

    script = tmp_path / "parent.py"
    pid_file = tmp_path / "workers.json"
    script.write_text(
        "import json, sys, time\n"
        "from pathlib import Path\n"
        "from spectrafit.parallel import start_pool\n"
        "def busy(marker):\n"
        "    Path(marker).write_text('running')\n"
        "    time.sleep(120)\n"
        "if __name__ == '__main__':\n"
        "    pool = start_pool(2)\n"
        "    target = Path(sys.argv[1])\n"
        "    markers = [target.with_name('busy-' + str(i)) for i in range(2)]\n"
        "    futures = [pool.submit(busy, str(marker)) for marker in markers]\n"
        "    while not all(marker.exists() for marker in markers):\n"
        "        time.sleep(0.05)\n"
        "    temporary = target.with_suffix('.tmp')\n"
        "    temporary.write_text(json.dumps(list(pool._processes)))\n"
        "    temporary.replace(target)\n"
        "    time.sleep(120)\n", encoding="utf-8")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handles = []
    with (tmp_path / "parent.log").open("w", encoding="utf-8") as output:
        parent = subprocess.Popen([sys.executable, str(script), str(pid_file)], stdout=output, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 30
            while not pid_file.exists() and parent.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert pid_file.exists(), (tmp_path / "parent.log").read_text(encoding="utf-8")
            pids = json.loads(pid_file.read_text(encoding="utf-8"))
            assert len(pids) == 2
            for pid in pids:
                handle = kernel.OpenProcess(0x100001, False, pid)  # synchronize + cleanup on test failure
                assert handle, f"Could not open test worker {pid}"
                handles.append(handle)
            parent.kill()  # only the parent created by this test
            parent.wait(timeout=5)
            for handle in handles:
                assert kernel.WaitForSingleObject(handle, 10000) == 0, "Worker survived its parent"
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait(timeout=5)
            for handle in handles:
                if kernel.WaitForSingleObject(handle, 0) != 0:
                    kernel.TerminateProcess(handle, 1)
                    kernel.WaitForSingleObject(handle, 5000)
                kernel.CloseHandle(handle)


def test_packaging_detects_build_computer_home_folder(tmp_path):
    pytest.importorskip("PyInstaller")
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "packaging" / "windows" / "package_windows.py"
    spec = importlib.util.spec_from_file_location("package_windows", script)
    package_windows = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(package_windows)
    app = tmp_path / "SpectraFit"
    metadata = app / "_internal" / "spectrafit-0.1.8.dist-info"
    metadata.mkdir(parents=True)
    (app / "README.txt").write_text("Extract into a short folder.", encoding="utf-8")
    (metadata / "METADATA").write_text("Name: spectrafit", encoding="utf-8")
    assert package_windows.personal_paths(app, "C:/Users/Ana Müller") == []
    (metadata / "direct_url.json").write_text(
        '{"url": "file:///C:/Users/Ana%20M%C3%BCller/OneDrive/SpectraFit"}', encoding="utf-8")
    (app / "notes.txt").write_text("Built in C:\\Users\\ana müller\\src", encoding="utf-16-le")
    expected = ["_internal/spectrafit-0.1.8.dist-info/direct_url.json", "notes.txt"]
    assert package_windows.personal_paths(app, "C:/Users/Ana Müller") == expected
    assert package_windows.personal_paths(app, "C:\\Users\\Ana Müller\\") == expected
