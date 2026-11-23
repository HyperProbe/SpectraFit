"""Smoke-test the uninstrumented production GUI without changing its code."""

import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", Path(__file__).resolve().parent)).resolve()


if __name__ == "__main__":
    working = ROOT / ".windows-baseline" / f"production-gui-{time.time_ns()}"
    working.mkdir(parents=True)
    home = working / "app-home"
    environment = os.environ.copy()
    windows = environment.get("SystemRoot", r"C:\Windows")
    environment.update(PATH=f"{windows}\\System32;{windows}", PYTHONHOME=r"Z:\missing-python-home",
                       PYTHONPATH=r"Z:\missing-python-modules", SPECTRAFIT_HOME=str(home), QT_API="pyside6")
    environment.pop("QT_QPA_PLATFORM", None)
    executable = ROOT / "artifacts/windows/SpectraFit/SpectraFit.exe"
    process = subprocess.Popen([str(executable), str(HERE / "settings-2.yaml")], cwd=working, env=environment)
    report = {"status": "running", "executable": str(executable), "pid": process.pid}
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            assert process.poll() is None, f"Desktop exited early: {process.returncode}"
            time.sleep(0.2)
        log = home / "logs" / "spectrafit.log"
        text = log.read_text(encoding="utf-8")
        assert "Starting SpectraFit" in text
        assert "could not start" not in text and "Traceback" not in text, text
        (HERE / "production-gui-startup.log").write_text(text, encoding="utf-8")
        report.update(status="passed", observed_running_seconds=15, startup_error_logged=False,
                      scope="Production GUI process startup only; no OS mouse/keyboard interaction or screenshot")
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        if process.poll() is None:
            # This test-created session has only loaded settings, not data/results.
            process.kill()
        process.wait(timeout=10)
        report["test_process_stopped"] = True
        (HERE / "production-gui-startup.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
