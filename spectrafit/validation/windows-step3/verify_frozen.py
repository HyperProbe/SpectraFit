"""Run genuine frozen binaries with Python paths removed, retaining raw evidence."""

import ctypes
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from ctypes import wintypes
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", Path(__file__).resolve().parent)).resolve()
APP = ROOT / "artifacts/windows/SpectraFit"
PROBE = ROOT / "artifacts/windows/SpectraFit-verify/SpectraFit-verify.exe"
WORK = ROOT / ".windows-baseline/step3-working-directory"


def invoke(name, command, env, timeout=240, windowed=False):
    print(f"Checking {name}...", flush=True)
    before = time.monotonic()
    with (HERE / f"{name}.log").open("wb") as output:
        streams = {} if windowed else {"stdout": output, "stderr": subprocess.STDOUT}
        if windowed:
            output.write(b"Windowed process launched without redirected standard handles; see app-home/logs.\n")
        result = subprocess.run([str(value) for value in command], cwd=WORK, env=env,
                                timeout=timeout, **streams)
    assert result.returncode == 0, f"{name}: exit {result.returncode}; see {name}.log"
    return {"exit_code": result.returncode, "elapsed_seconds": time.monotonic() - before}


def busy_workers(env):
    print("Checking frozen busy-worker cleanup...", flush=True)
    pid_file = HERE / "busy-worker-pids.json"
    # Use a unique directory so stale markers cannot satisfy a later run.
    marker_root = WORK / f"busy-{time.time_ns()}"
    marker_root.mkdir()
    target = marker_root / "workers.json"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    parent = subprocess.Popen([str(PROBE), "busy-parent", str(target)], cwd=WORK, env=env)
    handles = []
    try:
        deadline = time.monotonic() + 45
        while not target.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert target.exists(), "Frozen workers did not start both busy jobs"
        report = json.loads(target.read_text(encoding="utf-8"))
        assert len(report["pids"]) == 2
        for pid in report["pids"]:
            handle = kernel.OpenProcess(0x100001, False, pid)
            assert handle
            handles.append(handle)
        parent.kill()
        parent.wait(timeout=5)
        for handle in handles:
            assert kernel.WaitForSingleObject(handle, 10000) == 0, "Frozen worker survived its parent"
        report["status"] = "passed"
        pid_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        for handle in handles:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                kernel.TerminateProcess(handle, 1)
                kernel.WaitForSingleObject(handle, 5000)
            kernel.CloseHandle(handle)


def verify(report):
    WORK.mkdir(parents=True, exist_ok=True)
    report["executable_sha256"] = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                    for path in (APP / "SpectraFit.exe", APP / "SpectraFit-cli.exe", PROBE)}
    env = os.environ.copy()
    windows = env.get("SystemRoot", r"C:\Windows")
    env.update(PATH=f"{windows}\\System32;{windows}", PYTHONHOME=r"Z:\missing-python-home",
               PYTHONPATH=r"Z:\missing-python-modules", PYTHONNOUSERSITE="1",
               SPECTRAFIT_HOME=str(WORK / "app-home"), QT_API="pyside6",
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    env.pop("QT_QPA_PLATFORM", None)
    env.pop("MPLBACKEND", None)
    report["runtime_environment"] = {key: env[key] for key in ("PATH", "PYTHONHOME", "PYTHONPATH", "QT_API")}
    cli = APP / "SpectraFit-cli.exe"
    report["cli_help"] = invoke("cli-help", [cli, "--help"], env, 45)
    report["chromophores"] = invoke("chromophores", [cli, "chromophores"], env, 45)
    assert "HbO2" in (HERE / "chromophores.log").read_text(encoding="utf-8")
    report["template"] = invoke("new-config", [cli, "new-config", WORK / "template.yaml", "--force"], env, 45)
    report["resources"] = invoke("resources", [PROBE, "resources", HERE / "resources.json"], env, 60)
    resource = json.loads((HERE / "resources.json").read_text(encoding="utf-8"))
    assert resource["status"] == "passed", resource
    report["desktop_help"] = invoke("desktop-help", [APP / "SpectraFit.exe", "--help"], env, 45)
    report["busy_workers"] = busy_workers(env)

    baseline_file = ROOT / ".windows-baseline/real-cube-results/results.npz"
    config = yaml.safe_load((ROOT / "validation/windows-baseline/real-cube-settings.yaml").read_text(encoding="utf-8"))
    report["cli_fits"] = {}
    for cores in (1, 2):
        config["solver"]["cores"] = cores
        output = WORK / f"results-{cores}"
        config["output"]["folder"] = str(output)
        settings = HERE / f"settings-{cores}.yaml"
        settings.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
        measured = invoke(f"real-cube-{cores}-core", [cli, "run", settings], env)
        differences = {}
        with np.load(output / "results.npz", allow_pickle=False) as actual, np.load(baseline_file, allow_pickle=False) as expected:
            for name in expected.files:
                np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-8, equal_nan=True)
                if name != "wavelengths":
                    finite = np.isfinite(actual[name]) & np.isfinite(expected[name])
                    differences[name] = float(np.max(np.abs(actual[name][finite] - expected[name][finite])))
        info = json.loads((output / "fit_info.json").read_text(encoding="utf-8"))
        if cores == 2:
            assert any("2 processor cores" in note for note in info["notes"])
        measured.update(max_absolute_difference_vs_step1=differences, run_time_s=info["run_time_s"])
        report["cli_fits"][str(cores)] = measured
        (HERE / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    gui_settings = HERE / "settings-2.yaml"
    report["gui"] = invoke("frozen-gui", [PROBE, "gui", HERE / "frozen-gui.json", "--settings", gui_settings,
                                         "--baseline", baseline_file], env, 300, windowed=True)
    gui = json.loads((HERE / "frozen-gui.json").read_text(encoding="utf-8"))
    assert gui["status"] == "passed" and gui["frozen"], gui
    assert gui["started_without_stdout"] and gui["started_without_stderr"]


if __name__ == "__main__":
    evidence = {"status": "running"}
    try:
        verify(evidence)
        evidence["status"] = "passed"
    except Exception:
        evidence["status"] = "failed"
        evidence["traceback"] = traceback.format_exc()
        raise
    finally:
        (HERE / "verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
