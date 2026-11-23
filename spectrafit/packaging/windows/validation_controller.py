"""Self-contained Windows validation kit runner. Uses only the Python standard library."""

import argparse
import array
import ast
import ctypes
import hashlib
import json
import math
import os
import platform
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import traceback
import zipfile
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def read_npy(file):
    assert file.read(6) == b"\x93NUMPY", "Invalid NPY signature"
    major, minor = file.read(2)
    assert major in (1, 2, 3), (major, minor)
    length = struct.unpack("<H" if major == 1 else "<I", file.read(2 if major == 1 else 4))[0]
    header = ast.literal_eval(file.read(length).decode("utf-8" if major == 3 else "latin-1"))
    dtype = header["descr"]
    assert dtype in ("<f4", "<f8", ">f4", ">f8", "=f4", "=f8"), dtype
    values = array.array("f" if dtype.endswith("4") else "d")
    values.frombytes(file.read())
    if dtype[0] in "<>" and (dtype[0] == "<") != (sys.byteorder == "little"):
        values.byteswap()
    assert len(values) == math.prod(header["shape"])
    return header, values


def compare_npz(actual_path, expected_path, rtol, atol):
    differences = {}
    with zipfile.ZipFile(actual_path) as actual, zipfile.ZipFile(expected_path) as expected:
        assert set(actual.namelist()) == set(expected.namelist()), "Result array names differ"
        for name in expected.namelist():
            with actual.open(name) as file:
                ah, av = read_npy(file)
            with expected.open(name) as file:
                eh, ev = read_npy(file)
            assert ah["shape"] == eh["shape"] and ah["fortran_order"] == eh["fortran_order"], name
            maximum = 0.0
            for index, (left, right) in enumerate(zip(av, ev)):
                if math.isnan(left) and math.isnan(right):
                    continue
                assert not (math.isnan(left) or math.isnan(right)), f"{name}[{index}]: NaN mask differs"
                if left == right:
                    continue
                delta = abs(left - right)
                assert math.isfinite(delta) and delta <= atol + rtol * abs(right), \
                    f"{name}[{index}]: {left} vs {right}; difference {delta}"
                maximum = max(maximum, delta)
            differences[name.removesuffix(".npy")] = maximum
    return differences


def scalar_tiff_pages(path, frames):
    """Check classic TIFF IFDs without needing tifffile on the target PC."""
    with path.open("rb") as file:
        order = file.read(2)
        assert order in (b"II", b"MM"), "Invalid TIFF byte order"
        endian = "<" if order == b"II" else ">"
        assert struct.unpack(endian + "H", file.read(2))[0] == 42, "Expected classic TIFF"
        offset = struct.unpack(endian + "I", file.read(4))[0]
        seen = set()
        while offset:
            assert offset not in seen and len(seen) < frames, "Unexpected/cyclic TIFF pages"
            seen.add(offset)
            file.seek(offset)
            count = struct.unpack(endian + "H", file.read(2))[0]
            tags = {}
            for _ in range(count):
                tag, kind, length, value = struct.unpack(endian + "HHI4s", file.read(12))
                if tag in (258, 262, 277, 339):
                    assert kind == 3 and length == 1, f"Unexpected scalar TIFF tag {tag}"
                    tags[tag] = struct.unpack(endian + "H", value[:2])[0]
            assert tags.get(262) == 1, "TIFF map is not grayscale (RGB/RGBA time-series encoding)"
            assert tags.get(258) == 32 and tags.get(339) == 3, "TIFF map is not float32"
            assert tags.get(277, 1) == 1, "TIFF map has multiple colour samples"
            offset = struct.unpack(endian + "I", file.read(4))[0]
        assert len(seen) == frames, f"Expected {frames} grayscale frame pages, found {len(seen)}"
    return len(seen)


def environment_inventory():
    import winreg

    registry = []
    for hive_name, hive in (("HKCU", winreg.HKEY_CURRENT_USER), ("HKLM", winreg.HKEY_LOCAL_MACHINE)):
        for key_name in (r"Software\Python\PythonCore", r"Software\WOW6432Node\Python\PythonCore"):
            try:
                with winreg.OpenKey(hive, key_name) as key:
                    for index in range(winreg.QueryInfoKey(key)[0]):
                        name = winreg.EnumKey(key, index)
                        registry.append({"hive": hive_name, "key": key_name, "version": name})
            except OSError:
                pass
    commands = {name: shutil.which(name) for name in ("python.exe", "python3.exe", "py.exe", "conda.exe")}
    indicators = registry or [path for path in commands.values() if path and "\\microsoft\\windowsapps\\" not in path.lower()]
    powershell = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32/WindowsPowerShell/v1.0/powershell.exe"
    try:
        gpu = subprocess.run([str(powershell), "-NoProfile", "-NonInteractive", "-Command",
                              "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress"],
                             capture_output=True, timeout=30, creationflags=subprocess.CREATE_NO_WINDOW)
        gpu_info = {"exit_code": gpu.returncode, "output": gpu.stdout.decode("utf-8", errors="replace").strip(),
                    "stderr": gpu.stderr.decode("utf-8", errors="replace").strip()}
    except Exception as error:
        gpu_info = {"error": str(error)}
    return {"os": platform.platform(), "machine": platform.machine(), "cpu_count": os.cpu_count(),
            "python_registry_entries": registry, "python_commands_on_path": commands,
            "python_indicators_detected": bool(indicators), "graphics": gpu_info}


def execute(command, cwd, env, log, timeout=180, windowed=False):
    started = time.monotonic()
    with log.open("wb") as file:
        streams = {} if windowed else {"stdout": file, "stderr": subprocess.STDOUT}
        if windowed:
            file.write(b"No standard handles redirected; see the application log.\n")
        process = subprocess.run([str(value) for value in command], cwd=cwd, env=env, timeout=timeout,
                                 creationflags=0 if windowed else subprocess.CREATE_NO_WINDOW, **streams)
    return {"exit_code": process.returncode, "seconds": time.monotonic() - started}


def worker_cleanup(probe, work, env):
    directory = work / f"worker-test-{time.time_ns()}"
    directory.mkdir()
    target = directory / "workers.json"
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    parent = subprocess.Popen([str(probe), "busy-parent", str(target)], cwd=work, env=env)
    handles = []
    try:
        deadline = time.monotonic() + 45
        while not target.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert target.exists(), "Both worker jobs did not start"
        pids = json.loads(target.read_text(encoding="utf-8"))["pids"]
        assert len(pids) == 2
        for pid in pids:
            handle = kernel.OpenProcess(0x100001, False, pid)
            assert handle
            handles.append(handle)
        parent.kill()
        parent.wait(timeout=10)
        for handle in handles:
            assert kernel.WaitForSingleObject(handle, 10000) == 0, "A worker survived its parent"
        return {"status": "passed", "worker_count": len(pids)}
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=10)
        for handle in handles:
            if kernel.WaitForSingleObject(handle, 0) != 0:
                kernel.TerminateProcess(handle, 1)
                kernel.WaitForSingleObject(handle, 5000)
            kernel.CloseHandle(handle)


def main(args):
    kit = args.kit.resolve() if args.kit else Path(sys.executable).resolve().parent
    context = args.context
    if context is None:
        print("Environment: C = fresh clean Windows, H = existing host, U = unknown")
        answer = input("Choose only C if this is a clean Windows PC/VM with no Python installed [U]: ").strip().lower()
        context = {"c": "clean-windows", "h": "host-rehearsal"}.get(answer, "unattested")
    output = args.results.resolve() if args.results else Path(os.environ.get("LOCALAPPDATA", Path.home())) / \
        "SpectraFitValidation" / "Runs" / (datetime.now().strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    output.mkdir(parents=True, exist_ok=False)
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "environment_context": context,
              "clean_environment_attested": context == "clean-windows", "automated_status": "running",
              "manual_checks": "pending", "clean_machine_validation": "pending", "checks": {}}
    report_file = output / "report.json"

    def save():
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def check(name, function):
        print(f"Checking {name}...", flush=True)
        try:
            value = function()
            report["checks"][name] = {"status": "passed", "details": value}
        except Exception:
            report["checks"][name] = {"status": "failed", "error": traceback.format_exc()}
        save()

    try:
        # Restore the normal per-process Windows DLL search before invoking
        # system tools or a different frozen application from this frozen runner.
        ctypes.windll.kernel32.SetDllDirectoryW(None)
        assert sys.platform == "win32" and struct.calcsize("P") == 8, "Requires 64-bit Windows"
        report["environment"] = environment_inventory()
        report["clean_environment_preconditions_met"] = context == "clean-windows" and \
            not report["environment"]["python_indicators_detected"]
        manifest = json.loads((kit / "kit-manifest.json").read_text(encoding="utf-8"))
        for name, expected in manifest["sha256"].items():
            path = (kit / name).resolve()
            assert path.is_relative_to(kit) and digest(path) == expected, f"Kit checksum mismatch: {name}"
        report["kit_integrity"] = "passed"
        report["portable_zip_sha256"] = manifest["portable_zip_sha256"]
        work = Path(tempfile.mkdtemp(prefix="sfcheck-"))
        report["working_directory"] = str(work)
        with zipfile.ZipFile(kit / "SpectraFit-portable.zip") as archive:
            for name in archive.namelist():
                assert (work / name).resolve().is_relative_to(work)
            assert max(len(str(work / name)) for name in archive.namelist()) < 260
            assert archive.testzip() is None
            archive.extractall(work)
        app = work / "SpectraFit"
        probe = app / "SpectraFit-verify.exe"
        shutil.copyfile(kit / "SpectraFit-verify.exe", probe)
        report["production_gui_path"] = str(app / "SpectraFit.exe")
        windows = os.environ.get("SystemRoot", r"C:\Windows")
        env = os.environ.copy()
        env.update(PATH=f"{windows}\\System32;{windows}", PYTHONHOME=r"Z:\missing-python-home",
                   PYTHONPATH=r"Z:\missing-python-modules", SPECTRAFIT_HOME=str(output / "app-home"),
                   QT_API="pyside6", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.pop("QT_QPA_PLATFORM", None)
        env.pop("MPLBACKEND", None)
        definitions = json.loads((kit / "fixtures/cases.json").read_text(encoding="utf-8"))

        def resource_check():
            path = output / "resources.json"
            result = execute([probe, "resources", path], work, env, output / "resources.log", windowed=True)
            data = json.loads(path.read_text(encoding="utf-8"))
            assert result["exit_code"] == 0 and data["status"] == "passed", data
            return data

        check("bundled-resources", resource_check)
        for case in definitions["cases"]:
            def numerical(case=case):
                folder = output / "cases" / case["name"]
                folder.mkdir(parents=True)
                result = execute([app / "SpectraFit-cli.exe", "run", kit / case["settings"], "-o", folder],
                                 work, env, folder / "console.log")
                assert result["exit_code"] == 0, "See console.log"
                differences = compare_npz(folder / "results.npz", kit / case["expected"], case["rtol"], case["atol"])
                for name in ("settings.yaml", "fit_info.json", "summary.csv", "pathlength_used.csv", "results.mat"):
                    assert (folder / name).stat().st_size > 0, name
                assert (folder / "results.mat").read_bytes().startswith(b"MATLAB")
                pages = {}
                with zipfile.ZipFile(kit / case["expected"]) as expected:
                    for name in differences:
                        if name != "wavelengths":
                            with expected.open(name + ".npy") as file:
                                header, _ = read_npy(file)
                            pages[name] = scalar_tiff_pages(folder / "tiff" / f"{name}.tif", header["shape"][0])
                result.update(array_max_abs_differences=differences, rtol=case["rtol"], atol=case["atol"],
                              tiff_grayscale_pages=pages,
                              comparison="Every NPZ value/NaN mask; MAT header; float32 grayscale TIFF frame pages")
                return result
            check(case["name"], numerical)
        for case in definitions["expected_errors"]:
            def rejection(case=case):
                log = output / f"{case['name']}.log"
                result = execute([app / "SpectraFit-cli.exe", "run", kit / case["settings"], "-o", output / "rejected"],
                                 work, env, log)
                assert result["exit_code"] != 0 and case["message"] in log.read_text(encoding="utf-8", errors="replace")
                return result
            check(f"reject-{case['name']}", rejection)
        if (os.cpu_count() or 1) >= 2:
            check("busy-worker-cleanup", lambda: worker_cleanup(probe, work, env))
            def gui_check():
                case, = [item for item in definitions["cases"] if item["gui"]]
                target = output / "gui.json"
                result = execute([probe, "gui", target, "--settings", kit / case["settings"],
                                  "--baseline", kit / case["expected"]], work, env, output / "gui.log", 300, windowed=True)
                data = json.loads(target.read_text(encoding="utf-8"))
                assert result["exit_code"] == 0 and data["status"] == "passed", data
                assert data["started_without_stdout"] and data["started_without_stderr"]
                return {"exit_code": result["exit_code"], "seconds": result["seconds"],
                        "screenshot": "gui.png", "reference": case["expected"],
                        "array_max_abs_differences": data["max_absolute_difference_vs_step1"]}
            check("frozen-gui", gui_check)
        else:
            report["checks"]["two-core-gui-and-cleanup"] = {"status": "skipped", "reason": "Requires at least two logical processors"}
        states = {entry["status"] for entry in report["checks"].values()}
        report["automated_status"] = "failed" if "failed" in states else ("partial" if "skipped" in states else "passed")
    except Exception:
        report["automated_status"] = "failed"
        report["fatal_error"] = traceback.format_exc()
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save()
        print(f"Automated result: {report['automated_status']}")
        print(f"Results: {output}")
        print("Manual display/lifecycle checks and clean-machine review remain pending.")
        if not args.noninteractive:
            input("Press Enter to close this test runner.")
    return 0 if report["automated_status"] == "passed" else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--context", choices=("host-rehearsal", "clean-windows", "unattested"))
    parser.add_argument("--noninteractive", action="store_true")
    raise SystemExit(main(parser.parse_args()))
