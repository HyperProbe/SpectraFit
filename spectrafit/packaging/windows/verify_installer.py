"""Exercise fresh install, repair, installed launch and uninstall on this host."""

import hashlib
import json
import os
import subprocess
import tempfile
import time
import traceback
import winreg
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", ROOT / "validation/windows-final")).resolve()
KEY = r"Software\Microsoft\Windows\CurrentVersion\Uninstall\{4D45D0C1-A8FA-4B22-909C-D341902BBD46}_is1"


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def registration():
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, KEY, 0, winreg.KEY_READ | winreg.KEY_WOW64_64KEY) as key:
            return {name: winreg.QueryValueEx(key, name)[0] for name in ("InstallLocation", "DisplayVersion")}
    except FileNotFoundError:
        return None


def library_snapshot():
    root = Path(os.environ["LOCALAPPDATA"]) / "SpectraFit" / "chromophores"
    return {p.relative_to(root).as_posix(): sha(p) for p in root.rglob("*") if p.is_file()}


def check_payload(folder, manifest):
    for name, info in manifest["files"].items():
        path = (folder / name).resolve()
        assert path.is_relative_to(folder.resolve()) and sha(path) == info["sha256"], name
    assert len(list(folder.glob("unins*.exe"))) == 1
    return len(manifest["files"])


def main(report):
    assert registration() is None, "Existing SpectraFit install: use a separate test account instead."
    build = json.loads((EVIDENCE / "installer-build.json").read_text(encoding="utf-8"))
    installer = Path(build["installer"])
    assert sha(installer) == build["sha256"]
    manifest = json.loads((Path(build["staging_directory"]) / "SpectraFit/BUILD-MANIFEST.json").read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp(prefix="sfinst-"))
    installed = work / "SpectraFit"
    before = library_snapshot()
    report.update(environment_context="development-host", clean_machine_validation="Deferred by user; next step",
                  working_directory=str(work), installer_sha256=build["sha256"], version=build["version"])
    command = [str(installer), "/SP-", "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/NOICONS",
               "/TASKS=", f"/DIR={installed}", f"/LOG={EVIDENCE / 'installer-install.log'}"]
    subprocess.run(command, check=True, timeout=180)
    assert Path(registration()["InstallLocation"]).resolve() == installed.resolve()
    assert registration()["DisplayVersion"] == build["version"]
    report["fresh_install_payload_files_verified"] = check_payload(installed, manifest)
    assert sha(installed / "BUILD-MANIFEST.json") == sha(Path(build["staging_directory"]) / "SpectraFit/BUILD-MANIFEST.json")
    # Only these new test-created files are changed. No existing installation is used.
    marker = installed / "analysis-results/keep.txt"
    marker.parent.mkdir()
    marker.write_text("test analysis result: preserve on repair and uninstall", encoding="utf-8")
    settings = installed / "user-settings.yaml"
    settings.write_text("test_setting: preserve\n", encoding="utf-8")
    (installed / "README.html").unlink()
    repair = [value if not value.startswith("/LOG=") else f"/LOG={EVIDENCE / 'installer-repair.log'}" for value in command]
    subprocess.run(repair, check=True, timeout=180)
    report["same_version_repair_payload_files_verified"] = check_payload(installed, manifest)
    assert marker.read_text(encoding="utf-8").startswith("test analysis") and settings.is_file()
    windows = os.environ.get("SystemRoot", r"C:\Windows")
    environment = os.environ.copy()
    environment.update(PATH=f"{windows}\\System32;{windows}", PYTHONHOME=r"Z:\missing-python-home",
                       PYTHONPATH=r"Z:\missing-python-modules", SPECTRAFIT_HOME=str(work / "user-data"))
    environment.pop("QT_QPA_PLATFORM", None)
    for name, arguments in (("help", ["--help"]), ("chromophores", ["chromophores"])):
        result = subprocess.run([str(installed / "SpectraFit-cli.exe"), *arguments], cwd=work,
                                env=environment, capture_output=True, timeout=45)
        (EVIDENCE / f"installed-cli-{name}.log").write_bytes(result.stdout + result.stderr)
        assert result.returncode == 0, name
        if name == "chromophores":
            assert b"HbO2" in result.stdout
    process = subprocess.Popen([str(installed / "SpectraFit.exe")], cwd=work, env=environment)
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            assert process.poll() is None, f"Installed GUI exited: {process.returncode}"
            time.sleep(0.2)
        log = (work / "user-data/logs/spectrafit.log").read_text(encoding="utf-8")
        assert "Starting SpectraFit" in log and "Traceback" not in log and "could not start" not in log
        (EVIDENCE / "installed-gui.log").write_text(log, encoding="utf-8")
        report["installed_gui_startup_seconds"] = 15
    finally:
        if process.poll() is None:
            process.kill()  # Test-created empty session: no analysis or unsaved work.
        process.wait(timeout=15)
    uninstaller, = installed.glob("unins*.exe")
    subprocess.run([str(uninstaller), "/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART",
                    f"/LOG={EVIDENCE / 'installer-uninstall.log'}"], check=True, timeout=180)
    assert registration() is None
    assert not (installed / "SpectraFit.exe").exists() and not (installed / "_internal").exists()
    assert marker.read_text(encoding="utf-8").startswith("test analysis")
    assert settings.read_text(encoding="utf-8") == "test_setting: preserve\n"
    assert library_snapshot() == before
    report.update(status="passed", installed_cli_checks="passed", uninstall_registration_removed=True,
                  owned_application_files_removed=True, untracked_results_and_settings_preserved=True,
                  existing_custom_chromophore_library_unchanged=True,
                  manual_installer_and_shortcut_checks="Pending clean Windows desktop testing")


if __name__ == "__main__":
    evidence = {"status": "running"}
    try:
        main(evidence)
    except Exception:
        evidence.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        (EVIDENCE / "installer-verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
