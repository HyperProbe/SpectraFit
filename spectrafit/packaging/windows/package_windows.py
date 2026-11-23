"""Package only the verified production bundle and record checksums."""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

from PyInstaller.archive.readers import CArchiveReader

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", ROOT / "validation/windows-step3")).resolve()
APP = ROOT / "artifacts/windows/SpectraFit"
PROBE = ROOT / "artifacts/windows/SpectraFit-verify"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


if __name__ == "__main__":
    verification = json.loads((EVIDENCE / "verification.json").read_text(encoding="utf-8"))
    assert verification["status"] == "passed", "Complete frozen verification before packaging"
    desktop = json.loads((EVIDENCE / "production-gui-startup.json").read_text(encoding="utf-8"))
    assert desktop["status"] == "passed"
    executable_paths = [APP / "SpectraFit.exe", APP / "SpectraFit-cli.exe", PROBE / "SpectraFit-verify.exe"]
    for path in executable_paths:
        assert sha(path) == verification["executable_sha256"][path.name], "Executable changed after verification"

    archives = [CArchiveReader(str(path)) for path in executable_paths]
    pyz_hashes = [hashlib.sha256(archive.extract("PYZ.pyz")).hexdigest() for archive in archives]
    assert len(set(pyz_hashes)) == 1, "Production and verification module archives differ"
    for archive, entry in zip(archives, ("spectrafit_gui", "spectrafit_cli", "frozen_probe")):
        application_entries = set(archive.toc) & {"spectrafit_gui", "spectrafit_cli", "frozen_probe"}
        assert application_entries == {entry}, application_entries

    # The probe was built alongside production from the same inputs. Check all
    # shared libraries/data, not just the module archive, before relying on it.
    payload = {}
    for path in sorted((APP / "_internal").rglob("*")):
        if path.is_file():
            relative = path.relative_to(APP)
            digest = sha(path)
            assert digest == sha(PROBE / relative), f"Verification payload differs: {relative}"
            payload[relative.as_posix()] = {"bytes": path.stat().st_size, "sha256": digest}
    assert len(payload) == sum(path.is_file() for path in (PROBE / "_internal").rglob("*"))

    shutil.copyfile(ROOT / "packaging/windows/PORTABLE-README.txt", APP / "README.txt")
    shutil.copyfile(ROOT / "README.html", APP / "README.html")
    shutil.copyfile(ROOT / "LICENSE", APP / "LICENSE")
    for path in APP.iterdir():
        if path.is_file() and path.name != "BUILD-MANIFEST.json":
            payload[path.name] = {"bytes": path.stat().st_size, "sha256": sha(path)}
    source_hashes = {path.relative_to(ROOT).as_posix(): sha(path)
                     for folder in (ROOT / "src/spectrafit", ROOT / "launchers", ROOT / "packaging/windows")
                     for path in sorted(folder.rglob("*"))
                     if path.is_file() and "__pycache__" not in path.parts}
    manifest = {"application": "SpectraFit", "version": metadata.version("spectrafit"),
                "platform": "Windows x64", "python_version": sys.version,
                "built_at_utc": datetime.now(timezone.utc).isoformat(),
                "pyinstaller_version": metadata.version("pyinstaller"),
                "hooks_version": metadata.version("pyinstaller-hooks-contrib"),
                "common_python_archive_sha256": pyz_hashes[0],
                "verification_payload_matches_production": True,
                "build_environment_distributions": {d.metadata["Name"]: d.version for d in metadata.distributions()},
                "source_sha256": source_hashes, "files": payload,
                "signed": False, "clean_machine_validation": "Deferred by user; next step"}
    (APP / "BUILD-MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    archive_path = ROOT / "artifacts" / f"SpectraFit-{manifest['version']}-windows-x64.zip"
    print(f"Creating {archive_path}", flush=True)
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6,
                         strict_timestamps=False) as archive:
        for path in sorted(APP.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(APP.parent).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        assert archive.testzip() is None
        assert not any("SpectraFit-verify" in name or "frozen_probe" in name for name in archive.namelist())
        # A nested checkout plus Jedi's autocomplete stub paths can exceed the
        # classic Windows path limit. Verify normal extraction in a short folder.
        unpacked = Path(tempfile.mkdtemp(prefix="sf-"))
        for name in archive.namelist():
            assert (unpacked / name).resolve().is_relative_to(unpacked.resolve())
        assert max(len(str(unpacked / name)) for name in archive.namelist()) < 260
        archive.extractall(unpacked)
    environment = os.environ.copy()
    windows = environment.get("SystemRoot", r"C:\Windows")
    environment.update(PATH=f"{windows}\\System32;{windows}", PYTHONHOME=r"Z:\missing-python-home",
                       PYTHONPATH=r"Z:\missing-python-modules", SPECTRAFIT_HOME=str(unpacked / "app-home"))
    smoke = subprocess.run([str(unpacked / "SpectraFit/SpectraFit-cli.exe"), "chromophores"],
                           cwd=unpacked, env=environment, capture_output=True, timeout=45)
    assert smoke.returncode == 0 and b"HbO2" in smoke.stdout, smoke.stderr
    checksum = sha(archive_path)
    archive_path.with_suffix(".zip.sha256").write_text(f"{checksum}  {archive_path.name}\n", encoding="ascii")
    result = {"status": "passed", "zip": str(archive_path), "zip_bytes": archive_path.stat().st_size,
              "zip_sha256": checksum, "file_count": len(payload) + 1,
              "uncompressed_bytes": sum(path.stat().st_size for path in APP.rglob("*") if path.is_file()),
              "verification_payload_matches_production": True,
              "relocated_zip_cli_exit_code": smoke.returncode,
              "extraction_directory": str(unpacked),
              "common_python_archive_sha256": pyz_hashes[0]}
    (EVIDENCE / "package-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
