"""Stage and package the self-contained clean-Windows test kit."""

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STAGE = ROOT / "artifacts" / "validation-kit"


def sha(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true")
    parser.add_argument("--rehearsal", type=Path)
    args = parser.parse_args()
    STAGE.mkdir(parents=True, exist_ok=True)
    application_version = version("spectrafit")
    portable = ROOT / "artifacts" / f"SpectraFit-{application_version}-windows-x64.zip"
    evidence = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", ROOT / "validation/windows-step3"))
    reference = json.loads((evidence / "package-verification.json").read_text(encoding="utf-8"))
    assert sha(portable) == reference["zip_sha256"]
    sources = {"SpectraFit-portable.zip": portable,
               "SpectraFit-verify.exe": ROOT / "artifacts/windows/SpectraFit-verify/SpectraFit-verify.exe",
               "Validate-Windows.exe": ROOT / "artifacts/step4-tools/Validate-Windows.exe",
               "README.txt": ROOT / "packaging/windows/validation-kit/README.txt",
               "MANUAL-CHECKLIST.md": ROOT / "packaging/windows/validation-kit/MANUAL-CHECKLIST.md"}
    fixtures = ROOT / ".windows-baseline/step4-fixtures"
    for path in fixtures.rglob("*"):
        if path.is_file():
            sources["fixtures/" + path.relative_to(fixtures).as_posix()] = path
    for name, source in sources.items():
        destination = STAGE / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "portable_zip_sha256": reference["zip_sha256"],
                "fixture_scope": "Synthetic inputs only", "clean_machine_validation": "pending",
                "sha256": {name: sha(STAGE / name) for name in sorted(sources)}}
    (STAGE / "kit-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Staged kit: {STAGE}", flush=True)
    if args.stage_only:
        raise SystemExit(0)
    assert args.rehearsal is not None, "Supply the host rehearsal report before packaging"
    rehearsal = json.loads(args.rehearsal.read_text(encoding="utf-8"))
    assert rehearsal["automated_status"] == "passed" and rehearsal["environment_context"] == "host-rehearsal"
    assert rehearsal["portable_zip_sha256"] == reference["zip_sha256"]
    target = ROOT / "artifacts" / f"SpectraFit-{application_version}-validation-kit-windows-x64.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in sorted([*sources, "kit-manifest.json"]):
            archive.write(STAGE / name, "SpectraFit-validation-kit/" + name)
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    checksum = sha(target)
    target.with_suffix(".zip.sha256").write_text(f"{checksum}  {target.name}\n", encoding="ascii")
    result = {"status": "prepared", "kit": str(target), "zip_bytes": target.stat().st_size,
              "sha256": checksum, "portable_zip_sha256": reference["zip_sha256"],
              "host_rehearsal": "passed", "clean_machine_validation": "deferred by user"}
    (evidence / "kit-verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
