"""Create a per-user installer from the exact verified portable ZIP."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
import zipfile
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", ROOT / "validation/windows-final")).resolve()


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iscc", type=Path, required=True, help="Inno Setup 6.7.4 ISCC.exe")
    args = parser.parse_args()
    compiler = args.iscc.resolve(strict=True)
    release = version("spectrafit")
    portable = ROOT / "artifacts" / f"SpectraFit-{release}-windows-x64.zip"
    verified = json.loads((EVIDENCE / "package-verification.json").read_text(encoding="utf-8"))
    assert verified["status"] == "passed" and sha(portable) == verified["zip_sha256"]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    # Short staging path also supports Windows' classic filename-length limit.
    stage = Path(tempfile.mkdtemp(prefix="sfsetup-"))
    with zipfile.ZipFile(portable) as archive:
        assert archive.testzip() is None
        for name in archive.namelist():
            assert (stage / name).resolve().is_relative_to(stage.resolve())
        archive.extractall(stage)
    bundle = stage / "SpectraFit"
    manifest = json.loads((bundle / "BUILD-MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["version"] == release
    assert {p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()} == \
        set(manifest["files"]) | {"BUILD-MANIFEST.json"}
    for name, info in manifest["files"].items():
        path = (bundle / name).resolve()
        assert path.is_relative_to(bundle.resolve()) and sha(path) == info["sha256"], name
    command = [str(compiler), "/Qp", f"/DAppVersion={release}", f"/DBundleDir={bundle}",
               f"/DOutputDir={ROOT / 'artifacts'}", str(ROOT / "packaging/windows/SpectraFit.iss")]
    with (EVIDENCE / "installer-build.log").open("wb") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
    target = ROOT / "artifacts" / f"SpectraFit-{release}-windows-x64-setup.exe"
    checksum = sha(target)
    target.with_suffix(".exe.sha256").write_text(f"{checksum}  {target.name}\n", encoding="ascii")
    report = {"status": "built", "version": release, "installer": str(target),
              "bytes": target.stat().st_size, "sha256": checksum,
              "portable_zip_sha256": verified["zip_sha256"], "payload_files": len(manifest["files"]) + 1,
              "compiler": str(compiler), "compiler_sha256": sha(compiler), "signed": False,
              "clean_machine_validation": "Deferred by user; next step", "staging_directory": str(stage)}
    (EVIDENCE / "installer-build.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
