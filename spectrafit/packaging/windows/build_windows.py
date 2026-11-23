"""Build the portable distribution using the active, pinned Python environment."""

import os
import subprocess
import sys
from pathlib import Path


if __name__ == "__main__":
    if sys.platform != "win32":
        raise SystemExit("Build this Windows distribution on Windows.")
    root = Path(__file__).resolve().parents[2]
    dist = (root / "artifacts" / "windows").resolve()
    work = (root / ".windows-baseline" / "pyinstaller-step3").resolve()
    for target in (dist, work):
        if root not in target.parents:
            raise SystemExit(f"Build target leaves the application workspace: {target}")
    evidence = Path(os.environ.get("SPECTRAFIT_VALIDATION_DIR", root / "validation/windows-step3")).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(QT_API="pyqt6", QT_QPA_PLATFORM="offscreen", PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
    command = [sys.executable, "-m", "PyInstaller", "--noconfirm", "--distpath", str(dist),
               "--workpath", str(work), str(root / "packaging/windows/SpectraFit.spec")]
    log = evidence / "build.log"
    print(f"Building Windows application. Log: {log}", flush=True)
    with log.open("wb") as output:
        result = subprocess.run(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT)
    print(f"PyInstaller exit code: {result.returncode}", flush=True)
    if not result.returncode:
        print(f"Portable application: {dist / 'SpectraFit'}", flush=True)
        print(f"Separate verification executable: {dist / 'SpectraFit-verify'}", flush=True)
    raise SystemExit(result.returncode)
