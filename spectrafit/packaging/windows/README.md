# Windows application builds

This builds a one-folder application containing `SpectraFit.exe`,
`SpectraFit-cli.exe`, and their shared `_internal` runtime. It does not create an
installer by itself or publish a release. Python is embedded in the bundle.
The per-user installer described below wraps the verified portable payload.

## Reproduce

Use Windows x64 and CPython 3.12.16. From the `spectrafit/` directory, create a
build environment and install the captured environment plus build-tool versions:

```powershell
py -3.12 -m venv .venv-build
.\.venv-build\Scripts\python.exe -m pip install -r packaging\windows\requirements-environment.txt -r packaging\windows\requirements-build.txt
.\.venv-build\Scripts\python.exe -m pip install --no-deps --editable .
.\.venv-build\Scripts\python.exe tools\build_guide.py
.\.venv-build\Scripts\python.exe packaging\windows\build_windows.py
```

The captured environment includes test tools so source checks can use the same
versions. PyInstaller excludes pytest and the alternative Qt bindings from the
application. The build selects PyQt6 and the QtAgg plotting backend. It collects
napari's dynamic modules, plugin manifests, distribution metadata and application
data explicitly. Frozen Python uses UTF-8 mode; the runtime hook selects the
shipped Qt binding.

`build_windows.py` runs PyInstaller with a clean PATH (the build Python and the
Windows folders only), so libraries from other programs on the build computer,
such as a Java runtime's `freetype.dll` and `ucrtbase.dll`, are not collected.
The spec stops the build if any library still comes from outside the build
Python or Windows. It also leaves out installer records such as
`direct_url.json`, which contain the build folder and user name, and
`package_windows.py` refuses to package if the application's own files or package
metadata mention the builder's home folder.

Set `SPECTRAFIT_VALIDATION_DIR` to an absolute directory to keep a release's
evidence separate, for example `validation/windows-final`. Use the same value for
the build, frozen checks, portable packaging, installer and kit commands.

The build specification has one dependency analysis and one Python module
archive shared by three entry scripts. Each executable contains exactly its own
entry script and the common runtime hooks. `SpectraFit-verify.exe` is collected
into a separate directory, never into the portable user ZIP. It drives the real
desktop launcher and Qt workers without adding a test switch, external-code
loader or automation hook to either production executable.

Build outputs:

- `artifacts/windows/SpectraFit/`: production GUI and CLI.
- `artifacts/windows/SpectraFit-verify/`: separate frozen verifier.
- `.windows-baseline/pyinstaller-step3/`: PyInstaller analysis/cache.
- `validation/windows-step3/build.log`: build output.

PyInstaller's relevant documentation:
[spec files/runtime options](https://pyinstaller.org/en/stable/spec-files.html),
[hooks/metadata](https://pyinstaller.org/en/stable/hooks.html), and
[Qt/Matplotlib hook configuration](https://pyinstaller.org/en/stable/hooks-config.html).

## Verify and package

The local real-data verifier requires the confirmed settings and saved step-1
result maps under `validation/windows-baseline/` and `.windows-baseline/`. The
input measurement/reference files are not part of the repository or distribution.
For another measurement, establish equivalent baseline evidence and adjust the
verification inputs. Do not claim the original cube checks for a different input.

```powershell
.\.venv-build\Scripts\python.exe validation\windows-step3\verify_frozen.py
.\.venv-build\Scripts\python.exe validation\windows-step3\verify_desktop_startup.py
.\.venv-build\Scripts\python.exe packaging\windows\package_windows.py
```

Verification launches actual compiled executables with Python directories
removed from PATH and invalid external Python paths. It checks resources, command
entry points, parent-death handling during active worker jobs, serial/parallel
real-data maps, and a no-console frozen GUI workflow. Reports retain failures.
The verifier produces an application screenshot for visual inspection.
The separate startup smoke check launches the unmodified production GUI, observes
it for 15 seconds without startup errors, and stops that test-created session.

Packaging requires passing verification and matching executable hashes. It
compares the complete production/verification shared payloads and embedded
Python archives, adds the guide/readme/license and file-hash manifest, creates a
ZIP, checks its CRCs, extracts it into another directory and runs the relocated
CLI. The verification executable and local measurements/results are excluded.
Extraction is tested in a short system temporary folder: a deeply nested
checkout combined with autocomplete stub filenames can exceed Windows' classic
path limit. Users should extract into a short folder path as the portable readme
explains. The first nested extraction failure is retained in the verification log.

The artifact is `artifacts/SpectraFit-<version>-windows-x64.zip`, with a sibling SHA-256
file. The build is unsigned. A clean machine and additional Windows/graphics
configurations still require the next release-validation step.

## Build and verify the per-user installer

Use the official [Inno Setup 6.7.4 compiler](https://github.com/jrsoftware/issrc/releases/tag/is-6_7_4).
The compiler setup used here had a valid Authenticode signature from Pyrsys B.V.
and SHA-256 `a549929319f9120716677c685535ca4671e4b937b5fdc87e2585f32d8eb30e8c`.
The compiler is a build tool, not an application dependency. It is not downloaded
or installed automatically by the scripts.

```powershell
.\.venv-build\Scripts\python.exe packaging\windows\build_installer.py --iscc "C:\path\to\ISCC.exe"
.\.venv-build\Scripts\python.exe packaging\windows\verify_installer.py
```

`build_installer.py` checks the verified portable ZIP checksum and every payload
file, then compiles `SpectraFit.iss`. The output is
`artifacts/SpectraFit-<version>-windows-x64-setup.exe` with an adjacent SHA-256
file. The application and installer are unsigned.

The installer targets Windows x64 (Windows 10 build 19041 or later; the tested
host is Windows 11). It installs for the current user under
`%LOCALAPPDATA%\Programs\SpectraFit`, adds Start menu entries, offers an optional
desktop shortcut, and registers its uninstaller. It does not change PATH or
file associations. Close the app before updating or uninstalling it.
Uninstallation keeps custom chromophores, settings and analysis results.

The installer verifier refuses to run if SpectraFit is already registered in
the current account. It installs to a new short temporary path without shortcuts,
checks every payload hash and uninstall registration, repairs a deliberately
removed guide, tests the installed CLI and GUI startup without external Python,
then uninstalls. It checks that registration and owned program files are removed,
test-created results/settings survive, and the existing custom chromophore
library is unchanged. This host check does not certify the interactive installer
wizard, shortcuts, cross-version upgrades or a clean Windows desktop.

## Prepare the deferred clean-machine validation kit

The kit contains synthetic fixtures, source-reference results, the portable ZIP,
a separate frozen verifier, a standalone test runner and a manual checklist.
It needs no installed Python. User measurements and references are excluded.

For a refresh, set `SPECTRAFIT_VALIDATION_DIR` to an absolute evidence directory
before building, verifying and packaging the application above. Generate the
synthetic fixtures with `validation/windows-step4/prepare_fixtures.py`, then build
`packaging/windows/ValidationController.spec` with PyInstaller (output directory
`artifacts/step4-tools`). Stage with
`packaging/windows/build_validation_kit.py --stage-only`. Run the staged
`Validate-Windows.exe --context host-rehearsal --noninteractive --results <new-directory>`.
Audit its MAT/TIFF exports with `validation/windows-step4/audit_exports.py <new-directory>`.
Finally run `build_validation_kit.py --rehearsal <new-directory>/report.json`.

The resulting `artifacts/SpectraFit-<version>-validation-kit-windows-x64.zip`
includes instructions for a later clean Windows run. A passing host rehearsal
does not complete clean-machine or manual validation; both remain pending.
