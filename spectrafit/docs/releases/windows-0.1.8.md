# Windows 0.1.8 release record

Status: **published** as [v0.1.8](https://github.com/HyperProbe/SpectraFit/releases/tag/v0.1.8) on 24 November 2026, from build commit `ba7d0ed`. The release page has user-facing notes; this file keeps the technical build and test record.

The four rebuilt assets are staged in an [unpublished upstream draft](https://github.com/HyperProbe/SpectraFit/releases/tag/untagged-0aea43f8ec1f4675d99b), visible to repository maintainers. Their uploaded sizes and GitHub SHA-256 digests were checked against the local verified packages. The draft targets the build commit below. Maintain that target when publishing so the source archive corresponds to the binaries.

The source fixes are already merged in PR #5. This change proposes release notes and checksums only; the binaries are release assets and are not committed to Git. After review, a maintainer can publish the draft as a prerelease. The older release in the contributor fork contains the earlier build and is not the source of these assets.

The proposed release notes follow.

---

Windows x64 preview rebuilt from upstream commit **ba7d0ed68ecd75ffd9850b3bed9ddd3bfdb6ed1a**, after merging PR #5 (Windows build hygiene).

### Downloads

- **Recommended:** `SpectraFit-0.1.8-windows-x64-setup.exe` installs for the current user, with no separate Python installation required.
- **Portable alternative:** extract `SpectraFit-0.1.8-windows-x64.zip` to a short path and run `SpectraFit.exe`. Keep the full extracted folder, including `_internal`.
- Each download has an adjacent `.sha256` checksum file.

### Included changes

The build restricts DLL search paths and rejects libraries collected from outside the Python environment or Windows. Installer metadata containing personal build paths is excluded, and packaging checks for personal-path leakage. This rebuild also includes the merged installation-guide and PolyForm Noncommercial licence updates. The mathematical fitting model is unchanged.

### Validation completed on the development Windows host

- Python 3.12.16, pinned dependencies, PyInstaller 6.22.3, Inno Setup 6.7.4.
- Source tests: **78 passed, 1 skipped** (optional comparison against absent original research code).
- Frozen single-core CLI, two-core CLI, and GUI each matched all **23 saved real-data baseline maps exactly**, with maximum absolute difference **0**.
- Frozen resources, worker cleanup, and production GUI startup passed.
- Portable ZIP integrity, shared runtime consistency, and relocated CLI startup with external Python paths disabled passed.
- Installer fresh installation, same-version repair, installed CLI/GUI startup, and uninstall passed. Test-created results/settings and the existing custom chromophore library were preserved.
- Independent audit matched **51 bundled source hashes** to the target upstream commit. No root `api-ms-win-*.dll`, root `freetype.dll`, recursive `direct_url.json`, or personal-path findings from the packaging scan remained.

Private measurement data, reference data, and local validation records are not included in these downloads. The optional validation kit from the earlier fork release was not rebuilt for this release.

### Preview limitations

The installer and executables are **unsigned**; Windows may display publisher or reputation prompts. Clean Windows machine/VM testing remains **deferred and unperformed**. Manual installer/shortcut checks, additional graphics drivers and DPI configurations, downstream-tool checks, and cross-version upgrade tests remain pending.

PyInstaller still reports a missing `scipy.special._cdflib` hidden import and `MSVCR90.dll` dependencies of legacy PyOpenGL GLUT/GLE binaries. The tested workflows passed despite these warnings. A napari point-rendering fallback was used successfully on the development host.

### SHA-256

```text
6993c2f8dde9eaa7a363497ea981af45c71ac3a5c7e406eda430ac34bc64baba  SpectraFit-0.1.8-windows-x64-setup.exe
d9e9aa8272694b468847a063e1f70f1375299fb29f9c0c19901f66a357bdd3e3  SpectraFit-0.1.8-windows-x64.zip
```
