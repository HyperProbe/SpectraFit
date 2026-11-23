SpectraFit - Windows x64 portable application (version in BUILD-MANIFEST.json)

Extract the entire ZIP into a short folder path (for example, the Apps folder
in your user profile), then double-click SpectraFit.exe. Deeply nested paths
can exceed Windows' path limit for bundled autocomplete data files.
Keep the _internal folder alongside both executables. Moving just an EXE
does not move the application. No Python installation is required by the bundle.

The offline user guide is available from the User guide button. A copy is
also included at _internal/README.html.

For command-line use, open a terminal in this folder:
  .\SpectraFit-cli.exe chromophores
  .\SpectraFit-cli.exe new-config settings.yaml
  .\SpectraFit-cli.exe run settings.yaml

Use quotes around paths containing spaces. Settings-relative data paths are
resolved relative to the settings file.

Desktop logs and added chromophores use %LOCALAPPDATA%\SpectraFit by default.
Set SPECTRAFIT_HOME to select another application-data directory.
Analysis results are written to the folder selected in the app/settings.

This build is unsigned. Clean-machine and broader Windows validation are
separate release-validation steps. See BUILD-MANIFEST.json for build details
and checksums. Licensing information is included in _internal/LICENSE and
the collected distribution metadata.
