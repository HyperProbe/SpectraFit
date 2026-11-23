SpectraFit Windows validation kit

1. Extract this kit ZIP into a short folder in your user profile.
2. Double-click Validate-Windows.exe. No Python installation is needed.
3. Select C only for a fresh clean Windows environment with no Python installed.
   Select H for an existing development/analysis PC, or U if unsure.
4. Allow the automatic tests to finish. A synthetic-data GUI opens and closes.
5. Read the printed Results path and report.json. Complete MANUAL-CHECKLIST.md.

The kit and application are unsigned. Use your organisation's normal approval
process for unsigned research software.

The runner creates a short temporary application directory and stores results in
%LOCALAPPDATA%\SpectraFitValidation\Runs. It records OS/graphics details, Python
installation indicators, test logs, numerical comparisons and a GUI screenshot.
The temporary application remains available at the path printed in report.json
for the manual checks. No administrator privileges or OS feature changes are
required by the runner. Results are written locally.

This kit contains only synthetic measurement data. It includes the unchanged
portable application ZIP identified by kit-manifest.json, a separate frozen
verification executable and source-reference synthetic output maps. The kit uses
the same production executables as that portable ZIP.

Automated checks:
  - Every output array and NaN mask in 24 numerical cases, including all three
    fit models, absolute/differential modes, exact SciPy solving, MATLAB v5/v7.3,
    ENVI BIL/BIP/BSQ calibration, references, time series, masks, pathlength models
    and custom chromophores. Tolerances are rtol=1e-6, atol=1e-8 against source.
  - Three expected-invalid inputs, bundled resources, two-core worker cleanup
    and a no-console frozen GUI with a synthetic 64x64 image.
  - MAT file/header checks and float32 grayscale TIFF frame-page checks. The kit
    does not independently decode their numerical pixel values; this was
    additionally audited during host rehearsal.

The GUI verifier shares the production runtime and module archive. Its raw
comparison field retains the historical name max_absolute_difference_vs_step1;
in this kit its reference is the bundled synthetic source-reference NPZ file,
not the user's real cube.

A green automated result alone does not certify a clean machine, every driver,
display scaling, or all manual lifecycle checks. The report keeps clean-machine
validation and manual checks pending for review. Python registry/PATH checks
are indicators, not proof of a computer's software history.

Advanced usage:
  Validate-Windows.exe --context clean-windows --noninteractive
  Validate-Windows.exe --context host-rehearsal --noninteractive
  Validate-Windows.exe --results "C:\Users\me\Desktop\validation-results"

The selected results directory must not already exist, preserving earlier runs.
