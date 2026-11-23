# Manual Windows validation record

Record the machine/VM, Windows version, graphics driver, display scaling and date.
Use the application path and test fixtures recorded in `report.json`. Run as a
normal user. Keep the automatic report and screenshot with this completed record.

Do not call an existing development PC a clean-machine test. For the clean test,
record how the Windows environment was provisioned and confirm that Python and
SpectraFit were not installed beforehand. The runner's registry/PATH inventory
is supporting evidence; it cannot establish the machine's complete history.

| Check | Pass/fail/not tested | Evidence or observed problem |
|---|---|---|
| Production `SpectraFit.exe` opens normally, without a terminal window | | |
| Load `fixtures/cases/gui-two-core.yaml`, then Load data and Run fit | | |
| File/folder dialogs work with spaces and Unicode paths | | |
| Change reference pixel, draw ROI/crop and inspect measured/fitted spectra | | |
| Select result maps, adjust contrast and use side-by-side mode | | |
| Panels, controls and labels remain usable at the machine's normal scaling | | |
| If other scaling configurations are available, record their separate results | | |
| User guide opens the local offline HTML guide | | |
| Save/load settings round trip works from a user-writable directory | | |
| Add a copy of `fixtures/data/dye.csv`; it remains available after restart | | |
| Invalid input produces a readable error and permits a subsequent valid run | | |
| Stop during fitting returns control and a new fit can start | | |
| Close during fitting: record window/process/worker behaviour and any delay | | |
| Reopen after stopping/closing; no stale test workers remain | | |
| NPZ/MAT/TIFF exports open in the intended downstream tools, when available | | |
| Time-series TIFFs are interpreted as frames, not colour channels | | |
| Ordinary application use works without an internet connection | | |

Use synthetic data for the initial lifecycle checks. To make a fit long enough
to test Stop/Close, make a **copy alongside the original GUI settings** (preserving
its relative data paths). Set reference scattering `method: grid` and increase its
grid steps. Record those settings and the observed stop/close latency.
The automatic force-termination worker check does not replace a normal GUI-close
test.

Keep failures and untested items visible. Clean-machine and release validation
remain pending until the automatic results and these manual checks are reviewed.
