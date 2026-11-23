"""Separate frozen verification entry point; never included in the user ZIP."""

import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path


def resources(output):
    from importlib import metadata
    import spectrafit
    from spectrafit.chromophores import ChromophoreLibrary
    from spectrafit.paths import find_guide
    from spectrafit.pathlength import gray_matter_table
    from npe2 import PluginManager

    root = Path(sys._MEIPASS)
    guide = find_guide()
    assert guide is not None and guide.is_relative_to(root)
    assert guide.read_text(encoding="utf-8").startswith("<!DOCTYPE html>")
    library = ChromophoreLibrary(include_user_dir=False)
    assert len(library.names()) == 10
    manager = PluginManager()
    manager.discover()
    plugins = [manifest.name for manifest in manager.iter_manifests()]
    assert "spectrafit" in plugins and "napari" in plugins, plugins
    assert (root / "spectrafit/data/config_template.yaml").is_file()
    assert sys.flags.utf8_mode == 1
    assert os.environ["QT_API"] == "pyqt6"
    return {"status": "passed", "frozen": sys.frozen, "executable": sys.executable,
            "utf8_mode": sys.flags.utf8_mode, "qt_api": os.environ["QT_API"],
            "bundle_root": str(root), "module_path": spectrafit.__file__, "guide": str(guide),
            "versions": {name: metadata.version(name) for name in ("spectrafit", "napari", "numpy", "scipy", "h5py", "PyQt6")},
            "chromophores": library.names(), "pathlength_samples": len(gray_matter_table()[0]), "plugins": plugins}


def busy(marker):
    Path(marker).write_text("running", encoding="utf-8")
    time.sleep(120)


def busy_parent(output):
    from spectrafit.parallel import start_pool

    pool = start_pool(2)
    markers = [output.with_name(f"busy-{i}") for i in range(2)]
    futures = [pool.submit(busy, str(marker)) for marker in markers]
    while not all(marker.exists() for marker in markers):
        time.sleep(0.05)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps({"pids": list(pool._processes)}), encoding="utf-8")
    temporary.replace(output)
    time.sleep(120)


def gui(settings, baseline, output):
    from spectrafit import cli, launcher

    report = {"status": "running", "frozen": sys.frozen,
              "started_without_stdout": sys.stdout is None, "started_without_stderr": sys.stderr is None}
    started = time.monotonic()
    original = cli.cmd_gui

    def command(args):
        import napari
        import numpy as np
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication
        from spectrafit.napari_plugin.widget import SpectraFitWidget
        from spectrafit.paths import find_guide

        original_run = napari.run

        def run():
            viewer = napari.current_viewer()
            widget, = viewer.window._qt_window.findChildren(SpectraFitWidget)
            widget.autosave.setChecked(False)
            phase = ["loading"]
            timer = QTimer()
            timer.setInterval(200)

            def finish(error=None):
                timer.stop()
                report["status"] = "failed" if error else "passed"
                if error:
                    report["error"] = error
                report["elapsed_seconds"] = time.monotonic() - started
                output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                viewer.close()
                QApplication.instance().quit()

            def capture():
                try:
                    image = viewer.screenshot(path=str(output.with_suffix(".png")), canvas_only=False, flash=False)
                    report["screenshot_shape"] = list(image.shape)
                    finish()
                except Exception:
                    finish(traceback.format_exc())

            def poll():
                try:
                    if time.monotonic() - started > 240:
                        raise TimeoutError(widget.status.text())
                    if widget._worker is not None:
                        return
                    if phase[0] == "loading":
                        assert widget.data is not None, widget.status.text()
                        report["loaded_shape"] = list(widget.data.cube.shape)
                        phase[0] = "fitting"
                        widget._run_fit()
                    elif phase[0] == "fitting":
                        result = widget.results
                        assert result is not None, widget.status.text()
                        assert any("2 processor cores" in note for note in result.notes)
                        differences = {}
                        with np.load(baseline, allow_pickle=False) as expected:
                            for name, values in result.maps.items():
                                np.testing.assert_allclose(values, expected[name], rtol=1e-6, atol=1e-8, equal_nan=True)
                                finite = np.isfinite(values) & np.isfinite(expected[name])
                                differences[name] = float(np.max(np.abs(values[finite] - expected[name][finite])))
                        report["max_absolute_difference_vs_step1"] = differences
                        report["fit_seconds"] = result.timing_s
                        report["notes"] = result.notes
                        report["guide_path"] = str(find_guide())
                        widget._focus_map("HbT")
                        row, col = np.unravel_index(np.nanargmax(result.maps["HHb"][0]), result.maps["HHb"][0].shape)
                        viewer.cursor.position = (float(row), float(col))
                        next(widget._on_click(viewer, type("Press", (), {"type": "mouse_press"})()), None)
                        widget.tabs.setCurrentIndex(4)
                        phase[0] = "rendering"
                        QTimer.singleShot(1000, capture)
                except Exception:
                    finish(traceback.format_exc())

            timer.timeout.connect(poll)
            timer.start()
            QTimer.singleShot(0, widget._load_data)
            original_run()

        napari.run = run
        return original(args)

    cli.cmd_gui = command
    launcher._show_startup_error = lambda message: report.update(startup_error=message)
    report["launcher_exit_code"] = launcher.gui_main([str(settings)])
    if report["launcher_exit_code"] or report["status"] != "passed":
        report["status"] = "failed"
    return report


if __name__ == "__main__":
    multiprocessing.freeze_support()
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("resources", "gui", "busy-parent"))
    parser.add_argument("output", type=Path)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        assert getattr(sys, "frozen", False)
        if args.mode == "busy-parent":
            busy_parent(args.output)
            raise SystemExit(0)
        report = resources(args.output) if args.mode == "resources" else gui(args.settings, args.baseline, args.output)
    except Exception:
        report = {"status": "failed", "traceback": traceback.format_exc()}
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    raise SystemExit(0 if report["status"] == "passed" else 1)
