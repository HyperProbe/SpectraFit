"""Console and no-console desktop entry points, shared by source and frozen builds."""

from __future__ import annotations

import io
import logging
import multiprocessing
import os
import sys
import tempfile
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .paths import app_home


class _LogStream(io.TextIOBase):
    """A file-like logging target when pythonw/windowed builds have no streams."""

    def __init__(self, level: int):
        self.level = level

    @property
    def encoding(self):
        return "utf-8"

    def writable(self):
        return True

    def write(self, text):
        if text.strip():
            logging.getLogger("spectrafit.output").log(self.level, text.rstrip())
        return len(text)


def _configure_logging() -> Path:
    log_path = app_home() / "logs" / "spectrafit.log"
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
    except OSError:
        # A read-only override/profile must not hide the original startup error.
        fd, name = tempfile.mkstemp(prefix="spectrafit-", suffix=".log")
        os.close(fd)
        log_path = Path(name)
        handler = RotatingFileHandler(log_path, maxBytes=2 * 1024 * 1024, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    return log_path


def _show_startup_error(message: str) -> None:
    # The Windows dialog also works when Qt itself could not be imported.
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        box = ctypes.WinDLL("user32", use_last_error=True).MessageBoxW
        box.argtypes = [wintypes.HWND, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.UINT]
        box.restype = ctypes.c_int
        box(None, message, "SpectraFit — startup error", 0x10)
    else:
        from qtpy.QtWidgets import QApplication, QMessageBox

        app = QApplication.instance() or QApplication([])
        QMessageBox.critical(None, "SpectraFit — startup error", message)


def console_main(argv=None) -> int:
    multiprocessing.freeze_support()
    from .cli import main

    return main(argv)


def gui_main(argv=None) -> int:
    # Divert spawned frozen workers before parsing arguments or importing Qt.
    multiprocessing.freeze_support()
    log_path = None
    try:
        log_path = _configure_logging()
        if sys.stdout is None:
            sys.stdout = _LogStream(logging.INFO)
        if sys.stderr is None:
            sys.stderr = _LogStream(logging.ERROR)
        logging.getLogger(__name__).info("Starting SpectraFit; executable=%s", sys.executable)
        from .cli import main

        code = main(["gui", *(sys.argv[1:] if argv is None else argv)])
        if code:
            raise RuntimeError("The graphical interface could not start. See the log for details.")
        return code
    except Exception as error:
        logging.getLogger(__name__).exception("SpectraFit could not start")
        details = f"SpectraFit could not start.\n\n{error}"
        if log_path:
            details += f"\n\nDetails were saved to:\n{log_path}"
        try:
            _show_startup_error(details)
        except Exception:
            logging.getLogger(__name__).exception("Could not display the startup error dialog")
        return 1
