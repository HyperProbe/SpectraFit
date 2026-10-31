"""SpectraFit dock widget for napari.

Tabs: 1 Data · 2 Reference · 3 Chromophores · 4 Model · 5 Run & results.
A second dock, the fit inspector, plots the spectrum and fit of any clicked pixel.
All settings map one-to-one to the YAML settings file (spectrafit.config.Config).
"""

from __future__ import annotations

import copy
import traceback
from pathlib import Path
from typing import List, Optional

import numpy as np
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
                            QFileDialog, QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                            QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
                            QSpinBox, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ..chromophores import (UNIT_DESCRIPTIONS, Chromophore, ChromophoreLibrary, absorption_matrix, builtin_dir,
                            collinearity_report, common_range, detection_limits, read_chromophore_file, user_dir,
                            write_chromophore_file)
from ..config import (MODE_LABELS, PATHLENGTH_LABELS, PATHLENGTH_MODES, REFERENCE_LABELS, REFERENCE_TYPES, Config)
from ..io import inspect_file, read_spectrum_file
from ..pathlength import JACQUES_TISSUES, global_pathlength

AXES_CHOICES = ["(automatic)", "HWB", "WHB", "BHW", "BWH", "HBW", "WBH", "HWBT", "WHBT", "BHWT", "THWB", "HWTB"]
MAIN_MAPS = ["HbT", "StO2", "diffCCO", "b", "delta_a", "residual_rms"]


# ---------------------------------------------------------------------------- small helpers
def _dspin(lo, hi, val, step=0.1, dec=3, suffix="") -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(dec)
    s.setSingleStep(step)
    s.setValue(val)
    if suffix:
        s.setSuffix(suffix)
    return s


def _ispin(lo, hi, val) -> QSpinBox:
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def _combo(items, data=None) -> QComboBox:
    c = QComboBox()
    for i, it in enumerate(items):
        c.addItem(it, data[i] if data else it)
    return c


def _set_combo_data(c: QComboBox, value) -> None:
    i = c.findData(value)
    if i >= 0:
        c.setCurrentIndex(i)


def _browse_row(edit: QLineEdit, callback) -> QWidget:
    w = QWidget()
    lay = QHBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.addWidget(edit, 1)
    b = QPushButton("Browse…")
    b.clicked.connect(callback)
    lay.addWidget(b)
    return w


def _hint(text: str) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setStyleSheet("color: #8a8a8a; font-size: 11px;")
    return lab


class _Signals(QObject):
    progress = Signal(float, str)


# ---------------------------------------------------------------------------- matplotlib panel
class PlotPanel(QWidget):
    def __init__(self, height=3.0):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.fig = Figure(figsize=(4.5, height), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        self.canvas.setMinimumHeight(int(height * 70))

    def clear(self, title=""):
        self.ax.clear()
        self.ax.set_title(title, fontsize=9)
        self.ax.tick_params(labelsize=8)

    def draw(self):
        self.ax.grid(alpha=0.3)
        if self.ax.get_legend_handles_labels()[0]:
            self.ax.legend(fontsize=6, loc="best", ncol=2, framealpha=0.6, handlelength=1.5, columnspacing=0.8)
        self.canvas.draw_idle()


class InspectorWidget(QWidget):
    """Shows the clicked pixel: measured attenuation, fitted model and each component."""

    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)
        self.info = QLabel("Click a pixel in the image to see its spectrum and fit.")
        self.info.setWordWrap(True)
        lay.addWidget(self.info)
        self.plot = PlotPanel(height=3.6)
        lay.addWidget(self.plot)
        self.show_components = QCheckBox("Show contribution of each chromophore")
        self.show_components.setChecked(True)
        lay.addWidget(self.show_components)
        self._last = None
        self.show_components.toggled.connect(lambda: self._last and self.show_fit(*self._last))

    def show_reflectance(self, wl, spec, title):
        self.plot.clear(title)
        self.plot.ax.plot(wl, spec, "k-", lw=1.2, label="reflectance")
        self.plot.ax.set_xlabel("Wavelength (nm)", fontsize=8)
        self.plot.ax.set_ylabel("Reflectance", fontsize=8)
        self.plot.draw()

    def show_fit(self, fit: dict, title: str, absolute: bool):
        self._last = (fit, title, absolute)
        self.plot.clear(title)
        ax = self.plot.ax
        wl = fit["wavelengths"]
        ax.plot(wl, fit["measured"], "k.", ms=4, label="measured")
        if fit["fitted"] is not None:
            ax.plot(wl, fit["fitted"], "r-", lw=1.5, label=f"fit (RMS {fit.get('residual_rms', np.nan):.2g})")
            if self.show_components.isChecked():
                for name, comp in fit["components"].items():
                    ax.plot(wl, comp, "--", lw=0.9, label=name)
        ax.set_xlabel("Wavelength (nm)", fontsize=8)
        ax.set_ylabel("Attenuation A = −ln R" if absolute else "ΔA = −ln(R / R_ref)", fontsize=8)
        self.plot.draw()


# ---------------------------------------------------------------------------- add-chromophore dialog
class AddChromophoreDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add a chromophore")
        lay = QFormLayout(self)
        lay.addRow(_hint("A text or CSV file with two columns: wavelength in nm, and the absorption value. "
                         "Header lines are ignored. The chromophore is saved to your library and stays available."))
        self.file = QLineEdit()
        lay.addRow("Spectrum file", _browse_row(self.file, self._browse))
        self.name = QLineEdit()
        self.name.setPlaceholderText("short name, e.g. ICG")
        lay.addRow("Name", self.name)
        self.label = QLineEdit()
        self.label.setPlaceholderText("description (optional)")
        lay.addRow("Description", self.label)
        self.unit = QComboBox()
        for k, d in UNIT_DESCRIPTIONS.items():
            self.unit.addItem(d, k)
        lay.addRow("Values are", self.unit)
        self.baseline = _dspin(0, 1e6, 0.0, 0.001, 5)
        lay.addRow("Typical tissue concentration", self.baseline)
        lay.addRow(_hint("Used only by the δ-P1 and Jacques pathlength models (to compute tissue absorption). "
                         "Leave 0 if unknown. Unit: mM, or volume fraction for 'pure substance' spectra."))
        self.source = QLineEdit()
        self.source.setPlaceholderText("reference / where the spectrum comes from (optional)")
        lay.addRow("Source", self.source)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addRow(bb)
        self.result_path: Optional[Path] = None

    def _browse(self):
        f, _ = QFileDialog.getOpenFileName(self, "Spectrum file", "", "Text files (*.txt *.csv *.dat);;All files (*)")
        if f:
            self.file.setText(f)
            if not self.name.text():
                self.name.setText(Path(f).stem)
            try:  # a file already in library format fills the form
                c = read_chromophore_file(f)
                self.name.setText(c.name)
                self.label.setText(c.label)
                _set_combo_data(self.unit, c.unit)
                self.baseline.setValue(c.baseline)
                self.source.setText(c.source)
            except Exception:
                pass

    def _accept(self):
        try:
            name = self.name.text().strip()
            if not name or not name.replace("_", "").replace("-", "").isalnum():
                raise ValueError("Use a short name with letters, digits, - or _ only.")
            wl, val = read_spectrum_file(self.file.text())
            c = Chromophore(name=name, wavelengths=wl, values=val, unit=self.unit.currentData(),
                            label=self.label.text().strip() or name, baseline=self.baseline.value(),
                            source=self.source.text().strip())
            self.result_path = write_chromophore_file(c, user_dir() / f"{name}.csv")
            self.accept()
        except Exception as e:
            QMessageBox.warning(self, "Cannot add chromophore", str(e))


# ---------------------------------------------------------------------------- main widget
class SpectraFitWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.data = None
        self.results = None
        self._worker = None
        self._stop = False
        self._signals = _Signals()
        self._signals.progress.connect(self._on_progress)
        self.library = ChromophoreLibrary()
        self._file_info = None

        outer = QVBoxLayout(self)
        top = QHBoxLayout()
        b_load = QPushButton("Load settings…")
        b_load.clicked.connect(self._load_settings_dialog)
        b_save = QPushButton("Save settings…")
        b_save.clicked.connect(self._save_settings_dialog)
        top.addWidget(b_load)
        top.addWidget(b_save)
        outer.addLayout(top)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_data_tab(), "1 Data")
        self.tabs.addTab(self._build_reference_tab(), "2 Reference")
        self.tabs.addTab(self._build_chromophore_tab(), "3 Chromophores")
        self.tabs.addTab(self._build_model_tab(), "4 Model")
        self.tabs.addTab(self._build_run_tab(), "5 Run")
        outer.addWidget(self.tabs, 1)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.status = QLabel("Start by choosing a data file in tab 1.")
        self.status.setWordWrap(True)
        outer.addWidget(self.progress)
        outer.addWidget(self.status)

        self.inspector = InspectorWidget()
        try:
            self.viewer.window.add_dock_widget(self.inspector, name="Fit inspector", area="right")
        except Exception:  # e.g. when napari opens us as a plugin before the window exists
            pass
        self.viewer.mouse_drag_callbacks.append(self._on_click)
        self.setMinimumWidth(460)
        self._update_visibility()
        self._refresh_chromophore_table(Config().chromophores.names)

    # ======================================================================== tab 1: data
    def _build_data_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("a .mat file, or a folder of .mat files (time series)")
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.file_edit, 1)
        b1 = QPushButton("File…")
        b1.clicked.connect(self._browse_data_file)
        b2 = QPushButton("Folder…")
        b2.clicked.connect(self._browse_data_folder)
        rl.addWidget(b1)
        rl.addWidget(b2)
        form.addRow("Data", row)
        self.var_combo = QComboBox()
        self.var_combo.setEditable(True)
        form.addRow("Cube variable", self.var_combo)
        self.axes_combo = _combo(AXES_CHOICES)
        self.axes_combo.setEditable(True)
        self.axes_combo.setToolTip("MATLAB order of the array: H rows, W columns, B bands, T time. Usually detected automatically.")
        form.addRow("Axis order", self.axes_combo)
        self.wl_mode = _combo(["Read from file (automatic)", "First wavelength and step"], ["auto", "startstep"])
        self.wl_mode.currentIndexChanged.connect(self._update_visibility)
        form.addRow("Wavelengths", self.wl_mode)
        self.wl_start = _dspin(100, 3000, 385, 1, 2, " nm")
        self.wl_step = _dspin(0.01, 100, 5, 0.5, 3, " nm")
        self.wl_row = QWidget()
        wr = QHBoxLayout(self.wl_row)
        wr.setContentsMargins(0, 0, 0, 0)
        wr.addWidget(QLabel("start"))
        wr.addWidget(self.wl_start)
        wr.addWidget(QLabel("step"))
        wr.addWidget(self.wl_step)
        form.addRow("", self.wl_row)
        self.kind_combo = _combo(["Reflectance (already calibrated)", "Raw counts (white/dark calibration)"],
                                 ["refl", "raw"])
        self.kind_combo.currentIndexChanged.connect(self._update_visibility)
        form.addRow("Values are", self.kind_combo)
        self.white_edit = QLineEdit()
        self.white_edit.setPlaceholderText("variable name in the file, or a .mat file")
        self.dark_edit = QLineEdit()
        self.dark_edit.setPlaceholderText("variable name in the file, or a .mat file")
        self.white_row = _browse_row(self.white_edit, lambda: self._browse_into(self.white_edit, "*.mat"))
        self.dark_row = _browse_row(self.dark_edit, lambda: self._browse_into(self.dark_edit, "*.mat"))
        form.addRow("White reference", self.white_row)
        form.addRow("Dark reference", self.dark_row)
        self.offset_auto = QCheckBox("automatic")
        self.offset_auto.setToolTip("0.1 for raw data (as in the HELICOID pipeline), 0 for reflectance data")
        self.offset_auto.setChecked(True)
        self.offset_spin = _dspin(-1, 10, 0.1, 0.01, 4)
        self.offset_auto.toggled.connect(self._update_visibility)
        orow = QWidget()
        ol = QHBoxLayout(orow)
        ol.setContentsMargins(0, 0, 0, 0)
        ol.addWidget(self.offset_auto)
        ol.addWidget(self.offset_spin)
        form.addRow("Offset added to R", orow)
        self.binning = _ispin(1, 32, 4)
        self.binning.setToolTip("Averages n×n pixels. 4 turns 2048×2048 into 512×512: 16× faster and less noisy.")
        form.addRow("Binning (n×n)", self.binning)
        self.frames_edit = QLineEdit()
        self.frames_edit.setPlaceholderText("all frames  (or e.g. 0, 50  or  0, 50, 2)")
        form.addRow("Time frames", self.frames_edit)
        lay.addLayout(form)

        crop_box = QGroupBox("Crop")
        cl = QHBoxLayout(crop_box)
        self.crop_label = QLabel("whole image")
        b_crop = QPushButton("Crop to drawn rectangle")
        b_crop.setToolTip("Draw a rectangle in the 'Crop' layer, then press this. The data is reloaded.")
        b_crop.clicked.connect(self._apply_crop)
        b_draw = QPushButton("Draw…")
        b_draw.clicked.connect(lambda: self._start_shape_layer("Crop", "add_rectangle", "yellow"))
        b_reset = QPushButton("Reset")
        b_reset.clicked.connect(self._reset_crop)
        cl.addWidget(self.crop_label, 1)
        cl.addWidget(b_draw)
        cl.addWidget(b_crop)
        cl.addWidget(b_reset)
        lay.addWidget(crop_box)
        self._crop = None

        self.load_button = QPushButton("Load data")
        self.load_button.setStyleSheet("font-weight: bold; padding: 6px;")
        self.load_button.clicked.connect(self._load_data)
        lay.addWidget(self.load_button)
        self.data_info = QPlainTextEdit()
        self.data_info.setReadOnly(True)
        self.data_info.setMaximumHeight(110)
        lay.addWidget(self.data_info)
        lay.addStretch(1)
        return w

    def _browse_data_file(self):
        f, _ = QFileDialog.getOpenFileName(self, "Hyperspectral data", "", "MATLAB files (*.mat);;All files (*)")
        if f:
            self.file_edit.setText(f)
            self._inspect(f)

    def _browse_data_folder(self):
        d = QFileDialog.getExistingDirectory(self, "Folder with one .mat file per time frame")
        if d:
            self.file_edit.setText(d)
            files = sorted(Path(d).glob("*.mat"))
            if files:
                self._inspect(str(files[0]), n_files=len(files))

    def _browse_into(self, edit: QLineEdit, pattern: str):
        f, _ = QFileDialog.getOpenFileName(self, "Select file", "", f"Files ({pattern});;All files (*)")
        if f:
            edit.setText(f)

    def _inspect(self, path: str, n_files: int = 1):
        try:
            info = inspect_file(path)
        except Exception as e:
            self.data_info.setPlainText(f"Could not read the file: {e}")
            return
        self._file_info = info
        self.var_combo.clear()
        for k, s in info["variables"].items():
            self.var_combo.addItem(f"{k}", k)
            self.var_combo.setItemData(self.var_combo.count() - 1, f"shape {list(s)}", Qt.ItemDataRole.ToolTipRole)
        _set_combo_data(self.var_combo, info["cube"])
        lines = ["Variables: " + ", ".join(f"{k} {list(s)}" for k, s in info["variables"].items()),
                 f"Cube: '{info['cube']}' {list(info['shape'])}, detected axis order {info['axes']}"]
        if info["wavelength_variable"]:
            lines.append(f"Wavelengths found in variable '{info['wavelength_variable']}'.")
        elif info["n_bands"] == 127:
            lines.append("No wavelength variable: 385–1015 nm in 5 nm steps will be assumed (change below if wrong).")
        else:
            lines.append("No wavelength variable found: set the first wavelength and step.")
            _set_combo_data(self.wl_mode, "startstep")
        if n_files > 1:
            lines.append(f"{n_files} files in the folder: one time frame each.")
        self.data_info.setPlainText("\n".join(lines))
        self.status.setText("Check the settings, then press 'Load data'.")

    # ------------------------------------------------------------------ load
    def _load_data(self):
        try:
            cfg = self.gather_config()
        except Exception as e:
            return self._error("Settings", e)
        self._run_in_thread(self._do_load, cfg.data, on_done=self._loaded, label="Loading")

    def _do_load(self, data_settings):
        from ..io import load_data

        return load_data(data_settings, progress=self._emit)

    def _loaded(self, data):
        self.data = data
        self.results = None
        self._remove_layers(prefix="SpectraFit: ")
        T = data.n_frames
        rgb = np.stack([data.rgb(frame=t) for t in range(T)]) if T > 1 else data.rgb()
        self._add_or_replace_image("SpectraFit: colour preview", rgb, rgb=True)
        # bands first: napari aligns trailing axes, so maps (T, H, W) share the time axis with the cube (B, T, H, W)
        cube = np.moveaxis(data.cube, -1, 0) if T > 1 else np.moveaxis(data.cube[0], -1, 0)
        lay = self._add_or_replace_image("SpectraFit: reflectance", cube, colormap="gray", visible=False)
        lay.metadata["wavelengths"] = data.wavelengths
        self._sync_geometry_layers()
        H, W = data.shape
        lines = [f"Loaded {len(data.files)} file(s); working image {H}×{W} pixels, {T} frame(s), "
                 f"{data.wavelengths.size} bands {data.wavelengths[0]:g}–{data.wavelengths[-1]:g} nm."] + data.notes
        self.data_info.setPlainText("\n".join(lines))
        self.status.setText("Data loaded. Next: choose the reference in tab 2.")
        self.viewer.reset_view()
        self._update_chromophore_plot()

    # ------------------------------------------------------------------ crop
    def _apply_crop(self):
        if self.data is None:
            return self._info("Load the data first.")
        poly = self._first_shape("Crop")
        if poly is None:
            return self._info("Draw a rectangle in the 'Crop' layer first (button 'Draw…').")
        rows, cols = poly[:, -2], poly[:, -1]
        r0, c0 = self.data.to_original(max(0, rows.min()), max(0, cols.min()))
        r1, c1 = self.data.to_original(rows.max(), cols.max())
        self._crop = [int(r0), int(np.ceil(r1)), int(c0), int(np.ceil(c1))]
        self.crop_label.setText(f"rows {self._crop[0]}–{self._crop[1]}, cols {self._crop[2]}–{self._crop[3]}")
        self._remove_layers(names=["Crop"])
        self._load_data()

    def _reset_crop(self):
        self._crop = None
        self.crop_label.setText("whole image")
        if self.data is not None:
            self._load_data()

    # ======================================================================== tab 2: reference
    def _build_reference_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.ref_type = _combo([REFERENCE_LABELS[t] for t in REFERENCE_TYPES], list(REFERENCE_TYPES))
        _set_combo_data(self.ref_type, "pixel")
        self.ref_type.currentIndexChanged.connect(self._update_visibility)
        form.addRow("Reference", self.ref_type)
        lay.addLayout(form)
        lay.addWidget(_hint("With a reference, changes ΔA = −ln(R/R_ref) are fitted (differential Beer–Lambert law). "
                            "Without, the absolute attenuation −ln R is fitted with concentrations ≥ 0."))

        self.ref_pixel_box = QGroupBox("Reference pixel")
        pl = QFormLayout(self.ref_pixel_box)
        b_pick = QPushButton("Pick on image")
        b_pick.setToolTip("Click once on the image to place the reference pixel.")
        b_pick.clicked.connect(self._start_pick_pixel)
        pl.addRow(b_pick)
        self.ref_row = _ispin(0, 100000, 0)
        self.ref_col = _ispin(0, 100000, 0)
        rr = QWidget()
        rl = QHBoxLayout(rr)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel("row"))
        rl.addWidget(self.ref_row)
        rl.addWidget(QLabel("column"))
        rl.addWidget(self.ref_col)
        pl.addRow("Position (original pixels)", rr)
        self.ref_row.valueChanged.connect(self._sync_geometry_layers)
        self.ref_col.valueChanged.connect(self._sync_geometry_layers)
        lay.addWidget(self.ref_pixel_box)

        self.ref_roi_box = QGroupBox("Reference region")
        rl2 = QVBoxLayout(self.ref_roi_box)
        b_roi = QPushButton("Draw region (polygon)")
        b_roi.clicked.connect(lambda: self._start_shape_layer("Reference region", "add_polygon", "orange"))
        rl2.addWidget(b_roi)
        rl2.addWidget(_hint("Click the corners, double-click to close. The mean spectrum inside is the reference."))
        lay.addWidget(self.ref_roi_box)

        self.ref_file_box = QGroupBox("Reference spectrum file")
        fl = QFormLayout(self.ref_file_box)
        self.ref_file = QLineEdit()
        fl.addRow(_browse_row(self.ref_file, lambda: self._browse_into(self.ref_file, "*.csv *.txt")))
        fl.addRow(_hint("Two columns: wavelength in nm, reflectance."))
        lay.addWidget(self.ref_file_box)

        self.ref_frame_box = QGroupBox("Reference frame(s)")
        ff = QFormLayout(self.ref_frame_box)
        self.ref_f0 = _ispin(0, 100000, 0)
        self.ref_f1 = _ispin(0, 100000, 0)
        ff.addRow("From frame", self.ref_f0)
        ff.addRow("To frame (inclusive)", self.ref_f1)
        self.ref_frame_hint = _hint("Each pixel is compared with itself in these frames (their mean). "
                                    "For a pixel or region reference in a time series, the 'From frame' is used.")
        ff.addRow(self.ref_frame_hint)
        lay.addWidget(self.ref_frame_box)

        b_show = QPushButton("Plot reference spectrum")
        b_show.clicked.connect(self._plot_reference)
        lay.addWidget(b_show)
        lay.addStretch(1)
        return w

    def _start_pick_pixel(self):
        if self.data is None:
            return self._info("Load the data first.")
        layer = self._get_points_layer()
        layer.data = np.empty((0, 2))
        layer.mode = "add"
        self.viewer.layers.selection.active = layer
        self.status.setText("Click on the image to set the reference pixel.")

    def _get_points_layer(self):
        name = "Reference pixel"
        if name in self.viewer.layers:
            return self.viewer.layers[name]
        layer = self.viewer.add_points(np.empty((0, 2)), name=name, size=max(1.0, max(self.data.shape) / 40),
                                       face_color="transparent", border_color="cyan", border_width=0.25,
                                       symbol="cross", ndim=2)
        layer.events.data.connect(self._points_changed)
        return layer

    def _points_changed(self, event=None):
        layer = self.viewer.layers["Reference pixel"]
        if len(layer.data) == 0 or self.data is None:
            return
        if len(layer.data) > 1:
            layer.data = layer.data[-1:]
            return
        r, c = layer.data[-1][-2:]
        orow, ocol = self.data.to_original(int(np.floor(r)), int(np.floor(c)))
        for s, v in ((self.ref_row, orow), (self.ref_col, ocol)):
            s.blockSignals(True)
            s.setValue(int(v))
            s.blockSignals(False)
        layer.mode = "pan_zoom"
        _set_combo_data(self.ref_type, "pixel")
        self.status.setText(f"Reference pixel set to row {int(orow)}, column {int(ocol)} (original image).")
        self._plot_reference()

    def _plot_reference(self):
        if self.data is None:
            return self._info("Load the data first.")
        try:
            from ..preprocess import build_reference

            cfg = self.gather_config()
            ref = build_reference(self.data, cfg.reference)
        except Exception as e:
            return self._error("Reference", e)
        if ref.spectrum is None:
            return self._info(ref.description)
        self.inspector.show_reflectance(self.data.wavelengths, ref.spectrum, "Reference spectrum")

    # ======================================================================== tab 3: chromophores
    def _build_chromophore_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.addWidget(_hint("Tick the absorbers to fit. The baseline is the typical tissue concentration, "
                            "used only by the δ-P1 / Jacques pathlength models."))
        self.chrom_table = QTableWidget(0, 5)
        self.chrom_table.setHorizontalHeaderLabels(["Fit", "Chromophore", "Baseline", "Unit", "± resolvable"])
        self.chrom_table.horizontalHeaderItem(4).setToolTip(
            "Smallest change the fit can resolve (1 standard deviation) for noise of 0.001 in ΔA, "
            "with the current wavelength range and model. Orange/red: large compared with the baseline.")
        self.chrom_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.chrom_table.verticalHeader().setVisible(False)
        self.chrom_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.chrom_table.itemChanged.connect(lambda *_: self._update_chromophore_plot())
        self.chrom_table.verticalHeader().setDefaultSectionSize(24)
        self.chrom_table.setMinimumHeight(24 * 11 + 30)
        lay.addWidget(self.chrom_table, 1)
        row = QHBoxLayout()
        b_add = QPushButton("Add chromophore from file…")
        b_add.clicked.connect(self._add_chromophore)
        b_folder = QPushButton("Open my chromophore folder")
        b_folder.clicked.connect(self._open_user_folder)
        row.addWidget(b_add)
        row.addWidget(b_folder)
        lay.addLayout(row)
        self.coll_label = QLabel("")
        self.coll_label.setWordWrap(True)
        lay.addWidget(self.coll_label)
        self.chrom_plot = PlotPanel(height=2.3)
        lay.addWidget(self.chrom_plot)
        return w

    def _refresh_chromophore_table(self, selected: List[str], baselines: Optional[dict] = None):
        self.library = ChromophoreLibrary(extra_files=getattr(self, "_extra_files", []))
        baselines = baselines or {}
        t = self.chrom_table
        t.blockSignals(True)
        t.setRowCount(0)
        for name in self.library.names():
            c = self.library.get(name)
            r = t.rowCount()
            t.insertRow(r)
            chk = QTableWidgetItem()
            chk.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            chk.setCheckState(Qt.CheckState.Checked if name in selected else Qt.CheckState.Unchecked)
            t.setItem(r, 0, chk)
            it = QTableWidgetItem(name)
            it.setData(Qt.ItemDataRole.UserRole, name)
            it.setToolTip(f"{c.label}\n{c.range[0]:g}–{c.range[1]:g} nm; {UNIT_DESCRIPTIONS[c.unit]}\n{c.source}")
            it.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(r, 1, it)
            b = QTableWidgetItem(f"{baselines.get(name, c.baseline):g}")
            t.setItem(r, 2, b)
            u = QTableWidgetItem(c.concentration_unit)
            u.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(r, 3, u)
            d = QTableWidgetItem("")
            d.setFlags(Qt.ItemFlag.ItemIsEnabled)
            t.setItem(r, 4, d)
        t.resizeColumnToContents(0)
        t.blockSignals(False)
        self._update_chromophore_plot()

    def _selected_chromophores(self):
        names, baselines = [], {}
        for r in range(self.chrom_table.rowCount()):
            name = self.chrom_table.item(r, 1).data(Qt.ItemDataRole.UserRole)
            try:
                b = float(self.chrom_table.item(r, 2).text())
            except ValueError:
                b = self.library.get(name).baseline
            if b != self.library.get(name).baseline:
                baselines[name] = b
            if self.chrom_table.item(r, 0).checkState() == Qt.CheckState.Checked:
                names.append(name)
        return names, baselines

    def _update_chromophore_plot(self):
        if not hasattr(self, "chrom_plot") or not hasattr(self, "range_lo"):
            return
        names, _ = self._selected_chromophores()
        self.chrom_plot.clear("Absorption of selected chromophores (per unit concentration)")
        if not names:
            self.coll_label.setText("Select at least one chromophore.")
            self.chrom_plot.draw()
            return
        chroms = [self.library.get(n) for n in names]
        lo, hi = common_range(chroms)
        lo, hi = max(lo, self.range_lo.value()), min(hi, self.range_hi.value())
        if hi <= lo:
            self.coll_label.setText("The fit range lies outside the tabulated range of the selected spectra.")
            self.chrom_plot.draw()
            return
        wl = self.data.wavelengths if self.data is not None else np.arange(lo, hi + 1, 1.0)
        wl = wl[(wl >= lo) & (wl <= hi)]
        if wl.size < 2:
            return
        M = absorption_matrix(chroms, wl)
        for k, n in enumerate(names):
            self.chrom_plot.ax.semilogy(wl, np.clip(M[:, k], 1e-6, None), lw=1.2, label=n)
        self.chrom_plot.ax.set_xlabel("Wavelength (nm)", fontsize=8)
        self.chrom_plot.ax.set_ylabel("cm⁻¹ per mM (or per fraction)", fontsize=8)
        self.chrom_plot.draw()
        self._update_detection_limits(names, chroms, wl, M)
        rep = collinearity_report(M, names)
        a, b, s = rep["most_similar"]
        colour = {"ok": "#2e7d32", "warning": "#e65100", "bad": "#c62828"}[rep["level"]]
        msg = {"ok": "Spectra are distinguishable in this range.",
               "warning": "Some spectra are similar in this range; check their standard-error maps.",
               "bad": "Spectra are nearly indistinguishable in this range; results for them will be unreliable."}
        pair = f" Most similar: {a} and {b} ({s:.3f})." if a else ""
        self.coll_label.setText(f"<span style='color:{colour}'>●</span> {msg[rep['level']]} "
                                f"Condition number {rep['condition_number']:.3g}.{pair}")

    def _update_detection_limits(self, names, chroms, wl, M):
        """Fill the '± resolvable' column for the current model (scattering terms included)."""
        from qtpy.QtGui import QColor

        cols = [M]
        mode = self.mode_combo.currentData() if hasattr(self, "mode_combo") else "no_scattering"
        lam0 = self.lambda0.value() if hasattr(self, "lambda0") else 500.0
        b = self.fixed_b.value() if mode == "fixed_b" else (self.b0.value() if hasattr(self, "b0") else 1.0)
        if mode != "no_scattering":
            S = (wl / lam0) ** (-b)
            cols.append(S[:, None])
            if mode == "free_scattering":
                cols.append((-S * np.log(wl / lam0))[:, None])
        if self.ref_type.currentData() == "none" and getattr(self, "abs_offset", None) and self.abs_offset.isChecked():
            cols.append(np.ones((wl.size, 1)))
        lim = detection_limits(np.hstack(cols))
        self.chrom_table.blockSignals(True)
        for r in range(self.chrom_table.rowCount()):
            name = self.chrom_table.item(r, 1).data(Qt.ItemDataRole.UserRole)
            item = self.chrom_table.item(r, 4)
            if name not in names:
                item.setText("")
                continue
            k = names.index(name)
            item.setText(f"{lim[k]:.2g}")
            base = chroms[k].baseline
            colour = None
            if base > 0:
                ratio = lim[k] / base
                colour = "#2e7d32" if ratio < 0.05 else ("#e65100" if ratio < 0.5 else "#c62828")
            item.setForeground(QColor(colour) if colour else QColor("#8a8a8a"))
        self.chrom_table.blockSignals(False)

    def _add_chromophore(self):
        dlg = AddChromophoreDialog(self)
        if dlg.exec() and dlg.result_path:
            names, baselines = self._selected_chromophores()
            new = read_chromophore_file(dlg.result_path).name
            self._refresh_chromophore_table(names + [new], baselines)
            self.status.setText(f"Added '{new}' to your chromophore library ({dlg.result_path}).")

    def _open_user_folder(self):
        from qtpy.QtCore import QUrl
        from qtpy.QtGui import QDesktopServices

        user_dir().mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(user_dir())))

    # ======================================================================== tab 4: model
    def _build_model_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        f1 = QFormLayout()
        self.range_lo = _dspin(200, 3000, 530, 5, 1, " nm")
        self.range_hi = _dspin(200, 3000, 750, 5, 1, " nm")
        self.range_lo.valueChanged.connect(self._update_chromophore_plot)
        self.range_hi.valueChanged.connect(self._update_chromophore_plot)
        rr = QWidget()
        rl = QHBoxLayout(rr)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.range_lo)
        rl.addWidget(QLabel("to"))
        rl.addWidget(self.range_hi)
        f1.addRow("Fit range", rr)
        self.exclude_edit = QLineEdit()
        self.exclude_edit.setPlaceholderText("e.g. 600-610, 700-705")
        f1.addRow("Leave out (nm)", self.exclude_edit)
        self.mode_combo = _combo([MODE_LABELS[m] for m in MODE_LABELS], list(MODE_LABELS))
        _set_combo_data(self.mode_combo, "free_scattering")
        self.mode_combo.currentIndexChanged.connect(self._update_visibility)
        self.mode_combo.currentIndexChanged.connect(self._update_chromophore_plot)
        f1.addRow("Model", self.mode_combo)
        self.fixed_b = _dspin(0, 10, 1.0, 0.05, 3)
        self.fixed_b_label = QLabel("Scattering power b")
        f1.addRow(self.fixed_b_label, self.fixed_b)
        lay.addLayout(f1)
        self.model_hint = _hint("")
        lay.addWidget(self.model_hint)

        self.refscat_box = QGroupBox("Scattering of the reference (a0, b0)")
        rf = QFormLayout(self.refscat_box)
        self.refscat_method = _combo(["Grid search (as in the paper code)", "Enter values"], ["grid", "manual"])
        self.refscat_method.currentIndexChanged.connect(self._update_visibility)
        rf.addRow(self.refscat_method)
        self.a0 = _dspin(0, 1e4, 0.0, 1, 3)
        self.b0 = _dspin(0, 10, 1.0, 0.05, 3)
        self.a_steps = _ispin(1, 200, 3)
        self.b_steps = _ispin(1, 200, 25)
        self.grid_bin = _ispin(1, 64, 8)
        self.a0_row = self._pair("a0", self.a0, "b0", self.b0)
        self.grid_row = self._pair("a steps", self.a_steps, "b steps", self.b_steps)
        rf.addRow(self.a0_row)
        rf.addRow(self.grid_row)
        self.grid_bin_label = QLabel("Use every n-th pixel")
        rf.addRow(self.grid_bin_label, self.grid_bin)
        lay.addWidget(self.refscat_box)

        pbox = QGroupBox("Optical pathlength")
        pf = QFormLayout(pbox)
        self.pl_mode = _combo([PATHLENGTH_LABELS[m] for m in PATHLENGTH_MODES], list(PATHLENGTH_MODES))
        self.pl_mode.currentIndexChanged.connect(self._update_visibility)
        pf.addRow(self.pl_mode)
        self.pl_const = _dspin(1e-4, 1e3, 1.0, 0.1, 4, " cm")
        self.pl_const_label = QLabel("Pathlength")
        pf.addRow(self.pl_const_label, self.pl_const)
        self.pl_file = QLineEdit()
        self.pl_file_row = _browse_row(self.pl_file, lambda: self._browse_into(self.pl_file, "*.csv *.txt"))
        self.pl_file_label = QLabel("File (nm, cm)")
        pf.addRow(self.pl_file_label, self.pl_file_row)
        self.t_a = _dspin(0, 1e4, 40.8, 1, 3)
        self.t_b = _dspin(0, 10, 3.089, 0.05, 3)
        self.t_g = _dspin(0, 0.999, 0.85, 0.01, 3)
        self.t_n = _dspin(1, 2, 1.36, 0.01, 3)
        self.tissue_row1 = self._pair("μs′ a", self.t_a, "b", self.t_b)
        self.tissue_row2 = self._pair("g", self.t_g, "n", self.t_n)
        pf.addRow(self.tissue_row1)
        pf.addRow(self.tissue_row2)
        self.jacques_tissue = _combo(list(JACQUES_TISSUES))
        self.jacques_label = QLabel("Jacques parameters")
        pf.addRow(self.jacques_label, self.jacques_tissue)
        self.pl_hint = _hint("")
        pf.addRow(self.pl_hint)
        b_pl = QPushButton("Plot pathlength")
        b_pl.clicked.connect(self._plot_pathlength)
        pf.addRow(b_pl)
        lay.addWidget(pbox)

        mbox = QGroupBox("Pixels to fit")
        mf = QFormLayout(mbox)
        self.mask_min_on = QCheckBox("skip pixels with mean reflectance below")
        self.mask_min = _dspin(0, 10, 0.02, 0.01, 4)
        self.mask_max_on = QCheckBox("skip pixels with mean reflectance above")
        self.mask_max = _dspin(0, 10, 0.95, 0.01, 4)
        mf.addRow(self.mask_min_on, self.mask_min)
        mf.addRow(self.mask_max_on, self.mask_max)
        self.mask_poly_on = QCheckBox("fit only inside the 'Fit region' shapes")
        b_fr = QPushButton("Draw…")
        b_fr.clicked.connect(lambda: (self._start_shape_layer("Fit region", "add_polygon", "lime"),
                                      self.mask_poly_on.setChecked(True)))
        mf.addRow(self.mask_poly_on, b_fr)
        lay.addWidget(mbox)

        self.adv_box = QGroupBox("Advanced")
        self.adv_box.setCheckable(True)
        self.adv_box.setChecked(False)
        af = QFormLayout(self.adv_box)
        self.a_max = _dspin(0.001, 1e5, 100, 1, 3)
        self.b_max = _dspin(0.1, 20, 5, 0.1, 3)
        af.addRow(self._pair("a max", self.a_max, "b max", self.b_max))
        self.scat_scale = _dspin(0.001, 1e4, 10, 1, 4)
        self.lambda0 = _dspin(100, 3000, 500, 10, 1, " nm")
        af.addRow("Scattering factor 1/(1−g)", self.scat_scale)
        af.addRow("λ0 of a·(λ/λ0)^−b", self.lambda0)
        self.abs_offset = QCheckBox("fit a constant offset (no reference)")
        self.abs_offset.setChecked(True)
        af.addRow(self.abs_offset)
        self.solver_method = _combo(["Fast (whole image)", "Exact (scipy, pixel by pixel — slow)"], ["fast", "exact"])
        af.addRow("Solver", self.solver_method)
        self.b_step = _dspin(0.005, 1, 0.05, 0.01, 3)
        af.addRow("b search step", self.b_step)
        for child in self.adv_box.findChildren(QWidget):
            child.setVisible(False)
        self.adv_box.toggled.connect(lambda on: [c.setVisible(on) for c in self.adv_box.findChildren(QWidget)])
        lay.addWidget(self.adv_box)
        lay.addStretch(1)
        return w

    @staticmethod
    def _pair(l1, w1, l2, w2) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(QLabel(l1))
        lay.addWidget(w1, 1)
        lay.addWidget(QLabel(l2))
        lay.addWidget(w2, 1)
        return w

    def _plot_pathlength(self):
        try:
            cfg = self.gather_config()
            names, _ = self._selected_chromophores()
            chroms = [self.library.get(n) for n in names]
            for n, v in cfg.chromophores.baselines.items():
                if n in names:
                    chroms[names.index(n)] = copy.copy(chroms[names.index(n)])
                    chroms[names.index(n)].baseline = v
            lo, hi = cfg.spectral.range
            wl = self.data.wavelengths if self.data is not None else np.arange(lo, hi + 1, 5.0)
            wl = wl[(wl >= max(lo, common_range(chroms)[0])) & (wl <= min(hi, common_range(chroms)[1]))]
            M = absorption_matrix(chroms, wl)
            pl = global_pathlength(cfg.pathlength, wl, M, np.array([c.baseline for c in chroms]))
        except Exception as e:
            return self._error("Pathlength", e)
        title = PATHLENGTH_LABELS[cfg.pathlength.mode]
        if cfg.pathlength.mode.endswith("_pixelwise"):
            title += ": first pass uses 1 cm; per-pixel curves are computed during the fit"
        self.inspector.plot.clear(title)
        self.inspector.plot.ax.plot(wl, pl, "b-")
        self.inspector.plot.ax.set_xlabel("Wavelength (nm)", fontsize=8)
        self.inspector.plot.ax.set_ylabel("Pathlength (cm)", fontsize=8)
        self.inspector.plot.draw()

    # ======================================================================== tab 5: run
    def _build_run_tab(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        f = QFormLayout()
        self.out_edit = QLineEdit(str(Path.home() / "spectrafit_results"))
        f.addRow("Save results to", _browse_row(self.out_edit, self._browse_out))
        fr = QWidget()
        fl = QHBoxLayout(fr)
        fl.setContentsMargins(0, 0, 0, 0)
        self.fmt = {}
        for k, label in (("npz", ".npz"), ("mat", ".mat"), ("tiff", "TIFF")):
            cb = QCheckBox(label)
            cb.setChecked(True)
            self.fmt[k] = cb
            fl.addWidget(cb)
        f.addRow("Formats", fr)
        self.autosave = QCheckBox("save after each run (with the settings file)")
        self.autosave.setChecked(True)
        f.addRow(self.autosave)
        lay.addLayout(f)
        row = QHBoxLayout()
        self.run_button = QPushButton("Run fit")
        self.run_button.setStyleSheet("font-weight: bold; padding: 8px;")
        self.run_button.clicked.connect(self._run_fit)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._request_stop)
        self.save_button = QPushButton("Save results")
        self.save_button.clicked.connect(self._save_results)
        row.addWidget(self.run_button, 2)
        row.addWidget(self.stop_button, 1)
        row.addWidget(self.save_button, 1)
        lay.addLayout(row)
        mh = QHBoxLayout()
        mh.addWidget(QLabel("Maps (click a name to view it):"), 1)
        self.side_by_side = QCheckBox("side by side")
        self.side_by_side.toggled.connect(self._toggle_grid)
        mh.addWidget(self.side_by_side)
        lay.addLayout(mh)
        self.map_list = QListWidget()
        self.map_list.itemChanged.connect(self._map_toggled)
        self.map_list.itemClicked.connect(lambda it: self._focus_map(it.data(Qt.ItemDataRole.UserRole)))
        lay.addWidget(self.map_list, 1)
        mrow = QHBoxLayout()
        b_all = QPushButton("Show all chromophores")
        b_all.clicked.connect(lambda: self._check_maps(lambda k: k in (self.results.chromophores if self.results else [])))
        b_none = QPushButton("Hide all")
        b_none.clicked.connect(lambda: self._check_maps(lambda k: False))
        mrow.addWidget(b_all)
        mrow.addWidget(b_none)
        lay.addLayout(mrow)
        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setMaximumHeight(140)
        lay.addWidget(self.notes)
        return w

    def _browse_out(self):
        d = QFileDialog.getExistingDirectory(self, "Folder for results", self.out_edit.text())
        if d:
            self.out_edit.setText(d)

    def _run_fit(self):
        if self.data is None:
            return self._info("Load the data first (tab 1).")
        try:
            cfg = self.gather_config()
            cfg.validate()
        except Exception as e:
            return self._error("Settings", e)
        n_px = int(np.prod(self.data.cube.shape[:3]))
        if cfg.solver.method == "exact" and n_px > 5000:
            ok = QMessageBox.question(self, "Exact solver", f"The exact solver fits {n_px} pixels one by one; this can "
                                      "take a long time. Use it for small regions (crop or fit region). Continue?")
            if ok != QMessageBox.StandardButton.Yes:
                return
        self._stop = False
        self.stop_button.setEnabled(True)
        self._run_in_thread(self._do_fit, cfg, on_done=self._fitted, label="Fitting")

    def _do_fit(self, cfg):
        from ..pipeline import run

        return run(cfg, data=self.data, progress=self._emit, should_stop=lambda: self._stop)

    def _request_stop(self):
        self._stop = True
        self.status.setText("Stopping…")

    def _fitted(self, results):
        self.results = results
        self.stop_button.setEnabled(False)
        self._remove_layers(prefix="SpectraFit map: ")
        self.map_list.blockSignals(True)
        self.map_list.clear()
        order = results.chromophores + [k for k in results.maps if k not in results.chromophores]
        default = set(MAIN_MAPS) | ({"HbO2", "HHb"} if "HbT" not in results.maps else set())
        if len(results.chromophores) <= 3:
            default |= set(results.chromophores)
        for k in order:
            it = QListWidgetItem(results.labels.get(k, k))
            it.setData(Qt.ItemDataRole.UserRole, k)
            it.setFlags(it.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            it.setCheckState(Qt.CheckState.Checked if k in default else Qt.CheckState.Unchecked)
            self.map_list.addItem(it)
        self.map_list.blockSignals(False)
        shown = [self.map_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(self.map_list.count())
                 if self.map_list.item(i).checkState() == Qt.CheckState.Checked]
        for k in reversed(shown):
            self._show_map(k)
        first = next((k for k in ("HbT", "StO2") if k in shown), shown[0] if shown else None)
        if first:
            self._focus_map(first)
        self.notes.setPlainText("\n".join(results.notes))
        msg = f"Fit finished in {results.timing_s:.1f} s."
        if self.autosave.isChecked():
            try:
                folder = self._output_folder()
                results.save(folder, self._formats())
                msg += f" Saved to {folder}."
            except Exception as e:
                self._error("Saving", e)
        self.status.setText(msg + " Click any pixel to inspect its fit.")
        self.tabs.setCurrentIndex(4)

    def _output_folder(self) -> Path:
        base = Path(self.out_edit.text()).expanduser()
        stem = Path(self.data.files[0]).stem if self.data else "results"
        import time

        return base / f"{stem}_{time.strftime('%Y%m%d_%H%M%S')}"

    def _formats(self):
        return [k for k, cb in self.fmt.items() if cb.isChecked()]

    def _save_results(self):
        if self.results is None:
            return self._info("Run a fit first.")
        folder = QFileDialog.getExistingDirectory(self, "Save results into folder", self.out_edit.text())
        if folder:
            try:
                self.results.save(folder, self._formats())
                self.status.setText(f"Saved to {folder}.")
            except Exception as e:
                self._error("Saving", e)

    def _check_maps(self, pred):
        for i in range(self.map_list.count()):
            it = self.map_list.item(i)
            it.setCheckState(Qt.CheckState.Checked if pred(it.data(Qt.ItemDataRole.UserRole)) else Qt.CheckState.Unchecked)

    def _map_toggled(self, item: QListWidgetItem):
        key = item.data(Qt.ItemDataRole.UserRole)
        name = f"SpectraFit map: {key}"
        if item.checkState() == Qt.CheckState.Checked:
            self._show_map(key)
        elif name in self.viewer.layers:
            self.viewer.layers.remove(name)

    def _toggle_grid(self, on: bool):
        self.viewer.grid.enabled = on
        if on:
            for layer in self.viewer.layers:
                layer.visible = layer.name.startswith("SpectraFit map: ")
        elif self.viewer.layers.selection.active is not None:
            self._focus_map(self.viewer.layers.selection.active.name.replace("SpectraFit map: ", ""))

    def _focus_map(self, key: str):
        """Show this map on top; hide the other maps unless side-by-side view is on."""
        name = f"SpectraFit map: {key}"
        if name not in self.viewer.layers:
            item = next((self.map_list.item(i) for i in range(self.map_list.count())
                         if self.map_list.item(i).data(Qt.ItemDataRole.UserRole) == key), None)
            if item is None:
                return
            item.setCheckState(Qt.CheckState.Checked)  # adds the layer
        layer = self.viewer.layers[name]
        if not self.side_by_side.isChecked():
            for other in self.viewer.layers:
                if other.name.startswith("SpectraFit map: "):
                    other.visible = other is layer
            self.viewer.layers.move(self.viewer.layers.index(layer), len(self.viewer.layers))
            for keep in ("Reference pixel", "Reference region", "Fit region", "Crop"):
                if keep in self.viewer.layers:
                    self.viewer.layers.move(self.viewer.layers.index(keep), len(self.viewer.layers))
        layer.visible = True
        self.viewer.layers.selection.active = layer
        self.status.setText(f"Showing {self.results.labels.get(key, key)}. Click a pixel to inspect its fit.")

    def _show_map(self, key: str):
        r = self.results
        arr = r.maps[key]
        data = arr if arr.shape[0] > 1 else arr[0]
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return
        lo, hi = np.percentile(finite, [1, 99])
        diverging = (not r.absolute and (key in r.chromophores or key in ("HbT", "diffCCO", "delta_a", "delta_b"))) \
            or key == "offset"
        if diverging:
            m = max(abs(lo), abs(hi), 1e-12)
            lo, hi, cmap = -m, m, "PiYG"
        elif key == "flag":
            lo, hi, cmap = 0, 2, "inferno"
        else:
            cmap = "viridis"
        if hi <= lo:
            hi = lo + 1e-12
        layer = self._add_or_replace_image(f"SpectraFit map: {key}", data, colormap=cmap, contrast_limits=(lo, hi),
                                           visible=self.side_by_side.isChecked())
        layer.metadata["label"] = r.labels.get(key, key)

    # ======================================================================== pixel inspector
    def _on_click(self, viewer, event):
        if event.type != "mouse_press" or self.data is None:
            return
        active = viewer.layers.selection.active
        if active is not None and getattr(active, "mode", "") in ("add", "add_polygon", "add_rectangle", "add_ellipse",
                                                                    "add_path", "add_line", "direct", "select"):
            return
        pos = viewer.cursor.position
        row, col = int(np.floor(pos[-2] + 0.5)), int(np.floor(pos[-1] + 0.5))
        H, W = self.data.shape
        if not (0 <= row < H and 0 <= col < W):
            return
        steps = viewer.dims.current_step  # time is the third axis from the end (layers align trailing axes)
        frame = int(steps[-3]) if self.data.n_frames > 1 and len(steps) >= 3 else 0
        frame = min(max(frame, 0), self.data.n_frames - 1)
        orow, ocol = self.data.to_original(row, col)
        where = f"pixel row {int(orow)}, col {int(ocol)}" + (f", frame {frame}" if self.data.n_frames > 1 else "")
        if self.results is None:
            self.inspector.show_reflectance(self.data.wavelengths, self.data.cube[frame, row, col], where)
            return
        fit = self.results.pixel_fit(self.data, frame, row, col)
        if fit is None:
            return
        vals = []
        for k in self.results.chromophores[:4] + (["b"] if "b" in self.results.maps else []):
            v = self.results.maps[k][frame, row, col]
            vals.append(f"{k} {v:.3g}")
        self.inspector.info.setText(where + ":  " + ",  ".join(vals))
        self.inspector.show_fit(fit, where, self.results.absolute)

    # ======================================================================== settings <-> widgets
    def gather_config(self) -> Config:
        cfg = Config()
        d = cfg.data
        d.file = self.file_edit.text().strip()
        if d.file and not any(ch in d.file for ch in "*?["):
            d.file = str(Path(d.file).expanduser().resolve())
        v = self.var_combo.currentData() or self.var_combo.currentText().strip()
        d.variable = v or None
        ax = self.axes_combo.currentText().strip()
        d.axes = None if ax.startswith("(") or not ax else ax.upper()
        if self.wl_mode.currentData() == "startstep":
            d.wavelengths = {"start": self.wl_start.value(), "step": self.wl_step.value()}
        d.is_reflectance = self.kind_combo.currentData() == "refl"
        d.white_reference = self.white_edit.text().strip() or None
        d.dark_reference = self.dark_edit.text().strip() or None
        d.normalisation_offset = None if self.offset_auto.isChecked() else self.offset_spin.value()
        d.binning = self.binning.value()
        d.crop = list(self._crop) if self._crop else None
        fr = self.frames_edit.text().replace(";", ",").strip()
        d.frames = [int(x) for x in fr.split(",") if x.strip()] if fr else None

        r = cfg.reference
        r.type = self.ref_type.currentData()
        r.pixel = [self.ref_row.value(), self.ref_col.value()]
        poly = self._first_shape("Reference region")
        if poly is not None and self.data is not None:
            r.roi = [list(map(float, self.data.to_original(p[-2], p[-1]))) for p in poly]
        else:
            r.roi = getattr(self, "_roi_original", None)
        r.file = self.ref_file.text().strip() or None
        if r.type == "frame":
            r.frame = [self.ref_f0.value(), self.ref_f1.value() + 1]
        elif self.data is not None and self.data.n_frames > 1:
            r.frame = [self.ref_f0.value()]
        else:
            r.frame = None

        names, baselines = self._selected_chromophores()
        builtin = builtin_dir().resolve()
        plain, extra = [], []
        for n in names:  # chromophores not shipped with the program are referenced by file, so settings travel
            c = self.library.get(n)
            if c.path and builtin not in Path(c.path).resolve().parents:
                extra.append(str(Path(c.path).resolve()))
            else:
                plain.append(n)
        cfg.chromophores.names = plain
        cfg.chromophores.extra_files = extra
        cfg.chromophores.baselines = baselines

        cfg.spectral.range = [self.range_lo.value(), self.range_hi.value()]
        cfg.spectral.exclude = _parse_intervals(self.exclude_edit.text())

        m = cfg.model
        m.mode = self.mode_combo.currentData()
        m.fixed_b = self.fixed_b.value()
        rs = m.reference_scattering
        rs.method = self.refscat_method.currentData()
        rs.a0, rs.b0 = self.a0.value(), self.b0.value()
        rs.a_steps, rs.b_steps, rs.grid_binning = self.a_steps.value(), self.b_steps.value(), self.grid_bin.value()
        m.a_max, m.b_max = self.a_max.value(), self.b_max.value()
        m.scattering_scale, m.lambda0 = self.scat_scale.value(), self.lambda0.value()
        m.absolute_offset = self.abs_offset.isChecked()

        p = cfg.pathlength
        p.mode = self.pl_mode.currentData()
        p.constant = self.pl_const.value()
        p.file = self.pl_file.text().strip() or None
        p.tissue.a, p.tissue.b, p.tissue.g, p.tissue.n = self.t_a.value(), self.t_b.value(), self.t_g.value(), self.t_n.value()
        p.jacques_tissue = self.jacques_tissue.currentText()

        cfg.solver.method = self.solver_method.currentData()
        cfg.solver.b_step = self.b_step.value()

        mk = cfg.mask
        mk.min_reflectance = self.mask_min.value() if self.mask_min_on.isChecked() else None
        mk.max_reflectance = self.mask_max.value() if self.mask_max_on.isChecked() else None
        mk.polygons = []
        if self.mask_poly_on.isChecked():
            if "Fit region" in self.viewer.layers and self.data is not None:
                for shape in self.viewer.layers["Fit region"].data:
                    mk.polygons.append([list(map(float, self.data.to_original(q[-2], q[-1]))) for q in shape])
            else:
                mk.polygons = getattr(self, "_mask_polys_original", []) or []
        cfg.output.folder = self.out_edit.text().strip()
        cfg.output.formats = self._formats()
        return cfg

    def apply_config(self, cfg: Config):
        d = cfg.data
        self.file_edit.setText(d.file or "")
        if d.file and Path(d.file).is_file():
            self._inspect(d.file)
        if d.variable:
            if self.var_combo.findData(d.variable) < 0:
                self.var_combo.addItem(d.variable, d.variable)
            _set_combo_data(self.var_combo, d.variable)
        self.axes_combo.setCurrentText(d.axes or AXES_CHOICES[0])
        if isinstance(d.wavelengths, dict):
            _set_combo_data(self.wl_mode, "startstep")
            self.wl_start.setValue(float(d.wavelengths["start"]))
            self.wl_step.setValue(float(d.wavelengths["step"]))
        else:
            _set_combo_data(self.wl_mode, "auto")
        _set_combo_data(self.kind_combo, "refl" if d.is_reflectance else "raw")
        self.white_edit.setText(d.white_reference or "")
        self.dark_edit.setText(d.dark_reference or "")
        self.offset_auto.setChecked(d.normalisation_offset is None)
        if d.normalisation_offset is not None:
            self.offset_spin.setValue(d.normalisation_offset)
        self.binning.setValue(int(d.binning))
        self._crop = list(d.crop) if d.crop else None
        self.crop_label.setText("whole image" if not self._crop else
                                f"rows {self._crop[0]}–{self._crop[1]}, cols {self._crop[2]}–{self._crop[3]}")
        self.frames_edit.setText(", ".join(map(str, d.frames)) if d.frames else "")

        r = cfg.reference
        _set_combo_data(self.ref_type, r.type)
        if r.pixel:
            self.ref_row.setValue(int(r.pixel[0]))
            self.ref_col.setValue(int(r.pixel[1]))
        self._roi_original = r.roi
        self.ref_file.setText(r.file or "")
        if r.frame:
            self.ref_f0.setValue(int(r.frame[0]))
            self.ref_f1.setValue(int(r.frame[1]) - 1 if len(r.frame) > 1 else int(r.frame[0]))

        self._extra_files = list(cfg.chromophores.extra_files)
        self._refresh_chromophore_table(list(cfg.chromophores.names) + self._extra_names(), cfg.chromophores.baselines)
        self.range_lo.setValue(cfg.spectral.range[0])
        self.range_hi.setValue(cfg.spectral.range[1])
        self.exclude_edit.setText(", ".join(f"{a:g}-{b:g}" for a, b in cfg.spectral.exclude))
        m = cfg.model
        _set_combo_data(self.mode_combo, m.mode)
        self.fixed_b.setValue(m.fixed_b)
        rs = m.reference_scattering
        _set_combo_data(self.refscat_method, rs.method)
        self.a0.setValue(rs.a0)
        self.b0.setValue(rs.b0)
        self.a_steps.setValue(rs.a_steps)
        self.b_steps.setValue(rs.b_steps)
        self.grid_bin.setValue(rs.grid_binning)
        self.a_max.setValue(m.a_max)
        self.b_max.setValue(m.b_max)
        self.scat_scale.setValue(m.scattering_scale)
        self.lambda0.setValue(m.lambda0)
        self.abs_offset.setChecked(m.absolute_offset)
        p = cfg.pathlength
        _set_combo_data(self.pl_mode, p.mode)
        self.pl_const.setValue(p.constant)
        self.pl_file.setText(p.file or "")
        self.t_a.setValue(p.tissue.a)
        self.t_b.setValue(p.tissue.b)
        self.t_g.setValue(p.tissue.g)
        self.t_n.setValue(p.tissue.n)
        self.jacques_tissue.setCurrentText(p.jacques_tissue)
        _set_combo_data(self.solver_method, cfg.solver.method)
        self.b_step.setValue(cfg.solver.b_step)
        mk = cfg.mask
        self.mask_min_on.setChecked(mk.min_reflectance is not None)
        if mk.min_reflectance is not None:
            self.mask_min.setValue(mk.min_reflectance)
        self.mask_max_on.setChecked(mk.max_reflectance is not None)
        if mk.max_reflectance is not None:
            self.mask_max.setValue(mk.max_reflectance)
        self._mask_polys_original = mk.polygons
        self.mask_poly_on.setChecked(bool(mk.polygons))
        if cfg.output.folder:
            self.out_edit.setText(cfg.output.folder)
        for k, cb in self.fmt.items():
            cb.setChecked(k in cfg.output.formats)
        self._update_visibility()
        self._sync_geometry_layers()

    def _extra_names(self) -> List[str]:
        out = []
        for f in getattr(self, "_extra_files", []):
            try:
                out.append(read_chromophore_file(f).name)
            except Exception:
                pass
        return out

    def load_settings(self, path: str):
        try:
            cfg = Config.load(path)
        except Exception as e:
            return self._error("Settings file", e)
        self.apply_config(cfg)
        self.status.setText(f"Settings loaded from {path}. Press 'Load data' in tab 1.")

    def _load_settings_dialog(self):
        f, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "Settings (*.yaml *.yml)")
        if f:
            self.load_settings(f)

    def _save_settings_dialog(self):
        f, _ = QFileDialog.getSaveFileName(self, "Save settings", "spectrafit_settings.yaml", "Settings (*.yaml)")
        if f:
            try:
                self.gather_config().save(f)
                self.status.setText(f"Settings saved to {f}. Run them later with:  spectrafit run {Path(f).name}")
            except Exception as e:
                self._error("Settings", e)

    # ======================================================================== geometry layers
    def _start_shape_layer(self, name, mode, colour):
        if self.data is None:
            return self._info("Load the data first.")
        if name not in self.viewer.layers:
            self.viewer.add_shapes(name=name, edge_color=colour, face_color=[0, 0, 0, 0], edge_width=1.5, ndim=2)
        layer = self.viewer.layers[name]
        self.viewer.layers.selection.active = layer
        layer.mode = mode
        self.status.setText(f"Draw in the '{name}' layer. Press Esc or choose the pan/zoom tool when done.")

    def _first_shape(self, name):
        if name not in self.viewer.layers or not len(self.viewer.layers[name].data):
            return None
        return np.asarray(self.viewer.layers[name].data[0])

    def _sync_geometry_layers(self):
        """Redraw reference pixel / regions from the original-pixel coordinates (e.g. after changing binning)."""
        if self.data is None:
            return
        r, c = self.data.to_working(self.ref_row.value(), self.ref_col.value())
        layer = self._get_points_layer()
        layer.events.data.disconnect(self._points_changed)
        layer.data = np.array([[np.floor(r), np.floor(c)]])
        layer.events.data.connect(self._points_changed)
        for name, polys in (("Reference region", [self._roi_original] if getattr(self, "_roi_original", None) else []),
                            ("Fit region", getattr(self, "_mask_polys_original", []) or [])):
            if polys and name not in self.viewer.layers:
                work = [np.array([self.data.to_working(p[0], p[1]) for p in poly]) for poly in polys]
                self.viewer.add_shapes(work, shape_type="polygon", name=name, edge_width=1.5, face_color=[0, 0, 0, 0],
                                       edge_color="orange" if name.startswith("Ref") else "lime")

    # ======================================================================== layers & threads
    def _add_or_replace_image(self, name, data, **kw):
        if name in self.viewer.layers:
            self.viewer.layers.remove(name)
        return self.viewer.add_image(data, name=name, **kw)

    def _remove_layers(self, prefix=None, names=None):
        for layer in list(self.viewer.layers):
            if (prefix and layer.name.startswith(prefix)) or (names and layer.name in names):
                self.viewer.layers.remove(layer)

    def _emit(self, frac, msg):
        self._signals.progress.emit(float(frac), str(msg))

    def _on_progress(self, frac, msg):
        self.progress.setValue(int(1000 * frac))
        self.status.setText(msg)

    def _run_in_thread(self, func, arg, on_done, label):
        from napari.qt.threading import create_worker

        if self._worker is not None:
            return self._info("Please wait until the current task has finished.")
        self.load_button.setEnabled(False)
        self.run_button.setEnabled(False)
        self.status.setText(f"{label}…")
        worker = create_worker(func, arg)

        def finished():
            self._worker = None
            self.load_button.setEnabled(True)
            self.run_button.setEnabled(True)
            self.stop_button.setEnabled(False)

        worker.returned.connect(on_done)
        worker.errored.connect(lambda e: self._error(label, e))
        worker.finished.connect(finished)
        self._worker = worker
        worker.start()

    # ======================================================================== misc
    def _update_visibility(self, *_):
        raw = self.kind_combo.currentData() == "raw"
        self.white_row.setEnabled(raw)
        self.dark_row.setEnabled(raw)
        self.offset_spin.setEnabled(not self.offset_auto.isChecked())
        self.wl_row.setVisible(self.wl_mode.currentData() == "startstep")
        t = self.ref_type.currentData()
        self.ref_pixel_box.setVisible(t == "pixel")
        self.ref_roi_box.setVisible(t == "roi")
        self.ref_file_box.setVisible(t == "file")
        ts = self.data is not None and self.data.n_frames > 1
        self.ref_frame_box.setVisible(t == "frame" or (ts and t in ("pixel", "roi")))
        for name, show in (("Reference pixel", t == "pixel"), ("Reference region", t == "roi")):
            if hasattr(self, "viewer") and name in self.viewer.layers:
                self.viewer.layers[name].visible = show
        mode = self.mode_combo.currentData()
        self.fixed_b.setVisible(mode == "fixed_b")
        self.fixed_b_label.setVisible(mode == "fixed_b")
        self.refscat_box.setVisible(mode == "free_scattering" and t != "none")
        grid = self.refscat_method.currentData() == "grid"
        self.a0_row.setVisible(not grid)
        self.grid_row.setVisible(grid)
        self.grid_bin.setVisible(grid)
        self.grid_bin_label.setVisible(grid)
        hints = {
            "no_scattering": "ΔA = PL · M Δc. Fast and linear; ignores scattering changes.",
            "fixed_b": "ΔA = PL · (M Δc + s·Δa·(λ/λ0)^−b) with b fixed: only the scattering amplitude change Δa is fitted. Linear and fast.",
            "free_scattering": "ΔA = PL · (M Δc + s·[a(λ/λ0)^−b − a0(λ/λ0)^−b0]): scattering amplitude a and power b are fitted per pixel (as in the paper code).",
        }
        self.model_hint.setText(hints[mode] + ("" if t != "none" else "  No reference: absolute A = −ln R, concentrations ≥ 0."))
        pm = self.pl_mode.currentData()
        self.pl_const.setVisible(pm == "constant")
        self.pl_const_label.setVisible(pm == "constant")
        self.pl_file_row.setVisible(pm == "table_file")
        self.pl_file_label.setVisible(pm == "table_file")
        tissue = pm in ("delta_p1", "jacques", "delta_p1_pixelwise", "jacques_pixelwise")
        self.tissue_row1.setVisible(tissue)
        self.tissue_row2.setVisible(pm.startswith("delta_p1"))
        jac = pm.startswith("jacques")
        self.jacques_tissue.setVisible(jac)
        self.jacques_label.setVisible(jac)
        pl_hints = {
            "constant": "The same pathlength at every wavelength (1 cm = unitary pathlength).",
            "gray_matter_table": "Measured gray-matter pathlength curve (450–950 nm).",
            "table_file": "Your own curve: two columns, wavelength (nm) and pathlength (cm).",
            "delta_p1": "From baseline tissue optics: μa = Σ baseline·spectrum, μs′ = a(λ/500)^−b.",
            "jacques": "From baseline tissue optics with the Jacques model.",
            "delta_p1_pixelwise": "Two passes: fit with 1 cm, then each pixel's own μa, μs′ give its pathlength.",
            "jacques_pixelwise": "Two passes: fit with 1 cm, then each pixel's own μa, μs′ give its pathlength.",
        }
        self.pl_hint.setText(pl_hints[pm])

    def _info(self, msg):
        self.status.setText(msg)

    def _error(self, where, e):
        self.stop_button.setEnabled(False)
        text = str(e) or e.__class__.__name__
        if e.__class__.__name__ == "Cancelled":
            self.status.setText("Stopped.")
            return
        self.status.setText(f"{where}: {text}")
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Warning)
        box.setWindowTitle(f"SpectraFit — {where}")
        box.setText(text)
        box.setDetailedText("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        box.show()


def _parse_intervals(text: str):
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.replace("–", "-").partition("-")
        out.append([float(a), float(b or a)])
    return out
