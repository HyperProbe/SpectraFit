"""Smoke test of the napari widget (skipped when napari is not installed).

Run headless on Linux with:  xvfb-run -a python -m pytest tests/test_gui.py
"""

import numpy as np
import pytest

napari = pytest.importorskip("napari")


def test_widget_full_workflow(tmp_path, qtbot):
    from spectrafit.napari_plugin.widget import SpectraFitWidget, _set_combo_data
    from spectrafit.synthetic import make_cube, write_mat

    sim = make_cube(shape=(24, 20), mode="fixed_b", fixed_b=1.2, seed=2, realistic=True)
    mat = write_mat(tmp_path / "cube.mat", sim["cube"], sim["wavelengths"])
    viewer = napari.Viewer(show=False)
    try:
        w = SpectraFitWidget(viewer)
        w.file_edit.setText(str(mat))
        w._inspect(str(mat))
        w.binning.setValue(1)
        cfg = w.gather_config()
        w._loaded(w._do_load(cfg.data))
        assert "SpectraFit: colour preview" in viewer.layers
        w.ref_row.setValue(0)
        w.ref_col.setValue(0)
        _set_combo_data(w.mode_combo, "fixed_b")
        w.fixed_b.setValue(1.2)
        w.out_edit.setText(str(tmp_path / "out"))
        cfg = w.gather_config()
        assert cfg.reference.pixel == [0, 0] and cfg.model.mode == "fixed_b"
        w._fitted(w._do_fit(cfg))
        assert "SpectraFit map: HbT" in viewer.layers
        est = w.results.maps["HbO2"][0]
        assert np.nanmax(np.abs(est - sim["HbO2"][0])) < 1e-4
        assert any((tmp_path / "out").rglob("results.npz"))
        # settings survive a save/load round trip through the widget
        cfg.save(tmp_path / "s.yaml")
        w2 = SpectraFitWidget(viewer)
        w2.load_settings(str(tmp_path / "s.yaml"))
        assert w2.gather_config().to_dict() == cfg.to_dict()
    finally:
        viewer.close()


def test_reference_marker_pick_and_drag(tmp_path, qtbot):
    """The cross snaps to the pixel it is drawn on, can be dragged, stays above the image after reloading."""
    from spectrafit.napari_plugin.widget import SpectraFitWidget
    from spectrafit.synthetic import make_cube, write_mat

    sim = make_cube(shape=(40, 40), mode="no_scattering", seed=1, realistic=True)
    mat = write_mat(tmp_path / "cube.mat", sim["cube"], sim["wavelengths"])
    viewer = napari.Viewer(show=False)
    try:
        w = SpectraFitWidget(viewer)
        w.file_edit.setText(str(mat))
        w.binning.setValue(2)
        w._loaded(w._do_load(w.gather_config().data))
        w._start_pick_pixel()
        viewer.layers["Reference pixel"].add(np.array([[10.7, 5.2]]))
        assert viewer.layers["Reference pixel"].data.tolist() == [[11.0, 5.0]]
        assert (w.ref_row.value(), w.ref_col.value()) == (22, 10)  # binning 2

        class Ev:
            type = "mouse_press"

        ev = Ev()
        viewer.cursor.position = (11.2, 4.9)
        gen = w._on_click(viewer, ev)
        next(gen)
        assert viewer.camera.mouse_pan is False
        ev.type = "mouse_move"
        viewer.cursor.position = (15.4, 8.6)
        next(gen)
        ev.type = "mouse_release"
        with pytest.raises(StopIteration):
            next(gen)
        assert viewer.camera.mouse_pan is True
        assert (w.ref_row.value(), w.ref_col.value()) == (30, 18)

        w._loaded(w._do_load(w.gather_config().data))  # reloading must not hide the cross
        assert viewer.layers[-1].name == "Reference pixel"
    finally:
        viewer.close()


def test_clickable_image_cues(tmp_path, qtbot):
    """Crosshair cursor over the image, hand over the reference cross, yellow square on the inspected pixel."""
    from spectrafit.napari_plugin.widget import INSPECTED, SpectraFitWidget
    from spectrafit.synthetic import make_cube, write_mat

    sim = make_cube(shape=(40, 40), mode="no_scattering", seed=1, realistic=True)
    mat = write_mat(tmp_path / "cube.mat", sim["cube"], sim["wavelengths"])
    viewer = napari.Viewer(show=False)
    try:
        w = SpectraFitWidget(viewer)
        w.file_edit.setText(str(mat))
        w.binning.setValue(1)
        w._loaded(w._do_load(w.gather_config().data))
        w.ref_row.setValue(5)
        w.ref_col.setValue(5)

        class Ev:
            type = "mouse_move"

        for pos, style in (((20, 20), "crosshair"), ((5, 5), "pointing"), ((-10, -10), "standard")):
            viewer.cursor.position = pos
            w._on_mouse_move(viewer, Ev())
            assert viewer.cursor.style == style

        active = viewer.layers.selection.active
        ev = Ev()
        ev.type = "mouse_press"
        viewer.cursor.position = (30.3, 12.8)
        gen = w._on_click(viewer, ev)
        with pytest.raises(StopIteration):
            next(gen)
        assert viewer.layers[INSPECTED].data.tolist() == [[30.0, 13.0]]
        assert np.all(viewer.layers[INSPECTED].size == 1)  # outlines exactly that one pixel
        assert viewer.layers.selection.active is active  # the marker must not steal the selection
        assert viewer.layers[-1].name == "Reference pixel"

        w._loaded(w._do_load(w.gather_config().data))  # a new image clears the old marker
        assert INSPECTED not in viewer.layers
    finally:
        viewer.close()


def test_map_list_tick_and_untick(tmp_path, qtbot):
    """Clicking a tick box ticks/unticks the map; clicking a name shows that map."""
    from qtpy.QtCore import QPoint, Qt
    from qtpy.QtTest import QTest

    from spectrafit.napari_plugin.widget import SpectraFitWidget
    from spectrafit.synthetic import make_cube, write_mat

    sim = make_cube(shape=(20, 20), mode="no_scattering", seed=1, realistic=True)
    mat = write_mat(tmp_path / "cube.mat", sim["cube"], sim["wavelengths"])
    viewer = napari.Viewer(show=False)
    try:
        w = SpectraFitWidget(viewer)
        w.file_edit.setText(str(mat))
        w.binning.setValue(1)
        w._loaded(w._do_load(w.gather_config().data))
        w.ref_row.setValue(0)
        w.ref_col.setValue(0)
        w.autosave.setChecked(False)
        w._fitted(w._do_fit(w.gather_config()))
        w.show()
        lw = w.map_list
        it = lw.item(0)
        key = it.data(Qt.ItemDataRole.UserRole)
        layer = f"SpectraFit map: {key}"

        def click(x):
            lw.scrollToItem(it)
            r = lw.visualItemRect(it)
            QTest.mouseClick(lw.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(r.left() + x, r.center().y()))

        def ticked():
            return it.checkState() == Qt.CheckState.Checked

        assert not ticked() and layer not in viewer.layers
        click(10)  # tick box
        assert ticked() and layer in viewer.layers
        click(10)  # tick box again: must untick
        assert not ticked() and layer not in viewer.layers
        click(80)  # the name: shows it
        assert ticked() and viewer.layers[layer].visible and viewer.layers.selection.active.name == layer
        click(80)  # the name again: stays shown
        assert ticked()
        w._check_maps(lambda k: False)  # "Hide all"
        assert not any(l.name.startswith("SpectraFit map: ") for l in viewer.layers)
        click(80)  # the name right after "Hide all" still shows it
        assert ticked() and layer in viewer.layers
    finally:
        viewer.close()


def test_new_image_starts_clean(tmp_path, qtbot, monkeypatch):
    """After a fit, loading another image clears the fit and everything drawn on the old image;
    reloading the same image (e.g. other binning) keeps crop and reference."""
    from qtpy.QtCore import Qt
    from qtpy.QtWidgets import QMessageBox

    from spectrafit.napari_plugin.widget import INSPECTED, SpectraFitWidget
    from spectrafit.synthetic import make_cube, write_envi, write_mat

    a = make_cube(shape=(20, 20), mode="no_scattering", seed=1, realistic=True)
    b = make_cube(shape=(16, 18), mode="no_scattering", seed=2, realistic=True)
    mat_a = write_mat(tmp_path / "a.mat", a["cube"], a["wavelengths"])
    mat_b = write_mat(tmp_path / "b.mat", b["cube"], b["wavelengths"])
    envi_dir = tmp_path / "envi"
    envi_dir.mkdir()
    write_envi(envi_dir / "raw", (a["cube"][0] * 1000).astype("u2"), a["wavelengths"])
    write_envi(envi_dir / "whiteReference", np.full((1, 20, a["wavelengths"].size), 1000, "u2"), a["wavelengths"])
    viewer = napari.Viewer(show=False)
    try:
        w = SpectraFitWidget(viewer)
        w.binning.setValue(1)
        w.autosave.setChecked(False)
        w.file_edit.setText(str(mat_a))
        w._loaded(w._do_load(w.gather_config().data))
        w.ref_row.setValue(3)
        w.ref_col.setValue(4)
        w._crop = [0, 18, 0, 18]
        w._fitted(w._do_fit(w.gather_config()))
        lw = w.map_list
        assert lw.count() > 0 and any(l.name.startswith("SpectraFit map: ") for l in viewer.layers)
        w.inspector.info.setText("pixel row 1, col 1: HbO2 0.1")

        # same image again (e.g. other binning): fit cleared, crop and reference kept
        w._loaded(w._do_load(w.gather_config().data))
        assert w.results is None and lw.count() == 0
        assert w._crop == [0, 18, 0, 18] and w._ref_confirmed and "Reference pixel" in viewer.layers

        # another image: everything from the first one is gone
        w._fitted(w._do_fit(w.gather_config()))
        w.file_edit.setText(str(mat_b))
        cfg = w.gather_config()
        w._loaded(w._do_load(cfg.data))
        assert w.results is None and lw.count() == 0 and w.notes.toPlainText() == ""
        assert not any(l.name.startswith("SpectraFit map: ") for l in viewer.layers) and INSPECTED not in viewer.layers
        assert w.inspector.info.text().startswith("Click a pixel") and w.inspector._last is None
        assert not viewer.text_overlay.visible and w.progress.value() == 0
        assert w._crop is None and not w.undo_crop_button.isEnabled() and "Reference pixel" not in viewer.layers
        assert w.data.shape == (16, 18)  # loaded without the old crop
        assert "previous image were cleared" in w.data_info.toPlainText()

        # running now asks before reusing the old reference pixel; "No" goes to tab 2
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.No)
        started = []
        monkeypatch.setattr(w, "_run_in_thread", lambda *a, **k: started.append(1))
        w._run_fit()
        assert not started and w.tabs.currentIndex() == 1
        monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.StandardButton.Yes)
        w._run_fit()
        assert started and w._ref_confirmed and "Reference pixel" in viewer.layers

        # white/dark references found next to an ENVI file are not carried over to the next file
        w._inspect(str(envi_dir / "raw.hdr"))
        assert w.white_edit.text() and w.kind_combo.currentData() == "raw"
        w._inspect(str(mat_a))
        assert w.white_edit.text() == "" and w.kind_combo.currentData() == "refl"
    finally:
        viewer.close()


def test_error_maps_not_listed_and_pathlength_plot(tmp_path, qtbot):
    """se_* maps are saved but not listed; per-pixel pathlength plots show real per-pixel curves after a fit."""
    from qtpy.QtCore import Qt

    from spectrafit.napari_plugin.widget import INSPECTED, SpectraFitWidget, _set_combo_data
    from spectrafit.synthetic import make_cube, write_mat

    sim = make_cube(shape=(24, 24), mode="no_scattering", seed=3, realistic=True, pattern="tissue")
    mat = write_mat(tmp_path / "t.mat", sim["cube"], sim["wavelengths"])
    r, c = (int(v) for v in sim["reference_pixel"])
    viewer = napari.Viewer(show=False)
    try:
        w = SpectraFitWidget(viewer)
        w.file_edit.setText(str(mat))
        w.binning.setValue(1)
        w.autosave.setChecked(False)
        w._loaded(w._do_load(w.gather_config().data))
        w.ref_row.setValue(r)
        w.ref_col.setValue(c)
        _set_combo_data(w.mode_combo, "no_scattering")
        _set_combo_data(w.pl_mode, "delta_p1_pixelwise")

        w._plot_pathlength()  # before the fit: example curve, not the flat 1 cm of the first pass
        lines = w.inspector.plot.ax.get_lines()
        assert len(lines) == 1 and np.ptp(lines[0].get_ydata()) > 0
        assert "baseline" in w.inspector.info.text()

        w._fitted(w._do_fit(w.gather_config()))
        keys = [w.map_list.item(i).data(Qt.ItemDataRole.UserRole) for i in range(w.map_list.count())]
        assert "residual_rms" in keys and not any(k.startswith("se_") for k in keys)
        assert any(k.startswith("se_") for k in w.results.maps)  # still saved
        assert any(n.startswith("Typical standard error") for n in w.results.notes)

        w._mark_inspected(5, 7)
        w._plot_pathlength()  # after the fit: reference pixel and inspected pixel
        labels = [l.get_label() for l in w.inspector.plot.ax.get_lines()]
        assert labels == ["reference pixel", "pixel row 5, col 7"]
        y = w.inspector.plot.ax.get_lines()[1].get_ydata()
        assert np.allclose(y, w.results.pathlength[0, 5, 7])
    finally:
        viewer.close()
