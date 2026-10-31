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
