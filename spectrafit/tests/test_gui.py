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
