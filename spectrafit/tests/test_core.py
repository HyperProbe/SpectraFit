"""Tests on synthetic data with known answers."""

import math
from pathlib import Path

import numpy as np
import pytest

from spectrafit.chromophores import (Chromophore, ChromophoreLibrary, absorption_matrix, read_chromophore_file,
                                     write_chromophore_file)
from spectrafit.config import Config
from spectrafit.io import load_data
from spectrafit.pathlength import global_pathlength
from spectrafit.pipeline import run
from spectrafit.synthetic import make_cube, write_mat

CHROMS = ["HbO2", "HHb", "oxCCO", "redCCO", "water", "fat"]


def _cfg(tmp_path, mat, mode="free_scattering", ref="pixel", **kw):
    cfg = Config()
    cfg.data.file = str(mat)
    cfg.data.binning = 1
    cfg.reference.type = ref
    cfg.reference.pixel = [0, 0]
    cfg.model.mode = mode
    cfg.output.folder = str(tmp_path / "out")
    for k, v in kw.items():
        sec, attr = k.split("__")
        setattr(getattr(cfg, sec), attr, v)
    return cfg


def _rel_err(est, true):
    m = np.isfinite(est)
    scale = max(np.abs(true[m]).max(), 1e-12)
    return np.abs(est[m] - true[m]).max() / scale


# ------------------------------------------------------------------ chromophores
def test_unit_conversion_matches_research_code():
    lib = ChromophoreLibrary(include_user_dir=False)
    c = lib.get("HbO2")
    wl = np.array([550.0, 600.0])
    raw = np.interp(wl, c.wavelengths, c.values)
    assert np.allclose(c.absorption(wl), raw * math.log(10) / 1000)  # research code: *LN10 /1000
    w = lib.get("water")
    assert np.allclose(w.absorption(wl), np.interp(wl, w.wavelengths, w.values))


def test_custom_chromophore_roundtrip(tmp_path):
    wl = np.arange(450, 800, 2.0)
    ch = Chromophore("dye", wl, np.exp(-((wl - 620) / 30) ** 2) * 1e4, "eps10_per_mM_cm", baseline=0.01)
    p = write_chromophore_file(ch, tmp_path / "dye.csv")
    back = read_chromophore_file(p)
    assert back.name == "dye" and back.unit == "eps10_per_mM_cm" and back.baseline == 0.01
    assert np.allclose(back.absorption([620.0]), 1e4 * math.log(10))
    with pytest.raises(ValueError):
        back.absorption([440.0])  # outside its range


# ------------------------------------------------------------------ reading data
@pytest.mark.parametrize("v73", [False, True])
def test_mat_versions_and_wavelengths(tmp_path, v73):
    sim = make_cube(shape=(12, 10), mode="no_scattering")
    p = write_mat(tmp_path / "c.mat", sim["cube"], sim["wavelengths"], v73=v73)
    cfg = Config()
    cfg.data.file = str(p)
    d = load_data(cfg.data)
    assert d.cube.shape == (1, 12, 10, 127)
    assert np.allclose(d.cube, sim["cube"], rtol=1e-6)
    assert np.allclose(d.wavelengths, sim["wavelengths"])


def test_axis_order_binning_crop(tmp_path):
    sim = make_cube(shape=(16, 12), mode="no_scattering")
    cube = sim["cube"][0]
    from scipy.io import savemat

    savemat(tmp_path / "b.mat", {"img": np.transpose(cube, (2, 0, 1))})  # B×H×W, no wavelength variable
    cfg = Config()
    cfg.data.file = str(tmp_path / "b.mat")
    cfg.data.wavelengths = {"start": 385, "step": 5}
    cfg.data.crop = [2, 14, 1, 11]
    cfg.data.binning = 2
    d = load_data(cfg.data)
    assert d.axes == "BHW"
    expected = cube[2:14, 1:11].reshape(6, 2, 5, 2, 127).mean(axis=(1, 3))
    assert np.allclose(d.cube[0], expected, rtol=1e-6)
    assert d.to_working(2, 1) == (0, 0) and d.to_original(1, 1) == (4, 3)


def test_raw_calibration_with_offset(tmp_path):
    rng = np.random.default_rng(0)
    H, W, B = 6, 5, 127
    R = rng.uniform(0.1, 0.8, (H, W, B))
    white = rng.uniform(900, 1000, (1, W, B))
    dark = rng.uniform(5, 10, (1, W, B))
    raw = R * (white - dark) + dark
    from scipy.io import savemat

    savemat(tmp_path / "raw.mat", {"raw": raw, "white": white, "dark": dark,
                                   "wavelengths": np.arange(385, 1016, 5.0)})
    cfg = Config()
    cfg.data.file = str(tmp_path / "raw.mat")
    cfg.data.variable = "raw"
    cfg.data.is_reflectance = False
    cfg.data.white_reference = "white"
    cfg.data.dark_reference = "dark"
    d = load_data(cfg.data)
    assert np.allclose(d.cube[0], R + 0.1, rtol=1e-5)  # default offset 0.1 for raw data, as in HELICOID code


def test_time_series_4d_and_files(tmp_path):
    sim = make_cube(shape=(8, 8), mode="no_scattering", n_frames=3)
    write_mat(tmp_path / "ts.mat", sim["cube"], sim["wavelengths"])
    cfg = Config()
    cfg.data.file = str(tmp_path / "ts.mat")
    d = load_data(cfg.data)
    assert d.axes == "HWBT" and d.cube.shape == (3, 8, 8, 127)
    assert np.allclose(d.cube, sim["cube"], rtol=1e-6)
    for t in range(3):
        write_mat(tmp_path / f"frame_{t + 1}.mat", sim["cube"][t:t + 1], sim["wavelengths"])
    cfg.data.file = str(tmp_path / "frame_*.mat")
    cfg.data.frames = [1, 3]
    d2 = load_data(cfg.data)
    assert d2.cube.shape == (2, 8, 8, 127) and np.allclose(d2.cube, sim["cube"][1:3], rtol=1e-6)


# ------------------------------------------------------------------ recovery
@pytest.mark.parametrize("mode", ["no_scattering", "fixed_b", "free_scattering"])
@pytest.mark.parametrize("absolute", [False, True])
def test_recovery_all_modes(tmp_path, mode, absolute):
    sim = make_cube(shape=(14, 14), mode=mode, absolute=absolute, a0=20, b0=1.2, fixed_b=1.2, seed=1)
    mat = write_mat(tmp_path / "s.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat, mode=mode, ref="none" if absolute else "pixel")
    cfg.spectral.range = [500, 900]
    cfg.model.fixed_b = 1.2
    cfg.model.reference_scattering.method = "manual"
    cfg.model.reference_scattering.a0, cfg.model.reference_scattering.b0 = 20, 1.2
    res = run(cfg)
    for name in CHROMS:  # data are stored as float32, which limits the precision
        assert _rel_err(res.maps[name][0], sim[name][0]) < 2e-3, name
    if mode == "fixed_b":
        key = "a" if absolute else "delta_a"
        assert _rel_err(res.maps[key][0], sim[key][0]) < 2e-3
    if mode == "free_scattering":
        assert _rel_err(res.maps["a"][0], sim["a"][0]) < 1e-3
        assert _rel_err(res.maps["b"][0], sim["b"][0]) < 1e-3
    if absolute:
        assert _rel_err(res.maps["offset"][0], sim["offset"][0]) < 1e-3
    assert np.nanmax(res.maps["residual_rms"]) < 1e-5 * max(1.0, float(np.nanmax(np.abs(-np.log(sim["cube"])))))


def test_grid_search_finds_reference_scattering(tmp_path):
    sim = make_cube(shape=(24, 24), mode="free_scattering", a0=33.333333, b0=1.6, seed=4)
    mat = write_mat(tmp_path / "g.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat)
    cfg.model.reference_scattering.grid_binning = 2
    res = run(cfg)
    assert abs(res.a0 - 100 / 3) < 1e-6 and abs(res.b0 - 1.6) < 1e-9
    assert _rel_err(res.maps["HbO2"][0], sim["HbO2"][0]) < 1e-3


@pytest.mark.parametrize("pl_mode", ["gray_matter_table", "delta_p1", "jacques"])
def test_recovery_with_tissue_pathlength(tmp_path, pl_mode):
    cfg0 = Config()
    cfg0.pathlength.mode = pl_mode
    lib = ChromophoreLibrary(include_user_dir=False)
    chroms = [lib.get(n) for n in CHROMS]
    wl = np.arange(385, 1016, 5.0)
    sel = (wl >= 530) & (wl <= 750)
    M = absorption_matrix(chroms, wl[sel])
    PL_fit = global_pathlength(cfg0.pathlength, wl[sel], M, np.array([c.baseline for c in chroms]))
    PL_full = np.ones_like(wl)
    PL_full[sel] = PL_fit
    sim = make_cube(shape=(10, 10), mode="free_scattering", pathlength=PL_full, seed=2)
    mat = write_mat(tmp_path / "p.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat)
    cfg.pathlength.mode = pl_mode
    cfg.model.reference_scattering.method = "manual"
    cfg.model.reference_scattering.a0, cfg.model.reference_scattering.b0 = 20, 1.2
    res = run(cfg)
    assert _rel_err(res.maps["HbO2"][0], sim["HbO2"][0]) < 1e-3
    assert _rel_err(res.maps["b"][0], sim["b"][0]) < 1e-3


@pytest.mark.parametrize("pl_mode", ["delta_p1_pixelwise", "jacques_pixelwise"])
@pytest.mark.parametrize("extra", [False, True])
def test_pixelwise_pathlength_runs_with_any_chromophore_set(tmp_path, pl_mode, extra):
    sim = make_cube(shape=(10, 10), mode="free_scattering", seed=5)
    mat = write_mat(tmp_path / "x.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat)
    cfg.pathlength.mode = pl_mode
    cfg.model.reference_scattering.method = "manual"
    cfg.model.reference_scattering.a0, cfg.model.reference_scattering.b0 = 20, 1.2
    if extra:  # a new chromophore with no baseline, plus cytochrome c
        wl = np.arange(400, 1001, 1.0)
        dye = Chromophore("dye", wl, 2e3 * np.exp(-((wl - 650) / 20) ** 2), "eps10_per_mM_cm")
        cfg.chromophores.extra_files = [str(write_chromophore_file(dye, tmp_path / "dye.csv"))]
        cfg.chromophores.names = CHROMS + ["oxCytc"]
    res = run(cfg)
    assert res.pathlength.ndim == 4 and np.all(res.pathlength > 0)
    assert np.isfinite(res.maps["HbO2"]).all()


def test_fast_matches_exact_solver(tmp_path):
    sim = make_cube(shape=(5, 5), mode="free_scattering", noise=0.003, seed=7)
    mat = write_mat(tmp_path / "e.mat", sim["cube"], sim["wavelengths"])
    out = {}
    for method in ("fast", "exact"):
        cfg = _cfg(tmp_path, mat)
        cfg.solver.method = method
        cfg.model.reference_scattering.method = "manual"
        cfg.model.reference_scattering.a0, cfg.model.reference_scattering.b0 = 20, 1.2
        out[method] = run(cfg)
    # with noise both are local optima of the same problem; the fast one should never be worse
    assert np.all(out["fast"].maps["residual_rms"] <= out["exact"].maps["residual_rms"] * (1 + 1e-5) + 1e-8)
    assert _rel_err(out["fast"].maps["HbO2"][0], out["exact"].maps["HbO2"][0]) < 0.05


def test_matches_research_code_model(tmp_path):
    """Our free-scattering model and the research code's compute_delta_A give the same ΔA."""
    import importlib.util

    rc_file = Path(__file__).resolve().parents[2] / "src" / "scattering_model" / "scatter.py"
    if not rc_file.exists():
        pytest.skip("research code not present (standalone copy of the spectrafit folder)")
    spec = importlib.util.spec_from_file_location("rc_scatter", rc_file)
    rc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rc)
    from spectrafit.model import Problem

    lib = ChromophoreLibrary(include_user_dir=False)
    wl = np.arange(530, 751, 5.0)
    M = absorption_matrix([lib.get(n) for n in CHROMS], wl)
    PL = np.linspace(0.5, 1.5, wl.size)
    dc = np.array([0.01, -0.005, 0.002, -0.001, 0.03, -0.02])
    p = Problem(wavelengths=wl, M=M, names=CHROMS, mode="free_scattering", absolute=False, a0=20, b0=1.2)
    ours = p.predict(np.append(dc, 25.0)[None], PL, np.array([1.5]))[0]
    theirs = rc.compute_delta_A(M=M, delta_c_i=dc, a_i=25.0, b_i=1.5, b_0=1.2, a_0=20, x=wl, PL=PL)
    assert np.allclose(ours, theirs, rtol=1e-12)


# ------------------------------------------------------------------ references, masks, export
def test_reference_types(tmp_path):
    sim = make_cube(shape=(12, 12), mode="no_scattering", seed=3)
    mat = write_mat(tmp_path / "r.mat", sim["cube"], sim["wavelengths"])
    # ROI around the reference pixel only
    cfg = _cfg(tmp_path, mat, mode="no_scattering", ref="roi")
    cfg.reference.roi = [[0, 0], [0, 1], [1, 1], [1, 0]]
    r1 = run(cfg)
    assert _rel_err(r1.maps["HbO2"][0], sim["HbO2"][0]) < 1e-4
    # reference from a file
    spec = tmp_path / "ref.csv"
    np.savetxt(spec, np.column_stack([sim["wavelengths"], sim["cube"][0, 0, 0]]), delimiter=",",
               header="wavelength_nm,reflectance")
    cfg = _cfg(tmp_path, mat, mode="no_scattering", ref="file")
    cfg.reference.file = str(spec)
    r2 = run(cfg)
    assert _rel_err(r2.maps["HbO2"][0], sim["HbO2"][0]) < 1e-4


def test_time_series_frame_reference(tmp_path):
    sim = make_cube(shape=(8, 8), mode="fixed_b", n_frames=3, seed=8)
    # make frame 0 the per-pixel reference: dividing by it gives ΔA(t) - ΔA(0)
    mat = write_mat(tmp_path / "t.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat, mode="fixed_b", ref="frame")
    cfg.reference.frame = [0]
    cfg.model.fixed_b = 1.2
    res = run(cfg)
    assert res.maps["HbO2"].shape == (3, 8, 8)
    expected = sim["HbO2"][2] - sim["HbO2"][0]
    assert _rel_err(res.maps["HbO2"][2], expected) < 1e-4
    assert np.nanmax(np.abs(res.maps["HbO2"][0])) < 1e-8


def test_mask_exclude_and_export(tmp_path):
    sim = make_cube(shape=(10, 10), mode="no_scattering", seed=9)
    mat = write_mat(tmp_path / "m.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat, mode="no_scattering")
    cfg.mask.polygons = [[[0, 0], [0, 5], [5, 5], [5, 0]]]
    cfg.spectral.exclude = [[600, 620]]
    res = run(cfg)
    assert not np.any(np.isin(res.wavelengths, [600, 605, 610, 615, 620]))
    hb = res.maps["HbO2"][0]
    assert np.isfinite(hb[:5, :5]).all() and np.isnan(hb[6:, :]).all()
    files = res.save()
    names = {Path(f).name for f in files}
    assert {"settings.yaml", "results.npz", "results.mat", "summary.csv", "fit_info.json", "HbO2.tif"} <= names
    back = Config.load(Path(cfg.output.folder) / "settings.yaml")
    assert back.mask.polygons == cfg.mask.polygons
    from scipy.io import loadmat

    m = loadmat(Path(cfg.output.folder) / "results.mat")
    assert m["HbO2"].shape == (10, 10)


def test_config_aliases_and_validation():
    cfg = Config.from_dict({"pathlength": {"mode": "gray_matter_jacques"}})
    assert cfg.pathlength.mode == "jacques"
    with pytest.raises(ValueError):
        Config.from_dict({"model": {"mode": "magic"}})
    with pytest.raises(ValueError):
        Config.from_dict({"modle": {}})


def test_collinearity_warning(tmp_path):
    sim = make_cube(shape=(6, 6), mode="no_scattering", seed=10)
    mat = write_mat(tmp_path / "c.mat", sim["cube"], sim["wavelengths"])
    cfg = _cfg(tmp_path, mat, mode="no_scattering")
    cfg.chromophores.names = CHROMS + ["oxCytb", "redCytb", "oxCytc", "redCytc"]
    res = run(cfg)
    assert res.collinearity["condition_number"] > 1
    assert any("hard to separate" in n for n in res.notes) == (res.collinearity["level"] != "ok")


def test_parallel_matches_serial(tmp_path):
    """Several cores give exactly the same maps and a0/b0 as one core."""
    sim = make_cube(shape=(64, 64), mode="free_scattering", a0=100 / 3, b0=1.2, noise=0.002, seed=11, realistic=True)
    mat = write_mat(tmp_path / "p.mat", sim["cube"], sim["wavelengths"])
    out = {}
    for cores in (1, 2):
        cfg = _cfg(tmp_path, mat)
        cfg.solver.cores = cores
        cfg.model.reference_scattering.grid_binning = 4
        out[cores] = run(cfg)
    assert (out[1].a0, out[1].b0) == (out[2].a0, out[2].b0)
    assert any("2 processor cores" in n for n in out[2].notes)
    for k in ("HbO2", "HHb", "oxCCO", "b", "residual_rms", "se_HbO2"):
        np.testing.assert_allclose(out[2].maps[k], out[1].maps[k], rtol=1e-10, atol=1e-14, equal_nan=True)


# ------------------------------------------------------------------ ENVI (HELICOID format)
@pytest.mark.parametrize("interleave", ["bil", "bip", "bsq"])
def test_envi_layouts(tmp_path, interleave):
    from spectrafit.synthetic import write_envi

    sim = make_cube(shape=(12, 10), mode="no_scattering", realistic=True)
    cube = sim["cube"][0].astype(np.float32)
    write_envi(tmp_path / "cube", cube, sim["wavelengths"], interleave=interleave, dtype="<f4")
    cfg = Config()
    cfg.data.file = str(tmp_path / "cube.hdr")  # the header or the data file can be chosen
    d = load_data(cfg.data)
    assert d.cube.shape == (1, 12, 10, 127)
    assert np.allclose(d.cube[0], cube) and np.allclose(d.wavelengths, sim["wavelengths"])


def test_helicoid_layout_raw_counts(tmp_path):
    """A HELICOID sample folder: 12-bit counts, one-line white/dark references, old Mac line endings."""
    from spectrafit.synthetic import write_envi

    rng = np.random.default_rng(3)
    H, W, B = 16, 14, 127
    wl = np.linspace(400.5, 1000.7, B)
    R = rng.uniform(0.05, 0.4, (H, W, B))
    white = rng.uniform(600, 1400, (1, W, B)).round()
    dark = rng.uniform(17, 43, (1, W, B)).round()
    raw = (R * (white - dark) + dark).round()
    folder = tmp_path / "004-02"
    folder.mkdir()
    write_envi(folder / "raw", raw, wl, mac_line_endings=True)
    write_envi(folder / "whiteReference", white, wl, mac_line_endings=True)
    write_envi(folder / "darkReference", dark, wl, mac_line_endings=True)
    from spectrafit.io import inspect_file

    info = inspect_file(folder / "raw")
    assert info["format"] == "ENVI" and info["white"].endswith("whiteReference.hdr")
    cfg = Config()
    cfg.data.file = str(folder)  # the sample folder itself
    cfg.data.is_reflectance = False
    cfg.data.binning = 2
    cfg.data.crop = [2, 14, 0, 14]
    d = load_data(cfg.data)
    expected = ((raw - dark) / (white - dark) + 0.1)[2:14].reshape(6, 2, 7, 2, B).mean(axis=(1, 3))
    assert d.cube.shape == (1, 6, 7, B)
    assert np.allclose(d.cube[0], expected, rtol=1e-5)
    assert any("references found next to the data" in n for n in d.notes)
    cfg.data.is_reflectance = True
    assert any("look like raw counts" in n for n in load_data(cfg.data).notes)


def test_helicoid_fit_runs(tmp_path):
    from spectrafit.synthetic import write_envi

    sim = make_cube(shape=(20, 20), mode="fixed_b", fixed_b=1.2, seed=4, realistic=True,
                    wavelengths=np.linspace(400.5, 1000.7, 826))
    write_envi(tmp_path / "raw", sim["cube"][0], sim["wavelengths"], dtype="<f4")
    cfg = _cfg(tmp_path, tmp_path / "raw", mode="fixed_b")
    cfg.model.fixed_b = 1.2
    res = run(cfg)
    assert res.wavelengths.size > 290  # every HELICOID band in 530–750 nm is used
    assert _rel_err(res.maps["HbO2"][0], sim["HbO2"][0]) < 2e-3
