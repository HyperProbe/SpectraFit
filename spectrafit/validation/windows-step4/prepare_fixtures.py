"""Create public synthetic validation inputs and source-reference result maps."""

import json
import os
from importlib.metadata import version
from pathlib import Path

import h5py
import numpy as np
from scipy.io import savemat

from spectrafit.chromophores import Chromophore, ChromophoreLibrary, absorption_matrix, write_chromophore_file
from spectrafit.config import Config
from spectrafit.pathlength import global_pathlength
from spectrafit.pipeline import run
from spectrafit.synthetic import make_cube, write_envi, write_mat

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / ".windows-baseline/step4-fixtures"


def relative(path):
    return os.path.relpath(path, FIXTURES / "cases").replace("/", "\\")


def config(path, mode="free_scattering", absolute=False):
    cfg = Config()
    cfg.data.file = relative(path)
    cfg.spectral.range = [500.0, 900.0]
    cfg.reference.type = "none" if absolute else "pixel"
    cfg.reference.pixel = None if absolute else [0, 0]
    cfg.model.mode = mode
    cfg.model.fixed_b = 1.2
    cfg.model.reference_scattering.method = "manual"
    cfg.model.reference_scattering.a0 = 20.0
    cfg.model.reference_scattering.b0 = 1.2
    cfg.solver.cores = 1
    cfg.output.folder = "generated-results"
    return cfg


def main():
    for folder in ("data", "cases", "expected"):
        (FIXTURES / folder).mkdir(parents=True, exist_ok=True)
    os.environ["SPECTRAFIT_HOME"] = str(ROOT / ".windows-baseline/step4-generation-home")
    cases = []

    def add(name, cfg, truth=None, description="", gui=False):
        path = FIXTURES / "cases" / f"{name}.yaml"
        cfg.save(path)
        source = Config.load(path)
        source.solver.cores = 1
        result = run(source)
        expected = FIXTURES / "expected" / f"{name}.npz"
        np.savez_compressed(expected, wavelengths=result.wavelengths, **result.maps)
        item = {"name": name, "settings": f"fixtures/cases/{name}.yaml",
                "expected": f"fixtures/expected/{name}.npz", "description": description,
                "rtol": 1e-6, "atol": 1e-8, "map_count": len(result.maps), "gui": gui}
        if truth is not None:
            errors = {}
            for chromophore in source.chromophores.names:
                actual, target = result.maps[chromophore], truth[chromophore]
                finite = np.isfinite(actual)
                error = float(np.max(np.abs(actual[finite] - target[finite])) / max(np.max(np.abs(target[finite])), 1e-12))
                assert error < 2e-3, (name, chromophore, error)
                errors[chromophore] = error
            item["source_truth_relative_errors"] = errors
        cases.append(item)
        print(f"Prepared {name}", flush=True)

    for absolute in (False, True):
        for mode in ("no_scattering", "fixed_b", "free_scattering"):
            name = f"{'absolute' if absolute else 'differential'}-{mode}"
            sim = make_cube(shape=(14, 14), mode=mode, absolute=absolute, seed=1)
            data = write_mat(FIXTURES / "data" / f"{name} ü.mat", sim["cube"], sim["wavelengths"])
            add(name, config(data, mode, absolute), sim, "MATLAB v5; known synthetic concentrations")

    sim = make_cube(shape=(5, 5), mode="free_scattering", noise=0.003, seed=7)
    path = write_mat(FIXTURES / "data/exact.mat", sim["cube"], sim["wavelengths"])
    cfg = config(path)
    cfg.solver.method = "exact"
    add("exact-scipy-solver", cfg, description="SciPy pixelwise nonlinear solver")

    sim = make_cube(shape=(12, 12), mode="no_scattering", seed=3)
    data = write_mat(FIXTURES / "data/hdf5-cube.mat", sim["cube"], sim["wavelengths"], v73=True)
    spectrum = sim["cube"][0, 0, 0]
    references = {}
    references["npy"] = FIXTURES / "data/reference.npy"
    np.save(references["npy"], spectrum)
    references["mat5"] = FIXTURES / "data/reference-v5.mat"
    savemat(references["mat5"], {"reference_spectrum": spectrum[None]})
    references["mat73"] = FIXTURES / "data/reference-v73.mat"
    with h5py.File(references["mat73"], "w") as file:
        file.create_dataset("reference_spectrum", data=spectrum[:, None])
    for kind, reference in references.items():
        cfg = config(data, "no_scattering")
        cfg.reference.type, cfg.reference.file = "file", relative(reference)
        add(f"hdf5-reference-{kind}", cfg, description=f"MATLAB v7.3 cube; {kind} reference")
    selected = (sim["wavelengths"] >= 510) & (sim["wavelengths"] <= 900)
    legacy = FIXTURES / "data/reference-79.npy"
    np.save(legacy, spectrum[selected])
    cfg = config(data, "no_scattering")
    cfg.reference.type, cfg.reference.file = "file", relative(legacy)
    cfg.spectral.range = [510, 900]
    add("legacy-79-band-reference", cfg, description="79-band reference aligned with 127-band cube")

    for interleave in ("bil", "bip", "bsq"):
        folder = FIXTURES / "data" / f"envi-{interleave}"
        calibration = folder / "calibration"
        calibration.mkdir(parents=True, exist_ok=True)
        white = np.full((1, 12, 127), 1000, np.float32)
        dark = np.full_like(white, 10)
        raw = sim["cube"][0] * (white - dark) + dark
        write_envi(folder / "raw", raw, sim["wavelengths"], interleave=interleave, dtype="<f4",
                   mac_line_endings=(interleave == "bil"))
        write_envi(calibration / "whiteReference", white, sim["wavelengths"], dtype="<f4")
        write_envi(calibration / "darkReference", dark, sim["wavelengths"], dtype="<f4")
        cfg = config(folder / "raw.hdr", "no_scattering")
        cfg.data.is_reflectance, cfg.data.normalisation_offset = False, 0.0
        cfg.data.white_reference = relative(calibration / "whiteReference.hdr")
        cfg.data.dark_reference = relative(calibration / "darkReference.hdr")
        if interleave == "bil":
            cfg.data.binning, cfg.data.crop = 2, [2, 12, 0, 12]
            cfg.reference.type, cfg.reference.file = "file", relative(references["npy"])
        add(f"envi-calibration-{interleave}", cfg, description="ENVI raw calibration; relative backslash paths")

    sim = make_cube(shape=(8, 8), mode="fixed_b", n_frames=3, seed=8)
    path = write_mat(FIXTURES / "data/time-series.mat", sim["cube"], sim["wavelengths"])
    cfg = config(path, "fixed_b")
    cfg.reference.type, cfg.reference.frame = "frame", [0]
    add("time-series-4d", cfg, description="Three-frame MATLAB cube; per-pixel baseline frame")
    frames = FIXTURES / "data/frames"
    frames.mkdir(exist_ok=True)
    for index in range(3):
        write_mat(frames / f"frame{index}.mat", sim["cube"][index:index + 1], sim["wavelengths"])
    cfg.data.file = relative(frames / "*.mat")
    add("time-series-relative-glob", cfg, description="Settings-relative glob; three MATLAB frame files")

    sim = make_cube(shape=(10, 10), mode="no_scattering", seed=9)
    path = write_mat(FIXTURES / "data/regions.mat", sim["cube"], sim["wavelengths"])
    cfg = config(path, "no_scattering")
    cfg.reference.type = "roi"
    cfg.reference.roi = [[-0.49, -0.49], [-0.49, 0.49], [0.49, 0.49], [0.49, -0.49]]
    cfg.mask.polygons = [[[0, 0], [0, 5], [5, 5], [5, 0]]]
    cfg.spectral.exclude = [[600, 620]]
    add("roi-mask-excluded-bands", cfg, description="ROI reference, polygon mask and omitted wavelengths")

    library = ChromophoreLibrary(include_user_dir=False)
    chroms = [library.get(name) for name in Config().chromophores.names]
    for mode in ("gray_matter_table", "delta_p1", "jacques"):
        cfg = config(path)
        cfg.pathlength.mode = mode
        wavelengths = np.arange(385.0, 1016.0, 5.0)
        mask = (wavelengths >= 500) & (wavelengths <= 900)
        pl = np.ones_like(wavelengths)
        pl[mask] = global_pathlength(cfg.pathlength, wavelengths[mask],
                                    absorption_matrix(chroms, wavelengths[mask]), np.array([c.baseline for c in chroms]))
        sample = make_cube(shape=(10, 10), mode="free_scattering", pathlength=pl, seed=2)
        file = write_mat(FIXTURES / "data" / f"{mode}.mat", sample["cube"], wavelengths)
        cfg.data.file = relative(file)
        add(f"pathlength-{mode}", cfg, description="Tissue/global optical pathlength model")
    sample = make_cube(shape=(10, 10), mode="free_scattering", seed=5)
    file = write_mat(FIXTURES / "data/pixelwise.mat", sample["cube"], sample["wavelengths"])
    for mode in ("delta_p1_pixelwise", "jacques_pixelwise"):
        cfg = config(file)
        cfg.pathlength.mode = mode
        add(f"pathlength-{mode}", cfg, description="Two-pass pixelwise optical pathlength")

    dye = Chromophore("dye", np.arange(400.0, 1001.0),
                      2000 * np.exp(-((np.arange(400.0, 1001.0) - 650) / 20) ** 2), "eps10_per_mM_cm")
    dye_file = write_chromophore_file(dye, FIXTURES / "data/dye.csv")
    cfg = config(file)
    cfg.chromophores.extra_files = [relative(dye_file)]
    cfg.chromophores.names += ["oxCytc"]
    add("custom-chromophore", cfg, description="Custom CSV absorber plus built-in cytochrome c")

    sample = make_cube(shape=(64, 64), mode="free_scattering", noise=0.001, seed=11, realistic=True)
    file = write_mat(FIXTURES / "data/gui-64.mat", sample["cube"], sample["wavelengths"])
    cfg = config(file)
    cfg.solver.cores = 2
    add("gui-two-core", cfg, description="64x64 synthetic GUI/multiprocessing fixture", gui=True)
    errors = []
    for name, mutate, message in (
        ("invalid-model", lambda c: setattr(c.model, "mode", "invalid"), "model.mode"),
        ("missing-data", lambda c: setattr(c.data, "file", r"..\data\missing.mat"), "File not found"),
        ("insufficient-bands", lambda c: setattr(c.spectral, "range", [530, 535]), "Only 2 wavelengths"),
    ):
        c = config(file)
        mutate(c)
        c.save(FIXTURES / "cases" / f"{name}.yaml")
        errors.append({"name": name, "settings": f"fixtures/cases/{name}.yaml", "message": message})
    manifest = {"fixture_scope": "Synthetic only; no user measurement/reference data", "cases": cases,
                "expected_errors": errors, "versions": {name: version(name) for name in ("spectrafit", "numpy", "scipy", "h5py")}}
    (FIXTURES / "cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Prepared {len(cases)} numerical cases and {len(errors)} error cases", flush=True)


if __name__ == "__main__":
    main()
