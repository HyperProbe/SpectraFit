"""Command line:  spectrafit gui | run | new-config | example | chromophores"""

from __future__ import annotations

import argparse
import shutil
import sys
from importlib import resources
from pathlib import Path


def _progress(frac: float, msg: str) -> None:
    bar = "#" * int(30 * frac)
    sys.stdout.write(f"\r[{bar:<30}] {100 * frac:5.1f}%  {msg[:60]:<60}")
    sys.stdout.flush()
    if frac >= 1:
        sys.stdout.write("\n")


def cmd_run(args) -> int:
    from .config import Config
    from .pipeline import run

    cfg = Config.load(args.config)
    if args.output:
        cfg.output.folder = str(Path(args.output).resolve())
    files = None
    if args.batch:
        from .io import resolve_files

        files = resolve_files(args.batch)
    targets = files or [cfg.data.file]
    for i, f in enumerate(targets):
        cfg.data.file = f
        out = Path(cfg.output.folder)
        if files:
            out = out / Path(f).stem
            print(f"[{i + 1}/{len(targets)}] {f}")
        res = run(cfg, progress=_progress)
        for n in res.notes:
            print("  -", n)
        paths = res.save(out)
        print(f"Results written to {out} ({len(paths)} files)")
    return 0


def cmd_new_config(args) -> int:
    src = Path(str(resources.files("spectrafit") / "data" / "config_template.yaml"))
    dst = Path(args.path)
    if dst.exists() and not args.force:
        print(f"{dst} already exists (use --force to overwrite)")
        return 1
    shutil.copyfile(src, dst)
    print(f"Wrote {dst}. Edit it, then run:  spectrafit run {dst}")
    return 0


def cmd_example(args) -> int:
    from .synthetic import make_cube, write_mat

    folder = Path(args.folder)
    folder.mkdir(parents=True, exist_ok=True)
    # simulated tissue with blood vessels; a0 lies on the default grid-search grid, so the fit reproduces the truth
    sim = make_cube(shape=(160, 160), mode="free_scattering", a0=100 / 3, b0=1.2, noise=0.001, seed=3,
                    realistic=True, pattern="tissue")
    ref_r, ref_c = (int(v) for v in sim["reference_pixel"])
    write_mat(folder / "example_cube.mat", sim["cube"], sim["wavelengths"])
    import numpy as np

    np.savez_compressed(folder / "example_truth.npz",
                        **{k: v for k, v in sim.items() if k not in ("cube",)})
    src = Path(str(resources.files("spectrafit") / "data" / "config_template.yaml"))
    text = src.read_text(encoding="utf-8").replace("file: my_measurement.mat", "file: example_cube.mat")
    text = text.replace("pixel: [100, 100]", f"pixel: [{ref_r}, {ref_c}]").replace("binning: 4 ", "binning: 1 ")
    (folder / "example_settings.yaml").write_text(text, encoding="utf-8")
    print(f"Example data written to {folder}:")
    print(f"  example_cube.mat        simulated 160×160 tissue image with blood vessels, 385–1015 nm, "
          f"reference pixel at row {ref_r}, col {ref_c}")
    print("  example_truth.npz       the true maps used to simulate it")
    print("  example_settings.yaml   settings to analyse it")
    print(f"Try:  spectrafit run {folder / 'example_settings.yaml'}    or open it in the GUI")
    return 0


def cmd_chromophores(args) -> int:
    from .chromophores import ChromophoreLibrary, user_dir

    lib = ChromophoreLibrary()
    print(f"{'name':<10} {'range (nm)':<14} {'unit':<16} {'baseline':<10} label")
    for n in lib.names():
        c = lib.get(n)
        print(f"{n:<10} {c.range[0]:>5g}–{c.range[1]:<7g} {c.unit:<16} {c.baseline:<10g} {c.label}")
    print(f"\nAdd your own: put files in {user_dir()} (see the guide for the format).")
    return 0


def cmd_gui(args) -> int:
    try:
        import napari
    except ImportError:
        print("The graphical interface needs napari. Install with:  pip install 'spectrafit[gui]'")
        return 1
    from .napari_plugin.widget import SpectraFitWidget

    viewer = napari.Viewer(title="SpectraFit")
    widget = SpectraFitWidget(viewer)
    viewer.window.add_dock_widget(widget, name="SpectraFit", area="right")
    if args.config:
        widget.load_settings(args.config)
    napari.run()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="spectrafit", description="Spectral unmixing of hyperspectral images.")
    sub = p.add_subparsers(dest="cmd")
    g = sub.add_parser("gui", help="open the graphical interface (napari)")
    g.add_argument("config", nargs="?", help="settings file to load")
    r = sub.add_parser("run", help="run an analysis from a settings file")
    r.add_argument("config")
    r.add_argument("-o", "--output", help="output folder (overrides the settings file)")
    r.add_argument("--batch", help="folder or pattern of .mat files: run the same settings on each")
    n = sub.add_parser("new-config", help="write a commented settings file to edit")
    n.add_argument("path", nargs="?", default="spectrafit_settings.yaml")
    n.add_argument("--force", action="store_true")
    e = sub.add_parser("example", help="create a synthetic example dataset with settings")
    e.add_argument("folder", nargs="?", default="spectrafit_example")
    sub.add_parser("chromophores", help="list available chromophores")
    args = p.parse_args(argv)
    if args.cmd is None:
        args.cmd, args.config = "gui", None
    return {"gui": cmd_gui, "run": cmd_run, "new-config": cmd_new_config, "example": cmd_example,
            "chromophores": cmd_chromophores}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
