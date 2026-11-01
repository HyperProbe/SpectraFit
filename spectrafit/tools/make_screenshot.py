"""Make the annotated screenshot of the app used in the user guide (docs/screenshot.png).

Run from the spectrafit folder (on Linux without a screen: xvfb-run -a python tools/make_screenshot.py):

    python tools/make_screenshot.py                       # simulated example (default)
    python tools/make_screenshot.py --data FILE_OR_FOLDER --ref ROW COL --credit "Image: ..."

It opens the app, loads the data, fits it, shows the ΔHbT map, inspects one vessel pixel,
and draws numbered labels that the user guide explains.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "docs" / "screenshot.png"
COLOUR = (255, 140, 0)  # orange: different from the yellow inspected-pixel square and the cyan reference cross


def _font(size):
    from PIL import ImageFont
    import matplotlib

    path = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
    return ImageFont.truetype(str(path), size)


def _annotate(img, boxes, badges, note):
    """Frame each box, put its number badge at the given point (in free space), and a note on the image."""
    from PIL import Image, ImageDraw

    im = Image.fromarray(img[..., :3])
    d = ImageDraw.Draw(im)
    font = _font(22)
    r = 17
    for n, (x, y, w, h) in boxes.items():
        d.rectangle([x, y, x + w, y + h], outline=COLOUR, width=3)
    for n, (cx, cy) in badges.items():
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLOUR, outline=(0, 0, 0), width=2)
        d.text((cx, cy), str(n), fill=(0, 0, 0), font=font, anchor="mm")
    if note:
        x, y, w, h = boxes[2]
        f = _font(16)
        tw = d.textlength(note, font=f)
        d.rounded_rectangle([x + 10, y + h - 40, x + 26 + tw, y + h - 12], radius=6, fill=(0, 0, 0))
        d.text((x + 18, y + h - 26), note, fill=(255, 255, 255), font=f, anchor="lm")
    return im


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="data file or folder (default: the simulated example)")
    ap.add_argument("--ref", nargs=2, type=int, metavar=("ROW", "COL"), help="reference pixel (original image)")
    ap.add_argument("--settings", help="settings file to load instead of the defaults")
    ap.add_argument("--credit", default=None, help="note shown on the image (default: 'Simulated data')")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import napari
    from qtpy.QtWidgets import QApplication

    from spectrafit.cli import main as cli
    from spectrafit.napari_plugin.widget import SpectraFitWidget

    work = Path(tempfile.mkdtemp())
    settings = args.settings
    if args.data is None:
        cli(["example", str(work)])
        settings = str(work / "example_settings.yaml")
        note = args.credit or "Simulated data"
    else:
        note = args.credit

    viewer = napari.Viewer(title="SpectraFit")
    w = SpectraFitWidget(viewer)
    viewer.window.add_dock_widget(w, name="SpectraFit", area="right")
    win = viewer.window._qt_window
    win.resize(1500, 940)
    win.show()
    app = QApplication.instance()

    def pump(s=0.3):
        t = time.time()
        while time.time() - t < s:
            app.processEvents()
            time.sleep(0.02)

    def wait():
        while w._worker is not None:
            pump(0.2)
        pump(0.6)

    if settings:
        w.load_settings(settings)
    if args.data:
        w.file_edit.setText(args.data)
        from spectrafit.io import resolve_files

        w._inspect(resolve_files(args.data)[0])
    w.autosave.setChecked(False)
    w.out_edit.setText("spectrafit_results")
    w._load_data()
    wait()
    if args.ref:
        w.ref_row.setValue(args.ref[0])
        w.ref_col.setValue(args.ref[1])
    w._ref_confirmed = True
    w._run_fit()
    wait()
    if w.results is None:
        raise SystemExit("The fit did not finish: " + w.status.text())
    w._focus_map("HbT")
    # inspect a pixel in the middle of a vein (most deoxyhaemoglobin)
    hhb = np.nan_to_num(w.results.maps["HHb"][0])
    row, col = np.unravel_index(np.argmax(hhb), hhb.shape)

    class Press:
        type = "mouse_press"

    viewer.cursor.position = (float(row), float(col))
    gen = w._on_click(viewer, Press())
    try:
        next(gen)
    except StopIteration:
        pass
    w.tabs.setCurrentIndex(4)
    viewer.cursor.position = (float(row), float(col))  # the status bar then shows the inspected pixel's value
    pump(1.0)

    img = viewer.screenshot(canvas_only=False, flash=False)
    scale = img.shape[1] / win.width()

    def box(widget):
        p = widget.mapTo(win, widget.rect().topLeft())
        return tuple(int(round(v * scale)) for v in (p.x(), p.y(), widget.width(), widget.height()))

    canvas = viewer.window._qt_viewer.canvas.native
    boxes = {1: box(w.tabs.tabBar()), 2: box(canvas), 3: box(w.inspector.plot), 4: box(w.map_list)}
    (x1, y1, w1, h1), (x2, y2, w2, h2), (x3, y3, w3, h3), (x4, y4, w4, h4) = (boxes[k] for k in (1, 2, 3, 4))
    badges = {1: (x1 + w1 + 26, y1 + h1 // 2),  # right of the tab bar
              2: (x2 + w2 - 28, y2 + 28),  # top-right of the image area (black)
              3: (x3 + 26, y3 + 26),  # top-left of the plot (white margin)
              4: (x4 + int(0.62 * w4), y4 - 21)}  # above the list, between its title and 'side by side'
    im = _annotate(img, boxes, badges, note)
    if im.width > 1400:
        im = im.resize((1400, round(im.height * 1400 / im.width)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    im.convert("P", palette=1, colors=256).save(args.out, optimize=True)  # adaptive palette keeps it small
    print(f"Saved {args.out} ({os.path.getsize(args.out) // 1024} KB)")
    viewer.close()


if __name__ == "__main__":
    main()
