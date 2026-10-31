"""Reading hyperspectral cubes from MATLAB .mat files and ENVI files.

Handles both MATLAB file versions (v5/v7 via scipy, v7.3 via h5py) and ENVI
(a binary file plus a .hdr text header, as used by the HELICOID database).
Finds the cube and its wavelengths automatically when possible, calibrates raw
counts with white/dark references, and crops and bins while reading, so a
2048×2048×127 cube never has to be held in memory at full resolution.

Internally a cube is always float32 with axes (T, H, W, B): time, rows,
columns, bands. A single image has T = 1.
"""

from __future__ import annotations

import glob
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import DataSettings

log = logging.getLogger(__name__)

HYPERPROBE_DEFAULT_WAVELENGTHS = np.arange(385.0, 1015.0 + 0.1, 5.0)  # 127 bands
PREFERRED_CUBE_NAMES = ("Ref_hyper", "cube", "hsi", "data", "img", "image", "I", "R")
WAVELENGTH_NAMES = ("wavelengths", "wavelength", "wl", "wls", "lambda", "lambdas", "Wavelengths", "Lambda", "WL")

Progress = Optional[Callable[[float, str], None]]


# --------------------------------------------------------------------------- .mat access
class MatFile:
    """Uniform read access to variables of a v5 or v7.3 .mat file, in MATLAB axis order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"File not found: {self.path}")
        self._h5 = None
        self._v5: Optional[dict] = None
        try:
            import h5py

            if h5py.is_hdf5(self.path):
                self._h5 = h5py.File(self.path, "r")
        except ImportError:  # pragma: no cover
            pass
        if self._h5 is None:
            from scipy.io import loadmat

            self._v5 = {k: v for k, v in loadmat(self.path).items() if not k.startswith("__")}

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def variables(self) -> Dict[str, Tuple[int, ...]]:
        """Numeric variables and their shapes (MATLAB order)."""
        out: Dict[str, Tuple[int, ...]] = {}
        if self._h5 is not None:
            import h5py

            def visit(name, obj):
                if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "fiub" and "#" not in name:
                    out[name] = tuple(reversed(obj.shape))

            self._h5.visititems(visit)
        else:
            for k, v in self._v5.items():
                if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
                    out[k] = tuple(v.shape)
        return out

    def shape(self, name: str) -> Tuple[int, ...]:
        if self._h5 is not None:
            return tuple(reversed(self._h5[name].shape))
        return tuple(self._v5[name].shape)

    def read(self, name: str, index: Optional[Tuple] = None) -> np.ndarray:
        """Read a variable (optionally a sub-block given as a tuple of slices in MATLAB order)."""
        if self._h5 is not None:
            ds = self._h5[name]
            if index is None:
                arr = ds[...]
            else:
                arr = ds[tuple(reversed(index))]
            return np.asarray(arr).transpose()
        arr = self._v5[name]
        return np.asarray(arr if index is None else arr[index])


# --------------------------------------------------------------------------- ENVI files
ENVI_DTYPES = {1: "u1", 2: "i2", 3: "i4", 4: "f4", 5: "f8", 12: "u2", 13: "u4", 14: "i8", 15: "u8"}


def envi_header_path(path: str | Path) -> Optional[Path]:
    """The .hdr belonging to `path` (which may be the header itself or the binary file), if any."""
    p = Path(path)
    if p.suffix.lower() == ".hdr":
        return p if p.is_file() else None
    if p.suffix.lower() == ".mat":
        return None
    for cand in (p.with_name(p.name + ".hdr"), p.with_suffix(".hdr")):
        if cand.is_file():
            return cand
    return None


def is_envi(path: str | Path) -> bool:
    return envi_header_path(path) is not None


def read_envi_header(path: str | Path) -> Dict[str, str]:
    """Parse an ENVI header. Copes with Windows, Unix and old Mac line endings, and {multi-line} values."""
    text = Path(path).read_bytes().decode("latin-1").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in text.split("\n")]
    if not lines or not lines[0].upper().startswith("ENVI"):
        raise ValueError(f"{path} is not an ENVI header (it should start with 'ENVI').")
    out: Dict[str, str] = {}
    i = 1
    while i < len(lines):
        line = lines[i]
        i += 1
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if value.startswith("{") and "}" not in value:
            parts = [value]
            while i < len(lines) and "}" not in lines[i]:
                parts.append(lines[i])
                i += 1
            if i < len(lines):
                parts.append(lines[i])
                i += 1
            value = " ".join(parts)
        out[key.strip().lower()] = value
    return out


def _envi_list(value: str) -> List[str]:
    return [v.strip() for v in value.strip().strip("{}").split(",") if v.strip()]


class EnviFile:
    """Read access to an ENVI cube with the same interface as MatFile.

    The cube appears as variable 'cube' with axes (H, W, B) = (lines, samples, bands);
    the wavelengths from the header appear as variable 'wavelength'.
    """

    def __init__(self, path: str | Path):
        hdr = envi_header_path(path)
        if hdr is None:
            raise FileNotFoundError(f"No ENVI header (.hdr) found for {path}")
        self.header = h = read_envi_header(hdr)
        self.path = self._data_path(hdr)
        try:
            samples, lines, bands = int(h["samples"]), int(h["lines"]), int(h["bands"])
            dtype = np.dtype(ENVI_DTYPES[int(h.get("data type", "4"))])
        except KeyError as e:
            raise ValueError(f"{hdr.name}: missing or unsupported header entry {e}") from None
        dtype = dtype.newbyteorder(">" if int(h.get("byte order", "0")) == 1 else "<")
        offset = int(h.get("header offset", "0"))
        self.interleave = h.get("interleave", "bsq").strip().lower()
        shapes = {"bsq": (bands, lines, samples), "bil": (lines, bands, samples), "bip": (lines, samples, bands)}
        if self.interleave not in shapes:
            raise ValueError(f"{hdr.name}: unknown interleave '{self.interleave}'")
        expected = offset + int(np.prod(shapes[self.interleave])) * dtype.itemsize
        if self.path.stat().st_size != expected:
            raise ValueError(f"{self.path.name} has {self.path.stat().st_size} bytes, but its header describes "
                             f"{expected} ({lines}×{samples}×{bands}, {dtype.name}).")
        self._mm = np.memmap(self.path, dtype=dtype, mode="r", offset=offset, shape=shapes[self.interleave])
        self._shape = (lines, samples, bands)
        self.wavelengths = None
        if "wavelength" in h:
            wl = np.array([float(v) for v in _envi_list(h["wavelength"])])
            if wl.size == bands:
                if "micro" in h.get("wavelength units", "").lower() or wl.max() < 20:
                    wl = wl * 1000.0
                self.wavelengths = wl

    @staticmethod
    def _data_path(hdr: Path) -> Path:
        stem = hdr.with_suffix("")
        for cand in [stem] + [stem.with_suffix(ext) for ext in (".raw", ".img", ".dat", ".bin", ".bil", ".bsq", ".bip")]:
            if cand.is_file():
                return cand
        raise FileNotFoundError(f"Found {hdr.name} but not the data file next to it (expected '{stem.name}').")

    def close(self) -> None:
        self._mm = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def variables(self) -> Dict[str, Tuple[int, ...]]:
        out: Dict[str, Tuple[int, ...]] = {"cube": self._shape}
        if self.wavelengths is not None:
            out["wavelength"] = (self.wavelengths.size,)
        return out

    def shape(self, name: str) -> Tuple[int, ...]:
        return self.variables()[name]

    def read(self, name: str, index: Optional[Tuple] = None) -> np.ndarray:
        if name == "wavelength":
            return self.wavelengths.copy()
        r, c, b = index if index is not None else (slice(None),) * 3
        if self.interleave == "bil":
            return np.asarray(self._mm[r, b, c]).transpose(0, 2, 1)
        if self.interleave == "bip":
            return np.asarray(self._mm[r, c, b])
        return np.asarray(self._mm[b, r, c]).transpose(1, 2, 0)


def find_envi_references(path: str | Path) -> Tuple[Optional[str], Optional[str]]:
    """White and dark reference files stored next to an ENVI cube (HELICOID: whiteReference, darkReference)."""
    folder = Path(path).parent
    found = {"white": None, "dark": None}
    for hdr in sorted(folder.glob("*.hdr")):
        name = hdr.stem.lower()
        for kind in found:
            if found[kind] is None and name.startswith(kind) and "ref" in name:
                try:
                    EnviFile._data_path(hdr)
                    found[kind] = str(hdr)
                except FileNotFoundError:
                    pass
    return found["white"], found["dark"]


def open_data_file(path: str | Path):
    """MatFile for .mat files, EnviFile for ENVI (binary + .hdr)."""
    return EnviFile(path) if is_envi(path) else MatFile(path)


# --------------------------------------------------------------------------- result object
@dataclass
class HyperspectralData:
    cube: np.ndarray
    """Reflectance, float32, axes (T, H, W, B), after crop and binning."""
    wavelengths: np.ndarray
    original_shape: Tuple[int, int]
    """(H, W) before crop and binning."""
    crop: Tuple[int, int, int, int]
    """(row_start, row_stop, col_start, col_stop) actually used, original pixels."""
    binning: int
    files: List[str]
    variable: str
    axes: str
    frame_indices: List[int] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return self.cube.shape[0]

    @property
    def shape(self) -> Tuple[int, int]:
        return self.cube.shape[1], self.cube.shape[2]

    def to_working(self, row: float, col: float) -> Tuple[float, float]:
        """Original pixel coordinates -> coordinates in the (cropped, binned) working image."""
        return (row - self.crop[0]) / self.binning, (col - self.crop[2]) / self.binning

    def to_original(self, row: float, col: float) -> Tuple[float, float]:
        return row * self.binning + self.crop[0], col * self.binning + self.crop[2]

    def rgb(self, bands=(630.0, 540.0, 460.0), frame: int = 0) -> np.ndarray:
        """Simple colour preview from three bands, scaled to 0–1 per channel (2–98 percentile)."""
        chans = []
        for wl in bands:
            i = int(np.argmin(np.abs(self.wavelengths - wl)))
            c = self.cube[frame, :, :, i].astype(np.float32)
            lo, hi = np.nanpercentile(c, [2, 98])
            chans.append(np.clip((c - lo) / (hi - lo + 1e-12), 0, 1))
        return np.stack(chans, axis=-1)


# --------------------------------------------------------------------------- helpers
def resolve_files(spec: str) -> List[str]:
    """A single file, a folder (all .mat inside) or a glob pattern; sorted in natural order."""
    if not spec:
        raise ValueError("No data file selected.")
    p = Path(spec).expanduser()
    if p.is_dir():
        files = [str(f) for f in p.glob("*.mat")]
        if not files:  # a folder holding one ENVI cube (HELICOID: raw + raw.hdr + white/dark references)
            cubes = [h for h in p.glob("*.hdr") if not h.stem.lower().startswith(("white", "dark"))]
            if len(cubes) == 1:
                files = [str(cubes[0])]
    elif any(ch in spec for ch in "*?["):
        files = glob.glob(str(p))
    else:
        files = [str(p)]
    if not files:
        raise FileNotFoundError(f"No .mat or ENVI files found for '{spec}'")

    def natural(s):
        return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

    return sorted(files, key=natural)


def find_cube_variable(variables: Dict[str, Tuple[int, ...]]) -> str:
    candidates = {k: s for k, s in variables.items() if len([d for d in s if d > 1]) >= 3}
    if not candidates:
        raise ValueError(
            "No 3D or 4D array found in the file. Variables: "
            + ", ".join(f"{k} {list(s)}" for k, s in variables.items())
        )
    for name in PREFERRED_CUBE_NAMES:
        if name in candidates:
            return name
    return max(candidates, key=lambda k: int(np.prod(candidates[k])))


def find_wavelength_variable(mat: MatFile, variables: Dict[str, Tuple[int, ...]], lengths: Sequence[int]):
    """Look for a vector of plausible wavelengths whose length matches one of the cube axes."""
    ordered = [n for n in WAVELENGTH_NAMES if n in variables] + [n for n in variables if n not in WAVELENGTH_NAMES]
    for name in ordered:
        shape = variables[name]
        if len([d for d in shape if d > 1]) != 1:
            continue
        n = max(shape)
        if n not in lengths or n < 3:
            continue
        v = np.asarray(mat.read(name), dtype=float).ravel()
        if np.all(np.diff(v) > 0) and 200 <= v[0] and v[-1] <= 3000:
            return name, v
    return None, None


def detect_axes(shape: Tuple[int, ...], n_bands: Optional[int]) -> str:
    """Guess MATLAB axis order from the shape. Returns e.g. 'HWB' or 'HWBT'."""
    nd = len(shape)
    if nd not in (3, 4):
        raise ValueError(f"Expected a 3D (image) or 4D (time series) array, got shape {list(shape)}")
    if n_bands is None and list(shape).count(len(HYPERPROBE_DEFAULT_WAVELENGTHS)) == 1:
        n_bands = len(HYPERPROBE_DEFAULT_WAVELENGTHS)  # the HyperProbe camera: 127 bands
    if n_bands is not None and list(shape).count(n_bands) == 1:
        b = shape.index(n_bands)
    elif nd == 3:
        b = int(np.argmin(shape))
    else:
        b = 2  # default MATLAB layout H×W×B×T
    rest = [i for i in range(nd) if i != b]
    letters = [""] * nd
    letters[b] = "B"
    if nd == 3:
        letters[rest[0]], letters[rest[1]] = "H", "W"
    else:
        # time is the smallest of the remaining axes unless the layout is ambiguous
        t = min(rest, key=lambda i: shape[i]) if n_bands is not None else 3
        hw = [i for i in rest if i != t]
        letters[t] = "T"
        letters[hw[0]], letters[hw[1]] = "H", "W"
    return "".join(letters)


def _resolve_wavelengths(settings: DataSettings, mat: MatFile, variables, shape) -> Tuple[np.ndarray, List[str]]:
    notes: List[str] = []
    wl_spec = settings.wavelengths
    if isinstance(wl_spec, dict):
        start, step = float(wl_spec["start"]), float(wl_spec["step"])
        axes = settings.axes or detect_axes(shape, None)
        nb = shape[axes.index("B")]
        return start + step * np.arange(nb), notes
    if isinstance(wl_spec, (list, tuple)):
        return np.asarray(wl_spec, dtype=float), notes
    if isinstance(wl_spec, str) and wl_spec:
        if wl_spec.lower().endswith((".csv", ".txt")):
            return np.loadtxt(wl_spec, delimiter=None if wl_spec.endswith(".txt") else ",", ndmin=1).ravel(), notes
        return np.asarray(mat.read(wl_spec), dtype=float).ravel(), notes
    name, wl = find_wavelength_variable(mat, variables, shape)
    if wl is not None:
        notes.append(f"Wavelengths read from variable '{name}' ({wl[0]:g}–{wl[-1]:g} nm, {len(wl)} bands).")
        return wl, notes
    if len(HYPERPROBE_DEFAULT_WAVELENGTHS) in shape:
        notes.append("No wavelength variable found; assuming 385–1015 nm in 5 nm steps (127 bands).")
        return HYPERPROBE_DEFAULT_WAVELENGTHS.copy(), notes
    raise ValueError(
        "Could not find the wavelengths. Set them in the Data settings "
        "(first wavelength and step, or the name of the wavelength variable)."
    )


def _reference_array(spec: Optional[str], mat: MatFile, axes: str, n_bands: int) -> Optional[np.ndarray]:
    """White/dark reference as an array broadcastable to (H, W, B) — shape (h, w, B) with h, w ∈ {1, H}/{1, W}."""
    if not spec:
        return None
    if spec.lower().endswith(".mat") or is_envi(spec):
        with open_data_file(spec) as other:
            if isinstance(other, EnviFile):
                var = "cube"
            else:
                var = find_cube_variable(other.variables()) if len(other.variables()) > 1 else list(other.variables())[0]
            arr = np.asarray(other.read(var), dtype=np.float32)
    else:
        arr = np.asarray(mat.read(spec), dtype=np.float32)
    shape = arr.shape
    if n_bands not in shape:
        raise ValueError(f"Reference '{spec}' has shape {list(shape)}; none of its axes has {n_bands} bands.")
    b = len(shape) - 1 - list(reversed(shape)).index(n_bands) if shape.count(n_bands) > 1 else shape.index(n_bands)
    arr = np.moveaxis(arr, b, -1)
    if arr.ndim == 1:
        return arr.reshape(1, 1, -1)
    if arr.ndim == 2:  # (W, B) line or (1, B)
        return arr.reshape(1, arr.shape[0], -1)
    if arr.ndim == 3:
        # keep cube order H, W according to the main cube's axes
        return arr
    return arr.reshape(-1, arr.shape[-2], arr.shape[-1])[:1]


def _bin2d(block: np.ndarray, b: int) -> np.ndarray:
    """Average b×b pixels; block has axes (h, w, B) with h, w multiples of b."""
    if b == 1:
        return block
    h, w, nb = block.shape
    return block.reshape(h // b, b, w // b, b, nb).mean(axis=(1, 3))


# --------------------------------------------------------------------------- main entry
def inspect_file(path: str | Path) -> dict:
    """Quick look at a .mat or ENVI file for the GUI: variables, guessed cube, axes, wavelengths, references."""
    with open_data_file(path) as mat:
        variables = mat.variables()
        cube = find_cube_variable(variables)
        shape = variables[cube]
        name, wl = find_wavelength_variable(mat, variables, shape)
        n_bands = len(wl) if wl is not None else (127 if 127 in shape else None)
        axes = detect_axes(shape, n_bands)
    info = {"variables": variables, "cube": cube, "shape": shape, "axes": axes,
            "wavelength_variable": name, "n_bands": n_bands, "format": "ENVI" if is_envi(path) else "MATLAB",
            "white": None, "dark": None}
    if is_envi(path):
        info["white"], info["dark"] = find_envi_references(path)
    return info


def load_data(settings: DataSettings, progress: Progress = None) -> HyperspectralData:
    files = resolve_files(settings.file)
    binning = max(1, int(settings.binning))
    notes: List[str] = []

    if not settings.is_reflectance and not settings.white_reference and is_envi(files[0]):
        white, dark = find_envi_references(files[0])
        if white:
            settings.white_reference = white
            settings.dark_reference = settings.dark_reference or dark
            notes.append(f"White/dark references found next to the data: {Path(white).name}"
                         + (f", {Path(dark).name}." if dark else " (no dark reference)."))
    with open_data_file(files[0]) as mat:
        variables = mat.variables()
        var = settings.variable or find_cube_variable(variables)
        if var not in variables:
            raise ValueError(f"Variable '{var}' not found. Available: {', '.join(variables)}")
        shape = variables[var]
        wavelengths, wl_notes = _resolve_wavelengths(settings, mat, variables, shape)
        notes += wl_notes
        axes = (settings.axes or detect_axes(shape, len(wavelengths))).upper()
        if sorted(axes) not in (sorted("HWB"), sorted("HWBT")) or len(axes) != len(shape):
            raise ValueError(f"Axis order '{axes}' does not fit an array with shape {list(shape)}")
        if shape[axes.index("B")] != len(wavelengths):
            raise ValueError(
                f"The cube has {shape[axes.index('B')]} bands but {len(wavelengths)} wavelengths were given."
            )
        if not settings.axes:
            notes.append(f"Array '{var}' {list(shape)} read with axis order {axes} (H rows, W columns, B bands, T time).")
        H, W = shape[axes.index("H")], shape[axes.index("W")]
        nB = len(wavelengths)
        white = dark = None
        if not settings.is_reflectance:
            white = _reference_array(settings.white_reference, mat, axes, nB)
            dark = _reference_array(settings.dark_reference, mat, axes, nB)
            if white is None:
                raise ValueError("Raw data needs a white reference (variable name or .mat file).")

    # frames: either a T axis inside one file, or one file per frame
    n_t_in_file = shape[axes.index("T")] if "T" in axes else 1
    if len(files) > 1 and n_t_in_file > 1:
        raise ValueError("Several files each containing a time series are not supported; use one or the other.")
    total_frames = len(files) if len(files) > 1 else n_t_in_file
    frame_idx = list(range(total_frames))
    if settings.frames:
        fr = list(settings.frames) + [1] * (3 - len(settings.frames))
        frame_idx = list(range(total_frames))[slice(fr[0], fr[1], fr[2])]
        if not frame_idx:
            raise ValueError(f"No frames selected by frames={settings.frames} (the data has {total_frames}).")

    r0, r1, c0, c1 = settings.crop if settings.crop else (0, H, 0, W)
    r0, c0 = max(0, int(r0)), max(0, int(c0))
    r1, c1 = min(H, int(r1)), min(W, int(c1))
    r1 = r0 + (r1 - r0) // binning * binning
    c1 = c0 + (c1 - c0) // binning * binning
    if r1 <= r0 or c1 <= c0:
        raise ValueError("Crop region is empty (or smaller than one bin).")

    offset = settings.normalisation_offset
    if offset is None:
        offset = 0.0 if settings.is_reflectance else 0.1
        if offset:
            notes.append(f"Normalisation offset {offset:g} added after white/dark calibration (as in the HELICOID pipeline).")

    min_refl = settings.min_reflectance
    if min_refl is None:  # as in the research code: HELICOID (raw) 1e-2, Biopsy (reflectance) 1e-3
        min_refl = 1e-3 if settings.is_reflectance else 1e-2

    Hb, Wb = (r1 - r0) // binning, (c1 - c0) // binning
    out = np.empty((len(frame_idx), Hb, Wb, nB), dtype=np.float32)
    rows_per_block = binning * max(1, 64 // binning)  # about 64 rows (≈70 MB for 2048 columns × 127 bands)

    total_steps = len(frame_idx) * int(np.ceil((r1 - r0) / rows_per_block))
    step = 0
    for ti, t in enumerate(frame_idx):
        fpath = files[t] if len(files) > 1 else files[0]
        with open_data_file(fpath) as mat:
            for rb in range(r0, r1, rows_per_block):
                re_ = min(r1, rb + rows_per_block)
                index = []
                for a in axes:
                    index.append({"H": slice(rb, re_), "W": slice(c0, c1), "B": slice(None),
                                  "T": slice(t, t + 1) if len(files) == 1 else slice(None)}[a])
                block = mat.read(var, tuple(index)).astype(np.float32)
                order = [axes.index(a) for a in "HWB"] + ([axes.index("T")] if "T" in axes else [])
                block = np.transpose(block, order)
                if block.ndim == 4:
                    block = block[..., 0]
                if not settings.is_reflectance:
                    wsl = white[_ref_rows(white, rb, re_), _ref_cols(white, c0, c1)]
                    dsl = dark[_ref_rows(dark, rb, re_), _ref_cols(dark, c0, c1)] if dark is not None else 0.0
                    denom = wsl - dsl
                    denom = np.where(np.abs(denom) < 1e-12, np.nan, denom)
                    block = (block - dsl) / denom
                block = block + offset
                block = np.where(np.isfinite(block) & (block > 0), block, min_refl)
                out[ti, (rb - r0) // binning:(re_ - r0) // binning] = _bin2d(block, binning)
                step += 1
                if progress:
                    progress(step / total_steps, f"Reading data ({step}/{total_steps})")

    if binning > 1:
        notes.append(f"Binned {binning}×{binning}: working image is {Hb}×{Wb} pixels.")
    if settings.is_reflectance:
        typical = float(np.nanmedian(out[0, ::max(1, Hb // 64), ::max(1, Wb // 64)]))
        if typical > 2.0:
            notes.append(f"Warning: the values (median {typical:.4g}) look like raw counts, not reflectance. "
                         "Set 'Values are' to raw counts and give white/dark references.")
    return HyperspectralData(
        cube=out, wavelengths=np.asarray(wavelengths, float), original_shape=(H, W),
        crop=(r0, r1, c0, c1), binning=binning, files=files, variable=var, axes=axes,
        frame_indices=frame_idx, notes=notes,
    )


def _ref_rows(ref: np.ndarray, r0: int, r1: int):
    return slice(None) if ref.shape[0] == 1 else slice(r0, r1)


def _ref_cols(ref: np.ndarray, c0: int, c1: int):
    return slice(None) if ref.shape[1] == 1 else slice(c0, c1)


def read_spectrum_file(path: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Two-column text/CSV file (wavelength_nm, value); header lines are skipped."""
    rows = []
    with Path(path).open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s or s[0] in "#%":
                continue
            parts = s.replace(",", " ").replace(";", " ").split()
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue
    if len(rows) < 2:
        raise ValueError(f"{path}: expected two numeric columns (wavelength, value)")
    a = np.array(rows)
    order = np.argsort(a[:, 0])
    return a[order, 0], a[order, 1]
