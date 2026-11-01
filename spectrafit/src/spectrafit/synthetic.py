"""Synthetic hyperspectral cubes with known concentrations, for testing and for trying the program."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .chromophores import ChromophoreLibrary
from .io import HYPERPROBE_DEFAULT_WAVELENGTHS


def _blob(shape, rng, n=3) -> np.ndarray:
    """Smooth random pattern in [-1, 1]."""
    H, W = shape
    rr, cc = np.mgrid[0:H, 0:W]
    out = np.zeros(shape)
    for _ in range(n):
        r0, c0 = rng.uniform(0, H), rng.uniform(0, W)
        s = rng.uniform(0.15, 0.4) * min(H, W)
        out += rng.uniform(-1, 1) * np.exp(-((rr - r0) ** 2 + (cc - c0) ** 2) / (2 * s * s))
    return out / (np.abs(out).max() + 1e-12)


def _vessel_tree(shape, rng, n_trunks=4):
    """Branching vessels as smooth random curves.

    Returns, per pixel, how much of it is vessel (0..1) and whether that vessel is an artery.
    """
    H, W = shape
    rr, cc = np.mgrid[0:H, 0:W].astype(float)
    fill = np.zeros(shape)
    artery = np.zeros(shape, bool)

    def draw(r, c, angle, radius, length, is_artery, depth):
        nonlocal fill, artery
        step = 1.5
        for i in range(int(length / step)):
            angle += rng.normal(0, 0.12)
            r += step * np.sin(angle)
            c += step * np.cos(angle)
            if not (-10 < r < H + 10 and -10 < c < W + 10):
                return
            d = np.hypot(rr - r, cc - c)
            tube = 1 / (1 + np.exp((d - radius) / 0.6))  # soft-edged disc
            new = tube > fill
            fill = np.maximum(fill, tube)
            artery = np.where(new & (tube > 0.5), is_artery, artery)
            if depth < 2 and rng.random() < 0.025:  # side branch
                draw(r, c, angle + rng.choice([-1, 1]) * rng.uniform(0.6, 1.2), radius * 0.6,
                     length * 0.45, is_artery, depth + 1)
            radius = max(0.8, radius * 0.995)

    for k in range(n_trunks):
        side = rng.integers(4)  # start on an edge and run across the image
        t = rng.uniform(0.15, 0.85)
        r, c, angle = [(0, t * W, np.pi / 2), (H - 1, t * W, -np.pi / 2),
                       (t * H, 0, 0.0), (t * H, W - 1, np.pi)][side]
        draw(r, c, angle + rng.normal(0, 0.3), rng.uniform(2.0, 3.5) * min(H, W) / 160,
             1.3 * max(H, W), k % 2 == 0, 0)
    return fill, artery


def _tissue_fields(shape, rng, names):
    """Absolute tissue maps (mM or volume fraction): vessels on a capillary background."""
    fill, artery = _vessel_tree(shape, rng)
    # blood volume fraction: capillary background about 3 %, vessels mostly blood
    background = 0.03 + 0.003 * _blob(shape, rng, n=4)
    blood = background + (0.6 - background) * fill
    # oxygen saturation: arteries 97 %, veins 60 %, tissue 70 % with one less-oxygenated patch
    patch = np.clip(_blob(shape, rng, n=1), 0, 1)
    s_tissue = 0.72 - 0.08 * patch
    sat = np.where(fill > 0.05, np.where(artery, 0.97, 0.60), s_tissue)
    sat = s_tissue + (sat - s_tissue) * fill
    hbt = 2.3 * blood * 0.004  # mM of haemoglobin seen at the surface (scaled so vessels look dark, not black)
    out = {"HbO2": hbt * sat, "HHb": hbt * (1 - sat)}
    smooth = {"oxCCO": (0.0064, 0.0008), "redCCO": (0.0016, 0.0003), "water": (0.73, 0.01), "fat": (0.1, 0.01)}
    for n in names:
        if n not in out:
            base, amp = smooth.get(n, (0.0, 0.0005))
            out[n] = base + amp * _blob(shape, rng)
    return out, fill


def make_cube(shape=(64, 64), wavelengths: Optional[np.ndarray] = None,
              chromophores: Sequence[str] = ("HbO2", "HHb", "oxCCO", "redCCO", "water", "fat"),
              mode: str = "free_scattering", absolute: bool = False, n_frames: int = 1,
              a0: float = 20.0, b0: float = 1.2, fixed_b: float = 1.2, scattering_scale: float = 10.0,
              lambda0: float = 500.0, pathlength: Optional[np.ndarray] = None, offset: float = 0.3,
              noise: float = 0.0, seed: int = 0, realistic: bool = False,
              pattern: str = "blobs") -> Dict[str, np.ndarray]:
    """Forward-simulate reflectance with the same model the fit uses.

    Differential cubes have a reference pixel at (0, 0) whose spectrum equals the
    reference: every map is zero there (Δc = 0, a = a0, b = b0).
    Returns the cube (T, H, W, B) and the true maps.
    realistic=True keeps changes tissue-like (|ΔA| of a few tenths), as in the example dataset.
    pattern="tissue" (differential only) draws branching vessels on capillary tissue instead of
    smooth blobs; the reference pixel is then the least-perfused tissue pixel (truth["reference_pixel"]),
    so blood only adds absorption and the reflectance stays below 1.
    """
    rng = np.random.default_rng(seed)
    wl = HYPERPROBE_DEFAULT_WAVELENGTHS.copy() if wavelengths is None else np.asarray(wavelengths, float)
    lib = ChromophoreLibrary(include_user_dir=False)
    chroms = [lib.get(n) for n in chromophores]
    # tabulated range only covers 400–1000 nm: hold the end values outside it
    M = np.column_stack([c.absorption(np.clip(wl, *c.range)) for c in chroms])
    PL = np.ones_like(wl) if pathlength is None else np.asarray(pathlength, float)
    H, W = shape
    T = n_frames
    truth: Dict[str, np.ndarray] = {}
    scale = {"mM": 0.004, "fraction": 0.01} if realistic else {"mM": 0.02, "fraction": 0.05}
    amp_a, amp_b, amp_da = (0.05, 0.004, 0.05) if realistic else (2.0, 0.15, 1.5)
    if pattern == "tissue":  # gentle scattering changes: vessels, not scattering, should dominate the image
        amp_a, amp_b, amp_da = 0.01, 0.001, 0.01
    base = np.array([c.baseline for c in chroms])
    ref = (0, 0)
    if pattern == "tissue":
        if absolute:
            raise ValueError("pattern='tissue' is only available for differential (reference) cubes")
        fields, fill = _tissue_fields(shape, rng, [c.name for c in chroms])
        hbt = fields.get("HbO2", 0) + fields.get("HHb", 0)
        quiet = np.where(fill < 0.01, hbt, np.inf)  # least blood, away from vessels
        H, W = shape
        margin = np.full(shape, np.inf)
        margin[H // 8:H - H // 8, W // 8:W - W // 8] = 0  # not at the very edge
        ref = tuple(int(i) for i in np.unravel_index(np.argmin(quiet + margin), shape))
        truth["reference_pixel"] = np.array(ref)
    for k, c in enumerate(chroms):
        if pattern == "tissue":
            v = fields[c.name] - fields[c.name][ref]
            truth[c.name] = np.stack([v * (1 + 0.1 * t) for t in range(T)])
            continue
        pat = np.stack([_blob(shape, rng) * (1 + 0.5 * t) for t in range(T)])
        if absolute:
            v = base[k] + 0.5 * base[k] * pat if base[k] > 0 else 0.002 * (pat + 1)
        else:
            v = scale.get(c.concentration_unit, 0.02) * pat
        truth[c.name] = v
    r = wl / lambda0
    if mode == "fixed_b":
        if absolute:  # absolute attenuation must stay realistic (A ~ 1): small scattering amplitude
            truth["a"] = 0.06 + 0.02 * np.stack([_blob(shape, rng) for _ in range(T)])
        else:
            truth["delta_a"] = amp_da * np.stack([_blob(shape, rng) for _ in range(T)])
    elif mode == "free_scattering":
        if absolute:
            truth["a"] = 0.06 + 0.02 * np.stack([_blob(shape, rng) for _ in range(T)])
        else:
            truth["a"] = a0 + amp_a * np.stack([_blob(shape, rng) for _ in range(T)])
        truth["b"] = b0 + amp_b * np.stack([_blob(shape, rng) for _ in range(T)])
        if not absolute:
            truth["a"][(slice(None),) + ref] = a0
            truth["b"][(slice(None),) + ref] = b0
    if not absolute:
        for k in truth:
            if k not in ("a", "b", "reference_pixel"):
                truth[k][(slice(None),) + ref] = 0.0
    C = np.stack([truth[c.name] for c in chroms], axis=-1)  # T,H,W,K
    absorb = C @ M.T
    if mode == "fixed_b":
        key = "a" if absolute else "delta_a"
        scat = scattering_scale * truth[key][..., None] * r ** (-fixed_b)
    elif mode == "free_scattering":
        scat = scattering_scale * truth["a"][..., None] * r[None, None, None, :] ** (-truth["b"][..., None])
        if not absolute:
            scat = scat - scattering_scale * a0 * r ** (-b0)
    else:
        scat = 0.0
    A = PL * (absorb + scat)
    if absolute:
        truth["offset"] = offset + 0.05 * np.stack([_blob(shape, rng) for _ in range(T)])
        A = A + truth["offset"][..., None]
        R = np.exp(-A)
    else:
        if pattern == "tissue":
            # diffuse reflectance of tissue: R ≈ exp(−k·√(μa/μs')), so blood makes blue/green dark and red bright
            mua = np.maximum(M @ base, 0)
            mus = 20.0 * (wl / 500.0) ** -1.2
            R_ref = 0.6 * np.exp(-1.3 * np.sqrt(mua / mus))
        else:
            # reference reflectance: a plausible tissue-like spectrum
            R_ref = 0.4 * np.exp(-0.02 * (M @ base))  # smooth, tissue-like, never near zero
        R = R_ref * np.exp(-A)
        truth["reference_spectrum"] = R_ref
    if noise:
        R = R * (1 + noise * rng.standard_normal(R.shape))
    truth["wavelengths"] = wl
    truth["cube"] = R.astype(np.float32)
    return truth


def write_mat(path: str | Path, cube_thwb: np.ndarray, wavelengths: np.ndarray, variable: str = "Ref_hyper",
              v73: bool = False, extra: Optional[dict] = None) -> Path:
    """Save as a MATLAB file with the cube in H×W×B (or H×W×B×T) order."""
    path = Path(path)
    arr = cube_thwb[0] if cube_thwb.shape[0] == 1 else np.moveaxis(cube_thwb, 0, -1)
    data = {variable: arr, "wavelengths": np.asarray(wavelengths, float).reshape(1, -1), **(extra or {})}
    if v73:
        import h5py

        with h5py.File(path, "w", userblock_size=512) as f:
            for k, v in data.items():
                f.create_dataset(k, data=np.asarray(v).transpose())
        header = b"MATLAB 7.3 MAT-file, created by spectrafit".ljust(116) + b"\x00" * 8 + b"\x00\x02" + b"IM"
        with open(path, "r+b") as fh:
            fh.write(header.ljust(512, b"\x00"))
    else:
        from scipy.io import savemat

        savemat(path, data, do_compression=True)
    return path


def write_envi(path: str | Path, cube_hwb: np.ndarray, wavelengths: Optional[np.ndarray] = None,
               interleave: str = "bil", dtype: str = "<u2", mac_line_endings: bool = False) -> Path:
    """Save an (H, W, B) array as ENVI (binary file `path` + `path`.hdr), e.g. to mimic HELICOID data."""
    path = Path(path)
    H, W, B = cube_hwb.shape
    order = {"bsq": (2, 0, 1), "bil": (0, 2, 1), "bip": (0, 1, 2)}[interleave]
    np.ascontiguousarray(np.transpose(cube_hwb, order)).astype(dtype).tofile(path)
    codes = {"u1": 1, "i2": 2, "i4": 3, "f4": 4, "f8": 5, "u2": 12, "u4": 13}
    lines = ["ENVI", "description = {synthetic}", f"samples = {W}", f"lines = {H}", f"bands = {B}",
             "header offset = 0", "file type = ENVI Standard", f"data type = {codes[np.dtype(dtype).str[1:]]}",
             f"interleave = {interleave}", f"byte order = {0 if np.dtype(dtype).byteorder in '<=|' else 1}"]
    if wavelengths is not None:
        lines += ["wavelength units = nm", "wavelength = {"] + \
                 [("," if i else "") + f"{w:g}" for i, w in enumerate(wavelengths)] + ["}"]
    sep = "\r" if mac_line_endings else "\n"
    with open(path.with_name(path.name + ".hdr"), "w", newline="") as fh:
        fh.write(sep.join(lines) + sep)
    return path
