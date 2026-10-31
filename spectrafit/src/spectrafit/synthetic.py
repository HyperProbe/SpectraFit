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


def make_cube(shape=(64, 64), wavelengths: Optional[np.ndarray] = None,
              chromophores: Sequence[str] = ("HbO2", "HHb", "oxCCO", "redCCO", "water", "fat"),
              mode: str = "free_scattering", absolute: bool = False, n_frames: int = 1,
              a0: float = 20.0, b0: float = 1.2, fixed_b: float = 1.2, scattering_scale: float = 10.0,
              lambda0: float = 500.0, pathlength: Optional[np.ndarray] = None, offset: float = 0.3,
              noise: float = 0.0, seed: int = 0, realistic: bool = False) -> Dict[str, np.ndarray]:
    """Forward-simulate reflectance with the same model the fit uses.

    Differential cubes have a reference pixel at (0, 0) whose spectrum equals the
    reference: every map is zero there (Δc = 0, a = a0, b = b0).
    Returns the cube (T, H, W, B) and the true maps.
    realistic=True keeps changes tissue-like (|ΔA| of a few tenths), as in the example dataset.
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
    base = np.array([c.baseline for c in chroms])
    for k, c in enumerate(chroms):
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
            truth["a"][:, 0, 0] = a0
            truth["b"][:, 0, 0] = b0
    if not absolute:
        for k in truth:
            if k not in ("a", "b"):
                truth[k][:, 0, 0] = 0.0
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
