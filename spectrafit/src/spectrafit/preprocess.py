"""From reflectance to the attenuation that is fitted: wavelength selection,
reference spectrum (or none), and pixel masks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .config import MaskSettings, ReferenceSettings, SpectralSettings
from .io import HyperspectralData, read_spectrum_file


def select_wavelengths(wavelengths: np.ndarray, spectral: SpectralSettings,
                       valid_range: Tuple[float, float]) -> Tuple[np.ndarray, List[str]]:
    """Indices of the bands used in the fit."""
    notes = []
    lo, hi = spectral.range
    vlo, vhi = valid_range
    if lo < vlo - 1e-9 or hi > vhi + 1e-9:
        notes.append(
            f"Fit range {lo:g}–{hi:g} nm limited to {max(lo, vlo):g}–{min(hi, vhi):g} nm, "
            "where all selected chromophore spectra are tabulated."
        )
        lo, hi = max(lo, vlo), min(hi, vhi)
    keep = (wavelengths >= lo - 1e-9) & (wavelengths <= hi + 1e-9)
    for a, b in spectral.exclude or []:
        keep &= ~((wavelengths >= min(a, b)) & (wavelengths <= max(a, b)))
    idx = np.flatnonzero(keep)
    if idx.size < 3:
        raise ValueError(f"Only {idx.size} wavelengths left in the fit range; widen the range.")
    return idx, notes


def polygon_mask(shape: Tuple[int, int], polygons_working: Sequence[np.ndarray]) -> np.ndarray:
    """Boolean (H, W) mask of pixel centres inside any of the polygons (vertices as (row, col))."""
    from matplotlib.path import Path as MplPath

    H, W = shape
    rr, cc = np.mgrid[0:H, 0:W]
    pts = np.column_stack([rr.ravel() + 0.5, cc.ravel() + 0.5])
    mask = np.zeros(H * W, bool)
    for poly in polygons_working:
        poly = np.asarray(poly, float)
        if len(poly) < 3:
            continue
        mask |= MplPath(poly).contains_points(pts)
    return mask.reshape(H, W)


def _to_working_poly(data: HyperspectralData, poly) -> np.ndarray:
    return np.array([data.to_working(r, c) for r, c in poly], float)


def build_mask(data: HyperspectralData, mask: MaskSettings, band_idx: np.ndarray) -> np.ndarray:
    """Pixels to fit: (T, H, W) bool."""
    T, H, W, _ = data.cube.shape
    m = np.ones((T, H, W), bool)
    if mask.polygons:
        pm = polygon_mask((H, W), [_to_working_poly(data, p) for p in mask.polygons])
        m &= pm[None]
    if mask.min_reflectance is not None or mask.max_reflectance is not None:
        mean_r = data.cube[..., band_idx].mean(axis=-1)
        if mask.min_reflectance is not None:
            m &= mean_r >= mask.min_reflectance
        if mask.max_reflectance is not None:
            m &= mean_r <= mask.max_reflectance
    return m


@dataclass
class Reference:
    type: str
    spectrum: Optional[np.ndarray]
    """(B,) on the data wavelengths, or None for 'none' and 'frame'."""
    frames: Optional[Tuple[int, int]] = None
    pixel_working: Optional[Tuple[int, int]] = None
    roi_working: Optional[np.ndarray] = None
    description: str = ""
    file_range: Optional[Tuple[float, float]] = None


def build_reference(data: HyperspectralData, ref: ReferenceSettings) -> Reference:
    T, H, W, B = data.cube.shape
    t = ref.type
    frame = 0
    if ref.frame:
        frame = int(ref.frame[0])
        if not 0 <= frame < T:
            raise ValueError(f"Reference frame {frame} does not exist (the data has {T} frames).")
    if t == "none":
        return Reference("none", None, description="No reference: absolute attenuation −ln R is fitted.")
    if t == "pixel":
        if not ref.pixel:
            raise ValueError("Choose a reference pixel (click on the image) or another reference type.")
        r, c = data.to_working(*ref.pixel)
        r, c = int(np.floor(r)), int(np.floor(c))
        if not (0 <= r < H and 0 <= c < W):
            raise ValueError(f"Reference pixel {ref.pixel} lies outside the (cropped) image.")
        return Reference("pixel", data.cube[frame, r, c, :].astype(float), pixel_working=(r, c),
                         description=f"Reference pixel (row {ref.pixel[0]}, col {ref.pixel[1]}), frame {frame}.")
    if t == "roi":
        if not ref.roi or len(ref.roi) < 3:
            raise ValueError("Draw a reference region (at least 3 corners) or choose another reference type.")
        poly = _to_working_poly(data, ref.roi)
        m = polygon_mask((H, W), [poly])
        if not m.any():
            raise ValueError("The reference region contains no pixels of the (cropped) image.")
        spec = data.cube[frame][m].astype(float).mean(axis=0)
        return Reference("roi", spec, roi_working=poly,
                         description=f"Mean of {int(m.sum())} pixels in the reference region, frame {frame}.")
    if t == "file":
        if not ref.file:
            raise ValueError("Select the reference spectrum file.")
        wl, val = read_spectrum_file(ref.file)
        if np.any(val <= 0):
            raise ValueError(f"The reference spectrum in {ref.file} must be positive (it is a reflectance).")
        return Reference("file", np.interp(data.wavelengths, wl, val), file_range=(float(wl[0]), float(wl[-1])),
                         description=f"Reference spectrum from {ref.file}.")
    if t == "frame":
        if T < 2:
            raise ValueError("Reference type 'frame' needs a time series (more than one frame).")
        fr = ref.frame or [0]
        f0, f1 = (int(fr[0]), int(fr[0]) + 1) if len(fr) == 1 else (int(fr[0]), int(fr[1]))
        if not (0 <= f0 < f1 <= T):
            raise ValueError(f"Reference frames {fr} are outside 0–{T - 1}.")
        return Reference("frame", None, frames=(f0, f1),
                         description=f"Each pixel referenced to itself, mean of frames {f0}–{f1 - 1}.")
    raise ValueError(f"Unknown reference type '{t}'")



def attenuation(data: HyperspectralData, reference: Reference, band_idx: np.ndarray,
                frame_slice: slice = slice(None)) -> np.ndarray:
    """−ln(R / R_ref) (or −ln R without reference) on the selected bands, shape (T', H, W, W_fit), float32."""
    R = data.cube[frame_slice][..., band_idx]
    if reference.type == "none":
        return -np.log(R)
    if reference.type == "frame":
        f0, f1 = reference.frames
        Rref = data.cube[f0:f1][..., band_idx].mean(axis=0)
        return -np.log(R / Rref[None])
    if reference.type == "file":
        lo, hi = data.wavelengths[band_idx].min(), data.wavelengths[band_idx].max()
        flo, fhi = reference.file_range
        if lo < flo - 1e-9 or hi > fhi + 1e-9:
            raise ValueError(f"The reference spectrum file covers {flo:g}–{fhi:g} nm, "
                             f"but the fit range is {lo:g}–{hi:g} nm.")
    return -np.log(R / reference.spectrum[band_idx].astype(np.float32))
