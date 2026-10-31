"""Optical pathlength PL(λ) in cm.

Tissue-level models (one curve for the whole image) need the baseline tissue
absorption μa(λ) = M · c_baseline and reduced scattering μs'(λ) = a·(λ/500)^-b.
Per-pixel models take μa and μs' of every pixel from a first fit with unit
pathlength, as in the research code. Because μa is computed from whatever
chromophores are selected (each with its own baseline concentration), all
modes work with any chromophore set, including new ones.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import numpy as np

from . import _pathlength_formulas as F
from .config import PathlengthSettings
from .io import read_spectrum_file

JACQUES_TISSUES = tuple(F.JACQUES_M_PARAMS)
MIN_MU = 1e-9


def carp_boundary_factor(n: float) -> float:
    """Boundary factor k(n) of the δ-P1 (CARP) approximation."""
    return -0.13755 * n**3 + 4.3390 * n**2 - 4.90466 * n + 1.6896


def tissue_mu_a(M: np.ndarray, baselines: np.ndarray) -> np.ndarray:
    mu_a = M @ baselines
    if np.any(~np.isfinite(mu_a)) or np.any(mu_a <= 0):
        bad = float(np.min(mu_a))
        raise ValueError(
            "The baseline tissue absorption μa = M·c_baseline must be positive at every wavelength "
            f"(minimum is {bad:.3g} cm⁻¹). Check the baseline concentrations of the selected chromophores."
        )
    return mu_a


def mu_s_reduced(wavelengths: np.ndarray, a, b, lambda0: float = 500.0) -> np.ndarray:
    wl = np.asarray(wavelengths, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    return a[..., None] * (wl / lambda0) ** (-b[..., None]) if a.ndim else a * (wl / lambda0) ** (-b)


def _dA_dmua(model: str, mu_a, mu_s, settings: PathlengthSettings) -> np.ndarray:
    mu_a = np.clip(np.asarray(mu_a, float), MIN_MU, None)
    mu_s = np.clip(np.asarray(mu_s, float), MIN_MU, None)
    with np.errstate(all="ignore"):
        if model == "delta_p1":
            t = settings.tissue
            pl = F.carp_dA_dmua(mu_a, mu_s, t.g, carp_boundary_factor(t.n))
        else:
            key = settings.jacques_tissue
            if key not in F.JACQUES_M_PARAMS:
                raise ValueError(f"Unknown Jacques tissue '{key}'. Choose one of: {', '.join(JACQUES_TISSUES)}")
            pl = F.jacques_dA_dmua(mu_a, mu_s, *F.JACQUES_M_PARAMS[key])
    return np.asarray(pl, float)


def gray_matter_table() -> tuple:
    path = Path(str(resources.files("spectrafit") / "data" / "pathlengths" / "gray_matter.csv"))
    return read_spectrum_file(path)


def _from_table(wl_tab, pl_tab, wavelengths, what: str) -> np.ndarray:
    wl = np.asarray(wavelengths, float)
    if wl.min() < wl_tab[0] - 1e-9 or wl.max() > wl_tab[-1] + 1e-9:
        raise ValueError(
            f"The {what} covers {wl_tab[0]:g}–{wl_tab[-1]:g} nm; the fit range "
            f"{wl.min():g}–{wl.max():g} nm is outside it."
        )
    return np.interp(wl, wl_tab, pl_tab)


def global_pathlength(settings: PathlengthSettings, wavelengths: np.ndarray, M: np.ndarray,
                      baselines: np.ndarray) -> np.ndarray:
    """PL(λ) shared by all pixels. For per-pixel modes this is the tissue-level first guess."""
    mode = settings.mode
    wl = np.asarray(wavelengths, float)
    if mode == "constant" or mode.endswith("_pixelwise"):
        if mode.endswith("_pixelwise"):
            return np.ones_like(wl)  # first pass uses unit pathlength, as in the research code
        if settings.constant <= 0:
            raise ValueError("The constant pathlength must be positive.")
        return np.full_like(wl, float(settings.constant))
    if mode == "gray_matter_table":
        return _from_table(*gray_matter_table(), wl, "gray-matter pathlength table")
    if mode == "table_file":
        return _from_table(*read_spectrum_file(settings.file), wl, f"pathlength file {settings.file}")
    if mode in ("delta_p1", "jacques"):
        mu_a = tissue_mu_a(M, baselines)
        mu_s = mu_s_reduced(wl, settings.tissue.a, settings.tissue.b)
        pl = _dA_dmua(mode, mu_a, mu_s, settings)
        _check_pl(pl, mode)
        return pl
    raise ValueError(f"Unknown pathlength mode '{mode}'")


def pixelwise_pathlength(settings: PathlengthSettings, wavelengths: np.ndarray, M: np.ndarray,
                         mu_a: np.ndarray, mu_s: np.ndarray) -> np.ndarray:
    """Per-pixel PL (N × W) from per-pixel μa and μs' (both N × W, cm⁻¹)."""
    model = "delta_p1" if settings.mode == "delta_p1_pixelwise" else "jacques"
    pl = _dA_dmua(model, mu_a, mu_s, settings)
    bad = ~np.isfinite(pl) | (pl <= 0)
    if bad.any():
        # fall back to the tissue-level curve where the model is outside its validity range
        mu_a_t = np.clip(np.nanmedian(mu_a, axis=0), MIN_MU, None)
        mu_s_t = np.clip(np.nanmedian(mu_s, axis=0), MIN_MU, None)
        fallback = _dA_dmua(model, mu_a_t, mu_s_t, settings)
        pl = np.where(bad, fallback[None, :], pl)
    return pl


def _check_pl(pl: np.ndarray, mode: str) -> None:
    if np.any(~np.isfinite(pl)) or np.any(pl <= 0):
        raise ValueError(
            f"The {mode} pathlength is not positive everywhere in this wavelength range; "
            "the model is outside its validity range (check tissue parameters and baselines)."
        )
