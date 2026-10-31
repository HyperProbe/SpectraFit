"""Chromophore spectra: loading, unit conversion and the built-in library.

Every chromophore is a small text file with a few ``# key: value`` header lines
followed by two columns (wavelength in nm, value)::

    # name: HbO2
    # label: Oxyhaemoglobin
    # unit: eps10_per_M_cm
    # concentration_unit: mM
    # baseline: 0.0646
    wavelength_nm,value
    400,266232
    ...

Supported units (``unit:``), and how they are converted to the natural-log
absorption coefficient per unit concentration used in the fit:

=====================  =====================================  ==================  ==============
unit                   meaning                                factor              concentration
=====================  =====================================  ==================  ==============
eps10_per_M_cm         molar extinction, base 10, cm^-1 M^-1  ln(10) / 1000       mM
eps10_per_mM_cm        molar extinction, base 10, cm^-1 mM^-1 ln(10)              mM
mua_per_mM_cm          absorption (natural log), cm^-1 mM^-1  1                   mM
mua_per_cm             absorption of the pure substance       1                   volume fraction
=====================  =====================================  ==================  ==============

Only the header line ``unit`` is required; ``name`` defaults to the file name,
``baseline`` (the typical tissue concentration, used by the pathlength models)
defaults to 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from importlib import resources
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

LN10 = math.log(10.0)

UNITS = {
    "eps10_per_M_cm": (LN10 / 1000.0, "mM"),
    "eps10_per_mM_cm": (LN10, "mM"),
    "mua_per_mM_cm": (1.0, "mM"),
    "mua_per_cm": (1.0, "fraction"),
}

UNIT_DESCRIPTIONS = {
    "eps10_per_M_cm": "Molar extinction coefficient, base 10 [cm⁻¹ M⁻¹] (e.g. Prahl tables)",
    "eps10_per_mM_cm": "Molar extinction coefficient, base 10 [cm⁻¹ mM⁻¹]",
    "mua_per_mM_cm": "Specific absorption coefficient, natural log [cm⁻¹ mM⁻¹]",
    "mua_per_cm": "Absorption coefficient of the pure substance [cm⁻¹] (concentration = volume fraction)",
}

DEFAULT_SET = ["HbO2", "HHb", "oxCCO", "redCCO", "water", "fat"]


@dataclass
class Chromophore:
    """One absorber with its spectrum on its own wavelength grid."""

    name: str
    wavelengths: np.ndarray
    values: np.ndarray
    unit: str
    label: str = ""
    baseline: float = 0.0
    source: str = ""
    path: Optional[str] = None
    concentration_unit: str = field(default="")

    def __post_init__(self) -> None:
        if self.unit not in UNITS:
            raise ValueError(
                f"Chromophore '{self.name}': unknown unit '{self.unit}'. "
                f"Use one of: {', '.join(UNITS)}"
            )
        self.wavelengths = np.asarray(self.wavelengths, dtype=float)
        self.values = np.asarray(self.values, dtype=float)
        if self.wavelengths.ndim != 1 or self.wavelengths.shape != self.values.shape:
            raise ValueError(f"Chromophore '{self.name}': wavelengths and values must be 1D and equal length")
        order = np.argsort(self.wavelengths)
        self.wavelengths = self.wavelengths[order]
        self.values = self.values[order]
        if not self.concentration_unit:
            self.concentration_unit = UNITS[self.unit][1]
        if not self.label:
            self.label = self.name

    @property
    def range(self) -> tuple:
        return float(self.wavelengths[0]), float(self.wavelengths[-1])

    def absorption(self, wavelengths: Sequence[float]) -> np.ndarray:
        """Natural-log absorption per unit concentration on the requested grid.

        Values outside the tabulated range raise an error rather than being
        extrapolated silently.
        """
        wl = np.asarray(wavelengths, dtype=float)
        lo, hi = self.range
        if wl.min() < lo - 1e-9 or wl.max() > hi + 1e-9:
            raise ValueError(
                f"Chromophore '{self.name}' is only tabulated from {lo:g} to {hi:g} nm, "
                f"but {wl.min():g}–{wl.max():g} nm was requested."
            )
        factor = UNITS[self.unit][0]
        out = np.interp(wl, self.wavelengths, self.values) * factor
        out[out < 0] = 0.0
        return out


def read_chromophore_file(path: str | Path, **overrides) -> Chromophore:
    """Read a chromophore file (header lines + two columns, comma/space/tab separated)."""
    path = Path(path)
    meta: Dict[str, str] = {}
    rows: List[tuple] = []
    with path.open("r", encoding="utf-8-sig") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#") or s.startswith("%"):
                body = s.lstrip("#%").strip()
                if ":" in body:
                    k, v = body.split(":", 1)
                    meta[k.strip().lower()] = v.strip()
                continue
            parts = s.replace(",", " ").replace(";", " ").split()
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except (ValueError, IndexError):
                continue  # column header such as "wavelength_nm,value"
    if not rows:
        raise ValueError(f"No numeric data found in {path}")
    data = np.array(rows)
    meta.update({k: str(v) for k, v in overrides.items() if v is not None})
    if "unit" not in meta:
        raise ValueError(
            f"{path.name}: missing '# unit: ...' header line. Use one of: {', '.join(UNITS)}"
        )
    return Chromophore(
        name=meta.get("name", path.stem),
        label=meta.get("label", ""),
        wavelengths=data[:, 0],
        values=data[:, 1],
        unit=meta["unit"],
        baseline=float(meta.get("baseline", 0.0) or 0.0),
        source=meta.get("source", ""),
        concentration_unit=meta.get("concentration_unit", ""),
        path=str(path),
    )


def write_chromophore_file(chrom: Chromophore, path: str | Path) -> Path:
    """Save a chromophore in the library format (used by 'Add chromophore' in the GUI)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        fh.write(f"# name: {chrom.name}\n# label: {chrom.label}\n# unit: {chrom.unit}\n")
        fh.write(f"# concentration_unit: {chrom.concentration_unit}\n# baseline: {chrom.baseline}\n")
        if chrom.source:
            fh.write(f"# source: {chrom.source}\n")
        fh.write("wavelength_nm,value\n")
        for w, v in zip(chrom.wavelengths, chrom.values):
            fh.write(f"{w:g},{v:.8g}\n")
    return path


def builtin_dir() -> Path:
    return Path(str(resources.files("spectrafit") / "data" / "chromophores"))


def user_dir() -> Path:
    """Folder where chromophores added through the GUI are stored."""
    return Path.home() / ".spectrafit" / "chromophores"


class ChromophoreLibrary:
    """All chromophores known to the program: built-in ones, the user's folder, and extra files."""

    def __init__(self, extra_files: Iterable[str | Path] = (), include_user_dir: bool = True):
        self.items: Dict[str, Chromophore] = {}
        for f in sorted(builtin_dir().glob("*.csv")):
            self.add(read_chromophore_file(f))
        if include_user_dir and user_dir().exists():
            for f in sorted(user_dir().glob("*.csv")) + sorted(user_dir().glob("*.txt")):
                self.add(read_chromophore_file(f))
        for f in extra_files:
            self.add(read_chromophore_file(f))

    def add(self, chrom: Chromophore) -> None:
        self.items[chrom.name] = chrom  # later sources override earlier ones

    def names(self) -> List[str]:
        builtin_order = DEFAULT_SET + ["oxCytb", "redCytb", "oxCytc", "redCytc"]
        known = [n for n in builtin_order if n in self.items]
        return known + sorted(n for n in self.items if n not in known)

    def get(self, name: str) -> Chromophore:
        if name not in self.items:
            raise KeyError(f"Unknown chromophore '{name}'. Available: {', '.join(self.names())}")
        return self.items[name]

    def __contains__(self, name: str) -> bool:
        return name in self.items


def absorption_matrix(chromophores: Sequence[Chromophore], wavelengths: Sequence[float]) -> np.ndarray:
    """Matrix M (n_wavelengths × n_chromophores) of natural-log absorption per unit concentration."""
    return np.column_stack([c.absorption(wavelengths) for c in chromophores])


def common_range(chromophores: Sequence[Chromophore]) -> tuple:
    lo = max(c.range[0] for c in chromophores)
    hi = min(c.range[1] for c in chromophores)
    return lo, hi


def collinearity_report(M: np.ndarray, names: Sequence[str]) -> dict:
    """How well the selected spectra can be told apart on this wavelength grid.

    Returns the condition number of the column-normalised matrix and the most
    similar pair (cosine similarity). A condition number above ~1e3 or a
    similarity above ~0.99 means the fit cannot reliably separate them.
    """
    norms = np.linalg.norm(M, axis=0)
    norms[norms == 0] = 1.0
    Mn = M / norms
    cond = float(np.linalg.cond(Mn)) if M.shape[1] > 0 else 1.0
    worst = (None, None, 0.0)
    for i in range(M.shape[1]):
        for j in range(i + 1, M.shape[1]):
            s = float(abs(Mn[:, i] @ Mn[:, j]))
            if s > worst[2]:
                worst = (names[i], names[j], s)
    level = "ok"
    if cond > 1e4 or worst[2] > 0.995:
        level = "bad"
    elif cond > 1e3 or worst[2] > 0.98:
        level = "warning"
    return {"condition_number": cond, "most_similar": worst, "level": level}


def detection_limits(X: np.ndarray, noise: float = 1e-3) -> np.ndarray:
    """Standard error of each fitted quantity for white noise of size `noise` in the attenuation.

    X is the design matrix (wavelengths × fitted quantities, e.g. the absorption spectra
    plus scattering columns). This is the smallest change the data can resolve
    (1 standard deviation); compare it with the concentration changes you expect.
    """
    G = X.T @ X
    try:
        cov = np.linalg.inv(G)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(G)
    return noise * np.sqrt(np.clip(np.diag(cov), 0, None))
