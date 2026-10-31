"""Run settings. One ``Config`` describes a complete analysis and is saved as YAML
next to every result, so any figure can be reproduced or the same settings
reused on other files.

All pixel coordinates in the config are (row, column) in the ORIGINAL image,
before cropping and binning, so a config stays valid when binning changes.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MODES = ("no_scattering", "fixed_b", "free_scattering")
MODE_LABELS = {
    "no_scattering": "Linear, no scattering",
    "fixed_b": "Linear, scattering with fixed power b",
    "free_scattering": "Scattering amplitude a and power b free",
}

PATHLENGTH_MODES = (
    "constant",
    "gray_matter_table",
    "table_file",
    "delta_p1",
    "jacques",
    "delta_p1_pixelwise",
    "jacques_pixelwise",
)
PATHLENGTH_LABELS = {
    "constant": "Constant (e.g. 1 = unitary pathlength)",
    "gray_matter_table": "Gray-matter table (wavelength dependent)",
    "table_file": "Table from my own file",
    "delta_p1": "δ-P1 model, tissue baseline",
    "jacques": "Jacques model, tissue baseline",
    "delta_p1_pixelwise": "δ-P1 model, per pixel (two passes)",
    "jacques_pixelwise": "Jacques model, per pixel (two passes)",
}
# Names used by the research code (scripts/plot_helicoid_molecular_maps.py --pathlength-mode)
PATHLENGTH_ALIASES = {
    "ones": "constant",
    "pathlength_from_wavelength": "gray_matter_table",
    "gray_matter_delta_p1": "delta_p1",
    "gray_matter_jacques": "jacques",
    "pixelwise_delta_p1_from_unitary": "delta_p1_pixelwise",
    "pixelwise_jacques_from_unitary": "jacques_pixelwise",
}
REFERENCE_TYPES = ("none", "pixel", "roi", "file", "frame")
REFERENCE_LABELS = {
    "none": "No reference (absolute attenuation)",
    "pixel": "Single pixel",
    "roi": "Mean of a region",
    "file": "Spectrum from a file",
    "frame": "Earlier frame of a time series (each pixel vs. itself)",
}


@dataclass
class DataSettings:
    file: str = ""
    """.mat file; ENVI file (data file or its .hdr, e.g. HELICOID 'raw'); a folder or glob pattern of .mat files
    (one per time frame); or a folder holding one ENVI cube."""
    variable: Optional[str] = None
    """Name of the cube inside the .mat file. Empty = detect automatically (prefers 'Ref_hyper')."""
    wavelengths: Optional[Any] = None
    """Empty = look for a wavelength variable in the file; else a variable name,
    a list of values, or {start: 385, step: 5}."""
    axes: Optional[str] = None
    """Axis order in MATLAB terms: letters H, W, B (bands) and T (time), e.g. 'HWB' or 'HWBT'.
    Empty = detect automatically."""
    is_reflectance: bool = True
    """True: values are already reflectance. False: raw counts, calibrate with white/dark."""
    white_reference: Optional[str] = None
    """Variable name in the same file, or another .mat or ENVI file (raw data only). For ENVI data, empty =
    use whiteReference/darkReference files next to the cube when they exist (HELICOID layout)."""
    dark_reference: Optional[str] = None
    normalisation_offset: Optional[float] = None
    """Added to the reflectance before taking the log. Empty = 0.1 for raw data
    (as in the HELICOID pipeline), 0 for reflectance data."""
    min_reflectance: Optional[float] = None
    """Zero or negative reflectance values are replaced by this, to avoid the log of zero.
    Empty = 0.01 for raw data (HELICOID pipeline), 0.001 for reflectance data (Biopsy pipeline)."""
    crop: Optional[List[int]] = None
    """[row_start, row_stop, col_start, col_stop] in original pixels."""
    binning: int = 1
    """Average n×n pixels. 4 turns a 2048×2048 image into 512×512."""
    frames: Optional[List[int]] = None
    """[start, stop] or [start, stop, step] for time series. Empty = all frames."""


@dataclass
class ReferenceSettings:
    type: str = "pixel"
    pixel: Optional[List[int]] = None
    """[row, col] in original pixels."""
    roi: Optional[List[List[float]]] = None
    """Polygon vertices [[row, col], ...] in original pixels."""
    file: Optional[str] = None
    """Two-column file: wavelength_nm, reflectance."""
    frame: Optional[List[int]] = None
    """Reference frame(s) [k] or [start, stop] (mean) for type 'frame', or the frame
    the pixel/region is taken from for a time series."""


@dataclass
class ChromophoreSettings:
    names: List[str] = field(default_factory=lambda: ["HbO2", "HHb", "oxCCO", "redCCO", "water", "fat"])
    extra_files: List[str] = field(default_factory=list)
    """Additional chromophore files, used in addition to the built-in library."""
    baselines: Dict[str, float] = field(default_factory=dict)
    """Override the baseline (typical tissue) concentration of a chromophore."""


@dataclass
class SpectralSettings:
    range: List[float] = field(default_factory=lambda: [530.0, 750.0])
    exclude: List[List[float]] = field(default_factory=list)
    """Wavelength intervals to leave out, e.g. [[600, 610]]."""


@dataclass
class ScatteringReference:
    method: str = "grid"
    """'grid' = estimate a0, b0 of the reference by grid search (as in the paper code); 'manual' = use a0, b0 below."""
    a0: float = 0.0
    b0: float = 1.0
    a_steps: int = 3
    b_steps: int = 25
    grid_binning: int = 8
    """Extra subsampling of the image during the grid search (speed)."""


@dataclass
class ModelSettings:
    mode: str = "free_scattering"
    fixed_b: float = 1.0
    """Scattering power used in 'fixed_b' mode (same for reference and pixels; Δa is fitted)."""
    reference_scattering: ScatteringReference = field(default_factory=ScatteringReference)
    a_max: float = 100.0
    b_max: float = 5.0
    scattering_scale: float = 10.0
    """Factor in front of the scattering term, 1/(1-g) with g = 0.9 in the paper code."""
    lambda0: float = 500.0
    """Reference wavelength (nm) of the power law a·(λ/λ0)^-b."""
    absolute_offset: bool = True
    """Absolute mode (no reference): fit a wavelength-independent offset G."""


@dataclass
class TissueSettings:
    a: float = 40.8
    b: float = 3.089
    g: float = 0.85
    n: float = 1.36


@dataclass
class PathlengthSettings:
    mode: str = "constant"
    constant: float = 1.0
    file: Optional[str] = None
    """Two-column file wavelength_nm, pathlength_cm (mode 'table_file')."""
    tissue: TissueSettings = field(default_factory=TissueSettings)
    jacques_tissue: str = "gray matter"
    """Parameter set of the Jacques model: 'gray matter', 'tumor', 'blood vessel' or 'general'."""


@dataclass
class SolverSettings:
    method: str = "fast"
    """'fast' = vectorised solver for the whole image; 'exact' = scipy least_squares pixel by pixel
    (as in the paper code; slow, for checking small regions)."""
    b_step: float = 0.05
    """Grid step for the scattering power b before local refinement (free scattering, fast solver)."""
    chunk_size: int = 20000
    cores: Optional[int] = None
    """Processor cores used for fitting. Empty or 0 = all but one (keeps the computer usable); 1 = no parallel work."""


@dataclass
class MaskSettings:
    min_reflectance: Optional[float] = None
    max_reflectance: Optional[float] = None
    """Exclude pixels whose mean reflectance in the fit range is outside these limits
    (dark background, specular highlights)."""
    polygons: List[List[List[float]]] = field(default_factory=list)
    """Fit only inside these polygons ([[row, col], ...] in original pixels). Empty = whole image."""


@dataclass
class OutputSettings:
    folder: str = "spectrafit_results"
    formats: List[str] = field(default_factory=lambda: ["npz", "tiff", "mat"])


@dataclass
class Config:
    data: DataSettings = field(default_factory=DataSettings)
    reference: ReferenceSettings = field(default_factory=ReferenceSettings)
    chromophores: ChromophoreSettings = field(default_factory=ChromophoreSettings)
    spectral: SpectralSettings = field(default_factory=SpectralSettings)
    model: ModelSettings = field(default_factory=ModelSettings)
    pathlength: PathlengthSettings = field(default_factory=PathlengthSettings)
    solver: SolverSettings = field(default_factory=SolverSettings)
    mask: MaskSettings = field(default_factory=MaskSettings)
    output: OutputSettings = field(default_factory=OutputSettings)

    # ------------------------------------------------------------------ I/O
    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            fh.write("# SpectraFit settings. Edit and run with:  spectrafit run <this file>\n")
            yaml.safe_dump(self.to_dict(), fh, sort_keys=False, allow_unicode=True)
        return path

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        cfg = _from_dict(cls, d or {})
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as fh:
            d = yaml.safe_load(fh) or {}
        cfg = cls.from_dict(d)
        # relative paths in a config file are relative to the config file
        base = Path(path).resolve().parent
        cfg.resolve_paths(base)
        return cfg

    def resolve_paths(self, base: Path) -> None:
        def fix(p):
            if not p:
                return p
            q = Path(p).expanduser()
            return str(q if q.is_absolute() or "*" in p else (base / q))

        self.data.file = fix(self.data.file)
        for attr in ("white_reference", "dark_reference"):
            v = getattr(self.data, attr)
            if v and (v.endswith(".mat") or "/" in v):
                setattr(self.data, attr, fix(v))
        self.reference.file = fix(self.reference.file)
        self.pathlength.file = fix(self.pathlength.file)
        self.chromophores.extra_files = [fix(f) for f in self.chromophores.extra_files]
        self.output.folder = fix(self.output.folder)

    def validate(self) -> None:
        self.pathlength.mode = PATHLENGTH_ALIASES.get(self.pathlength.mode, self.pathlength.mode)
        _check(self.model.mode in MODES, f"model.mode must be one of {MODES}")
        _check(self.pathlength.mode in PATHLENGTH_MODES, f"pathlength.mode must be one of {PATHLENGTH_MODES}")
        _check(self.reference.type in REFERENCE_TYPES, f"reference.type must be one of {REFERENCE_TYPES}")
        _check(self.solver.method in ("fast", "exact"), "solver.method must be 'fast' or 'exact'")
        _check(self.model.reference_scattering.method in ("grid", "manual"),
               "model.reference_scattering.method must be 'grid' or 'manual'")
        _check(len(self.spectral.range) == 2 and self.spectral.range[0] < self.spectral.range[1],
               "spectral.range must be [min_nm, max_nm]")
        _check(int(self.data.binning) >= 1, "data.binning must be 1 or larger")
        _check(len(self.chromophores.names) + len(self.chromophores.extra_files) > 0,
               "select at least one chromophore")
        if self.data.crop is not None:
            _check(len(self.data.crop) == 4, "data.crop must be [row_start, row_stop, col_start, col_stop]")
        if self.pathlength.mode == "table_file":
            _check(bool(self.pathlength.file), "pathlength.file is required for mode 'table_file'")


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _from_dict(cls, d: dict):
    kwargs = {}
    names = {f.name: f for f in fields(cls)}
    unknown = set(d) - set(names)
    if unknown:
        raise ValueError(f"Unknown setting(s) in section '{cls.__name__}': {', '.join(sorted(unknown))}")
    for name, f in names.items():
        if name not in d:
            continue
        value = d[name]
        default = f.default_factory() if f.default_factory is not dataclasses.MISSING else f.default  # type: ignore
        if is_dataclass(default) and isinstance(value, dict):
            value = _from_dict(type(default), value)
        kwargs[name] = value
    return cls(**kwargs)
