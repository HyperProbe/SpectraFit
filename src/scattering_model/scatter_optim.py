from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union
import pickle
import warnings
from loguru import logger
import numpy as np
import scipy
from sympy import lambdify
from src.dataset.helicoid_dataset import HelicoidDataset
from src.scattering_model.scatter import compute_delta_A_error
from src.constants import A_STEPS, B_STEPS, MAX_A, MAX_B
from tqdm import tqdm
import multiprocessing as mp
import gc

from src.wavelength_selection.enums import SampleType
from src.molecules import REFERENCE_SPECTRUM_T1, MoleculeMode, Molecules
from src.dataset.biopsy1_dataset import Biopsy1Dataset
from src.dataset.biopsy2_dataset import Biopsy2Dataset
from src.wavelength_selection.metrics import nrmse


PATHLENGTH_FILE = (
    Path(__file__).resolve().parents[2]
    / "data/pathlengths/gray_matter_pl_interpolated_spline.txt"
)

CARP_PICKLE_FILE = Path(__file__).resolve().parents[2] / "data/pathlengths/carp.pickle"
JACQUES_PICKLE_FILE = Path(__file__).resolve().parents[2] / "data/pathlengths/jacques.pickle"
JACQUES_M_PARAMS_FILE = Path(__file__).resolve().parents[2] / "data/pathlengths/m_parameters.pickle"

GRAY_MATTER_A = 40.8
GRAY_MATTER_B = 3.089
GRAY_MATTER_G = 0.85
GRAY_MATTER_N = 1.36


def _gray_matter_baseline_vector(num_molecules: int) -> np.ndarray:
    """Return the gray-matter baseline concentrations for the current molecule set."""

    if num_molecules == 10:
        return np.array(
            [
                0.0646,
                0.0114,
                0.0064,
                0.0016,
                0.0,
                0.0,
                0.0,
                0.0,
                0.73,
                0.10,
            ],
            dtype=float,
        )
    if num_molecules == 6:
        return np.array([0.0646, 0.0114, 0.0064, 0.0016, 0.73, 0.10], dtype=float)
    if num_molecules == 5:
        return np.array([0.0646, 0.0114, 0.0064, 0.0016, 0.10], dtype=float)
    if num_molecules == 4:
        return np.array([0.0646, 0.0114, 0.0064, 0.0016], dtype=float)

    raise ValueError(
        f"No gray-matter baseline vector defined for {num_molecules} molecules."
    )


def compute_gray_matter_mu_a(M: np.ndarray) -> np.ndarray:
    """Compute the gray-matter tissue absorption coefficient mu_a(lambda)."""

    M = np.asarray(M, dtype=float)
    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {M.shape}")

    c_gray = _gray_matter_baseline_vector(M.shape[1])
    if M.shape[1] != c_gray.shape[0]:
        raise ValueError(
            f"M has {M.shape[1]} columns but gray-matter baseline has {c_gray.shape[0]} entries."
        )

    mu_a = M @ c_gray
    if not np.isfinite(mu_a).all():
        raise ValueError("Computed gray-matter mu_a contains non-finite values.")
    if np.any(mu_a <= 0):
        raise ValueError("Computed gray-matter mu_a must be strictly positive.")

    return mu_a


def compute_gray_matter_mu_s_red(
    wavelengths: np.ndarray,
    a: float = GRAY_MATTER_A,
    b: float = GRAY_MATTER_B,
) -> np.ndarray:
    """Compute the gray-matter reduced scattering coefficient mu_s'(lambda)."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    mu_s_red = a * (wavelengths / 500.0) ** (-b)
    if not np.isfinite(mu_s_red).all():
        raise ValueError("Computed gray-matter mu_s_red contains non-finite values.")
    if np.any(mu_s_red <= 0):
        raise ValueError("Computed gray-matter mu_s_red must be strictly positive.")
    return mu_s_red


def _carp_boundary_factor(n: float) -> float:
    """Boundary factor used by the CARP / delta-P1 approximation."""

    return -0.13755 * n**3 + 4.3390 * n**2 - 4.90466 * n + 1.6896


@lru_cache(maxsize=1)
def _load_carp_lp_function(carp_pickle_path: str = str(CARP_PICKLE_FILE)):
    """Load the symbolic CARP / delta-P1 derivative dA/dmu_a and lambdify it."""

    with Path(carp_pickle_path).open("rb") as f:
        dA_dmu_a, _dA_dmu_s_red = pickle.load(f)

    mu_a_sym = next(symbol for symbol in dA_dmu_a.free_symbols if str(symbol) == "mu_a")
    mu_s_red_sym = next(
        symbol for symbol in dA_dmu_a.free_symbols if str(symbol) == "mu_s_red"
    )
    g_sym = next(symbol for symbol in dA_dmu_a.free_symbols if str(symbol) == "g")
    k_sym = next(symbol for symbol in dA_dmu_a.free_symbols if str(symbol) == "k")

    dA_dmu_a_np = lambdify(
        (mu_a_sym, mu_s_red_sym, g_sym, k_sym),
        dA_dmu_a,
        modules="numpy",
    )

    def lp_func(
        mu_a: np.ndarray,
        mu_s_red: np.ndarray,
        g: float = GRAY_MATTER_G,
        n: float = GRAY_MATTER_N,
    ) -> np.ndarray:
        mu_a = np.asarray(mu_a, dtype=float)
        mu_s_red = np.asarray(mu_s_red, dtype=float)

        if mu_a.shape != mu_s_red.shape:
            raise ValueError(
                f"mu_a and mu_s_red must have the same shape. Got {mu_a.shape} and {mu_s_red.shape}."
            )
        if np.any(mu_a <= 0):
            raise ValueError("mu_a must be positive for the delta-P1 derivative.")
        if np.any(mu_s_red <= 0):
            raise ValueError("mu_s_red must be positive for the delta-P1 derivative.")

        k = _carp_boundary_factor(n)
        lp = dA_dmu_a_np(mu_a, mu_s_red, g, k)
        return np.asarray(lp, dtype=float)

    return lp_func


@lru_cache(maxsize=1)
def _load_jacques_m_params(path: str = str(JACQUES_M_PARAMS_FILE)) -> dict[str, np.ndarray]:
    with Path(path).open("rb") as f:
        m_params, _A_vals, _N_vals, _dref_vals = pickle.load(f)
    return m_params


@lru_cache(maxsize=1)
def _load_jacques_lp_function(jacques_pickle_path: str = str(JACQUES_PICKLE_FILE)):
    """Load symbolic Jacques derivative dA/dmu_a and lambdify it."""

    with Path(jacques_pickle_path).open("rb") as f:
        dA_dmu_a, _dA_dmu_s_red = pickle.load(f)

    symbol_map = {str(symbol): symbol for symbol in dA_dmu_a.free_symbols}
    required_symbols = ["mu_a", "mu_s_red", "m1", "m2", "m3"]
    missing_symbols = [name for name in required_symbols if name not in symbol_map]
    if missing_symbols:
        raise ValueError(
            f"Jacques derivative pickle {jacques_pickle_path} is missing symbol(s): {', '.join(missing_symbols)}"
        )

    mu_a_sym = symbol_map["mu_a"]
    mu_s_red_sym = symbol_map["mu_s_red"]
    m1_sym = symbol_map["m1"]
    m2_sym = symbol_map["m2"]
    m3_sym = symbol_map["m3"]

    dA_dmu_a_np = lambdify(
        (mu_a_sym, mu_s_red_sym, m1_sym, m2_sym, m3_sym),
        dA_dmu_a,
        modules="numpy",
    )

    def lp_func(mu_a: np.ndarray, mu_s_red: np.ndarray, m_params: np.ndarray) -> np.ndarray:
        mu_a = np.asarray(mu_a, dtype=float)
        mu_s_red = np.asarray(mu_s_red, dtype=float)
        m_params = np.asarray(m_params, dtype=float)

        if mu_a.shape != mu_s_red.shape:
            raise ValueError(
                f"mu_a and mu_s_red must have the same shape. Got {mu_a.shape} and {mu_s_red.shape}."
            )
        if m_params.shape != (3,):
            raise ValueError(f"m_params must have shape (3,), got {m_params.shape}")
        if np.any(mu_a <= 0):
            raise ValueError("mu_a must be positive for the Jacques derivative.")
        if np.any(mu_s_red <= 0):
            raise ValueError("mu_s_red must be positive for the Jacques derivative.")

        lp = dA_dmu_a_np(mu_a, mu_s_red, m_params[0], m_params[1], m_params[2])
        lp = np.asarray(lp, dtype=float)
        if not np.isfinite(lp).all():
            raise ValueError("Computed Jacques pathlength contains non-finite values.")
        if np.any(lp <= 0):
            raise ValueError("Computed Jacques pathlength must be strictly positive.")
        return lp

    return lp_func


def compute_delta_p1_pathlength_from_wavelength(
    M: np.ndarray,
    wavelengths: np.ndarray,
    g: float = GRAY_MATTER_G,
    n: float = GRAY_MATTER_N,
    a: float = GRAY_MATTER_A,
    b: float = GRAY_MATTER_B,
) -> np.ndarray:
    """Compute the gray-matter delta-P1 pathlength used by the optimizer."""

    mu_a = compute_gray_matter_mu_a(M)
    mu_s_red = compute_gray_matter_mu_s_red(wavelengths, a=a, b=b)
    lp_func = _load_carp_lp_function()
    return lp_func(mu_a=mu_a, mu_s_red=mu_s_red, g=g, n=n)


def compute_jacques_pathlength_from_wavelength(
    M: np.ndarray,
    wavelengths: np.ndarray,
    jacques_m_key: str = "gray matter",
    jacques_pickle_path: Union[str, Path] = JACQUES_PICKLE_FILE,
    m_params_path: Union[str, Path] = JACQUES_M_PARAMS_FILE,
) -> np.ndarray:
    """Compute the gray-matter Jacques pathlength used by the optimizer."""

    mu_a = compute_gray_matter_mu_a(M)
    mu_s_red = compute_gray_matter_mu_s_red(wavelengths)

    m_params_dict = _load_jacques_m_params(str(m_params_path))
    if jacques_m_key not in m_params_dict:
        raise ValueError(
            f"Jacques m_params key '{jacques_m_key}' was not found in {m_params_path}. Available keys: {', '.join(map(str, m_params_dict.keys()))}"
        )

    m_params = np.asarray(m_params_dict[jacques_m_key], dtype=float)
    lp_func = _load_jacques_lp_function(str(jacques_pickle_path))
    return lp_func(mu_a=mu_a, mu_s_red=mu_s_red, m_params=m_params)


def load_unitary_pathlength_results(
    sample_id: str,
    results_root: Union[str, Path] = "/home/cihank/hsi-biopsy/results/helicoid_molecular_maps/reference_params",
) -> tuple[np.ndarray, np.ndarray]:
    """Load unitary-pathlength coefficients and scatter parameters for one sample."""

    results_root = Path(results_root)
    coef_path = results_root / f"{sample_id}_coef_list.npy"
    scatter_path = results_root / f"{sample_id}_scatter_params.npy"

    missing = [str(path) for path in (coef_path, scatter_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing unitary pathlength result file(s) for sample_id={sample_id}: {', '.join(missing)}"
        )

    coef_pl1 = np.load(coef_path)
    scatter_params_pl1 = np.load(scatter_path)
    return coef_pl1, scatter_params_pl1


def compute_pixelwise_delta_p1_pathlength_from_unitary_results(
    M: np.ndarray,
    wavelengths: np.ndarray,
    coef_pl1: np.ndarray,
    scatter_params_pl1: np.ndarray,
    c_baseline: Optional[np.ndarray] = None,
    g: float = GRAY_MATTER_G,
    n: float = GRAY_MATTER_N,
    min_abs_concentration: float = 1e-9,
    min_mu_a: float = 1e-9,
    min_mu_s_red: float = 1e-9,
) -> np.ndarray:
    """Compute pixelwise delta-P1 pathlength curves from unitary-pathlength outputs."""

    M = np.asarray(M, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    coef_pl1 = np.asarray(coef_pl1, dtype=float)
    scatter_params_pl1 = np.asarray(scatter_params_pl1, dtype=float)

    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {M.shape}")
    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths must be 1D, got shape {wavelengths.shape}")
    if M.shape[0] != wavelengths.shape[0]:
        raise ValueError(
            f"M.shape[0] must match wavelengths.shape[0]; got {M.shape[0]} and {wavelengths.shape[0]}"
        )
    if coef_pl1.ndim not in (2, 3):
        raise ValueError(
            f"coef_pl1 must be either flattened (N, n_molecules) or image-shaped (H, W, n_molecules); got {coef_pl1.shape}"
        )
    if scatter_params_pl1.ndim not in (2, 3):
        raise ValueError(
            f"scatter_params_pl1 must be either flattened (N, 2) or image-shaped (H, W, 2); got {scatter_params_pl1.shape}"
        )
    if coef_pl1.shape[:-1] != scatter_params_pl1.shape[:-1]:
        raise ValueError(
            f"coef_pl1 and scatter_params_pl1 must have matching leading dimensions; got {coef_pl1.shape} and {scatter_params_pl1.shape}"
        )
    if coef_pl1.shape[-1] != M.shape[1]:
        raise ValueError(
            f"coef_pl1 last dimension must match M.shape[1]; got {coef_pl1.shape[-1]} and {M.shape[1]}"
        )
    if scatter_params_pl1.shape[-1] != 2:
        raise ValueError(
            f"scatter_params_pl1 last dimension must be 2 for [a, b]; got {scatter_params_pl1.shape[-1]}"
        )
    if c_baseline is None:
        c_baseline = _gray_matter_baseline_vector(M.shape[1])
    else:
        c_baseline = np.asarray(c_baseline, dtype=float)
        if c_baseline.ndim != 1:
            raise ValueError(f"c_baseline must be 1D, got shape {c_baseline.shape}")
        if c_baseline.shape[0] != M.shape[1]:
            raise ValueError(
                f"c_baseline length must match M.shape[1]; got {c_baseline.shape[0]} and {M.shape[1]}"
            )

    c_abs = np.clip(c_baseline + coef_pl1, min_abs_concentration, None)
    mu_a = np.clip(c_abs @ M.T, min_mu_a, None)

    a_pixel = scatter_params_pl1[..., 0]
    b_pixel = scatter_params_pl1[..., 1]
    mu_s_red = a_pixel[..., None] * (wavelengths[None, :] / 500.0) ** (-b_pixel[..., None])
    mu_s_red = np.clip(mu_s_red, min_mu_s_red, None)

    lp_func = _load_carp_lp_function()
    PL = lp_func(mu_a=mu_a, mu_s_red=mu_s_red, g=g, n=n)
    PL = np.asarray(PL, dtype=float)

    if not np.isfinite(PL).all():
        raise ValueError("Computed pixelwise delta-P1 pathlength contains non-finite values.")
    if np.any(PL <= 0):
        raise ValueError("Computed pixelwise delta-P1 pathlength must be strictly positive.")

    return PL


def compute_pixelwise_jacques_pathlength_from_unitary_results(
    M: np.ndarray,
    wavelengths: np.ndarray,
    coef_pl1: np.ndarray,
    scatter_params_pl1: np.ndarray,
    c_baseline: Optional[np.ndarray] = None,
    jacques_m_key: str = "general",
    jacques_pickle_path: Union[str, Path] = JACQUES_PICKLE_FILE,
    m_params_path: Union[str, Path] = JACQUES_M_PARAMS_FILE,
    min_abs_concentration: float = 1e-9,
    min_mu_a: float = 1e-9,
    min_mu_s_red: float = 1e-9,
) -> np.ndarray:
    """Compute pixelwise Jacques pathlength curves from unitary-pathlength outputs."""

    M = np.asarray(M, dtype=float)
    wavelengths = np.asarray(wavelengths, dtype=float)
    coef_pl1 = np.asarray(coef_pl1, dtype=float)
    scatter_params_pl1 = np.asarray(scatter_params_pl1, dtype=float)

    if M.ndim != 2:
        raise ValueError(f"M must be 2D, got shape {M.shape}")
    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths must be 1D, got shape {wavelengths.shape}")
    if M.shape[0] != wavelengths.shape[0]:
        raise ValueError(
            f"M.shape[0] must match wavelengths.shape[0]; got {M.shape[0]} and {wavelengths.shape[0]}"
        )
    if coef_pl1.ndim not in (2, 3):
        raise ValueError(
            f"coef_pl1 must be either flattened (N, n_molecules) or image-shaped (H, W, n_molecules); got {coef_pl1.shape}"
        )
    if scatter_params_pl1.ndim not in (2, 3):
        raise ValueError(
            f"scatter_params_pl1 must be either flattened (N, 2) or image-shaped (H, W, 2); got {scatter_params_pl1.shape}"
        )
    if coef_pl1.shape[:-1] != scatter_params_pl1.shape[:-1]:
        raise ValueError(
            f"coef_pl1 and scatter_params_pl1 must have matching leading dimensions; got {coef_pl1.shape} and {scatter_params_pl1.shape}"
        )
    if coef_pl1.shape[-1] != M.shape[1]:
        raise ValueError(
            f"coef_pl1 last dimension must match M.shape[1]; got {coef_pl1.shape[-1]} and {M.shape[1]}"
        )
    if scatter_params_pl1.shape[-1] != 2:
        raise ValueError(
            f"scatter_params_pl1 last dimension must be 2 for [a, b]; got {scatter_params_pl1.shape[-1]}"
        )
    if c_baseline is None:
        c_baseline = _gray_matter_baseline_vector(M.shape[1])
    else:
        c_baseline = np.asarray(c_baseline, dtype=float)
        if c_baseline.ndim != 1:
            raise ValueError(f"c_baseline must be 1D, got shape {c_baseline.shape}")
        if c_baseline.shape[0] != M.shape[1]:
            raise ValueError(
                f"c_baseline length must match M.shape[1]; got {c_baseline.shape[0]} and {M.shape[1]}"
            )

    c_abs = np.clip(c_baseline + coef_pl1, min_abs_concentration, None)
    mu_a = np.clip(c_abs @ M.T, min_mu_a, None)

    a_pixel = scatter_params_pl1[..., 0]
    b_pixel = scatter_params_pl1[..., 1]
    mu_s_red = a_pixel[..., None] * (wavelengths[None, :] / 500.0) ** (-b_pixel[..., None])
    mu_s_red = np.clip(mu_s_red, min_mu_s_red, None)

    m_params_dict = _load_jacques_m_params(str(m_params_path))
    if jacques_m_key not in m_params_dict:
        raise ValueError(
            f"Jacques m_params key '{jacques_m_key}' was not found in {m_params_path}. Available keys: {', '.join(map(str, m_params_dict.keys()))}"
        )

    m_params = np.asarray(m_params_dict[jacques_m_key], dtype=float)
    lp_func = _load_jacques_lp_function(str(jacques_pickle_path))
    PL = lp_func(mu_a=mu_a, mu_s_red=mu_s_red, m_params=m_params)
    PL = np.asarray(PL, dtype=float)

    if not np.isfinite(PL).all():
        raise ValueError("Computed pixelwise Jacques pathlength contains non-finite values.")
    if np.any(PL <= 0):
        raise ValueError("Computed pixelwise Jacques pathlength must be strictly positive.")

    return PL


def save_pixelwise_pathlength_debug(
    wavelengths: np.ndarray,
    PL: np.ndarray,
    save_path: Union[str, Path],
    max_pixels: int = 20,
) -> None:
    """Save pixelwise pathlength curves and a compact summary next to them."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    PL = np.asarray(PL, dtype=float)
    save_path = Path(save_path)
    if save_path.exists() and save_path.is_dir():
        base_path = save_path / "pixelwise_pathlength_debug"
    elif save_path.suffix:
        base_path = save_path.with_suffix("")
    else:
        base_path = save_path

    if PL.ndim == 1:
        PL_for_summary = PL[None, :]
    elif PL.ndim == 2:
        PL_for_summary = PL
    elif PL.ndim == 3:
        PL_for_summary = PL.reshape(-1, PL.shape[-1])
    else:
        raise ValueError(f"PL must be 1D, 2D, or 3D, got shape {PL.shape}")

    if wavelengths.ndim != 1:
        raise ValueError(f"wavelengths must be 1D, got shape {wavelengths.shape}")
    if PL_for_summary.shape[-1] != wavelengths.shape[0]:
        raise ValueError(
            f"PL wavelength dimension must match wavelengths; got {PL_for_summary.shape[-1]} and {wavelengths.shape[0]}"
        )

    summary = np.column_stack(
        [
            wavelengths,
            np.mean(PL_for_summary, axis=0),
            np.std(PL_for_summary, axis=0),
            np.min(PL_for_summary, axis=0),
            np.max(PL_for_summary, axis=0),
        ]
    )
    np.savetxt(
        base_path.with_suffix(".csv"),
        summary,
        delimiter=",",
        header="wavelength_nm,mean_PL,std_PL,min_PL,max_PL",
        comments="",
    )

    payload = {
        "wavelengths": wavelengths,
        "PL": PL,
    }
    if PL_for_summary.shape[0] > 0:
        payload["PL_first_pixels"] = PL_for_summary[: max_pixels].copy()

    np.savez_compressed(base_path.with_suffix(".npz"), **payload)


@lru_cache(maxsize=1)
def _load_pathlength_table(pathlength_file: str) -> tuple[np.ndarray, np.ndarray]:
    table = np.loadtxt(pathlength_file)
    return table[:, 0], table[:, 1] / 10.0


def compute_pathlength_from_wavelength(x, pathlength_file: Union[str, Path] = PATHLENGTH_FILE):
    wavelengths, pathlength = _load_pathlength_table(str(pathlength_file))
    return np.interp(np.asarray(x), wavelengths, pathlength)


def prepare_pathlength_for_optimization(
    sample_id: str,
    M: np.ndarray,
    wavelengths: np.ndarray,
    delta_A: np.ndarray,
    pathlength_mode: str = "gray_matter_delta_p1",
    unitary_results_root: Union[str, Path] = "/home/cihank/hsi-biopsy/results/helicoid_molecular_maps/reference_params",
    pathlength_debug_dir: Optional[Union[str, Path]] = None,
    jacques_m_key: Optional[str] = None,
    jacques_pickle_path: Union[str, Path] = JACQUES_PICKLE_FILE,
    jacques_m_params_path: Union[str, Path] = JACQUES_M_PARAMS_FILE,
) -> np.ndarray:
    """Prepare the pathlength array used by the scattering optimizer."""

    wavelengths = np.asarray(wavelengths, dtype=float)
    delta_A = np.asarray(delta_A, dtype=float)

    if pathlength_mode == "ones":
        PL = np.ones(wavelengths.shape[0], dtype=float)
    elif pathlength_mode == "pathlength_from_wavelength":
        PL = compute_pathlength_from_wavelength(wavelengths)
    elif pathlength_mode == "gray_matter_delta_p1":
        PL = compute_delta_p1_pathlength_from_wavelength(M, wavelengths)
    elif pathlength_mode == "gray_matter_jacques":
        if jacques_m_key is None:
            jacques_m_key = "gray matter"
        PL = compute_jacques_pathlength_from_wavelength(
            M,
            wavelengths,
            jacques_m_key=jacques_m_key,
            jacques_pickle_path=jacques_pickle_path,
            m_params_path=jacques_m_params_path,
        )
        if pathlength_debug_dir is not None:
            save_pixelwise_pathlength_debug(
                wavelengths,
                PL,
                Path(pathlength_debug_dir) / f"{sample_id}_gray_matter_jacques_PL",
            )
    elif pathlength_mode == "pixelwise_delta_p1_from_unitary":
        coef_pl1, scatter_params_pl1 = load_unitary_pathlength_results(
            sample_id,
            unitary_results_root,
        )
        if coef_pl1.shape[:2] != delta_A.shape[:2]:
            raise ValueError(
                "Unitary pathlength results must use the same image shape as delta_A. "
                "Please use the same dataset, coarseness, and preprocessing when computing the unitary results."
            )
        PL_img = compute_pixelwise_delta_p1_pathlength_from_unitary_results(
            M,
            wavelengths,
            coef_pl1,
            scatter_params_pl1,
        )
        if PL_img.shape[:2] != delta_A.shape[:2]:
            raise ValueError(
                "Computed pixelwise pathlength image does not match the current delta_A shape. "
                "Please use the same dataset, coarseness, and preprocessing when computing the unitary results."
            )
        PL = PL_img.reshape(-1, PL_img.shape[-1])
        if pathlength_debug_dir is not None:
            save_pixelwise_pathlength_debug(
                wavelengths,
                PL_img,
                Path(pathlength_debug_dir) / f"{sample_id}_pixelwise_pathlength_debug",
            )
    elif pathlength_mode == "pixelwise_jacques_from_unitary":
        if jacques_m_key is None:
            jacques_m_key = "general"
        coef_pl1, scatter_params_pl1 = load_unitary_pathlength_results(
            sample_id,
            unitary_results_root,
        )
        if coef_pl1.shape[:2] != delta_A.shape[:2]:
            raise ValueError(
                "Unitary pathlength results must use the same image shape as delta_A. "
                "Please use the same dataset, coarseness, and preprocessing when computing the unitary results."
            )
        PL_img = compute_pixelwise_jacques_pathlength_from_unitary_results(
            M,
            wavelengths,
            coef_pl1,
            scatter_params_pl1,
            jacques_m_key=jacques_m_key,
            jacques_pickle_path=jacques_pickle_path,
            m_params_path=jacques_m_params_path,
        )
        if PL_img.shape[:2] != delta_A.shape[:2]:
            raise ValueError(
                "Computed pixelwise pathlength image does not match the current delta_A shape. "
                "Please use the same dataset, coarseness, and preprocessing when computing the unitary results."
            )
        PL = PL_img.reshape(-1, PL_img.shape[-1])
        if pathlength_debug_dir is not None:
            save_pixelwise_pathlength_debug(
                wavelengths,
                PL_img,
                Path(pathlength_debug_dir) / f"{sample_id}_pixelwise_jacques_PL",
            )
    else:
        raise ValueError(
            f"Unsupported pathlength_mode '{pathlength_mode}'. Expected one of: ones, pathlength_from_wavelength, gray_matter_delta_p1, gray_matter_jacques, pixelwise_delta_p1_from_unitary, pixelwise_jacques_from_unitary."
        )

    if not np.isfinite(PL).all():
        raise ValueError("Prepared pathlength contains non-finite values.")
    if np.any(PL <= 0):
        raise ValueError("Prepared pathlength must be strictly positive.")
    return np.asarray(PL, dtype=float)


def optimize_image_parallel(params_t1, *args, PL=None, executor=None, disable_tqdm=False):
    """
    Equivalent to old optimize_image/helicoid_optimisation_ti but using parallelization with ProcessPoolExecutor (parallel pixel optim).

    Using fixed a and b of reference spectrum - Infer the change in concentration of molecules and scattering parameters given absorption coefficients
    and change of attenuation utilizing the differential modified Beer-Lambert Law with scattering.

    Args:
        params_t1 (float,float): (a_t1, b_t1) where a_t1 is the scattering coefficient of the reference spectrum and b_t1 is the scattering power of the reference spectrum
        *args: [B, M, x] where B is the flattened delta A image, M is the absorption matrix, x is the wavelengths
        executor: Optional external ProcessPoolExecutor to reuse

    Returns:
        (float, NDArray, NDArray, NDArray): Error (mean of errors_list), Concentrations, Scattering parameters, errors_list

    """

    a_t1, b_t1 = params_t1[0], params_t1[1]
    B = args[0]
    """flattened delta A image -> B[i] = delta A at index i (1D with length len(wl))"""

    M = args[1]
    x = args[2]
    if PL is None:
        PL = compute_delta_p1_pathlength_from_wavelength(M, x)
    else:
        PL = np.asarray(PL, dtype=float)
        if PL.ndim == 1:
            if PL.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Global PL must have length {x.shape[0]}, got {PL.shape[0]}"
                )
        elif PL.ndim == 2:
            if PL.shape != B.shape:
                raise ValueError(
                    f"Pixelwise PL must have shape {B.shape}, got {PL.shape}"
                )
        else:
            raise ValueError(f"PL must be 1D or 2D, got shape {PL.shape}")
        if not np.isfinite(PL).all():
            raise ValueError("PL must contain only finite values")
        if np.any(PL <= 0):
            raise ValueError("PL must be strictly positive")
    pl_is_pixelwise = PL.ndim == 2
    if len(args) > 3:
        num_workers = args[3]
    else:
        num_workers = min(16, mp.cpu_count())

    num_molecules = M.shape[1]

    #  Define bounds for optimization

    ### Original bounds from kevin's codebase (not sure if they make sense or are even the latest)
    # left_bound = np.append(np.ones(N_HELICOID_MOLECULES) * (-np.inf), [-np.inf, 0])
    # right_bound = np.append(np.ones(N_HELICOID_MOLECULES) * np.inf, [np.inf, MAX_B])

    ### Adapted bounds that make more sense to me:
    # delta concentration changes can be +/- and are not bound => +- inf
    # scattering parameters are bound by our proposed reasonable ranges (for a (0.1; MAX_A) and b (0; MAX_B)),
    # however we do not infer the delta, but rather the absolute values of the scattering parameters
    left_bound = np.append(np.ones(num_molecules) * (-np.inf), [0, 0])
    right_bound = np.append(np.ones(num_molecules) * np.inf, [MAX_A, MAX_B])

    # start least squares optimization with scatter params of reference spectrum (probably irrelevant, could be 0 as well)
    current_x = np.zeros(num_molecules + 2)
    current_x[-2] = a_t1
    current_x[-1] = b_t1

    # init arrays with size matching number of pixels (first dim of B)
    coef_list = np.zeros((B.shape[0], num_molecules))
    scattering_params_list = np.zeros((B.shape[0], 2))
    errors_scatter = np.zeros((B.shape[0]))

    # Prepare optimization parameters as a single structure to reduce serialization overhead
    opt_params = {
        "b_t1": b_t1,
        "a_t1": a_t1,
        "M": M,
        "x": x,
        "current_x": current_x,
        "left_bound": left_bound,
        "right_bound": right_bound,
    }

    def use_external_executor():
        """Use external executor (no context manager)"""
        futures = []
        for i, b_i in enumerate(B):
            pl_i = PL[i] if pl_is_pixelwise else PL
            future = executor.submit(optimize_single_b_with_params, b_i, opt_params, pl_i)
            futures.append((i, future))

        for i, future in tqdm(futures, disable=disable_tqdm):
            try:
                coef, scattering_params, error_scatter = future.result(timeout=30)
                coef_list[i, :] = coef
                scattering_params_list[i, :] = scattering_params
                errors_scatter[i] = error_scatter
            except Exception as e:
                print(f"Error processing pixel {i}: {e}")
                # Use default values for failed optimization
                coef_list[i, :] = current_x[:num_molecules]
                scattering_params_list[i, :] = current_x[num_molecules:]
                errors_scatter[i] = np.inf

    def use_internal_executor():
        """Create and manage internal executor"""
        with ProcessPoolExecutor(max_workers=num_workers) as exec_ctx:
            futures = []
            for i, b_i in enumerate(B):
                pl_i = PL[i] if pl_is_pixelwise else PL
                future = exec_ctx.submit(optimize_single_b_with_params, b_i, opt_params, pl_i)
                futures.append((i, future))

            for i, future in tqdm(
                futures, desc="Processing pixels", disable=disable_tqdm
            ):
                try:
                    coef, scattering_params, error_scatter = future.result(timeout=30)
                    coef_list[i, :] = coef
                    scattering_params_list[i, :] = scattering_params
                    errors_scatter[i] = error_scatter
                except Exception as e:
                    print(f"Error processing pixel {i}: {e}")
                    # Use default values for failed optimization
                    coef_list[i, :] = current_x[:num_molecules]
                    scattering_params_list[i, :] = current_x[num_molecules:]
                    errors_scatter[i] = np.inf

    # Choose execution path
    if executor is not None:
        logger.debug("Using provided executor for parallel optimization.")
        use_external_executor()
    else:
        use_internal_executor()

    # Force garbage collection to free memory
    gc.collect()

    error = np.mean(errors_scatter[errors_scatter != np.inf])
    return error, coef_list, scattering_params_list, errors_scatter


def optimize_single_b(b_i, b_t1, a_t1, M, x, PL, current_x, left_bound, right_bound):
    """
    Optimize the concentrations and scattering parameters for a single pixel.

    Args:
        b_i (NDArray): delta A at pixel i
        a_t1 (float): scattering coefficient of the reference spectrum
        b_t1 (float): scattering power of the reference spectrum
        M (NDArray): absorption matrix
        x (NDArray): wavelengths
        current_x (NDArray): current concentrations and scattering parameters
        left_bound (NDArray): left bound for optimization
        right_bound (NDArray): right bound for optimization

    Returns:
        Tuple[NDArray, NDArray, float]: concentrations, scattering parameters, error

    """
    num_molecules = M.shape[1]
    result = scipy.optimize.least_squares(
        compute_delta_A_error,  # error is squares (mse) internally, so it's correct that compute_delta_A_error returns the raw error (not abs or squared)
        current_x,
        args=(b_i, a_t1, b_t1, M, x, PL),
        bounds=(left_bound, right_bound),
    )
    coef = result.x[:num_molecules]
    scattering_params = result.x[num_molecules:]
    error_scatter = result.cost
    return coef, scattering_params, error_scatter


def optimize_single_b_with_params(b_i, opt_params, PL_i):
    """
    Optimized version that takes parameters as a dictionary to reduce serialization overhead.

    Args:
        b_i (NDArray): delta A at pixel i
        opt_params (dict): Dictionary containing all optimization parameters

    Returns:
        Tuple[NDArray, NDArray, float]: concentrations, scattering parameters, error
    """
    num_molecules = opt_params["M"].shape[1]

    try:
        result = scipy.optimize.least_squares(
            compute_delta_A_error,
            opt_params["current_x"],
            args=(
                b_i,
                opt_params["a_t1"],
                opt_params["b_t1"],
                opt_params["M"],
                opt_params["x"],
                PL_i,
            ),
            bounds=(opt_params["left_bound"], opt_params["right_bound"]),
        )
        coef = result.x[:num_molecules]
        scattering_params = result.x[num_molecules:]
        error_scatter = result.cost
    except Exception as e:
        # Return default values if optimization fails
        coef = opt_params["current_x"][:num_molecules]
        scattering_params = opt_params["current_x"][num_molecules:]
        error_scatter = np.inf

    return coef, scattering_params, error_scatter


def single_optim_scattering_with_wl_subset(
    data: dict,
    dataset: Union[Biopsy1Dataset, Biopsy2Dataset, HelicoidDataset],
    gt_path: Union[str, None],
    x_chosen,
    use_parallel: Optional[bool] = True,
    molecule_mode: MoleculeMode = MoleculeMode.ALL,
    disable_tqdm: bool = False,
):
    """
    Run coefficient optimization with specified subset of wavelengths and compare it against the ground truth run utilizing the full wavelength range.
    Will re-use the reference spectrum scattering coefficients from the ground truth data (all wavelengths, coarseness=1).

    Args:
        data (dict): Biopsy data
        dataset (Union[Biopsy1Dataset, Biopsy2Dataset, HelicoidDataset]): Dataset object that provides access to hyperspectral images and related metadata.
        gt_path (str): Path to the ground‐truth scattering parameters and coefficient file
        x_chosen (_type_):  Masked Array, Selected subset of wavelengths is True
        use_parallel (bool, optional): Use parallelization. Defaults to False.
        molecule_mode (MoleculeMode, optional): Molecule selection mode to apply during optimization (default: MoleculeMode.ALL).
        disable_tqdm (bool, optional): Disable tqdm progress bar. Defaults to False.
    Returns:
        Tuple[float, NDArray, NDArray]: error_wavelength_selection, params_found, params_found_GT_coarsened
    """

    #### load reference ground truth data
    # force coarseness to 1 to get the full img ground truth data as error baseline

    # img_ref = gt_data.img_ref[:, :, data.left_cut_index : data.right_cut_index]

    if isinstance(dataset, HelicoidDataset):
        sample_type = SampleType.HELICOID
        reference_spectrum_params = dataset.load_reference_params(
            id=data["id"], load_from_path=gt_path
        )
    else:
        reference_spectrum_params = REFERENCE_SPECTRUM_T1[molecule_mode]
        sample_type = SampleType.BIOPSY2
    delta_A = data["delta_A"][:, :, x_chosen]
    logger.debug(delta_A.shape)
    molecules = Molecules(
        left_cut=dataset.left_cut,
        right_cut=dataset.right_cut,
        molecule_mode=molecule_mode,
        sample_type=sample_type,
    )
    M = molecules.M[x_chosen, :]

    # flatten img_ref to 1D x wavelength
    b = delta_A.reshape(-1, delta_A.shape[-1])  # reshape to 1D x wavelength

    # run optimization with reduced wavelength range __x__!
    x = dataset.cut_wavelengths[x_chosen]
    if use_parallel:
        error, coef_list, scattering_params_list, _ = optimize_image_parallel(
            reference_spectrum_params, b, M, x, disable_tqdm=disable_tqdm
        )
    else:
        raise NotImplementedError("Sequential mode not implemented")
        # error, coef_list, scattering_params_list, _ = helicoid_optimisation_ti(
        #     # use x or x chosen?
        #     reference_spectrum_params,
        #     *(b, M, x),
        # )

    # build params_found array
    coef_list = np.array(coef_list)
    coef_list = coef_list.reshape(
        delta_A.shape[0], delta_A.shape[1], coef_list.shape[-1]
    )
    scattering_params = np.array(scattering_params_list)
    scattering_params = scattering_params.reshape(
        delta_A.shape[0], delta_A.shape[1], scattering_params.shape[-1]
    )
    params_found = np.concatenate((coef_list, scattering_params), axis=2)

    # compute error
    if gt_path is None:
        return error, coef_list, scattering_params
    else:
        # load ground truth data
        coef_list_GT, scattering_params_GT = dataset.load_delta_c_and_scatter_params(
            id=data["id"], load_from_path=gt_path
        )
        coef_list_GT_coarsened = coef_list_GT[
            :: dataset.coarseness, :: dataset.coarseness, :
        ]
        scattering_params_GT_coarsened = scattering_params_GT[
            :: dataset.coarseness, :: dataset.coarseness, :
        ]
        params_found_GT_coarsened = np.concatenate(
            (coef_list_GT_coarsened, scattering_params_GT_coarsened), axis=2
        )

    #! TODO ongoing experiment if scatter params should be included in the error calculation => no real analysis, just keeping it at concentrations only
    if coef_list_GT_coarsened.shape != coef_list.shape:
        logger.warning(
            "GT and found coef_list shapes differ, liekly due to different molecule sets"
        )
        logger.warning(
            f"GT shape: {coef_list_GT_coarsened.shape}, found shape: {coef_list.shape}"
        )
        error_wavelength_selection = None
    else:
        error_wavelength_selection = nrmse(
            y_true=coef_list_GT_coarsened,
            y_pred=coef_list,
        )
    # error_wavelength_selection = nrmse(
    #     y_true=params_found_GT_coarsened, y_pred=params_found
    # )

    return error_wavelength_selection, params_found, params_found_GT_coarsened


def optim_reference_spectrum_scatter_params(
    data: dict,
    dataset: Union[Biopsy1Dataset, Biopsy2Dataset, HelicoidDataset],
    molecule_mode: MoleculeMode = MoleculeMode.ALL,
    load_a_b_from_path: Optional[str] = None,
    coarseness: int = 1,
    pathlength_mode: str = "gray_matter_delta_p1",
    unitary_results_root: Union[str, Path] = "/home/cihank/hsi-biopsy/results/helicoid_molecular_maps/reference_params",
    pathlength_debug_dir: Optional[Union[str, Path]] = None,
    jacques_m_key: Optional[str] = None,
    jacques_pickle_path: Union[str, Path] = JACQUES_PICKLE_FILE,
    jacques_m_params_path: Union[str, Path] = JACQUES_M_PARAMS_FILE,
):
    """
    Infer the change in concentration of molecules and scattering parameters given absorption coefficients and change of attenuation utilizing the differential modified Beer-Lambert Law with scattering.

    Performs a brute force search over a grid of possible scattering parameters to find the optimal scattering parameters of the reference spectrum.
    Then the optimal concentrations and scattering parameters are computed given the fixed scattering parameters of the reference spectrum.


    Args:
        data (dict): Helicoid or Biopsy data dict
        dataset (Union[Biopsy1Dataset, Biopsy2Dataset]): Dataset object that provides access to hyperspectral images and related metadata.
        use_parallel (bool, optional): Use parallelization. Defaults to False.

    Returns:
        (NDarray, NDarray, NDarray, float, float, NDarray, Tuple[int,int], ?, ?): Concentrations, Scattering parameters, Errors, a_t1 (scat coef of reference spectrum), b_t1 (scat power of reference spectrum), delta A image, reference pixel, grid, error grid

    Compute: minimize e = MSE [M∆c + S - b]

    Based on the formula:  ∆A(v1, λ) = A(v1, λ) - A(v0, λ) = PL(λ)  ∑ μ_i(λ) · ∆c_i(v) + ∆μ_s(v, λ)

        where:
            - A(v1, λ) is the attenuation at time/pixel v1 and wavelength λ
            - A(v0, λ) is the attenuation at time/pixel v0 and wavelength λ
            - μ_i(λ) is the absorption coefficient of molecule i at wavelength λ
            - ∆c_i(v) is the change in concentration of molecule i at time/pixel v
            - ∆μ_s(v, λ) _?UNSURE?_ is the change in scattering at time/pixel v and wavelength λ
            - PL(λ) is the path length. Here we assume it is semi constant = 1.

        Simplifying Assumptions
            - PL(λ) = 1
            - ∆μ_s(v, λ): majority of scattering is caused by 1 contributor

    """
    # absorption_matrix
    sample_type = SampleType.BIOPSY2
    if isinstance(dataset, HelicoidDataset):
        sample_type = SampleType.HELICOID
    molecules = Molecules(
        left_cut=dataset.left_cut,
        right_cut=dataset.right_cut,
        molecule_mode=molecule_mode,
        sample_type=sample_type,
    )
    M = molecules.M

    delta_A = data["delta_A"]
    if delta_A is None:
        raise ValueError(
            "delta_A is None, please compute it first using dataset.compute_delta_A()"
        )
    logger.debug(delta_A.shape)
    x = dataset.cut_wavelengths
    sample_id = data["id"]

    # Make image into 1d x wl vector (Each element is a spectrum)
    b = delta_A.reshape(-1, delta_A.shape[-1])  # reshape to 1D x wavelength
    grid = None
    error_grid = None

    PL = prepare_pathlength_for_optimization(
        sample_id=sample_id,
        M=M,
        wavelengths=x,
        delta_A=delta_A,
        pathlength_mode=pathlength_mode,
        unitary_results_root=unitary_results_root,
        pathlength_debug_dir=pathlength_debug_dir,
        jacques_m_key=jacques_m_key,
        jacques_pickle_path=jacques_pickle_path,
        jacques_m_params_path=jacques_m_params_path,
    )

    if load_a_b_from_path is not None:
        # load a_t1, b_t1 from file
        a_t1 = np.load(load_a_b_from_path + "/a_t1.npy")
        b_t1 = np.load(load_a_b_from_path + "/b_t1.npy")
        logger.info(
            f"Loaded scattering parameters of reference spectrum (a,b): {a_t1}, {b_t1} from {load_a_b_from_path}"
        )
    else:
        ### brute force search over grid of possible scattering parameters ###
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brute.html#scipy.optimize.brute
        coarsened_delta_A = delta_A[::coarseness, ::coarseness, :]
        logger.debug(f"Coarsened delta_A shape: {coarsened_delta_A.shape}")
        b_coarsened = coarsened_delta_A.reshape(-1, coarsened_delta_A.shape[-1])
        if PL.ndim == 2:
            PL_img = PL.reshape(delta_A.shape[0], delta_A.shape[1], delta_A.shape[-1])
            PL_coarsened = PL_img[::coarseness, ::coarseness, :].reshape(-1, delta_A.shape[-1])
        else:
            PL_coarsened = PL
        result = scipy.optimize.brute(
            (optimize_image_parallel_error),
            (
                slice(0, MAX_A, MAX_A / A_STEPS),  # slice(0.1, MAX_A, MAX_A / A_STEPS),
                slice(0, MAX_B, MAX_B / B_STEPS),
            ),
            args=(b_coarsened, M, x, PL_coarsened),
            finish=None,
            full_output=True,
            workers=1,  # set to 1 since parallelization is already done in optimize_image_parallel, nested parallelism doesn't work
        )

        x_min = result[0]  # optimal a,b of reference spectrum
        a_t1, b_t1 = x_min

        error_min = result[1]
        grid = result[2]
        error_grid = result[3]

        logger.info(
            f"Optimal scattering parameters of reference spectrum (a,b): {x_min} with error: {error_min}"
        )

    ### compute the optimal concentrations and scattering parameters given the fixed scattering parameters of the reference spectrum ###
    error, coef_list, scattering_params_list, errors = optimize_image_parallel(
        (a_t1, b_t1),
        b,
        M,
        x,
        PL=PL,
    )

    # reshape results to original image shape
    coef_list = np.array(coef_list)
    coef_list = coef_list.reshape(
        delta_A.shape[0], delta_A.shape[1], coef_list.shape[-1]
    )
    scattering_params = np.array(scattering_params_list)
    scattering_params = scattering_params.reshape(
        delta_A.shape[0], delta_A.shape[1], scattering_params.shape[-1]
    )
    errors = np.array(errors)
    errors = errors.reshape(delta_A.shape[0], delta_A.shape[1])

    # return results
    # TODO: make into object?
    return (
        coef_list,
        scattering_params,
        errors,
        a_t1,
        b_t1,
        delta_A,
        data["reference_pixel"] if isinstance(dataset, HelicoidDataset) else None,
        grid,
        error_grid,
    )


def optimize_image_parallel_error(params_t1, B, M, x, PL=None):
    """
    Wrapper around optimize_image_parallel to only return the error, useful for scipy.optimize.brute etc.

    Args:
         params_t1 (float,float): (a_t1, b_t1) where a_t1 is the scattering coefficient of the reference spectrum and b_t1 is the scattering power of the reference spectrum
         *args: [B, M, x] where B is the flattened delta A image, M is the absorption matrix, x is the wavelengths

     Returns:
         float: mean error across all pixels
    """
    error, _, _, _ = optimize_image_parallel(params_t1, B, M, x, PL=PL)
    return error
