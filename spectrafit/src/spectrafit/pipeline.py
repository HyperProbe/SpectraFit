"""Run a complete analysis from a Config: load, reference, fit, derived maps, export."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from . import __version__
from .chromophores import (ChromophoreLibrary, absorption_matrix, collinearity_report, common_range,
                           read_chromophore_file)
from .config import Config
from .io import HyperspectralData, load_data
from .model import Problem, fit
from .parallel import resolve_cores, start_pool
from .pathlength import global_pathlength, mu_s_reduced, pixelwise_pathlength
from .preprocess import Reference, attenuation, build_mask, build_reference, select_wavelengths

log = logging.getLogger(__name__)
Progress = Optional[Callable[[float, str], None]]


class Cancelled(Exception):
    pass


@dataclass
class Results:
    config: Config
    maps: Dict[str, np.ndarray]
    """Every map has shape (T, H, W); NaN where not fitted."""
    labels: Dict[str, str]
    wavelengths: np.ndarray
    """Wavelengths used in the fit."""
    band_idx: np.ndarray
    chromophores: List[str]
    M: np.ndarray
    pathlength: np.ndarray
    """(W_fit,) shared or (T, H, W, W_fit) per pixel."""
    problem: Problem
    reference: Reference
    a0: float = 0.0
    b0: float = 0.0
    collinearity: dict = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)
    timing_s: float = 0.0
    grid: Optional[dict] = None

    @property
    def absolute(self) -> bool:
        return self.problem.absolute

    def pixel_fit(self, data: HyperspectralData, frame: int, row: int, col: int) -> Optional[dict]:
        """Measured and modelled attenuation of one pixel, with each component, for the inspector."""
        T, H, W = next(iter(self.maps.values())).shape
        if not (0 <= frame < T and 0 <= row < H and 0 <= col < W):
            return None
        y = attenuation(data, self.reference, self.band_idx, slice(frame, frame + 1))[0, row, col].astype(float)
        names = self.problem.linear_names()
        beta = np.array([self.maps[n][frame, row, col] for n in names])
        if not np.all(np.isfinite(beta)):
            return {"wavelengths": self.wavelengths, "measured": y, "fitted": None, "components": {}}
        PL = self.pathlength if self.pathlength.ndim == 1 else self.pathlength[frame, row, col]
        b = self.maps["b"][frame, row, col] if "b" in self.maps and self.problem.mode == "free_scattering" else None
        fitted = self.problem.predict(beta[None], PL, b)[0]
        comps = {}
        for k, n in enumerate(self.chromophores):
            comps[n] = PL * self.M[:, k] * beta[k]
        p = self.problem
        if p.has_scatter:
            bb = p.fixed_b if p.mode == "fixed_b" else b
            term = p.scattering_scale * PL * beta[p.K] * p.power(bb)
            if p.mode == "free_scattering" and not p.absolute:
                term = term - p.scattering_scale * PL * p.a0 * p.power(p.b0)
            comps["scattering"] = term
        if p.has_offset:
            comps["offset"] = np.full_like(y, beta[-1])
        return {"wavelengths": self.wavelengths, "measured": y, "fitted": fitted, "components": comps,
                "residual_rms": float(self.maps["residual_rms"][frame, row, col])}

    # ------------------------------------------------------------------ export
    def save(self, folder: str | Path | None = None, formats=None) -> List[Path]:
        folder = Path(folder or self.config.output.folder)
        folder.mkdir(parents=True, exist_ok=True)
        formats = formats or self.config.output.formats
        written: List[Path] = []
        safe = {k: _safe_name(k) for k in self.maps}
        written.append(self.config.save(folder / "settings.yaml"))
        meta = {
            "spectrafit_version": __version__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mode": self.problem.mode,
            "absolute": self.absolute,
            "chromophores": self.chromophores,
            "fit_wavelengths_nm": self.wavelengths.tolist(),
            "reference": self.reference.description,
            "reference_scattering_a0_b0": [self.a0, self.b0],
            "collinearity": self.collinearity,
            "notes": self.notes,
            "maps": {safe[k]: self.labels.get(k, k) for k in self.maps},
            "run_time_s": round(self.timing_s, 2),
        }
        p = folder / "fit_info.json"
        p.write_text(json.dumps(meta, indent=2, default=_json_default), encoding="utf-8")
        written.append(p)
        pl = self.pathlength if self.pathlength.ndim == 1 else np.nanmean(
            self.pathlength.reshape(-1, self.pathlength.shape[-1]), axis=0)
        p = folder / "pathlength_used.csv"
        np.savetxt(p, np.column_stack([self.wavelengths, pl]), delimiter=",",
                   header="wavelength_nm,pathlength_cm" + ("" if self.pathlength.ndim == 1 else " (mean over pixels)"),
                   comments="")
        written.append(p)
        written.append(self._summary_csv(folder / "summary.csv"))
        if "npz" in formats:
            p = folder / "results.npz"
            np.savez_compressed(p, wavelengths=self.wavelengths, **{safe[k]: v for k, v in self.maps.items()})
            written.append(p)
        if "mat" in formats:
            from scipy.io import savemat

            p = folder / "results.mat"
            savemat(p, {"wavelengths": self.wavelengths,
                        **{safe[k]: np.moveaxis(v, 0, -1) if v.shape[0] > 1 else v[0] for k, v in self.maps.items()}},
                    do_compression=True)
            written.append(p)
        if "tiff" in formats:
            import tifffile

            tdir = folder / "tiff"
            tdir.mkdir(exist_ok=True)
            for k, v in self.maps.items():
                p = tdir / f"{safe[k]}.tif"
                tifffile.imwrite(p, (v[0] if v.shape[0] == 1 else v).astype(np.float32))
                written.append(p)
        return written

    def _summary_csv(self, path: Path) -> Path:
        lines = ["map,label,frame,mean,std,median,p05,p95,n_pixels"]
        for k, v in self.maps.items():
            if v.dtype == bool:
                continue
            for t in range(v.shape[0]):
                x = v[t][np.isfinite(v[t])]
                if x.size == 0:
                    continue
                q = np.percentile(x, [5, 50, 95])
                lines.append(f"{k},\"{self.labels.get(k, k)}\",{t},{x.mean():.6g},{x.std():.6g},"
                             f"{q[1]:.6g},{q[0]:.6g},{q[2]:.6g},{x.size}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path


def _safe_name(k: str) -> str:
    s = re.sub(r"\W", "_", k)
    return s if s and not s[0].isdigit() else f"m_{s}"


def _json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------- run
def run(cfg: Config, data: Optional[HyperspectralData] = None, progress: Progress = None,
        should_stop: Optional[Callable[[], bool]] = None) -> Results:
    t_start = time.time()

    def report(frac, msg):
        if should_stop and should_stop():
            raise Cancelled("Stopped by user")
        if progress:
            progress(float(np.clip(frac, 0, 1)), msg)

    import multiprocessing

    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # We are inside a worker process that is still starting up: it is re-running the user's
        # script because the script lacks the  if __name__ == "__main__":  guard. Stop at once,
        # so the analysis is not repeated in every worker; the main process then uses one core.
        raise RuntimeError('SpectraFit: put the code of your script under  if __name__ == "__main__":')
    cfg.validate()
    if data is None:
        data = load_data(cfg.data, progress=lambda f, m: report(0.15 * f, m))

    # Several cores only pay off for the non-linear fits: starting the workers takes a second or two,
    # while the linear models finish in about a second on one core.
    n_cores = resolve_cores(cfg.solver.cores)
    heavy = cfg.model.mode == "free_scattering" or cfg.solver.method == "exact"
    n_pixels = int(np.prod(data.cube.shape[:3]))
    pool, pool_note = None, None
    if heavy and n_cores > 1 and n_pixels >= 4000:
        report(0.16, f"Starting {n_cores} worker processes")
        try:
            pool = start_pool(n_cores)
            pool_note = f"Fitted on {n_cores} processor cores."
        except Exception:  # e.g. called from a script without the  if __name__ == "__main__":  guard
            log.warning("Could not start worker processes; fitting on one core.", exc_info=True)
            pool_note = ("Could not start worker processes, so one core was used. When calling SpectraFit "
                         "from your own Python script, put the code under  if __name__ == \"__main__\":")
    try:
        results = _run(cfg, data, report, pool)
    finally:
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)
    results.timing_s = time.time() - t_start
    if pool_note:
        results.notes.append(pool_note)
    report(1.0, f"Done in {results.timing_s:.1f} s")
    return results


def _run(cfg: Config, data: HyperspectralData, report, pool) -> Results:
    notes: List[str] = list(data.notes)

    # ---- chromophores and wavelengths
    lib = ChromophoreLibrary(extra_files=cfg.chromophores.extra_files)
    extra_names = [read_chromophore_file(f).name for f in cfg.chromophores.extra_files]
    names = list(dict.fromkeys(list(cfg.chromophores.names) + extra_names))
    chroms = [lib.get(n) for n in names]
    for n, v in cfg.chromophores.baselines.items():
        if n in names:
            chroms[names.index(n)].baseline = float(v)
    band_idx, wl_notes = select_wavelengths(data.wavelengths, cfg.spectral, common_range(chroms))
    notes += wl_notes
    wl = data.wavelengths[band_idx]
    M = absorption_matrix(chroms, wl)
    coll = collinearity_report(M, names)
    if coll["level"] != "ok":
        a, b, s = coll["most_similar"]
        notes.append(f"Warning: spectra are hard to separate in this range (condition number "
                     f"{coll['condition_number']:.3g}; {a} and {b} similarity {s:.3f}). Results for them may be unreliable.")
    baselines = np.array([c.baseline for c in chroms])

    # ---- reference, attenuation, mask
    reference = build_reference(data, cfg.reference)
    absolute = reference.type == "none"
    y_img = attenuation(data, reference, band_idx)  # (T,H,W,Wf)
    mask = build_mask(data, cfg.mask, band_idx)
    if reference.type == "frame":
        notes.append("Time series: frames used as the reference are fitted too (they give ΔA ≈ 0).")
    T, H, W, Wf = y_img.shape
    y = y_img.reshape(-1, Wf)
    valid = mask.reshape(-1) & np.all(np.isfinite(y), axis=1)
    y = np.where(valid[:, None], y, np.nan)
    report(0.17, "Preparing fit")

    # ---- model
    m = cfg.model
    problem = Problem(wavelengths=wl, M=M, names=names, mode=m.mode, absolute=absolute,
                      offset=m.absolute_offset, scattering_scale=m.scattering_scale, lambda0=m.lambda0,
                      a_max=m.a_max, b_max=m.b_max, fixed_b=m.fixed_b)
    PL = global_pathlength(cfg.pathlength, wl, M, baselines)
    pixelwise = cfg.pathlength.mode.endswith("_pixelwise")
    grid_info = None
    s = cfg.solver

    # ---- reference scattering a0, b0 (free scattering with a reference)
    if m.mode == "free_scattering" and not absolute:
        rs = m.reference_scattering
        if rs.method == "manual":
            problem.a0, problem.b0 = float(rs.a0), float(rs.b0)
        else:
            problem.a0, problem.b0, grid_info = _grid_search_reference(
                problem, y_img, mask, PL, rs, s, lambda f, msg: report(0.17 + 0.23 * f, msg), pool)
            notes.append(f"Reference scattering from grid search: a0 = {problem.a0:.4g}, b0 = {problem.b0:.4g}.")

    # ---- fit (pass 1)
    span = (0.40, 0.65) if pixelwise else (0.40, 0.95)
    res = fit(problem, y, PL, method=s.method, b_step=s.b_step, chunk_size=s.chunk_size, executor=pool,
              progress=lambda f, msg: report(span[0] + (span[1] - span[0]) * f,
                                             msg if not pixelwise else f"{msg} (pass 1 of 2)"))

    # ---- pass 2 with per-pixel pathlength
    if pixelwise:
        mu_a, mu_s = _pixel_optics(problem, res, baselines, cfg)
        ok = np.all(np.isfinite(mu_a), axis=1) & np.all(np.isfinite(mu_s), axis=1)
        PLp = np.ones((y.shape[0], Wf))
        if ok.any():
            PLp[ok] = pixelwise_pathlength(cfg.pathlength, wl, M, mu_a[ok], mu_s[ok])
        res = fit(problem, y, PLp, method=s.method, b_step=s.b_step, chunk_size=s.chunk_size, executor=pool,
                  progress=lambda f, msg: report(0.65 + 0.30 * f, f"{msg} (pass 2 of 2, per-pixel pathlength)"))
        PL_out = PLp.reshape(T, H, W, Wf).astype(np.float32)
        notes.append("Per-pixel pathlength from a first fit with unit pathlength (two passes).")
    else:
        PL_out = PL

    # ---- data-driven warnings
    notes.append(reference.description)
    for n, base in zip(names, baselines):  # is each chromophore actually determined by the data?
        se = np.nanmedian(res[f"se_{n}"])
        if not np.isfinite(se) or se <= 0:
            continue
        if base > 0 and se > 0.1 * base:
            notes.append(f"Warning: {n} is poorly determined in this range: its standard error ({se:.2g}) is "
                         f"{100 * se / base:.0f}% of its typical tissue concentration ({base:g}). "
                         "Consider a wider wavelength range or leaving it out.")
        elif base <= 0 and absolute:
            est = np.nanmedian(np.abs(res[n]))
            if se > 0.5 * max(est, 1e-30):
                notes.append(f"Warning: {n} is poorly determined (median standard error {se:.2g}, "
                             f"median value {est:.2g}).")
    n_fail = int((~res["converged"] & valid).sum())
    if n_fail:
        notes.append(f"{n_fail} pixels did not converge; they are marked in the 'flag' map.")

    # ---- maps
    maps, labels = _make_maps(problem, res, names, chroms, baselines, (T, H, W), mask, cfg)
    results = Results(config=cfg, maps=maps, labels=labels, wavelengths=wl, band_idx=band_idx,
                      chromophores=names, M=M, pathlength=PL_out, problem=problem, reference=reference,
                      a0=problem.a0, b0=problem.b0, collinearity=coll, notes=notes, grid=grid_info)
    return results


def _grid_point_cost(problem: Problem, sub, PL, b_step, chunk_size) -> float:
    """Mean least-squares cost of all pixels for one (a0, b0) of the reference (runs in a worker)."""
    r = fit(problem, sub, PL, method="fast", b_step=b_step, chunk_size=chunk_size,
            refine_iterations=8, uncertainties=False)
    cost = 0.5 * (r["residual_rms"] ** 2) * sub.shape[1]  # = scipy least_squares 'cost'
    return float(np.nanmean(cost))


def _grid_search_reference(problem: Problem, y_img, mask, PL, rs, solver, report, pool=None):
    """Grid search for the reference scattering a0, b0, as in the research code (scipy.optimize.brute):
    for every (a0, b0) all pixels are fitted, and the pair with the lowest mean cost wins."""
    step = max(1, int(rs.grid_binning))
    sub = y_img[:, ::step, ::step].reshape(-1, y_img.shape[-1])
    msub = mask[:, ::step, ::step].reshape(-1)
    sub = sub[msub & np.all(np.isfinite(sub), axis=1)].astype(float)
    if sub.shape[0] == 0:
        raise ValueError("No pixels available for the reference scattering grid search.")
    a_grid = np.arange(0.0, problem.a_max, problem.a_max / max(1, rs.a_steps))
    b_grid = np.arange(0.0, problem.b_max, problem.b_max / max(1, rs.b_steps))
    err = np.zeros((a_grid.size, b_grid.size))
    total = err.size
    b_step = max(solver.b_step, 0.1)
    points = [(i, j, replace(problem, a0=float(a0), b0=float(b0)))
              for i, a0 in enumerate(a_grid) for j, b0 in enumerate(b_grid)]
    if pool is not None:
        futures = {pool.submit(_grid_point_cost, pr, sub, PL, b_step, solver.chunk_size): (i, j) for i, j, pr in points}
        try:
            for done, fut in enumerate(as_completed(futures), 1):
                err[futures[fut]] = fut.result()
                report(done / total, f"Reference scattering grid search ({done}/{total})")
        except BaseException:
            for f in futures:
                f.cancel()
            raise
    else:
        for done, (i, j, pr) in enumerate(points, 1):
            err[i, j] = _grid_point_cost(pr, sub, PL, b_step, solver.chunk_size)
            report(done / total, f"Reference scattering grid search ({done}/{total})")
    i, j = np.unravel_index(np.nanargmin(err), err.shape)
    return float(a_grid[i]), float(b_grid[j]), {"a": a_grid, "b": b_grid, "mean_cost": err}


def _pixel_optics(problem: Problem, res, baselines, cfg: Config):
    """Per-pixel μa and μs' from a first-pass fit, for the per-pixel pathlength models."""
    wl = problem.wavelengths
    C = np.column_stack([res[n] for n in problem.names])
    c_abs = C if problem.absolute else baselines[None, :] + C
    c_abs = np.clip(c_abs, 1e-9, None)
    mu_a = np.clip(c_abs @ problem.M.T, 1e-9, None)
    t = cfg.pathlength.tissue
    if problem.mode == "free_scattering":
        a, b = res["a"], res["b"]
    elif problem.mode == "fixed_b":
        a = res["a"] if problem.absolute else t.a + res["delta_a"]
        b = np.full_like(a, problem.fixed_b)
    else:
        a = np.full(C.shape[0], t.a)
        b = np.full(C.shape[0], t.b)
    mu_s = np.clip(mu_s_reduced(wl, np.asarray(a), np.asarray(b), problem.lambda0), 1e-9, None)
    return mu_a, mu_s


def _make_maps(problem: Problem, res, names, chroms, baselines, shape, mask, cfg: Config):
    T, H, W = shape
    maps: Dict[str, np.ndarray] = {}
    labels: Dict[str, str] = {}
    pre = "" if problem.absolute else "Δ"

    def put(key, arr, label):
        maps[key] = np.asarray(arr, np.float32).reshape(T, H, W)
        labels[key] = label

    for c in chroms:
        unit = "mM" if c.concentration_unit == "mM" else "volume fraction"
        put(c.name, res[c.name], f"{pre}{c.name} [{unit}]")
    if problem.mode == "fixed_b":
        if problem.absolute:
            put("a", res["a"], "Scattering amplitude a")
        else:
            put("delta_a", res["delta_a"], f"Δa (scattering amplitude change, b = {problem.fixed_b:g})")
    elif problem.mode == "free_scattering":
        put("a", res["a"], "Scattering amplitude a")
        put("b", res["b"], "Scattering power b")
        if not problem.absolute:
            put("delta_a", res["a"] - problem.a0, f"Δa = a − a0 (a0 = {problem.a0:.3g})")
            put("delta_b", res["b"] - problem.b0, f"Δb = b − b0 (b0 = {problem.b0:.3g})")
    if problem.has_offset:
        put("offset", res["offset"], "Offset G")

    # derived maps
    if "HbO2" in names and "HHb" in names:
        put("HbT", res["HbO2"] + res["HHb"], f"{pre}HbT = {pre}HbO2 + {pre}HHb [mM]")
        bo, bh = baselines[names.index("HbO2")], baselines[names.index("HHb")]
        with np.errstate(all="ignore"):
            if problem.absolute:
                st = 100 * res["HbO2"] / (res["HbO2"] + res["HHb"])
                put("StO2", st, "StO₂ [%]")
            else:
                st = 100 * (bo + res["HbO2"]) / (bo + bh + res["HbO2"] + res["HHb"])
                put("StO2", st, f"StO₂ [%] assuming baseline HbO2 = {bo:g} mM, HHb = {bh:g} mM")
    if "oxCCO" in names and "redCCO" in names:
        put("diffCCO", res["oxCCO"] - res["redCCO"], f"{pre}diffCCO = {pre}oxCCO − {pre}redCCO [mM]")

    put("residual_rms", res["residual_rms"], "Fit residual (RMS of attenuation)")
    for n in res["_param_names"]:
        put(f"se_{n}", res[f"se_{n}"], f"Standard error of {n}")
    flag = np.zeros(T * H * W, np.float32)
    valid = mask.reshape(-1)
    flag[res["at_bound"]] = 1
    flag[~res["converged"] & valid] = 2
    flag[~valid] = np.nan
    put("flag", flag, "Flag: 0 ok, 1 a parameter at its limit, 2 not converged")
    for k in maps:
        if k != "flag":
            maps[k][~mask] = np.nan
    return maps, labels
