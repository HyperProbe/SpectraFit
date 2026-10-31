"""Forward models and fitting.

Differential (a reference spectrum is given), y = ΔA = −ln(R / R_ref):

    no_scattering    ΔA = PL · M Δc
    fixed_b          ΔA = PL · (M Δc + s · Δa · (λ/λ0)^−b)                 b fixed, Δa fitted
    free_scattering  ΔA = PL · (M Δc + s · [a (λ/λ0)^−b − a0 (λ/λ0)^−b0])  a ∈ [0, a_max], b ∈ [0, b_max]

Absolute (no reference), y = A = −ln R, concentrations ≥ 0:

    no_scattering    A = PL · M c + G
    fixed_b          A = PL · (M c + s · a (λ/λ0)^−b) + G
    free_scattering  same as fixed_b with b fitted

s is the scattering scale factor (10 in the research code), G an optional
constant offset. Units: M in cm⁻¹ per concentration unit, PL in cm.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np

from .solvers import solve_box

Progress = Optional[Callable[[float, str], None]]

GOLDEN = (np.sqrt(5) - 1) / 2


@dataclass
class Problem:
    wavelengths: np.ndarray
    M: np.ndarray
    names: List[str]
    mode: str
    absolute: bool
    offset: bool = True
    scattering_scale: float = 10.0
    lambda0: float = 500.0
    a_max: float = 100.0
    b_max: float = 5.0
    fixed_b: float = 1.0
    a0: float = 0.0
    b0: float = 0.0

    # ------------------------------------------------------------ parameters
    @property
    def K(self) -> int:
        return self.M.shape[1]

    @property
    def has_scatter(self) -> bool:
        return self.mode != "no_scattering"

    @property
    def has_offset(self) -> bool:
        return self.absolute and self.offset

    def linear_names(self) -> List[str]:
        names = list(self.names)
        if self.has_scatter:
            names.append("delta_a" if (self.mode == "fixed_b" and not self.absolute) else "a")
        if self.has_offset:
            names.append("offset")
        return names

    def bounds(self):
        K = self.K
        lo = [0.0 if self.absolute else -np.inf] * K
        hi = [np.inf] * K
        if self.has_scatter:
            if self.mode == "fixed_b" and not self.absolute:
                lo.append(-np.inf)
                hi.append(np.inf)
            else:
                lo.append(0.0)
                hi.append(self.a_max)
        if self.has_offset:
            lo.append(-np.inf)
            hi.append(np.inf)
        return np.array(lo), np.array(hi)

    # ------------------------------------------------------------ design
    def power(self, b) -> np.ndarray:
        """(λ/λ0)^−b; b scalar → (W,), b (N,) → (N,W)."""
        r = self.wavelengths / self.lambda0
        b = np.asarray(b, float)
        return r ** (-b) if b.ndim == 0 else r[None, :] ** (-b[:, None])

    def design(self, PL: np.ndarray, b=None) -> np.ndarray:
        """Design matrix: (W,P) if PL is (W,) and b scalar, else (N,W,P)."""
        cols = []
        per_pixel = PL.ndim == 2 or (b is not None and np.ndim(b) == 1)
        PLx = PL if PL.ndim == 2 else (PL[None, :] if per_pixel else PL)
        if per_pixel:
            cols.append(PLx[:, :, None] * self.M[None])
        else:
            cols.append(PLx[:, None] * self.M)
        if self.has_scatter:
            bb = self.fixed_b if self.mode == "fixed_b" else b
            p = self.power(bb)
            e = self.scattering_scale * PLx * p
            if per_pixel and e.ndim == 1:
                e = np.broadcast_to(e, (PLx.shape[0] if PLx.ndim == 2 else np.size(b), e.shape[-1]))
            cols.append(e[..., None])
        if self.has_offset:
            shape = cols[0].shape[:-1] + (1,)
            cols.append(np.ones(shape))
        if per_pixel:
            n = max(c.shape[0] for c in cols)
            cols = [np.broadcast_to(c, (n,) + c.shape[1:]) for c in cols]
        return np.concatenate(cols, axis=-1)

    def target(self, y: np.ndarray, PL: np.ndarray) -> np.ndarray:
        """Move the known reference scattering term to the measurement side (free mode, differential)."""
        if self.mode == "free_scattering" and not self.absolute and self.a0 != 0.0:
            return y + self.scattering_scale * self.a0 * PL * self.power(self.b0)
        return y

    def predict(self, beta: np.ndarray, PL: np.ndarray, b=None) -> np.ndarray:
        X = self.design(PL, b)
        pred = beta @ X.T if X.ndim == 2 else np.einsum("nwp,np->nw", X, beta)
        if self.mode == "free_scattering" and not self.absolute and self.a0 != 0.0:
            pred = pred - self.scattering_scale * self.a0 * PL * self.power(self.b0)
        return pred

    def jacobian(self, beta: np.ndarray, PL: np.ndarray, b=None) -> np.ndarray:
        """Jacobian of the model w.r.t. all fitted parameters (linear ones, then b in free mode)."""
        X = self.design(PL, b)
        if self.mode != "free_scattering":
            return X
        if X.ndim == 2:
            X = np.broadcast_to(X, (beta.shape[0],) + X.shape)
        ia = self.K
        lnr = np.log(self.wavelengths / self.lambda0)
        dXb = -X[:, :, ia] * lnr[None, :] * beta[:, ia][:, None]
        return np.concatenate([X, dXb[..., None]], axis=-1)


# ---------------------------------------------------------------- fitting
def _chunks(n: int, size: int):
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))


def _pl_rows(PL: np.ndarray, sl: slice) -> np.ndarray:
    return PL[sl] if PL.ndim == 2 else PL


def fit(problem: Problem, y: np.ndarray, PL: np.ndarray, method: str = "fast", b_step: float = 0.05,
        chunk_size: int = 20000, progress: Progress = None, refine_iterations: int = 22,
        uncertainties: bool = True) -> Dict[str, np.ndarray]:
    """Fit every row of y (N × W). Returns parameter arrays (N,), residual RMS, flags and standard errors."""
    y = np.asarray(y)
    N, W = y.shape
    lin_names = problem.linear_names()
    P = len(lin_names)
    beta = np.full((N, P), np.nan)
    bvals = np.full(N, np.nan) if problem.mode == "free_scattering" else None
    converged = np.zeros(N, bool)
    at_bound = np.zeros((N, P + (1 if bvals is not None else 0)), bool)
    rms = np.full(N, np.nan)
    se = np.full((N, P + (1 if bvals is not None else 0)), np.nan)
    finite = np.all(np.isfinite(y), axis=1)
    lo, hi = problem.bounds()

    chunks = list(_chunks(N, chunk_size))
    for ci, sl in enumerate(chunks):
        yy = y[sl]
        ok = finite[sl]
        if not ok.any():
            continue
        idx = np.flatnonzero(ok) + sl.start
        yk = yy[ok].astype(float)
        PLk = PL[idx] if PL.ndim == 2 else PL

        def sub_progress(f, msg, _ci=ci):
            if progress:
                progress((_ci + f) / len(chunks), msg)

        if method == "exact":
            res = _fit_exact(problem, yk, PLk, lo, hi, sub_progress)
        elif problem.mode == "free_scattering":
            res = _fit_free_fast(problem, yk, PLk, lo, hi, b_step, refine_iterations, sub_progress)
        else:
            yt = problem.target(yk, PLk)
            X = problem.design(PLk)
            bk, conv, ab = solve_box(X, yt, lo, hi)
            res = {"beta": bk, "converged": conv, "at_bound": ab, "b": None}

        bk = res["beta"]
        beta[idx] = bk
        converged[idx] = res["converged"]
        bk_b = res["b"]
        if bvals is not None:
            bvals[idx] = bk_b
            ab_b = (np.abs(bk_b) < 1e-6) | (np.abs(bk_b - problem.b_max) < 1e-6)
            at_bound[idx] = np.column_stack([res["at_bound"], ab_b])
        else:
            at_bound[idx] = res["at_bound"]
        pred = problem.predict(bk, PLk, bk_b)
        r = yk - pred
        rss = np.sum(r * r, axis=1)
        rms[idx] = np.sqrt(rss / W)
        if uncertainties:
            se[idx] = _standard_errors(problem, bk, PLk, bk_b, rss)
        sub_progress(1.0, "Fitting")

    out: Dict[str, np.ndarray] = {name: beta[:, i] for i, name in enumerate(lin_names)}
    se_names = lin_names + (["b"] if bvals is not None else [])
    if bvals is not None:
        out["b"] = bvals
    out["residual_rms"] = rms
    out["converged"] = converged
    out["at_bound"] = at_bound.any(axis=1)
    for i, name in enumerate(se_names):
        out[f"se_{name}"] = se[:, i]
    out["_param_names"] = np.array(se_names)
    return out


def _standard_errors(problem: Problem, beta, PL, b, rss) -> np.ndarray:
    J = problem.jacobian(beta, PL, b)
    n, W = beta.shape[0], problem.wavelengths.size
    Pp = J.shape[-1]
    dof = max(W - Pp, 1)
    sigma2 = rss / dof
    JtJ = J.T @ J if J.ndim == 2 else np.matmul(J.transpose(0, 2, 1), J)
    if JtJ.ndim == 2:
        cov = np.linalg.pinv(JtJ)
        d = np.broadcast_to(np.diag(cov), (n, Pp))
    else:
        cov = np.linalg.pinv(JtJ)
        d = np.diagonal(cov, axis1=1, axis2=2)
    return np.sqrt(np.clip(d * sigma2[:, None], 0, None))


def _cost(problem: Problem, beta, y, PL, b) -> np.ndarray:
    r = y - problem.predict(beta, PL, b)
    return np.sum(r * r, axis=1)


def _fit_free_fast(problem: Problem, y, PL, lo, hi, b_step, refine_iterations, progress):
    """Free scattering power: grid over b (shared design per grid point), then golden-section refinement per pixel."""
    n = y.shape[0]
    yt = problem.target(y, PL)
    grid = np.arange(0.0, problem.b_max + 1e-9, b_step)
    if grid[-1] < problem.b_max:
        grid = np.append(grid, problem.b_max)
    best_cost = np.full(n, np.inf)
    best_b = np.zeros(n)
    best_beta = np.zeros((n, len(lo)))
    best_conv = np.zeros(n, bool)
    best_ab = np.zeros((n, len(lo)), bool)
    for gi, bg in enumerate(grid):
        X = problem.design(PL, float(bg))
        beta, conv, ab = solve_box(X, yt, lo, hi)
        pred = beta @ X.T if X.ndim == 2 else np.einsum("nwp,np->nw", X, beta)
        c = np.sum((yt - pred) ** 2, axis=1)
        better = c < best_cost
        best_cost[better], best_b[better] = c[better], bg
        best_beta[better], best_conv[better], best_ab[better] = beta[better], conv[better], ab[better]
        if progress and gi % 10 == 0:
            progress(0.6 * gi / len(grid), "Scanning scattering power b")

    # golden-section search on [b - step, b + step] for each pixel
    left = np.clip(best_b - b_step, 0.0, problem.b_max)
    right = np.clip(best_b + b_step, 0.0, problem.b_max)

    def evaluate(bv, rows=None):
        """Cost (and solution) at per-pixel b values, optionally for a subset of pixels."""
        if rows is None:
            rows = np.arange(n)
        PLr = PL[rows] if PL.ndim == 2 else PL
        X = problem.design(PLr, bv)
        beta, conv, ab = solve_box(X, yt[rows], lo, hi)
        pred = np.einsum("nwp,np->nw", X, beta)
        return np.sum((yt[rows] - pred) ** 2, axis=1), beta, conv, ab

    x1 = right - GOLDEN * (right - left)
    x2 = left + GOLDEN * (right - left)
    f1 = evaluate(x1)[0]
    f2 = evaluate(x2)[0]
    for it in range(refine_iterations):
        go_left = f1 < f2
        right = np.where(go_left, x2, right)
        left = np.where(go_left, left, x1)
        # the surviving interior point is reused; only the new one is evaluated
        x1_new = np.where(go_left, right - GOLDEN * (right - left), x2)
        x2_new = np.where(go_left, x1, left + GOLDEN * (right - left))
        f1_new = np.where(go_left, 0.0, f2)
        f2_new = np.where(go_left, f1, 0.0)
        r1 = np.flatnonzero(go_left)
        r2 = np.flatnonzero(~go_left)
        if r1.size:
            f1_new[r1] = evaluate(x1_new[r1], r1)[0]
        if r2.size:
            f2_new[r2] = evaluate(x2_new[r2], r2)[0]
        x1, x2, f1, f2 = x1_new, x2_new, f1_new, f2_new
        if progress:
            progress(0.6 + 0.4 * (it + 1) / refine_iterations, "Refining scattering power b")
    b_ref = 0.5 * (left + right)
    c_ref, beta_ref, conv_ref, ab_ref = evaluate(b_ref)
    use = c_ref < best_cost
    best_b[use] = b_ref[use]
    best_cost[use] = c_ref[use]
    best_beta[use], best_conv[use], best_ab[use] = beta_ref[use], conv_ref[use], ab_ref[use]
    best_beta, best_b = _gauss_newton_polish(problem, yt, PL, best_beta, best_b, best_cost, lo, hi)
    return {"beta": best_beta, "converged": best_conv, "at_bound": best_ab, "b": best_b}


def _gauss_newton_polish(problem: Problem, yt, PL, beta, b, cost, lo, hi, iterations: int = 4):
    """A few joint Gauss–Newton steps on (β, b) from the grid/golden-section solution.

    b and some chromophores (water in the visible) are strongly coupled, so b must be
    known very precisely; Gauss–Newton converges quadratically near the optimum.
    Steps are only accepted where they lower the cost; parameters on a bound stay there.
    """
    lo_all = np.append(lo, 0.0)
    hi_all = np.append(hi, problem.b_max)
    x = np.column_stack([beta, b])
    for _ in range(iterations):
        J = problem.jacobian(x[:, :-1], PL, x[:, -1])  # (N,W,P+1)
        r = yt - _pred_target(problem, x[:, :-1], PL, x[:, -1])
        on_bound = (np.isfinite(lo_all) & (x <= lo_all + 1e-12)) | (np.isfinite(hi_all) & (x >= hi_all - 1e-12))
        J = np.where(on_bound[:, None, :], 0.0, J)
        JtJ = np.matmul(J.transpose(0, 2, 1), J)
        g = np.einsum("nwp,nw->np", J, r)
        d = np.sqrt(np.clip(np.diagonal(JtJ, axis1=1, axis2=2), 1e-300, None))
        A = JtJ / (d[:, :, None] * d[:, None, :]) + 1e-12 * np.eye(JtJ.shape[-1])
        A = A + np.where(on_bound, 1.0, 0.0)[:, :, None] * np.eye(JtJ.shape[-1])[None]
        try:
            step = np.linalg.solve(A, (g / d)[..., None])[..., 0] / d
        except np.linalg.LinAlgError:
            break
        step = np.where(on_bound, 0.0, step)
        x_new = np.clip(x + step, lo_all, hi_all)
        r_new = yt - _pred_target(problem, x_new[:, :-1], PL, x_new[:, -1])
        c_new = np.sum(r_new * r_new, axis=1)
        better = np.isfinite(c_new) & (c_new < cost)
        if not better.any():
            break
        x[better], cost[better] = x_new[better], c_new[better]
    return x[:, :-1], x[:, -1]


def _pred_target(problem: Problem, beta, PL, b):
    """Model prediction on the 'target' side (reference scattering term not subtracted)."""
    X = problem.design(PL, b)
    return np.einsum("nwp,np->nw", X, beta) if X.ndim == 3 else beta @ X.T


def _fit_exact(problem: Problem, y, PL, lo, hi, progress):
    """Pixel-by-pixel scipy.optimize.least_squares, as in the research code (slow; for small regions)."""
    from scipy.optimize import least_squares

    n = y.shape[0]
    P = len(lo)
    free = problem.mode == "free_scattering"
    lo_all = np.append(lo, 0.0) if free else lo
    hi_all = np.append(hi, problem.b_max) if free else hi
    x0 = np.zeros(len(lo_all))
    if problem.absolute:
        x0[: problem.K] = 1e-6
    if problem.has_scatter and not (problem.mode == "fixed_b" and not problem.absolute):
        x0[problem.K] = min(max(problem.a0, 0.0), problem.a_max)
    if free:
        x0[-1] = min(max(problem.b0, 0.0), problem.b_max)
    beta = np.zeros((n, P))
    bvals = np.zeros(n) if free else None
    conv = np.zeros(n, bool)
    for i in range(n):
        PLi = PL[i] if PL.ndim == 2 else PL

        def resid(x, i=i, PLi=PLi):
            bb = x[-1] if free else None
            return problem.predict(x[None, :P], PLi, bb) [0] - y[i]

        r = least_squares(resid, x0, bounds=(lo_all, hi_all))
        beta[i] = r.x[:P]
        if free:
            bvals[i] = r.x[-1]
        conv[i] = r.success
        if progress and i % 50 == 0:
            progress(i / n, f"Exact fit, pixel {i + 1}/{n}")
    lo_f = np.where(np.isfinite(lo), lo, 0.0)
    hi_f = np.where(np.isfinite(hi), hi, 0.0)
    ab = (np.isfinite(lo) & (np.abs(beta - lo_f) < 1e-7)) | (np.isfinite(hi) & (np.abs(beta - hi_f) < 1e-7))
    return {"beta": beta, "converged": conv, "at_bound": ab, "b": bvals}
