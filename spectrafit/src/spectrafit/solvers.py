"""Batched bounded linear least squares.

Solves, for every pixel n independently,

    minimise ||y_n - X_n β||²   subject to  lo ≤ β ≤ hi

for many pixels at once, working on the normal equations G = XᵀX, h = Xᵀy
(G may be shared by all pixels or differ per pixel). Columns are rescaled to
unit norm first, which keeps the normal equations well conditioned.

Bounds are handled with an active-set method (variables are moved onto or
released from their bounds one at a time until the Karush–Kuhn–Tucker
conditions hold). It gives the exact constrained optimum; the few pixels that
do not converge within the iteration limit are re-solved with
``scipy.optimize.lsq_linear``.
"""

from __future__ import annotations

import numpy as np

RIDGE = 1e-12


def _solve(G: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Batched solve of G β = h. G: (N,P,P) or (P,P); h: (N,P)."""
    P = h.shape[-1]
    if G.ndim == 2:
        G = G + RIDGE * np.eye(P)
        try:
            return np.linalg.solve(G, h.T).T
        except np.linalg.LinAlgError:
            return (np.linalg.pinv(G) @ h.T).T
    G = G + RIDGE * np.eye(P)[None]
    try:
        return np.linalg.solve(G, h[..., None])[..., 0]
    except np.linalg.LinAlgError:
        return np.einsum("npq,nq->np", np.linalg.pinv(G), h)


def solve_box_gram(G: np.ndarray, h: np.ndarray, lo: np.ndarray, hi: np.ndarray, max_iter: int | None = None):
    """Bounded least squares in normal-equation form.

    Parameters
    ----------
    G : (P,P) or (N,P,P)   XᵀX
    h : (N,P)              Xᵀy
    lo, hi : (P,)          bounds (may be ±inf)

    Returns
    -------
    beta : (N,P)
    converged : (N,) bool
    at_bound : (N,P) bool   variable sits on one of its bounds
    """
    h = np.asarray(h, float)
    N, P = h.shape
    lo = np.broadcast_to(np.asarray(lo, float), (P,))
    hi = np.broadcast_to(np.asarray(hi, float), (P,))

    # --- column scaling
    diag = np.diagonal(G, axis1=-2, axis2=-1)
    s = np.sqrt(np.where(diag > 0, diag, 1.0))  # (P,) or (N,P)
    if G.ndim == 2:
        Gs = G / np.outer(s, s)
        hs = h / s
        los, his = lo * s, hi * s
        los = np.broadcast_to(los, (N, P))
        his = np.broadcast_to(his, (N, P))
    else:
        Gs = G / (s[:, :, None] * s[:, None, :])
        hs = h / s
        los, his = lo[None] * s, hi[None] * s

    beta = _solve(Gs, hs)
    bounded = np.isfinite(lo) | np.isfinite(hi)
    if not bounded.any():
        return beta / s, np.ones(N, bool), np.zeros((N, P), bool)

    status = np.zeros((N, P), np.int8)  # 0 free, -1 at lower, +1 at upper
    converged = np.zeros(N, bool)
    # pixels already feasible are done (unconstrained optimum satisfies the bounds)
    feasible = np.all((beta >= los - 1e-12) & (beta <= his + 1e-12), axis=1)
    converged |= feasible
    max_iter = max_iter or (4 * P + 10)
    eye = np.eye(P)

    for _ in range(max_iter):
        act = np.flatnonzero(~converged)
        if act.size == 0:
            break
        Gi = Gs if Gs.ndim == 2 else Gs[act]
        hi_ = hs[act]
        st = status[act]
        lo_a, hi_a = los[act], his[act]
        fixed_val = np.where(st == -1, lo_a, np.where(st == 1, hi_a, 0.0))
        free = st == 0
        # system for free variables, identity rows for fixed ones
        Gfull = np.broadcast_to(Gi, (act.size, P, P)) if Gi.ndim == 2 else Gi
        rhs = hi_ - np.einsum("npq,nq->np", Gfull, fixed_val)
        Gm = np.where(free[:, :, None] & free[:, None, :], Gfull, 0.0) + np.where(~free, 1.0, 0.0)[:, :, None] * eye[None]
        rhs = np.where(free, rhs, fixed_val)
        b = _solve(Gm, rhs)
        # 1) feasibility: fix the most violating free variable of each pixel
        viol = np.where(free, np.maximum(lo_a - b, 0) + np.maximum(b - hi_a, 0), 0.0)
        has_viol = viol.max(axis=1) > 1e-12
        if has_viol.any():
            rows = np.flatnonzero(has_viol)
            j = np.argmax(viol[rows], axis=1)
            side = np.where(b[rows, j] < lo_a[rows, j], -1, 1).astype(np.int8)
            st[rows, j] = side
        # 2) optimality for feasible pixels: release the bound variable with the wrong gradient sign
        ok = np.flatnonzero(~has_viol)
        if ok.size:
            bo = b[ok]
            grad = np.einsum("npq,nq->np", Gfull[ok], bo) - hi_[ok]
            sto = st[ok]
            tol = 1e-10 * (1.0 + np.abs(hi_[ok]).max(axis=1, keepdims=True))
            wrong = np.where(sto == -1, np.maximum(-grad - tol, 0), 0.0) + np.where(sto == 1, np.maximum(grad - tol, 0), 0.0)
            release = wrong.max(axis=1) > 0
            if release.any():
                r = np.flatnonzero(release)
                j = np.argmax(wrong[r], axis=1)
                sto[r, j] = 0
                st[ok] = sto
            done = ok[~release]
            beta[act[done]] = bo[~release]
            converged[act[done]] = True
        status[act] = st

    left = np.flatnonzero(~converged)
    if left.size:
        from scipy.optimize import lsq_linear

        for n in left:
            Gn = Gs if Gs.ndim == 2 else Gs[n]
            L = np.linalg.cholesky(Gn + 1e-10 * np.eye(P))
            rhs = np.linalg.solve(L, hs[n])
            res = lsq_linear(L.T, rhs, bounds=(los[n], his[n]), method="bvls")
            beta[n] = res.x
        converged[left] = True  # solved by the fallback

    beta = np.clip(beta, los, his)
    out = beta / s
    lo_f = np.where(np.isfinite(lo), lo, 0.0)
    hi_f = np.where(np.isfinite(hi), hi, 0.0)
    at_bound = (np.isfinite(lo) & (np.abs(out - lo_f) <= 1e-9 * (1 + np.abs(lo_f)))) | (
        np.isfinite(hi) & (np.abs(out - hi_f) <= 1e-9 * (1 + np.abs(hi_f))))
    return out, converged, at_bound


def solve_box(X: np.ndarray, y: np.ndarray, lo, hi):
    """Bounded least squares from the design matrix. X: (W,P) shared or (N,W,P); y: (N,W)."""
    if X.ndim == 2:
        G = X.T @ X
        h = y @ X
    else:
        G = np.matmul(X.transpose(0, 2, 1), X)
        h = np.einsum("nwp,nw->np", X, y)
    return solve_box_gram(G, h, lo, hi)
