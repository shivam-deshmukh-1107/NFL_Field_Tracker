"""Stage 5a - recover the yard lattice from a pencil of parallel field lines.

The problem
-----------
Yard lines are equally spaced on the ground but not in the image, some are
missing (occluded by players or the HUD), and a few detections are spurious.
Fitting consecutive indices directly fails because we do not know which yard
each detected line is.

The trick
---------
Intersect the pencil with a transversal to get 1-D positions ``s_i``.  If ``v``
is where that transversal meets the horizon, then

    u = 1 / (s - v)

is an **affine** function of ground distance.  Equally spaced ground positions
therefore become an equally spaced lattice in ``u``.  So we only have to search
two scalars - the horizon crossing ``v`` and the lattice pitch ``alpha`` - and
score each hypothesis with a circular resultant ("comb" score):

    R(alpha) = | sum_j exp(2*pi*i*u_j / alpha) | / n

``R`` is 1 when every point sits on the lattice and falls off gracefully as
points go missing, which is exactly the robustness we need.  Missing lines cost
nothing, and outliers only dilute the score.

Because a lattice of pitch ``alpha`` also scores well at ``alpha/2``,
``alpha/3`` and so on, we take the **largest** pitch whose score is within a
small tolerance of the best - the coarsest lattice that explains the data.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import geometry as G


@dataclass
class LatticeFit:
    v: float            # transversal coordinate of the horizon crossing
    alpha: float        # lattice pitch in u-space
    beta: float         # lattice phase in u-space
    score: float        # circular resultant in [0, 1]
    index: np.ndarray   # integer lattice index per input position
    resid: np.ndarray   # |index - real-valued index|, in lattice units
    n_in: int           # positions within tol of an integer index


def transversal_positions(segs: np.ndarray, ell: np.ndarray,
                          lo: float = -4000.0, hi: float = 6000.0):
    """Intersect each segment's line with the transversal ``ell``.

    Returns ``(s, keep)`` where ``s`` are the 1-D coordinates along ``ell``
    (its y coordinate for a vertical transversal) of the finite intersections.
    """
    if len(segs) == 0:
        return np.zeros(0), np.zeros(0, bool)
    L = G.seg_to_line(np.asarray(segs, float))
    P = np.cross(L, ell)
    w = P[:, 2]
    ok = np.abs(w) > 1e-9
    s = np.full(len(segs), np.nan)
    # For a vertical transversal the informative coordinate is y; for a
    # horizontal one it is x.  Pick whichever varies along ell.
    coord = 1 if abs(ell[0]) > abs(ell[1]) else 0
    s[ok] = P[ok, coord] / w[ok]
    keep = ok & np.isfinite(s) & (s > lo) & (s < hi)
    return s, keep


def cluster_1d(v: np.ndarray, tol: float = 7.0) -> np.ndarray:
    """Merge positions that are really the same line detected twice."""
    if len(v) == 0:
        return v
    v = np.sort(v)
    out, cur = [], [v[0]]
    for x in v[1:]:
        if x - cur[-1] <= tol:
            cur.append(x)
        else:
            out.append(float(np.mean(cur)))
            cur = [x]
    out.append(float(np.mean(cur)))
    return np.asarray(out)


def _comb_score(u: np.ndarray, alpha: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Circular resultant of ``u`` at each pitch in ``alpha`` (vectorised).

    Real ``cos``/``sin`` rather than ``exp(1j*phase)``: the complex exponential
    computes the same two values but through a slower path and into an array of
    twice the width, and this is evaluated a few hundred thousand times per
    frame - enough to be the single largest cost in a render.
    """
    phase = 2.0 * np.pi * (u[None, :] / alpha[:, None])
    c = np.cos(phase).mean(axis=1)
    s = np.sin(phase).mean(axis=1)
    return np.hypot(c, s), np.arctan2(s, c)


def fit(s: np.ndarray, n_alpha: int = 900, n_v: int = 220,
        max_steps: int = 70, min_steps: int = 4, tol: float = 0.14,
        pitch_tol: float = 0.02) -> LatticeFit | None:
    """Fit ``(v, alpha)`` to 1-D pencil positions.

    ``max_steps`` bounds how many lattice cells may span the observed data,
    keeping the search away from absurdly fine lattices that fit anything.
    ``min_steps`` is equally important in the other direction: a pitch wider
    than the data span puts every point in one cell and scores a perfect 1.0,
    so without a floor the "coarsest good pitch" rule always degenerates.
    """
    s = np.asarray(s, float)
    s = s[np.isfinite(s)]
    if len(s) < 5:
        return None
    s = np.sort(s)
    span = s[-1] - s[0]
    if span <= 1e-6:
        return None

    # Lines crowd together towards the horizon, so the horizon crossing lies
    # beyond the end with the smaller gaps.  That is a physical fact about
    # perspective, not something to be discovered by scoring: left to choose
    # freely the fitter sometimes selects the inverted branch, which puts the
    # horizon below the image and silently reverses the yard axis.  Decide the
    # branch from the gap trend, and only fall back to searching both when the
    # trend is too weak to call.
    gaps = np.diff(s)
    signs = (+1.0, -1.0)
    if len(gaps) >= 6:
        k = max(2, len(gaps) // 3)
        lo_gap = float(np.median(gaps[:k]))
        hi_gap = float(np.median(gaps[-k:]))
        if max(lo_gap, hi_gap) > 0:
            ratio = max(lo_gap, hi_gap) / max(min(lo_gap, hi_gap), 1e-9)
            if ratio >= 1.3:
                # dense end = smaller gaps; horizon lies beyond it
                signs = (+1.0,) if lo_gap < hi_gap else (-1.0,)

    best = None
    unit_alpha = np.geomspace(1.0 / max_steps, 1.0 / min_steps, n_alpha)
    for sign in signs:
        t = s if sign > 0 else -s[::-1]
        # Candidate horizon crossings beyond the dense end.  The offset floor
        # matters: as v approaches the data, 1/(t-v) blows up for the nearest
        # point, the u-span becomes dominated by that one sample, and every
        # other point collapses into a single lattice cell.
        v_cand = t[0] - np.geomspace(0.15 * span, 60.0 * span, n_v)
        for v in v_cand:
            u = 1.0 / (t - v)
            uspan = u.max() - u.min()
            if uspan <= 0:
                continue
            alpha = uspan * unit_alpha
            R, ang = _comb_score(u, alpha)
            k = int(np.argmax(R))
            if R[k] < 0.5:
                continue
            # coarsest pitch that still explains the data
            good = np.flatnonzero(R >= R[k] - pitch_tol)
            k = int(good[-1])
            a, ph = float(alpha[k]), float(ang[k])
            beta = -ph / (2.0 * np.pi) * a
            idx_real = (u - beta) / a
            idx = np.round(idx_real)
            resid = np.abs(idx_real - idx)
            inl = resid <= tol
            if inl.sum() < 4:
                continue
            iin = idx[inl]
            if np.ptp(iin) < min_steps:          # degenerate: all one cell
                continue
            # Reward distinct lattice cells explained; penalise collisions (two
            # image lines cannot be the same yard line) and penalise sparsely
            # populated lattices, which is how an over-fine pitch fits anything
            # by leaving most cells empty.
            n_distinct = len(np.unique(iin))
            span_cells = int(np.ptp(iin)) + 1
            quality = (n_distinct
                       - 0.5 * (int(inl.sum()) - n_distinct)
                       - 0.15 * (span_cells - n_distinct))
            cand = (quality, float(R[k]), sign, v, a, beta, idx, resid,
                    int(inl.sum()))
            if best is None or (quality, R[k]) > (best[0], best[1]):
                best = cand
    if best is None:
        return None

    quality, score, sign, v, a, beta, idx, resid, n_in = best
    if sign < 0:                       # undo the flip
        v = -v
        idx = idx[::-1]
        resid = resid[::-1]
        a = -a
        beta = -beta
    return LatticeFit(float(v), float(a), float(beta), float(score),
                      idx.astype(int), resid, n_in)


def refine(s: np.ndarray, f: LatticeFit, tol: float = 0.14) -> LatticeFit:
    """Least-squares refit of ``(alpha, beta)`` on inliers, ``v`` held fixed."""
    u = 1.0 / (np.sort(np.asarray(s, float)) - f.v)
    idx_real = (u - f.beta) / f.alpha
    idx = np.round(idx_real)
    inl = np.abs(idx_real - idx) <= tol
    if inl.sum() < 3:
        return f
    A = np.stack([idx[inl], np.ones(int(inl.sum()))], 1)
    (a, b), *_ = np.linalg.lstsq(A, u[inl], rcond=None)
    idx_real = (u - b) / a
    idx = np.round(idx_real)
    resid = np.abs(idx_real - idx)
    return LatticeFit(f.v, float(a), float(b), f.score, idx.astype(int),
                      resid, int((resid <= tol).sum()))


def fit_labelled(s: np.ndarray, k: np.ndarray, v_pred: float | None = None,
                 n_v: int = 500, span_lo: float = 0.6, span_hi: float = 200.0,
                 prior: float = 0.02, gap_ratio: float = 1.15) -> LatticeFit | None:
    """Fit ``(v, alpha, beta)`` when the lattice index of each position is known.

    Tracking already knows which yard line is which, so the lattice is
    over-determined: the pitch does not have to be *searched* for, it follows
    from least squares once the horizon crossing is chosen.  That removes the
    comb score, and with it the collision/sparsity heuristics that only exist
    to keep an unlabelled search honest.

    What remains is that ``v`` is badly conditioned on its own.  Pushing the
    horizon further away and stretching the pitch to match leaves the predicted
    line positions almost unchanged, so the objective is nearly flat along that
    ridge.  Measured on the clips, the cold fit hops along it: ``v`` moved by up
    to 111 px between neighbouring frames, and the pitch by up to 58%, while the
    lattice those two numbers describe did not move at all - the span of drawn
    yard labels changed by 0.0 yd.  Anything derived from ``(v, alpha)`` rather
    than from the labelled lines - the tick columns above all - inherits that
    jitter.

    Two changes make it stable.  The search runs in ``g = 1 / (s0 - v)``, which
    is bounded and tends to zero as the horizon recedes, so a grid over it
    spends its resolution where the fit is actually sensitive; and near-ties are
    broken towards the previous frame's value.  The prior is quadratic, so a
    genuine change in geometry is nearly free while a hop along the flat ridge
    is not.
    """
    s = np.asarray(s, float)
    k = np.asarray(k, float)
    m = np.isfinite(s) & np.isfinite(k)
    s, k = s[m], k[m]
    if len(s) < 4 or len(np.unique(k)) < 3:
        return None
    o = np.argsort(s)
    s, k = s[o], k[o]
    span = float(s[-1] - s[0])
    if span <= 1e-6:
        return None
    s0 = float(np.median(s))

    # Same physical branch rule as `fit`: lines crowd towards the horizon, so
    # it lies beyond the end whose gaps are smaller.
    gaps = np.diff(s)
    j = max(1, len(gaps) // 3)
    lo, hi = float(np.median(gaps[:j])), float(np.median(gaps[-j:]))
    signs = (-1.0, +1.0)
    if min(lo, hi) > 0 and max(lo, hi) / min(lo, hi) >= gap_ratio:
        signs = (-1.0,) if lo < hi else (+1.0,)

    kc = k - k.mean()
    kk = float(kc @ kc)
    if kk <= 1e-12:
        return None
    n_pts = len(k)
    g_scale = 1.0 / (span_lo * span)
    g_pred = (1.0 / (s0 - v_pred)
              if (v_pred is not None and np.isfinite(v_pred)
                  and abs(s0 - v_pred) > 1e-9) else None)

    best = None
    for sg in signs:
        v = s0 + sg * np.geomspace(span_lo * span, span_hi * span, n_v)
        d = s[None, :] - v[:, None]
        ok = np.abs(d).min(axis=1) > 1e-6
        if not ok.any():
            continue
        v, d = v[ok], d[ok]
        u = 1.0 / d
        a = (u @ kc) / kk
        good = np.abs(a) > 1e-15
        if not good.any():
            continue
        v, u, a = v[good], u[good], a[good]
        b = u.mean(axis=1) - a * k.mean()
        r = u - (a[:, None] * k[None, :] + b[:, None])
        # Median, not RMS.  One line carrying the wrong yard index is a 1-cell
        # error where the others are hundredths, and a squared cost lets it drag
        # the whole fit: measured in-sample residuals of 5-6 px on clip 1, where
        # the lattice could not reproduce the positions of the very lines it was
        # fitted to.  A median cost simply ignores it.
        resid = np.median(np.abs(r), axis=1) / np.abs(a)
        cost = resid.copy()
        if g_pred is not None:
            cost = cost + prior * ((1.0 / (s0 - v) - g_pred) / g_scale) ** 2
        i = int(np.argmin(cost))
        if best is None or cost[i] < best[0]:
            best = (float(cost[i]), float(v[i]), float(a[i]), float(b[i]),
                    float(resid[i]))

    if best is None:
        return None
    _, v, a, b, _ = best

    # Re-fit alpha and beta on the lines that agree, so the returned lattice is
    # set by the consensus rather than by every point equally.  Two passes is
    # enough - the first removes the gross label errors, the second settles.
    u = 1.0 / (s - v)
    for _ in range(2):
        pred = a * k + b
        r = np.abs(u - pred) / max(abs(a), 1e-15)
        keep = r <= max(0.25, 3.0 * float(np.median(r)))
        if keep.sum() < max(3, n_pts // 2):
            break
        A = np.stack([k[keep], np.ones(int(keep.sum()))], 1)
        try:
            (a2, b2), *_ = np.linalg.lstsq(A, u[keep], rcond=None)
        except np.linalg.LinAlgError:
            break
        if abs(a2) < 1e-15:
            break
        a, b = float(a2), float(b2)

    idx_real = (u - b) / a
    per = np.abs(idx_real - k)
    return LatticeFit(v, a, b, float(np.exp(-float(np.median(per)))),
                      k.astype(int), per, int((per <= 0.14).sum()))
