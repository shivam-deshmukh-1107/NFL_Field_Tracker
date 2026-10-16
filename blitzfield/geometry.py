"""Projective geometry helpers: line pencils, vanishing points, focal length.

Conventions
-----------
* A 2-D point is homogeneous ``(x, y, w)``; ``w == 0`` means a point at infinity
  (a pure direction), which happens whenever a set of image lines is exactly
  parallel - a real and common case here, so it is handled everywhere rather
  than being treated as degenerate.
* A segment is ``(x1, y1, x2, y2)`` in canonical frame pixels.

Focal-length recovery from orthogonal vanishing points used to live here and was
removed: it is not observable in this footage (see ``yardmap`` for why).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- #
# basic projective ops
# --------------------------------------------------------------------------- #
def seg_to_line(seg: np.ndarray) -> np.ndarray:
    """Homogeneous line through a segment, normalised so (a, b) is a unit vector."""
    x1, y1, x2, y2 = seg[..., 0], seg[..., 1], seg[..., 2], seg[..., 3]
    p1 = np.stack([x1, y1, np.ones_like(x1)], -1)
    p2 = np.stack([x2, y2, np.ones_like(x2)], -1)
    l = np.cross(p1, p2)
    n = np.linalg.norm(l[..., :2], axis=-1, keepdims=True)
    return l / np.maximum(n, EPS)


def seg_length(seg: np.ndarray) -> np.ndarray:
    return np.hypot(seg[..., 2] - seg[..., 0], seg[..., 3] - seg[..., 1])


def seg_midpoint(seg: np.ndarray) -> np.ndarray:
    return np.stack([(seg[..., 0] + seg[..., 2]) * 0.5,
                     (seg[..., 1] + seg[..., 3]) * 0.5], -1)


def seg_direction(seg: np.ndarray) -> np.ndarray:
    """Unit direction of each segment."""
    d = np.stack([seg[..., 2] - seg[..., 0], seg[..., 3] - seg[..., 1]], -1)
    return d / np.maximum(np.linalg.norm(d, axis=-1, keepdims=True), EPS)


def vp_angular_error(segs: np.ndarray, vp: np.ndarray) -> np.ndarray:
    """Angle (degrees) between each segment and the ray from it to ``vp``.

    Using an angular residual rather than an algebraic one keeps short and long
    segments comparable and stays finite for vanishing points far off-image.
    """
    mid = seg_midpoint(segs)
    d = seg_direction(segs)
    w = vp[2]
    if abs(w) > 1e-9 * max(abs(vp[0]), abs(vp[1]), 1.0):
        ray = vp[:2] / w - mid
    else:
        ray = np.broadcast_to(vp[:2], mid.shape).copy()
    ray = ray / np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), EPS)
    cos = np.abs((d * ray).sum(-1)).clip(0.0, 1.0)
    return np.degrees(np.arccos(cos))


def vp_angular_error_batch(mid: np.ndarray, d: np.ndarray,
                           V: np.ndarray) -> np.ndarray:
    """:func:`vp_angular_error` for many vanishing points at once.

    Returns an ``(m, n)`` array for ``m`` candidates and ``n`` segments.  Same
    arithmetic as the scalar form, evaluated as one batch: RANSAC scores
    hundreds of hypotheses against the same segments, and at a few hundred
    segments per frame the per-call numpy overhead dominated the actual work.
    ``mid`` and ``d`` are hoisted out because they do not depend on ``V``.
    """
    V = np.asarray(V, float).reshape(-1, 3)
    w = V[:, 2]
    finite = np.abs(w) > 1e-9 * np.maximum(
        np.maximum(np.abs(V[:, 0]), np.abs(V[:, 1])), 1.0)
    wsafe = np.where(finite, w, 1.0)
    ray = np.where(finite[:, None, None],
                   (V[:, :2] / wsafe[:, None])[:, None, :] - mid[None, :, :],
                   V[:, :2][:, None, :])
    ray = ray / np.maximum(np.linalg.norm(ray, axis=-1, keepdims=True), EPS)
    cos = np.abs((d[None, :, :] * ray).sum(-1)).clip(0.0, 1.0)
    return np.degrees(np.arccos(cos))


def refine_vp(segs: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares vanishing point of a pencil: null vector of stacked lines."""
    L = seg_to_line(segs)
    if weights is not None:
        L = L * weights[:, None]
    _, _, Vt = np.linalg.svd(L, full_matrices=False)
    return Vt[-1]


def ransac_vp(segs: np.ndarray, tau_deg: float = 1.5, iters: int = 600,
              min_inliers: int = 3, rng: np.random.Generator | None = None):
    """Find the dominant pencil of concurrent lines.

    Returns ``(vp, inlier_mask, score)``.  Score is total inlier segment length,
    so a few long yard lines beat many short noise fragments.
    """
    n = len(segs)
    if n < 2:
        return None, np.zeros(n, bool), 0.0
    rng = rng or np.random.default_rng(0)
    lengths = seg_length(segs)
    L = seg_to_line(segs)

    best = (None, np.zeros(n, bool), 0.0)
    # Sample proportionally to length: long segments are far more informative.
    p = lengths / lengths.sum()
    # Drawing and scoring are separated so the scoring can be batched.  The
    # draws stay in their original one-at-a-time order, which keeps the
    # hypothesis sequence - and so the result - identical to the scalar loop.
    cand = []
    for _ in range(iters):
        i, j = rng.choice(n, 2, replace=False, p=p)
        a, b = L[i], L[j]
        # Written out rather than np.cross: the generic version spends far more
        # time dispatching on axes than on three multiply-subtracts, and this
        # runs once per RANSAC hypothesis.
        v = np.array([a[1] * b[2] - a[2] * b[1],
                      a[2] * b[0] - a[0] * b[2],
                      a[0] * b[1] - a[1] * b[0]])
        if np.linalg.norm(v) < EPS:
            continue
        cand.append(v)
    if cand:
        mid, d = seg_midpoint(segs), seg_direction(segs)
        inl_all = vp_angular_error_batch(mid, d, np.asarray(cand)) < tau_deg
        scores = inl_all @ lengths
        enough = inl_all.sum(1) >= min_inliers
        for v, inl, score, ok in zip(cand, inl_all, scores, enough):
            if score > best[2] and ok:
                best = (v, inl, float(score))

    v, inl, score = best
    if v is None:
        return None, np.zeros(n, bool), 0.0
    # Two rounds of reweighted refinement.
    for _ in range(2):
        v = refine_vp(segs[inl], lengths[inl])
        err = vp_angular_error(segs, v)
        inl = err < tau_deg
        if inl.sum() < min_inliers:
            break
    return v, inl, float(lengths[inl].sum())


