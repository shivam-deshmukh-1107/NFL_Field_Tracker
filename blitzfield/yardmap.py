"""Mapping image points to yards along the field, and finding tick columns.

Two things survive here from a larger experiment.

``yard_of_points`` converts any image point into a down-field coordinate in
yards: the yard line through a point is the line joining it to the cross-field
vanishing point, and intersecting that with the transversal puts the point on
the same 1-D scale the lattice was fitted on.

``tick_columns`` groups the 1-yard tick marks into the columns they form down
the field.  What makes it work is a density test rather than a whole-yard test:
ticks exist at *every* yard, so a diagonal slash across the field also hits them
at whole yards - but it collects a few scattered ticks and leaves most yards in
its span empty, whereas a real column has a tick at nearly every yard.

The metric-rectification machinery that used to live alongside this was removed.
It aimed to recover focal length so that field *width* could be measured, and
that is not achievable from this footage: Blitz uses a low, near-ground camera,
so the cross-field vanishing point sits ~200 image widths from centre while the
down-field one sits at ~0.6.  Being near-perpendicular from the principal point
makes any focal estimate hypersensitive - 0 of 20 sampled frames had both
vanishing points usably placed.  See ``out/field_template.json``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from . import geometry as G
from .config import YARDS_PER_CELL
from .lattice import LatticeFit

if TYPE_CHECKING:          # annotation only; markings does not import us
    from .markings import FramePrep


def yard_of_points(P: np.ndarray, vp_cross, ell: np.ndarray,
                   fit: LatticeFit) -> np.ndarray:
    """Down-field coordinate, in yards, of each image point.

    The yard line through a point is the line joining it to the cross-field
    vanishing point; intersecting that with the transversal puts the point on
    the same 1-D scale the lattice was fitted on.
    """
    P = np.asarray(P, float).reshape(-1, 2)
    hom = np.concatenate([P, np.ones((len(P), 1))], 1)
    v = np.asarray(vp_cross, float)
    lines = np.cross(hom, v[None, :])
    X = np.cross(lines, ell[None, :])
    w = X[:, 2]
    coord = 1 if abs(ell[0]) > abs(ell[1]) else 0
    s = np.where(np.abs(w) > 1e-9, X[:, coord] / np.where(np.abs(w) > 1e-9, w, 1),
                 np.nan)
    u = 1.0 / (s - fit.v)
    return (u - fit.beta) / fit.alpha * YARDS_PER_CELL


# ------------------------------------------------------------------ #
# tick columns
# ------------------------------------------------------------------ #
@dataclass
class Column:
    seg: np.ndarray        # (4,) fitted line segment
    n_ticks: int
    yard_span: float
    density: float         # distinct yards occupied / yards spanned


def tick_columns(prep: "FramePrep", vp_cross, ell, fit: LatticeFit,
                 line_s: np.ndarray | None = None,
                 max_dim: int = 26, tol: float = 3.5, iters: int = 1500,
                 min_ticks: int = 5, min_span_yd: float = 6.0,
                 min_density: float = 0.45, min_sep_deg: float = 25.0, rng=None) -> tuple[list, np.ndarray]:
    """Group tick marks into down-field columns.

    ``line_s`` are the transversal positions of the detected yard lines.  Ticks
    outside that span are discarded, and this matters more than it looks: the
    down-field coordinate goes as ``1/(s - v)``, so a tick near the horizon maps
    to a near-infinite yard value.  Left unclamped, a single frame produced
    "ticks" spread over 146 yards, which destroys both the whole-yard test and
    the RANSAC sampling.

    Returns ``(columns, tick_points)``.
    """
    rng = rng or np.random.default_rng(0)
    # The *thin* mask, never the net's.  A hash mark is picked out here by its
    # size - a component of a few dozen pixels, no dimension over ~26 px - and
    # the learned line channel is trained to be thick and continuous, which is
    # what the line fitter wants and exactly what destroys that size test.
    # Measured on clip 4: 84 tick candidates and 2 columns from the top-hat
    # mask, 27 and 0 from the net's.  Two different questions, two masks.
    src = prep.tick_mask if prep.tick_mask is not None else prep.mask
    n, lbl, st, cent = cv2.connectedComponentsWithStats(src, 8)
    a = st[:, cv2.CC_STAT_AREA]
    dim = np.maximum(st[:, cv2.CC_STAT_WIDTH], st[:, cv2.CC_STAT_HEIGHT])
    sel = (a >= 4) & (a <= 260) & (dim <= max_dim)
    sel[0] = False
    P = cent[sel].astype(np.float64)
    if len(P) < min_ticks:
        return [], P

    # keep only ticks inside the span where yard lines were actually observed
    if line_s is not None and len(line_s) >= 2:
        lo, hi = float(np.min(line_s)), float(np.max(line_s))
        pad = 0.15 * (hi - lo)
        hom = np.concatenate([P, np.ones((len(P), 1))], 1)
        X = np.cross(np.cross(hom, np.asarray(vp_cross, float)[None, :]),
                     ell[None, :])
        w = X[:, 2]
        coord = 1 if abs(ell[0]) > abs(ell[1]) else 0
        with np.errstate(invalid="ignore", divide="ignore"):
            s_tick = np.where(np.abs(w) > 1e-9, X[:, coord] / w, np.nan)
        inside = np.isfinite(s_tick) & (s_tick >= lo - pad) & (s_tick <= hi + pad)
        P = P[inside]
        if len(P) < min_ticks:
            return [], P

    yd = yard_of_points(P, vp_cross, ell, fit)
    ok = np.isfinite(yd) & (np.abs(yd) < 400.0)
    P, yd = P[ok], yd[ok]
    if len(P) < min_ticks:
        return [], P

    cols, remaining = [], np.ones(len(P), bool)
    for _ in range(6):
        idx = np.flatnonzero(remaining)
        if len(idx) < min_ticks:
            break
        Q, Qy = P[idx], yd[idx]

        ia = rng.integers(0, len(Q), iters)
        ib = rng.integers(0, len(Q), iters)
        d = Q[ib] - Q[ia]
        norm = np.linalg.norm(d, axis=1)
        good = (norm >= 25.0) & (np.abs(Qy[ib] - Qy[ia]) >= 2.0)
        if not good.any():
            break
        ia, d, norm = ia[good], d[good], norm[good]
        nvec = np.stack([-d[:, 1], d[:, 0]], 1) / norm[:, None]
        dist = np.abs(np.einsum("kj,ij->ki", nvec, Q)
                      - (Q[ia] * nvec).sum(1)[:, None])
        inliers = dist < tol

        best = None
        for h in range(len(inliers)):
            m = inliers[h]
            if m.sum() < min_ticks:
                continue
            ys = Qy[m]
            whole = np.abs(ys - np.round(ys))
            yint = np.unique(np.round(ys[whole <= 0.3]))
            distinct = len(yint)
            if distinct < min_ticks:
                continue
            span = float(np.ptp(ys))
            if span < min_span_yd:
                continue
            # Density, not whole-yard-ness, is what separates a real column from
            # a diagonal slash.  Ticks exist at *every* yard, so any line across
            # the field also hits them at whole yards - but it collects a few
            # scattered ticks from different columns, leaving most yards in its
            # span empty.  A real column has a tick at nearly every yard.
            density = distinct / max(span + 1.0, 1.0)
            if density < min_density:
                continue
            quality = distinct * density
            if best is None or quality > best[0]:
                best = (quality, m, distinct, span, density)
        if best is None:
            break

        _, m, distinct, span, density = best
        S = Q[m]
        mu = S.mean(0)
        dirv = np.linalg.svd(S - mu)[2][0]
        t = (S - mu) @ dirv
        p0, p1 = mu + t.min() * dirv, mu + t.max() * dirv
        seg = np.array([p0[0], p0[1], p1[0], p1[1]])
        remaining[idx[m]] = False

        if vp_cross is not None and G.vp_angular_error(
                seg[None], np.asarray(vp_cross, float))[0] < min_sep_deg:
            continue
        cols.append(Column(seg, int(m.sum()), span, density))
    return cols, P
