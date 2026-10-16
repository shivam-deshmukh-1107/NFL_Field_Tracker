"""Stage 4 - extract field markings and the yard-line vanishing point.

Pipeline per frame
------------------
1.  ``FramePrep``            - one pass of the colour work (Lab, turf distance,
                               field region, occluders), cached and shared.
2.  ridge response           - top-hat gated on low chroma, normalised by a high
                               percentile *inside the field region*.
3.  thinning + LSD           - centreline segments rather than the two edges of
                               each ridge.
4.  flank classification     - a painted line has turf on both sides
                               (*interior*: yard lines, hash marks); a boundary
                               line has turf on one side and the apron or end
                               zone on the other (*boundary*: sidelines, goal
                               lines).  Anything else is rejected.
5.  vanishing point         - the yard lines form one dominant pencil, found by
                               RANSAC over the surviving segments.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from . import geometry as G
from .fieldcolor import TurfModel, white_line_response

INTERIOR, BOUNDARY = 0, 1

#: Probability above which :mod:`fieldnet` output counts as turf / as paint.
#: A plain 0.5 is meaningful here in a way no top-hat threshold ever was: the
#: net is trained with a per-pixel loss, so its output is a calibrated
#: probability rather than a filter response whose scale drifts with the frame.
NET_TURF_THR = 0.5
NET_LINE_THR = 0.5


def cc_filter(mask: np.ndarray, min_area: int = 0, max_area: int | None = None,
              max_dim: int | None = None) -> np.ndarray:
    """Keep connected components matching size constraints.

    Vectorised via fancy-indexing the label image with a per-label keep table;
    the obvious per-component Python loop is thousands of iterations on a
    speckled mask and dominates the whole frame budget.
    """
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    keep = np.ones(n, bool)
    keep[0] = False                                   # background
    a = stats[:, cv2.CC_STAT_AREA]
    if min_area:
        keep &= a >= min_area
    if max_area is not None:
        keep &= a <= max_area
    if max_dim is not None:
        keep &= np.maximum(stats[:, cv2.CC_STAT_WIDTH],
                           stats[:, cv2.CC_STAT_HEIGHT]) <= max_dim
    return keep[lbl]


def _otsu(values: np.ndarray) -> float:
    """Otsu threshold of a float array in [0, 1], returned in the same units."""
    if values.size < 64:
        return 0.25
    v = np.clip(values * 255.0, 0, 255).astype(np.uint8)
    t, _ = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(t) / 255.0


# --------------------------------------------------------------------------- #
# shared per-frame preprocessing
# --------------------------------------------------------------------------- #
@dataclass
class FramePrep:
    """All the colour-domain work for one frame, computed once."""

    frame: np.ndarray
    turf_mask: np.ndarray
    region: np.ndarray
    occluder: np.ndarray
    resp: np.ndarray
    mask: np.ndarray
    hud: np.ndarray
    region_closed: np.ndarray = None   # region + game-drawn lines; boundary use only
    tick_mask: np.ndarray = None       # thin, colour-derived; hash marks only
    off_turf: np.ndarray = None        # Mahalanobis > 3; shared, not recomputed
    colour_masks: dict = None          # game-drawn line hues, keyed by colour

    @staticmethod
    def build(frame: np.ndarray, turf: TurfModel, hud: np.ndarray | None = None,
              k: float = 3.0, open_ksize: int = 11, min_area: int = 150,
              dilate: int = 5, thr_lo: float = 0.15, thr_hi: float = 0.60,
              speckle_area: int = 22, net=None) -> "FramePrep":
        """Colour-domain preprocessing, optionally with a learned front end.

        ``net`` is a :class:`~blitzfield.fieldnet.FieldSegmenter`.  When given it
        supplies the turf mask and the line response in place of the Mahalanobis
        cut and the top-hat; with ``net=None`` this is exactly the colour
        pipeline, which stays the fallback whenever no weights are trained.

        The occluder mask is deliberately *not* taken from the net.  Players are
        marked ignore when the dataset is built - a moving body is turf in some
        frames of the voting window and not in others - so the net has no reason
        to carve them out and is free to predict the field straight through
        them.  That is the right answer for the tint, and the wrong one for
        "where must the line search not look", so that question keeps its own
        answer from the colour mask.
        """
        if hud is None:
            hud = np.zeros(frame.shape[:2], bool)
        maha = turf.mahalanobis(frame)
        tm_colour = (maha < k) & ~hud
        off_turf = maha > 3.0
        p_line = None
        if net is not None:
            from .fieldcolor import normalise_to_canon
            # tm_colour, not the raw Mahalanobis cut: it already excludes the
            # HUD, and the HUD's luminance has no business in a median over
            # turf pixels.
            src = (normalise_to_canon(frame, turf, tm_colour)
                   if getattr(net, "normalise", False) else frame)
            p_turf, p_line = net.predict(src)
            tm = (p_turf >= getattr(net, "turf_thr", NET_TURF_THR)) & ~hud
        else:
            tm = tm_colour

        def _close(mask_bool):
            mm = mask_bool.astype(np.uint8)
            mm = cv2.morphologyEx(mm, cv2.MORPH_CLOSE, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (25, 25)))
            mm = cv2.morphologyEx(mm, cv2.MORPH_OPEN, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (7, 7)))
            return cc_filter(mm, min_area=int(mm.size * 0.02))

        # The plain turf region.  Kept free of the game-drawn colour: anything
        # that consumes `region` as "where the field is" - notably the
        # scrimmage/first-down line fitter - degrades badly if transient banner
        # colour is allowed to widen it.
        region = _close(tm)

        # A second, extended region used *only* for boundary tracing.  The
        # coloured scrimmage line reads as "not turf" and carves a channel
        # across the field, which then surfaces as a cross-field region edge and
        # is mistaken for a goal line.  Only line-like coloured components are
        # folded in: taking every saturated pixel also swallows the light
        # out-of-bounds apron and erases the sideline.
        from .fieldcolor import coloured_line_masks
        colour_masks = coloured_line_masks(frame, turf, off_turf=off_turf)
        painted = np.zeros(tm.shape, bool)
        for _name, cmask in colour_masks.items():
            nn, ll, stt, _ = cv2.connectedComponentsWithStats(
                cmask.astype(np.uint8), 8)
            keep = np.zeros(nn, bool)
            a_ = stt[:, cv2.CC_STAT_AREA]
            w_ = stt[:, cv2.CC_STAT_WIDTH]
            h_ = stt[:, cv2.CC_STAT_HEIGHT]
            elong = np.maximum(w_, h_) / np.maximum(np.minimum(w_, h_), 1)
            keep = (a_ >= 300) & (elong >= 4.0)
            keep[0] = False
            painted |= keep[ll]
        region_closed = _close(tm | (painted & ~hud))

        # occluders: large off-turf blobs. The opening is wider than any painted
        # line, so markings drop out of this mask while solid objects survive.
        solid = cv2.morphologyEx((~tm_colour).astype(np.uint8), cv2.MORPH_OPEN,
                                 cv2.getStructuringElement(
                                     cv2.MORPH_ELLIPSE, (open_ksize, open_ksize)))
        occ = cc_filter(solid, min_area=min_area).astype(np.uint8)
        if dilate:
            occ = cv2.dilate(occ, cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1)))
        occ = occ.astype(bool)

        valid = region & ~occ & ~hud

        # The colour ridge mask, always.  Adaptive threshold: turf has a mottled
        # texture whose top-hat response varies frame to frame.  A fixed cut
        # either floods the mask with speckle - which makes LSD's density test
        # reject every rectangle and silently return zero segments - or misses
        # faint distant lines.
        resp_c = white_line_response(frame, valid=valid) * valid
        thr = float(np.clip(_otsu(resp_c[valid]) if valid.any() else 0.25,
                            thr_lo, thr_hi))
        tick_mask = cc_filter((resp_c >= thr).astype(np.uint8),
                              min_area=speckle_area).astype(np.uint8)

        if p_line is not None:
            resp = p_line * valid
            mask = cc_filter(
                (resp >= getattr(net, "line_thr", NET_LINE_THR)).astype(np.uint8),
                min_area=speckle_area).astype(np.uint8)
        else:
            resp, mask = resp_c, tick_mask
        return FramePrep(frame, tm, region, occ, resp, mask, hud,
                         region_closed, tick_mask, off_turf, colour_masks)


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #
def detect_segments(mask: np.ndarray, min_len: float = 18.0) -> np.ndarray:
    """Thin the mask to centrelines and run LSD on the skeleton."""
    thin = cv2.ximgproc.thinning(mask * 255,
                                 thinningType=cv2.ximgproc.THINNING_GUOHALL)
    img = cv2.dilate(thin, np.ones((2, 2), np.uint8))
    # REFINE_STD, not REFINE_ADV.  On a thinned skeleton the advanced NFA
    # refinement rejects almost every genuine line (0 vs 47 long segments on a
    # sample frame) and runs ~2000x slower (64s vs 0.03s per frame).
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = lsd.detect(img)[0]
    if lines is None or len(lines) == 0:
        return np.zeros((0, 4), np.float64)
    segs = np.asarray(lines, np.float64).reshape(-1, 4)
    return segs[G.seg_length(segs) >= min_len]


def merge_collinear(segs: np.ndarray, angle_tol_deg: float = 2.0,
                    perp_tol: float = 4.0, gap_tol: float = 80.0) -> np.ndarray:
    """Fuse fragments of the same painted line (players break lines in two)."""
    if len(segs) == 0:
        return segs
    segs = segs[np.argsort(-G.seg_length(segs))]
    n = len(segs)
    used = np.zeros(n, bool)
    # Hoisted out of the O(n^2) pair loop.  LSD returns a few hundred segments
    # per frame, so recomputing a direction and a length per *pair* was the
    # single largest cost in a render.
    D = G.seg_direction(segs)
    LN = G.seg_to_line(segs)
    LEN = G.seg_length(segs)
    PTS = segs.reshape(n, 2, 2)
    cos_tol = np.cos(np.radians(angle_tol_deg))
    out = []
    for i in range(n):
        if used[i]:
            continue
        used[i] = True
        di = D[i]
        # Both cheap rejections for every remaining candidate at once; only the
        # survivors reach the gap test, which has to stay sequential because it
        # depends on the group built so far.
        free = np.flatnonzero(~used[i + 1:]) + i + 1
        if len(free):
            aligned = np.abs(D[free] @ di) >= cos_tol
            free = free[aligned]
        if len(free):
            perp = np.abs(PTS[free] @ LN[i, :2] + LN[i, 2]).max(axis=1)
            free = free[perp <= perp_tol]

        ti = PTS[i] @ di
        lo_k, hi_k = int(np.argmin(ti)), int(np.argmax(ti))
        t_lo, t_hi = ti[lo_k], ti[hi_k]
        p_lo, p_hi = PTS[i][lo_k], PTS[i][hi_k]
        covered = float(LEN[i])
        for j in free:
            tj = PTS[j] @ di
            j_lo, j_hi = int(np.argmin(tj)), int(np.argmax(tj))
            n_lo, n_hi = min(t_lo, tj[j_lo]), max(t_hi, tj[j_hi])
            if (n_hi - n_lo) - (covered + float(LEN[j])) > gap_tol:
                continue
            if tj[j_lo] < t_lo:
                t_lo, p_lo = tj[j_lo], PTS[j][j_lo]
            if tj[j_hi] > t_hi:
                t_hi, p_hi = tj[j_hi], PTS[j][j_hi]
            covered += float(LEN[j])
            used[j] = True
        out.append([p_lo[0], p_lo[1], p_hi[0], p_hi[1]])
    return np.asarray(out, np.float64)


def classify_flanks(segs: np.ndarray, prep: FramePrep, offset: float = 5.0,
                    n_samples: int = 11, min_frac: float = 0.6,
                    boundary_min_len: float = 50.0) -> np.ndarray:
    """Label each segment INTERIOR / BOUNDARY, or -1 to reject.

    Painted markings are turf-flanked by construction, so requiring turf on both
    sides removes almost all structural false positives with no temporal model.
    Sidelines and goal lines are the deliberate exception: turf on one side, the
    apron or end zone on the other.  They are the main source of the down-field
    vanishing point, so rejecting them - as a naive both-sides test does - is
    what leaves the second pencil empty.
    """
    n = len(segs)
    if n == 0:
        return np.zeros(0, np.int8)
    h, w = prep.frame.shape[:2]
    tm = prep.turf_mask
    lengths = G.seg_length(segs)
    out = np.full(n, -1, np.int8)

    for i, s in enumerate(segs):
        p0, p1 = s[:2], s[2:]
        d = (p1 - p0) / max(np.linalg.norm(p1 - p0), 1e-9)
        nrm = np.array([-d[1], d[0]])
        ts = np.linspace(0.15, 0.85, n_samples)[:, None]
        base = p0 + ts * (p1 - p0)
        frac = []
        for sgn in (+1.0, -1.0):
            q = np.round(base + sgn * offset * nrm).astype(int)
            inb = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
            frac.append(float(tm[q[inb, 1], q[inb, 0]].mean()) if inb.sum() >= 3
                        else 0.0)
        lo, hi = min(frac), max(frac)
        if lo >= min_frac:
            out[i] = INTERIOR
        elif hi >= min_frac and lo <= 1.0 - min_frac and lengths[i] >= boundary_min_len:
            out[i] = BOUNDARY
    return out


# --------------------------------------------------------------------------- #
# field boundary (sidelines, goal lines)
# --------------------------------------------------------------------------- #
def boundary_segments(prep: FramePrep, min_len: float = 60.0,
                      border_margin: int = 6, eps: float = 3.0,
                      axis_tol_deg: float = 2.0) -> np.ndarray:
    """Straight edges of the field region - i.e. the sidelines and goal lines.

    Blitz does not draw the sideline as a thin white stripe: the entire
    out-of-bounds apron is light coloured, so the field edge is a *region
    boundary*, not a ridge.  A top-hat ridge filter is blind to it, which is why
    the down-field pencil stays empty if we only look for painted lines.

    Edges lying on the image border, on the HUD, or within ``axis_tol_deg`` of
    exactly horizontal/vertical (transient HUD banners are axis-aligned
    rectangles; field lines almost never are) are discarded.
    """
    src = prep.region_closed if prep.region_closed is not None else prep.region
    m = src.astype(np.uint8)
    h, w = m.shape
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hud_d = cv2.dilate(prep.hud.astype(np.uint8),
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    out = []
    for c in cnts:
        if cv2.contourArea(c) < 0.02 * m.size:
            continue
        ap = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        for i in range(len(ap)):
            p0, p1 = ap[i], ap[(i + 1) % len(ap)]
            d = p1 - p0
            L = float(np.hypot(*d))
            if L < min_len:
                continue
            ang = abs(np.degrees(np.arctan2(d[1], d[0]))) % 90.0
            if min(ang, 90.0 - ang) < axis_tol_deg:
                continue
            ts = np.linspace(0.0, 1.0, 15)[:, None]
            P = p0 + ts * d
            near_border = ((P[:, 0] < border_margin) | (P[:, 0] > w - 1 - border_margin)
                           | (P[:, 1] < border_margin) | (P[:, 1] > h - 1 - border_margin))
            if near_border.mean() > 0.3:
                continue
            q = np.round(P).astype(int)
            q[:, 0] = np.clip(q[:, 0], 0, w - 1)
            q[:, 1] = np.clip(q[:, 1], 0, h - 1)
            if hud_d[q[:, 1], q[:, 0]].mean() > 0.2:
                continue
            out.append([p0[0], p0[1], p1[0], p1[1]])
    return np.asarray(out, np.float64).reshape(-1, 4)


# --------------------------------------------------------------------------- #
# hash-mark columns
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# per-frame driver
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# per-frame driver
# --------------------------------------------------------------------------- #
@dataclass
class FrameMarkings:
    """What one frame yields: the yard-line pencil and the field boundary."""

    frame_index: int
    segs: np.ndarray                 # (N,4) surviving segments
    label: np.ndarray                # 0 = yard-line pencil, -1 = other
    boundary: np.ndarray = field(    # (K,4) field-region edges
        default_factory=lambda: np.zeros((0, 4)))
    vp_cross: object = None          # vanishing point of the yard-line pencil


def analyse_frame(frame: np.ndarray, turf: TurfModel, hud=None,
                  frame_index: int = -1, rng=None, net=None,
                  prep: "FramePrep" = None) -> FrameMarkings:
    """Extract the yard-line pencil and the field boundary from one frame.

    Only the *cross-field* pencil is recovered.  A second pencil for the
    down-field direction used to be fitted here as well, on the way to a focal
    length and a metric width; that is not recoverable from this footage, and
    nothing downstream consumed it, so the extra RANSAC was removed.  Tick
    columns - which give the down-field direction where it is actually needed -
    are found in :mod:`yardmap` using the yard lattice as a constraint.
    """
    rng = rng or np.random.default_rng(0)
    # Callers that already built one pass it in: with a net in play the prep is
    # a forward pass, not a handful of array ops, and building it twice per
    # frame doubles the cost of a render.
    if prep is None:
        prep = FramePrep.build(frame, turf, hud=hud, net=net)

    raw = detect_segments(prep.mask)
    segs = merge_collinear(raw)
    flank = classify_flanks(segs, prep) if len(segs) else np.zeros(0, np.int8)
    keep = flank >= 0
    segs, flank = segs[keep], flank[keep]

    label = np.full(len(segs), -1, np.int8)
    v1 = None
    if len(segs) >= 3:
        v1, in1, _ = G.ransac_vp(segs, tau_deg=1.5, iters=700, rng=rng)
        if v1 is not None:
            label[in1] = 0

    bnd = boundary_segments(prep)
    return FrameMarkings(frame_index=frame_index, segs=segs, label=label,
                         boundary=bnd, vp_cross=v1)
