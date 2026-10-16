"""Stage 6 - render an annotated video showing the detected field.

Draws, per frame:

* the playable field region (outline + tint)
* yard lines, labelled with their lattice index in yards
* 1-yard tick marks and the down-field columns they form
* field boundaries - sidelines and goal lines - distinguished by whether the
  edge runs along the field or across it
* the game-drawn scrimmage / first-down lines

Detection is per frame, but the overlay is held briefly when a frame fails so
that a single dropout does not make the annotation flicker.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import cv2
import numpy as np

from . import geometry as G, lattice as L, markings, tracker as tk, yardmap
from .config import LONG_CUT

COL_REGION = (60, 200, 60)
COL_YARD = (255, 230, 60)
COL_YARD_INFERRED = (150, 140, 40)   # placed by the lattice, not seen this frame
COL_TICK = (0, 220, 255)
COL_COLUMN = (255, 0, 255)
COL_SIDELINE = (0, 90, 255)
COL_GOAL = (0, 0, 255)
OUTSIDE_DIM = 0.45          # how far everything off the field is knocked back

#: Turf coverage above which a frame is treated as showing the playing field.
#:
#: Blitz draws its play-select menu over a turf backdrop, so something has to
#: separate that from a game.  The first attempt asked whether the turf was in
#: one piece, on the evidence that the menu grid leaves 16-18 fragments while a
#: wide gameplay shot leaves 2-3.  That was measured only on wide shots and it
#: is wrong: at close range a painted yard line is tens of pixels across, so the
#: lines themselves cut the field into bands, and zoomed gameplay scored 0.25
#: against menus reaching 0.54 - the statistic inverts exactly where it matters.
#:
#: Coverage over the whole capture is trimodal, and the cut goes in the valley:
#:
#:     0.000-0.025  14,078 frames  no turf at all (attract, splash, crowd)
#:     0.425-0.500  12,300 frames  the play-select menu   (lattice solves 6.5%)
#:     0.500-0.525     820 frames  <- valley              (solves 33%)
#:     0.525 and up                the field              (solves 46-69%)
FIELD_MIN_COVER = 0.52


# --------------------------------------------------------------------------- #
# field extent
# --------------------------------------------------------------------------- #
def field_mask(turf_mask: np.ndarray, close: int = 9,
               min_frac: float = 0.02,
               max_hole_frac: float = 0.02) -> np.ndarray:
    """Playable surface: where the turf actually is.

    Built from the raw turf mask, *not* from ``FramePrep.region``.  The region
    is closed with a 25 px kernel to bridge players, and closing large enough to
    span a player (40-80 px) inevitably also bridges the 30-60 px out-of-bounds
    apron - so the region reaches well past the sideline.  Measured on this
    footage the turf covers ~78% of frame, the region ~84%, and closing it
    further for display pushed it to ~94%: essentially the whole picture,
    including the apron and the crowd.

    Players are filled by *hole filling* instead, which closes anything fully
    enclosed by turf without growing the outer boundary at all.  A player
    standing on the sideline is not filled - a small cosmetic notch, versus
    silently claiming the field extends into the stands.
    """
    m = cv2.morphologyEx(turf_mask.astype(np.uint8), cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                   (close | 1, close | 1)))
    h, w = m.shape
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = m
    flood = ff.copy()
    cv2.floodFill(flood, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    holes = ((1 - flood) & (1 - ff))[1:-1, 1:-1].astype(np.uint8)

    # Fill only *small* enclosed holes.  Players and logos are small and sit on
    # the field, so filling them is right.  A transient play banner is also
    # enclosed by turf but is large, and absorbing it makes yard lines get drawn
    # straight across the HUD.
    small = markings.cc_filter(holes, max_area=int(holes.size * max_hole_frac))
    filled = (m.astype(bool) | small).astype(np.uint8)
    return markings.cc_filter(filled, min_area=int(filled.size * min_frac))


def field_is_coherent(mask: np.ndarray) -> bool:
    """Is the playing field on screen, rather than turf behind a menu?

    This is only half the test; :func:`analyse` also accepts any frame whose
    yard lattice solved, which is independent and stronger evidence.

    Where both are silent the answer is genuinely unavailable: between 0.15 and
    0.50 coverage with no lattice, a corner-of-the-field view and turf behind a
    play-select menu are not separable by any statistic measured here - they
    occupy the same range, and the menu peak holds ~12,000 frames of this
    capture.  Those frames are left unhighlighted rather than guessed.
    """
    return mask is not None and float(mask.mean()) >= FIELD_MIN_COVER


def clip_to_field(segs, idx, mask: np.ndarray, n: int = 160,
                  min_len: float = 25.0, extra=None):
    """Trim each line to the span that actually lies on the field.

    A tracked line is stored as an infinite line, so clipping it to the frame
    rectangle draws it straight across the sideline, the out-of-bounds apron and
    the crowd.  Yard lines stop at the sidelines, and the annotation has to as
    well - otherwise the overlay claims a field far wider than the one detected.
    """
    if segs is None or len(segs) == 0:
        return (segs, idx) if extra is None else (segs, idx, extra)
    h, w = mask.shape
    out_s, out_i, out_x = [], [], []
    ex = (np.zeros(len(segs), bool) if extra is None
          else np.asarray(extra).reshape(-1))
    for seg, k, xf in zip(np.asarray(segs, float), np.asarray(idx, float), ex):
        p0, p1 = seg[:2], seg[2:]
        t = np.linspace(0.0, 1.0, n)[:, None]
        P = p0 + t * (p1 - p0)
        q = np.round(P).astype(int)
        ok = (q[:, 0] >= 0) & (q[:, 0] < w) & (q[:, 1] >= 0) & (q[:, 1] < h)
        inside = np.zeros(n, bool)
        inside[ok] = mask[q[ok, 1], q[ok, 0]]
        # longest contiguous run on the field
        best_len, best = 0, None
        i = 0
        while i < n:
            if not inside[i]:
                i += 1
                continue
            j = i
            while j < n and inside[j]:
                j += 1
            if j - i > best_len:
                best_len, best = j - i, (i, j - 1)
            i = j
        if best is None:
            continue
        pa, pb = P[best[0]], P[best[1]]
        if float(np.hypot(*(pb - pa))) < min_len:
            continue
        out_s.append([pa[0], pa[1], pb[0], pb[1]])
        out_i.append(k)
        out_x.append(xf)
    S = np.asarray(out_s, float).reshape(-1, 4)
    I = np.asarray(out_i, float)
    return (S, I) if extra is None else (S, I, np.asarray(out_x, bool))


def snap_to_evidence(segs, idx, seen, resp, spacing, accept: float = 3.0,
                     keep: float = 1.8, prev=None, n: int = 140):
    """Pull each lattice-placed line onto the paint it predicts - or drop it.

    The lattice knows a yard line is *about* here; measured, it is a median 7.5
    px out, and pushing the construction through two transversals instead of the
    vanishing point did not improve that.  So the prediction is not good enough
    to draw on its own - but it is an excellent place to start looking, which
    turns inference back into detection.

    Each predicted line is swept sideways across a third of the local yard
    spacing and moved to the strongest ridge response it finds.  A line with no
    peak worth the name is dropped rather than drawn, so a gap in the lattice
    stays a gap unless the image actually supports filling it.

    The threshold has hysteresis, and that is not a detail.  Deciding each line
    afresh every frame makes anything sitting near the cut toggle on and off,
    and measured on the drawn set that was 20-33% of the labels turning over
    between neighbouring frames - twice as bad under pan+zoom as under a plain
    pan, which is exactly when it is most visible.  A line already recovered
    last frame only has to clear ``keep``; a new one has to clear ``accept``.
    """
    if segs is None or len(segs) == 0:
        return segs, idx, seen
    h, w = resp.shape
    reach = float(max(4.0, 0.35 * (spacing or 40.0)))
    offs = np.arange(-reach, reach + 1e-6, 1.0)
    out_s, out_i, out_o = [], [], []
    for sg, k, was_seen in zip(np.asarray(segs, float), np.asarray(idx, float),
                               np.asarray(seen, bool)):
        if was_seen:
            out_s.append(sg)
            out_i.append(k)
            out_o.append(True)
            continue
        p0, p1 = sg[:2], sg[2:]
        d = p1 - p0
        ln = float(np.hypot(*d))
        if ln < 30.0:
            continue
        nrm = np.array([-d[1], d[0]]) / ln
        base = p0 + np.linspace(0.06, 0.94, n)[:, None] * d
        vals = []
        for o in offs:
            P = np.round(base + o * nrm).astype(int)
            ok = ((P[:, 0] >= 0) & (P[:, 0] < w) & (P[:, 1] >= 0) & (P[:, 1] < h))
            vals.append(float(np.mean(resp[P[ok, 1], P[ok, 0]]))
                        if ok.sum() >= 25 else np.nan)
        v = np.asarray(vals, float)
        if not np.isfinite(v).any():
            continue
        j = int(np.nanargmax(v))
        floor = float(np.nanmedian(v))
        thr = keep if (prev is not None and round(float(k), 1) in prev) else accept
        if not np.isfinite(floor) or v[j] < thr * max(floor, 1e-3):
            continue
        sh = offs[j] * nrm
        out_s.append([sg[0] + sh[0], sg[1] + sh[1], sg[2] + sh[0], sg[3] + sh[1]])
        out_i.append(k)
        out_o.append(False)
    return (np.asarray(out_s, float).reshape(-1, 4), np.asarray(out_i, float),
            np.asarray(out_o, bool))


# --------------------------------------------------------------------------- #
# per-frame analysis bundle
# --------------------------------------------------------------------------- #
@dataclass
class Overlay:
    ok: bool = False
    region: object = None        # generous mask: motion features, number search
    field: object = None         # tight mask: what is actually turf
    yard_lines: object = None        # (N,4)
    yard_index: object = None        # (N,) yards, relative
    ticks: object = None             # (M,2)
    columns: list = dc_field(default_factory=list)
    sidelines: object = None
    goal_lines: object = None
    yard_observed: object = None  # per line: measured this frame, or from the lattice
    coloured: dict = dc_field(default_factory=dict)
    line_spacing: object = None  # median px between adjacent yard lines (5 yd)
    field_ok: bool = False       # the turf mask is a credible playing surface
    tracked: bool = False        # this frame's lines came from prediction alone
    n_detected: int = 0
    n_predicted: int = 0


def analyse(frame, turf, hud, rng=None, track: "tk.FieldTracker" = None,
            motion=None, net=None) -> Overlay:
    """Detect the field in one frame, optionally fused with tracked state.

    With a tracker, the drawn yard lines come from the *tracked* set rather than
    the raw detections: it is stable frame to frame, keeps its yard labelling,
    and interpolates lines that occlusion hid this frame.
    """
    from . import pipeline
    rng = rng or np.random.default_rng(0)
    ov = Overlay()
    prep = markings.FramePrep.build(frame, turf, hud=hud, net=net)
    ov.region = prep.region
    ov.field = field_mask(prep.turf_mask)
    # Extent and geometry are separate claims and are decided separately.  The
    # turf mask is reliable at zoom levels where the yard lattice is simply
    # underdetermined - the end-of-play close-ups in clip 1 show a plain field
    # with one yard line on it - and gating the highlight on the lattice threw
    # the reliable claim away with the unreliable one.
    ov.field_ok = field_is_coherent(ov.field)
    fm = markings.analyse_frame(frame, turf, hud=hud, rng=rng, prep=prep)


    h, w = frame.shape[:2]
    ell = np.array([1.0, 0.0, -w / 2])
    pen = fm.segs[fm.label == 0]
    Ls = G.seg_length(pen) if len(pen) else np.zeros(0)
    lng = pen[Ls >= LONG_CUT] if len(pen) else np.zeros((0, 4))
    ov.yard_lines = lng
    if (len(pen) == 0 or fm.vp_cross is None) and track is None:
        return ov

    # split the region boundary into along-field and across-field edges
    bnd = fm.boundary
    if len(bnd) and fm.vp_cross is not None:
        # Without a cross-field vanishing point there is nothing to measure the
        # boundary against, so it stays unclassified rather than being guessed.
        # Reachable whenever the tracker holds a lock through a frame whose own
        # detection failed - the early return above only covers the untracked case.
        err = G.vp_angular_error(bnd, np.asarray(fm.vp_cross, float))
        # Trim both to the neighbourhood of the detected field.  A boundary is
        # traced from the region contour, and on a play-select screen or behind
        # a result banner that contour follows the graphic rather than the turf
        # edge - which drew "sidelines" across menu panels and stadium walls.
        # Dilated rather than the bare mask because a true sideline lies exactly
        # on the edge, where a strict test drops it.
        near = cv2.dilate(ov.field.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17))
                          ).astype(bool)
        ov.goal_lines = clip_to_field(bnd[err <= 3.0],
                                      np.zeros(int((err <= 3.0).sum())), near,
                                      min_len=40.0)[0]
        ov.sidelines = clip_to_field(bnd[err > 3.0],
                                     np.zeros(int((err > 3.0).sum())), near,
                                     min_len=40.0)[0]

    # per-frame lattice, when there is enough evidence to fit one cold
    fit, sc = None, np.zeros(0)
    if len(lng) >= 5:
        s, k = L.transversal_positions(lng, ell)
        if k.any():
            sc = L.cluster_1d(s[k])
            fit = L.fit(sc)
            if fit is not None:
                fit = L.refine(sc, fit)

    if track is not None:
        st = track.step(lng, fm.vp_cross, fit, w, h, motion=motion)
        if st.ok:
            segs, idx, seen = st.segments(w, h, complete=True)
            segs, idx, seen = clip_to_field(segs, idx, ov.field, extra=seen)
            if len(segs) >= 2:
                sv = np.sort(L.transversal_positions(segs, ell)[0])
                sv = sv[np.isfinite(sv)]
                sp0 = float(np.median(np.diff(sv))) if len(sv) >= 2 else None
                segs, idx, seen = snap_to_evidence(
                    segs, idx, seen, prep.resp, sp0,
                    prev=getattr(track, "recovered_idx", None))
                track.recovered_idx = {round(float(a), 1)
                                       for a, s2 in zip(idx, seen) if not s2}
            ov.yard_lines, ov.yard_index, ov.yard_observed = segs, idx, seen
            ov.n_detected, ov.n_predicted = st.n_detected, st.n_predicted
            ov.tracked = st.from_prediction
            vp = st.vp_cross
            if st.fit is not None and vp is not None:
                s_all, k_all = L.transversal_positions(segs, ell)
                cols, ticks = yardmap.tick_columns(
                    prep, vp, ell, st.fit,
                    line_s=(s_all[k_all] if k_all.any() else None), rng=rng)
                # Columns get the same treatment as yard lines: a fitted line
                # is unbounded, and drawing it past the sideline is what makes
                # the hash columns look like stray diagonals.  Clipped one at a
                # time because a column that falls entirely off the field is
                # dropped, which would misalign a zipped result.
                keep = []
                for c in cols:
                    cs, _ = clip_to_field(c.seg[None], np.zeros(1), ov.field,
                                          min_len=40.0)
                    if len(cs):
                        c.seg = cs[0]
                        keep.append(c)
                cols = keep
                ov.columns, ov.ticks = cols, ticks
            # One line is not a field.  On a play-select screen the tracker
            # coasts for a few frames as its lines age out, and the last
            # survivor was being drawn across the menu with the field
            # highlighted behind it.  Two lines is the least that fixes a
            # lattice, so it is also the least that should be claimed.
            # Measured on the four clips: no frame is lost - the smallest
            # solved frame there carries ten.
            ov.ok = len(segs) >= 2
            if len(segs) >= 2:
                sv = np.sort(L.transversal_positions(segs, ell)[0])
                sv = sv[np.isfinite(sv)]
                if len(sv) >= 2:
                    ov.line_spacing = float(np.median(np.diff(sv)))
    elif fit is not None:
        mids = G.seg_midpoint(lng)
        ov.yard_index = yardmap.yard_of_points(mids, fm.vp_cross, ell, fit)
        cols, ticks = yardmap.tick_columns(prep, fm.vp_cross, ell, fit,
                                           line_s=sc, rng=rng)
        ov.columns, ov.ticks = cols, ticks
        ov.ok = True

    ov.coloured = pipeline.colour_lines(frame, turf, prep, fm.vp_cross)
    # A solved lattice is itself proof that the field is on screen - it needs
    # two yard lines lying on turf - and it is the evidence that carries the
    # awkward views coverage cannot: a field-goal attempt or a deep end-zone
    # angle puts the turf under half the frame, right where the menu sits.
    ov.field_ok = bool(ov.field_ok or ov.ok)
    return ov


def scale_overlay(ov: Overlay, sx: float, sy: float, shape) -> Overlay:
    """Map an overlay computed on the canonical frame onto a larger frame.

    Geometry is analysed at canonical size for consistency across sources, then
    scaled here so the annotation can be drawn at the source resolution rather
    than upsampling an already-drawn 960x720 image.
    """
    def seg(a):
        if a is None or len(a) == 0:
            return a
        b = np.asarray(a, float).copy()
        b[:, [0, 2]] *= sx
        b[:, [1, 3]] *= sy
        return b

    out = Overlay(ok=ov.ok, tracked=ov.tracked,
                  n_detected=ov.n_detected, n_predicted=ov.n_predicted,
                  line_spacing=ov.line_spacing, field_ok=ov.field_ok)
    out.field = (cv2.resize(ov.field.astype(np.uint8), (shape[1], shape[0]),
                            interpolation=cv2.INTER_NEAREST).astype(bool)
                 if ov.field is not None else None)
    out.region = (cv2.resize(ov.region.astype(np.uint8), (shape[1], shape[0]),
                             interpolation=cv2.INTER_NEAREST).astype(bool)
                  if ov.region is not None else None)
    out.yard_lines = seg(ov.yard_lines)
    out.yard_index = ov.yard_index
    out.yard_observed = ov.yard_observed
    out.sidelines = seg(ov.sidelines)
    out.goal_lines = seg(ov.goal_lines)
    if ov.ticks is not None and len(ov.ticks):
        t = np.asarray(ov.ticks, float).reshape(-1, 2).copy()
        t[:, 0] *= sx
        t[:, 1] *= sy
        out.ticks = t
    out.columns = [type(c)(np.array([c.seg[0] * sx, c.seg[1] * sy,
                                     c.seg[2] * sx, c.seg[3] * sy]),
                           c.n_ticks, c.yard_span, c.density) for c in ov.columns]
    out.coloured = {k: np.array([v[0] * sx, v[1] * sy, v[2] * sx, v[3] * sy])
                    for k, v in (ov.coloured or {}).items()}
    return out


# --------------------------------------------------------------------------- #
# drawing
# --------------------------------------------------------------------------- #
def _text(img, s, org, scale=0.5, col=(255, 255, 255), k=1.0):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale * k, (0, 0, 0),
                max(2, int(round(3 * k))), cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale * k, col,
                max(1, int(round(k))), cv2.LINE_AA)


def draw(frame: np.ndarray, ov: Overlay, label: str = "",
         motion=None) -> np.ndarray:
    """Annotate a frame.  Stroke and text scale with resolution, so the same
    overlay reads correctly at canonical size or at full source size."""
    vis = frame.copy()
    H, W = frame.shape[:2]
    k = W / 960.0
    t1, t2, t3 = max(1, int(round(2 * k))), max(2, int(round(3 * k))), max(1, int(round(4 * k)))

    # The tight turf-derived mask, so the highlight stops at the sideline.
    #
    # Nothing geometric is drawn unless the frame was actually solved.  A failed
    # frame still carries raw detections, and on a replay wipe or a team-select
    # screen those are turf texture and banner edges; drawing them claimed a
    # field where there was none.  "SEARCHING" in the corner is the honest
    # annotation for a frame we could not solve.
    fld = ov.field if ov.field is not None else ov.region
    if ov.field_ok and fld is not None and fld.any():
        m = fld.astype(bool)
        # Highlight by *contrast*, not by hue.  A translucent green wash is
        # invisible on green turf - which is exactly what the 18-minute capture
        # is, against the clips' teal - so the field keeps its own colour and
        # everything outside it is dimmed instead.  That reads identically on
        # any pitch colour, and it makes a wrong mask obvious rather than
        # subtle.
        out = ~m
        vis[out] = (vis[out] * OUTSIDE_DIM).astype(np.uint8)
        vis[m] = (0.90 * vis[m] + 0.10 * np.array(COL_REGION)).astype(np.uint8)
        # No mask contour is drawn.  The field boundary is a straight line, but a
        # per-pixel mask is not: where a player straddles the sideline he is not
        # enclosed by turf, so hole filling cannot remove him and any contour
        # must detour around him.  Raising the edge-length threshold does not
        # help - a player is taller than the threshold that would keep real
        # boundary edges.  The dimming carries the extent, and the *detected*
        # sidelines and goal lines below carry the boundary, fitted as lines.

    if not ov.ok:
        return _chrome(vis, ov, label, motion, k)

    play_col = {"orange_yellow": (0, 190, 255), "red": (40, 40, 255),
                "cyan": (255, 255, 0), "blue": (255, 120, 0)}
    for name, seg in (ov.coloured or {}).items():
        cv2.line(vis, (int(seg[0]), int(seg[1])), (int(seg[2]), int(seg[3])),
                 play_col.get(name, (255, 255, 255)), t2, cv2.LINE_AA)

    if ov.yard_lines is not None:
        obs = ov.yard_observed
        for j, s_ in enumerate(ov.yard_lines):
            seen = True if obs is None or j >= len(obs) else bool(obs[j])
            # Measured and inferred lines are both drawn - the lattice places
            # them equally well - but they are not the same claim, so the
            # inferred ones are drawn thinner and dimmer.
            cv2.line(vis, (int(s_[0]), int(s_[1])), (int(s_[2]), int(s_[3])),
                     COL_YARD if seen else COL_YARD_INFERRED,
                     t1 if seen else max(1, t1 - 1), cv2.LINE_AA)
            if ov.yard_index is not None and j < len(ov.yard_index):
                y = ov.yard_index[j]
                if np.isfinite(y):
                    mx, my = (s_[0] + s_[2]) / 2, (s_[1] + s_[3]) / 2
                    _text(vis, f"{y:+.0f}", (int(mx) + 4, int(my) - 4), 0.45,
                          COL_YARD if seen else COL_YARD_INFERRED, k)

    if ov.ticks is not None:
        for pt in np.asarray(ov.ticks).reshape(-1, 2):
            cv2.circle(vis, (int(pt[0]), int(pt[1])), max(2, int(round(2 * k))),
                       COL_TICK, -1, cv2.LINE_AA)

    for c in ov.columns or []:
        s_ = c.seg
        cv2.line(vis, (int(s_[0]), int(s_[1])), (int(s_[2]), int(s_[3])),
                 COL_COLUMN, t1, cv2.LINE_AA)

    for arr, col in ((ov.sidelines, COL_SIDELINE), (ov.goal_lines, COL_GOAL)):
        if arr is None:
            continue
        for s_ in arr:
            cv2.line(vis, (int(s_[0]), int(s_[1])), (int(s_[2]), int(s_[3])),
                     col, t3, cv2.LINE_AA)

    # Painted yard numbers are not drawn, and no longer detected.  The
    # connected-component detector that used to run here assumed a digit was
    # ~2 yd tall; measured on this footage the blobs sit at 0.81x the 5-yd line
    # spacing (median, n=200) and a directly measured "2" at 0.92x, so its size
    # window was centred below the digits and admitted hash marks instead.
    # Doing it properly means rectifying the turf flat so digits become upright,
    # then template matching as `hudtext` does for the HUD font.

    return _chrome(vis, ov, label, motion, k)


def _chrome(vis: np.ndarray, ov: Overlay, label: str, motion, k: float) -> np.ndarray:
    """Legend and status panel.  Drawn on every frame, solved or not."""
    H, W = vis.shape[:2]
    t2 = max(2, int(round(3 * k)))

    # ---- legend (bottom-left) ------------------------------------------- #
    items = [("field - off-field dimmed", COL_REGION),
             ("yard line - detected", COL_YARD),
             ("yard line - recovered", COL_YARD_INFERRED),
             ("1-yd ticks", COL_TICK), ("tick column", COL_COLUMN),
             ("sideline", COL_SIDELINE), ("goal line", COL_GOAL)]
    lh = int(round(15 * k))
    bw = int(round(200 * k))
    y0 = H - int(round(8 * k)) - lh * len(items)
    box = vis[y0 - lh:H - int(4 * k), int(4 * k):bw]
    vis[y0 - lh:H - int(4 * k), int(4 * k):bw] = (box * 0.35).astype(np.uint8)
    for i, (t, c) in enumerate(items):
        yy = y0 + lh * i
        cv2.line(vis, (int(10 * k), yy - int(4 * k)), (int(26 * k), yy - int(4 * k)),
                 c, t2, cv2.LINE_AA)
        _text(vis, t, (int(32 * k), yy), 0.38, (255, 255, 255), k)

    # ---- status (top-right, clear of the game's own scorebug) ------------ #
    if ov.ok:
        st = "PREDICTED" if ov.tracked else "TRACKING"
        stc = (0, 200, 255) if ov.tracked else (0, 255, 0)
    elif ov.field_ok:
        # The field is there and highlighted; the yard lattice is not solved.
        st = "FIELD ONLY"
        stc = (0, 220, 220)
    else:
        st = "SEARCHING"
        stc = (0, 140, 255)
    pw, ph = int(round(268 * k)), int(round(74 * k))
    bx, by = W - pw - int(6 * k), int(8 * k)
    panel = vis[by:by + ph, bx:bx + pw]
    vis[by:by + ph, bx:bx + pw] = (panel * 0.35).astype(np.uint8)
    _text(vis, label, (bx + int(6 * k), by + int(17 * k)), 0.40, (255, 255, 255), k)
    _text(vis, st, (bx + int(6 * k), by + int(35 * k)), 0.50, stc, k)
    # What was *drawn*, not what was detected: on an unsolved frame
    # ``yard_lines`` still holds raw candidates that the guard above rejected,
    # and reporting those beside "SEARCHING" reads as a contradiction.
    nl = 0 if (not ov.ok or ov.yard_lines is None) else len(ov.yard_lines)
    _text(vis, f"yard lines {nl} (det {ov.n_detected} / pred {ov.n_predicted})",
          (bx + int(6 * k), by + int(51 * k)), 0.36, (255, 255, 255), k)
    if motion is not None:
        mc = {"static": (120, 255, 120), "pan": (0, 220, 255),
              "zoom_in": (255, 180, 0), "zoom_out": (255, 180, 0),
              "pan+zoom": (255, 120, 255), "rotate": (255, 0, 220),
              "cut": (0, 0, 255)}.get(motion.label, (200, 200, 200))
        _text(vis, f"camera: {motion.label}  ({motion.speed_s:.1f} px/f, "
                   f"zoom {(motion.zoom_s - 1) * 100:+.2f}%, "
                   f"rot {motion.rot_s:+.2f}deg)",
              (bx + int(6 * k), by + int(67 * k)), 0.36, mc, k)
    return vis
