"""Temporal tracking of the field geometry.

Detection alone is memoryless, and measurement on the clips shows exactly what
that costs: up to 63 consecutive frames with no lattice at all, 10-20% of the
line set turning over between neighbouring frames (58% at the 90th percentile),
and ~28 px of horizon jitter frame to frame on a camera that is in fact moving
smoothly.  Those are three faces of one gap - nothing carries geometry forward.

The camera-motion estimator already recovers a similarity transform between
consecutive frames, so the previous line set can be propagated through it and
detection demoted from "solve the frame" to "correct the prediction".  A line
transforms by the inverse transpose of the point transform, so propagation is
exact for the estimated motion; error enters only via the motion estimate, which
is why prediction is trusted for a bounded number of frames and then dropped.

The asymmetry is what buys the robustness: establishing the lattice from nothing
needs five or more lines, but *correcting* a tracked one needs a single matched
line.  Frames that fail cold detection outright are usually still carrying two
or three usable segments.

Yard indices ride along with the lines, so once a line is labelled it keeps that
label for as long as it is tracked, and the lattice is re-fitted against the
*tracked* set rather than the raw detections - which is what takes the jitter
out of the reported horizon and pitch.
"""
from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from . import geometry as G, lattice as L
from .config import YARDS_PER_CELL


def warp_line(l: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Carry a homogeneous line through a point transform ``H``."""
    lw = np.linalg.inv(H).T @ l
    n = np.linalg.norm(lw[:2])
    return lw / n if n > 1e-12 else lw


def line_at(l: np.ndarray, ell: np.ndarray):
    """Where a line crosses the transversal, or None if parallel to it."""
    p = np.cross(l, ell)
    if abs(p[2]) < 1e-9:
        return None
    return float(p[1] / p[2]) if abs(ell[0]) > abs(ell[1]) else float(p[0] / p[2])


@dataclass
class TrackedField:
    ok: bool = False
    vp_cross: object = None
    fit: object = None
    lines: dict = dc_field(default_factory=dict)   # yard index -> homogeneous line
    ages: dict = dc_field(default_factory=dict)    # yard index -> frames unconfirmed
    age: int = 0                                   # frames since a real detection
    n_detected: int = 0
    n_predicted: int = 0
    index_origin: int = 0                          # cell index labels count from

    @property
    def from_prediction(self) -> bool:
        return self.n_detected == 0 and bool(self.lines)

    def _transversal_fit(self, x: float):
        """Fit the yard lattice along the vertical line ``x = const``."""
        ell = np.array([1.0, 0.0, -float(x)])
        pts = [(k, line_at(l, ell)) for k, l in self.lines.items()]
        pts = [(k, s) for k, s in pts if s is not None and np.isfinite(s)]
        if len(pts) < 4:
            return None
        ks = np.array([p[0] for p in pts], float)
        s = np.array([p[1] for p in pts], float)
        return L.fit_labelled(s, ks)

    def all_lines(self, w: int, h: int, extra: int = 12) -> dict:
        """Every yard line the fitted lattice puts on screen, seen or not.

        The lattice is a *model of the field*, not a summary of what happened to
        be visible: once ``(v, alpha, beta)`` is fitted and validated - and it is
        validated, to ~0.2 yd against the game's own down-and-distance - the
        position of a yard line no detector reported this frame is determined
        rather than guessed.  Drawing only the detected subset leaves holes in
        the middle of the lattice (measured: indices 6-11 and 14 tracked on one
        frame, with 12 and 13 simply missing) and hands downstream tracking a
        set whose membership changes every frame.

        Detected lines are kept exactly as measured; the gaps are filled in.
        Anything that lands off the field is dropped later by clipping, which is
        what bounds ``extra`` in practice.
        """
        out = dict(self.lines)
        if len(self.lines) < 4:
            return out
        # The docstring below used to argue that clipping to the field bounds
        # `extra` in practice.  It does not, and that assumption cost a whole
        # render: when the fitted pitch collapses towards zero every generated
        # line lands *inside* the field, so clipping removes none of them and
        # the overlay drew 17,550 "yard lines" on one frame.  A lattice wider
        # than a football field is not a lattice; fill nothing in.
        if max(self.lines) - min(self.lines) > MAX_CELLS:
            return out
        # Two transversals, not one plus the vanishing point.  Building the line
        # through a single crossing and the pencil point makes the whole result
        # hostage to that point's accuracy, and it is the least certain thing
        # here - the yard-line VP sits far off frame, so a fraction of a degree
        # of error swings the far end of the line by several pixels.  Measured
        # that way, inferred lines landed a median 7.5 px from the paint while
        # detected ones sat within 1.0 px.  Two crossings a half-frame apart are
        # each interpolated from nearby measurements, and the line through them
        # inherits only local error.
        xa, xb = 0.25 * w, 0.75 * w
        fa, fb = self._transversal_fit(xa), self._transversal_fit(xb)
        if fa is None or fb is None:
            return out
        ks = sorted(self.lines)
        lo = min(ks) - extra
        hi = min(max(ks) + extra, lo + MAX_CELLS)
        for k in range(lo, hi + 1):
            if k in out:
                continue
            ua, ub = fa.alpha * k + fa.beta, fb.alpha * k + fb.beta
            if abs(ua) < 1e-12 or abs(ub) < 1e-12:
                continue
            sa, sb = fa.v + 1.0 / ua, fb.v + 1.0 / ub
            if not (np.isfinite(sa) and np.isfinite(sb)):
                continue
            if max(abs(sa), abs(sb)) > 1e5:
                continue
            l = np.cross(np.array([xa, sa, 1.0]), np.array([xb, sb, 1.0]))
            n = float(np.linalg.norm(l[:2]))
            if n < 1e-12:
                continue
            out[k] = l / n
        return out

    def segments(self, w: int, h: int, complete: bool = False):
        """Yard lines as frame-clipped segments, with labels and provenance.

        Returns ``(segs, yards, observed)``; ``observed`` is False for a line
        the lattice supplied rather than a detector.
        """
        lines = self.all_lines(w, h) if complete else self.lines
        segs, idx, seen = [], [], []
        for k, l in sorted(lines.items()):
            a, b, c = l
            pts = []
            if abs(b) > 1e-6:
                pts += [(0.0, -c / b), (float(w), (-c - a * w) / b)]
            if abs(a) > 1e-6:
                pts += [(-c / a, 0.0), ((-c - b * h) / a, float(h))]
            inside = [p for p in pts if -3 <= p[0] <= w + 3 and -3 <= p[1] <= h + 3]
            if len(inside) < 2:
                continue
            p0, p1 = inside[0], inside[-1]
            if np.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 25:
                continue
            segs.append([p0[0], p0[1], p1[0], p1[1]])
            idx.append((k - self.index_origin) * YARDS_PER_CELL)
            seen.append(k in self.lines)
        return (np.asarray(segs, float).reshape(-1, 4), np.asarray(idx, float),
                np.asarray(seen, bool))


#: Widest span of yard-line cells a frame can possibly show, in 5-yard cells.
#: The field is 120 yd including both end zones - 24 cells - so anything past
#: this is not a field, it is a broken fit.  Nothing subtle is being excluded
#: here: the failure this catches spans *thousands* of cells.
MAX_CELLS = 30


class FieldTracker:
    """Carries the yard-line set, and its labelling, across frames."""

    def __init__(self, max_age: int = 12, match_px: float = 12.0,
                 match_deg: float = 4.0, line_max_age: int = 8,
                 max_lines: int = 16):
        #: Consecutive frames the state may survive with nothing detected.
        #: Prediction is for bridging a dropout - a player crossing the lines,
        #: a frame of motion blur - which lasts a handful of frames.  At the old
        #: 45 it was also bridging the *start of a play-select screen*: three
        #: quarters of a second of overlay drawn over a menu before the state
        #: expired.  Twelve frames is 0.2 s at 60 fps and still covers every
        #: real dropout; the clips never reach it at all.
        self.max_age = max_age
        self.match_px = match_px
        self.match_deg = match_deg
        #: A tracked line must be re-confirmed by detection this often.  Without
        #: it the set grows without bound: predictions are never contradicted,
        #: so stale lines - especially near the horizon, where warp error is
        #: worst - survive forever and fan out across the frame.
        self.line_max_age = line_max_age
        self.max_lines = max_lines
        self.state: TrackedField | None = None
        #: Cell index the drawn labels are measured from.  The lattice's own
        #: index has an arbitrary origin - it falls out of wherever ``beta``
        #: landed in the cold fit - and can be in the hundreds, so the overlay
        #: was labelling yard lines "+1585".  The labels must stay *stable*,
        #: which is why lines are never renamed once assigned, but nothing
        #: requires them to start from an arbitrary number, so the first lock
        #: fixes an origin and every label is relative to it.
        self.index_origin: int | None = None
        #: Yard indices recovered by search last frame.  Kept here rather than
        #: on the frame state because recovery happens in the renderer, after
        #: the tracker has run, and its only job is to give the acceptance test
        #: a memory - see `render.snap_to_evidence`.
        self.recovered_idx: set = set()

    def reset(self):
        self.state = None
        self.recovered_idx = set()
        self.index_origin = None

    # ---- prediction ------------------------------------------------------ #
    def _predict(self, motion):
        st = self.state
        if st is None or not st.lines:
            return None, None
        if getattr(motion, "label", None) == "cut":
            return None, None
        H = getattr(motion, "H", None) if motion is not None else None
        if H is None:
            H = np.eye(3)          # no estimate: assume the camera held still
        H = np.asarray(H, float)
        try:
            lines = {k: warp_line(np.asarray(l, float), H)
                     for k, l in st.lines.items()}
        except np.linalg.LinAlgError:
            return None, None
        vp = None
        if st.vp_cross is not None:
            v = H @ np.asarray(st.vp_cross, float)
            n = np.linalg.norm(v)
            if n > 1e-12:
                vp = v / n
        return lines, vp

    def _predict_v(self, motion, w: int):
        """Carry the previous horizon crossing through the camera motion.

        ``v`` is the transversal coordinate of the horizon, and the transversal
        is the vertical line x = w/2, so it is just the y of a point on that
        line - which the motion transform moves like any other point.  The
        result only has to be good enough to seed a prior, and the value it
        seeds is far more stable than re-deriving it from scratch each frame.
        """
        st = self.state
        if st is None or st.fit is None or not np.isfinite(st.fit.v):
            return None
        H = getattr(motion, "H", None) if motion is not None else None
        if getattr(motion, "label", None) == "cut":
            return None
        if H is None:
            return float(st.fit.v)
        p = np.asarray(H, float) @ np.array([w / 2.0, float(st.fit.v), 1.0])
        return float(p[1] / p[2]) if abs(p[2]) > 1e-9 else float(st.fit.v)

    # ---- main step ------------------------------------------------------- #
    def step(self, det_segs: np.ndarray, vp_cross, fit, w: int, h: int,
             motion=None) -> TrackedField:
        """Fuse this frame's detected segments with the propagated state."""
        ell = np.array([1.0, 0.0, -w / 2.0])
        pred, pred_vp = self._predict(motion)
        v_pred = self._predict_v(motion, w)

        det_lines = (G.seg_to_line(np.asarray(det_segs, float))
                     if len(det_segs) else np.zeros((0, 3)))
        det_s = [line_at(l, ell) for l in det_lines]

        lines: dict = {}
        ages: dict = {}
        n_det = 0
        prev_ages = self.state.ages if self.state else {}

        if pred:
            # Correct the prediction: match each detection to a tracked line.
            # One match is enough, which is why this survives frames that cold
            # detection cannot solve at all.
            pred_s = {k: line_at(l, ell) for k, l in pred.items()}
            lines = dict(pred)
            ages = {k: prev_ages.get(k, 0) + 1 for k in pred}
            used = set()
            for l, s in zip(det_lines, det_s):
                if s is None:
                    continue
                best, bestd = None, self.match_px
                for k, ps in pred_s.items():
                    if ps is None or k in used:
                        continue
                    if abs(ps - s) < bestd:
                        ang = np.degrees(np.arccos(
                            min(1.0, abs(float(l[:2] @ pred[k][:2])))))
                        if ang <= self.match_deg:
                            best, bestd = k, abs(ps - s)
                if best is not None:
                    lines[best] = l          # detection is more accurate
                    ages[best] = 0           # confirmed this frame
                    used.add(best)
                    n_det += 1

            # A detection that matched nothing extends the set, provided the
            # tracked lattice can give it an integer index.
            if fit is not None and n_det:
                for l, s in zip(det_lines, det_s):
                    if s is None or any(abs(line_at(lines[k], ell) - s) < self.match_px
                                        for k in lines
                                        if line_at(lines[k], ell) is not None):
                        continue
                    k = self._index_of(s, lines, ell)
                    if k is not None:
                        lines[k] = l
                        ages[k] = 0
                        n_det += 1

        if not lines:
            # Cold start: needs a real lattice to establish the labelling.
            if fit is None or not len(det_lines):
                self.state = None if (self.state and self.state.age > self.max_age) \
                    else self.state
                if self.state is not None:
                    self.state.age += 1
                    return self.state
                return TrackedField()
            for l, s in zip(det_lines, det_s):
                if s is None:
                    continue
                u = 1.0 / (s - fit.v)
                k = (u - fit.beta) / fit.alpha
                if abs(k - round(k)) <= 0.25:
                    lines[int(round(k))] = l
                    ages[int(round(k))] = 0
                    n_det += 1
            # Re-origin on *every* cold start, not just the first.  A cold
            # start derives its indices from a fresh lattice fit whose beta -
            # and so whose index base - is arbitrary and unrelated to the last
            # one.  Keeping the old origin across a re-lock is what put labels
            # like "-3560" back on screen after the first lock had been fine.
            if lines:
                self.index_origin = int(round(np.median(list(lines))))

        # Retire lines that detection has stopped confirming, and never carry
        # more than the field can plausibly show at once - keeping those nearest
        # the confirmed ones, since warp error grows with distance.
        lines = {k: v for k, v in lines.items()
                 if ages.get(k, 0) <= self.line_max_age}
        ages = {k: ages.get(k, 0) for k in lines}
        if len(lines) > self.max_lines:
            fresh = sorted(lines, key=lambda k: (ages[k], abs(k)))[:self.max_lines]
            lines = {k: lines[k] for k in fresh}
            ages = {k: ages[k] for k in fresh}

        if not lines:
            self.state = None
            return TrackedField()

        # Last line of defence.  If the tracked labels have spread wider than a
        # football field the lattice has lost its scale, and the right response
        # is to abandon it and re-acquire, not to keep drawing from it.
        if max(lines) - min(lines) > MAX_CELLS:
            self.reset()
            return TrackedField()

        fit_t = self._refit(lines, ell, v_pred)
        if fit_t is not None:
            n0 = len(lines)
            lines, ages = self._evict(lines, ages, fit_t, ell)
            if len(lines) != n0:
                fit_t = self._refit(lines, ell, v_pred) or fit_t
            n_det = min(n_det, len(lines))

        st = TrackedField(
            ok=True,
            vp_cross=vp_cross if vp_cross is not None else pred_vp,
            fit=fit_t or fit,
            lines=lines, ages=ages,
            age=0 if n_det else ((self.state.age + 1) if self.state else 1),
            n_detected=n_det,
            n_predicted=max(0, len(lines) - n_det),
            index_origin=self.index_origin or 0,
        )
        if n_det == 0 and st.age > self.max_age:
            self.state = None
            return TrackedField()
        self.state = st
        return st

    # ---- helpers --------------------------------------------------------- #
    def _index_of(self, s: float, lines: dict, ell) -> int | None:
        """Integer yard index for a new line, from the tracked set's own spacing."""
        pts = [(k, line_at(l, ell)) for k, l in lines.items()]
        pts = [(k, v) for k, v in pts if v is not None]
        if len(pts) < 3:
            return None
        pts.sort(key=lambda t: t[1])
        ks = np.array([p[0] for p in pts], float)
        vs = np.array([p[1] for p in pts], float)
        # fit u = alpha*k + beta with u = 1/(s - v0), v0 from the tracked lattice
        v0 = self.state.fit.v if (self.state and self.state.fit) else None
        if v0 is None or abs(s - v0) < 1e-9:
            return None
        u = 1.0 / (vs - v0)
        A = np.stack([ks, np.ones_like(ks)], 1)
        try:
            (a, b), *_ = np.linalg.lstsq(A, u, rcond=None)
        except np.linalg.LinAlgError:
            return None
        if abs(a) < 1e-12:
            return None
        k = (1.0 / (s - v0) - b) / a
        if abs(k - round(k)) > 0.25:
            return None
        # A degenerate pitch makes `a` tiny, and then a line a few pixels away
        # indexes thousands of cells off.  Such an index is not a yard line of
        # this field, whatever the arithmetic says.
        if not (ks.min() - MAX_CELLS <= k <= ks.max() + MAX_CELLS):
            return None
        return int(round(k))

    def _evict(self, lines: dict, ages: dict, f, ell, tol: float = 0.35,
               min_keep: int = 5):
        """Drop the one line the fitted lattice least accounts for.

        A line carrying the wrong yard index is worse than a missing one: it
        drags the fit, opens a phantom gap where nothing is actually absent, and
        keeps that label for as long as it is tracked.

        It is tempting to *relabel* it instead - the lattice does hand out a
        real-valued index - but that was tried and is unsafe: re-indexing the
        set lets the fit move the labels to suit itself, the next refit then
        agrees with the labels it just created, and the pitch walks away frame
        by frame.  Measured, clip 4's scrimmage-to-first-down distance drifted
        from 29 yd to 75.6 against a HUD that still said 29.  Labels are the one
        thing that must come from detection, so a bad line is only ever removed,
        never renamed - and only one per frame, so an unlucky fit cannot empty
        the set.
        """
        if len(lines) <= min_keep or abs(f.alpha) < 1e-15:
            return lines, ages
        worst, worst_r = None, tol
        for k, l in lines.items():
            s = line_at(l, ell)
            if s is None or not np.isfinite(s) or abs(s - f.v) < 1e-9:
                continue
            kr = (1.0 / (s - f.v) - f.beta) / f.alpha
            r = abs(kr - k)
            if r > worst_r:
                worst, worst_r = k, r
        if worst is None:
            return lines, ages
        lines = {k: v for k, v in lines.items() if k != worst}
        ages = {k: v for k, v in ages.items() if k != worst}
        return lines, ages

    def _refit(self, lines: dict, ell, v_pred=None):
        """Re-fit the lattice against the *tracked* set.

        Fitting the stable set rather than the raw per-frame detections is what
        removes the horizon and pitch jitter that made yard labels jump.

        The tracked set carries its yard labels, so the pitch does not have to
        be searched for - ``fit_labelled`` solves it in closed form and spends
        its effort on the one badly-conditioned parameter, the horizon, anchored
        to where the camera motion says it should be.  The unlabelled search is
        kept as the fallback for a set too small or too ragged to fit directly.
        """
        pts = [(k, line_at(l, ell)) for k, l in lines.items()]
        pts = [(k, v) for k, v in pts if v is not None]
        if len(pts) < 4:
            return None
        ks = np.array([p[0] for p in pts], float)
        s = np.array([p[1] for p in pts], float)
        f = L.fit_labelled(s, ks, v_pred=v_pred)
        if f is not None:
            return f
        s = np.sort(s)
        f = L.fit(s)
        return L.refine(s, f) if f is not None else None
