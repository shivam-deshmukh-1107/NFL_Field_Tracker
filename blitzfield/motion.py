"""Camera motion classification between consecutive frames.

Blitz's camera follows the ball: it pans along the field, dollies in and out,
swings, and sometimes holds still.  Knowing which is happening is useful
downstream - a static stretch means one calibration covers the whole segment, a
zoom means pixels-per-yard is changing, and a cut means history must be dropped.

Estimation
----------
Corners are tracked *on the turf* with Lucas-Kanade optical flow and a
similarity transform is fitted by RANSAC.  Two details matter:

* Restricting features to the learned field region keeps the HUD - which never
  moves regardless of the camera - from dragging the estimate toward zero.
  Learning that region beats a hardcoded UI rectangle: a fixed mask tuned to one
  capture's layout discarded 36% of the actual turf on this footage.
* Forward-backward consistency rejects tracks that slid along a line.  The field
  is full of long parallel edges, so the aperture problem is the dominant error
  mode here, and descriptor matching has no equivalent guard.

Measured against ORB descriptor matching on the same 80 consecutive frames, this
gives ~45% less frame-to-frame jitter (1.06 px vs 1.94 px), and noticeably
cleaner scale (sd 0.0032 vs 0.0051) and rotation (sd 0.08 deg vs 0.27 deg) -
which matters because those feed thresholds of the same order.

Classification
--------------
Raw parameters are median-filtered *before* thresholding, rather than
classifying each frame and voting on the labels.  Smoothing the continuous
signal keeps magnitude information a vote throws away, and a median (not a mean)
stays robust to the occasional bad RANSAC fit.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

STATIC_PX = 0.8          # median displacement below this is a held camera
ZOOM_EPS = 0.0035        # |scale - 1| above this is a dolly / zoom
ROT_EPS_DEG = 0.35       # rotation above this is a camera roll / orbit
CUT_PX = 45.0            # displacement this large *and* few tracks means a cut
CUT_TRACKS = 30

#: What each label implies for anything consuming the geometry downstream.
MEANING = {
    "static": "camera held; one calibration covers the whole segment",
    "pan": "camera translates; line positions shift but pixels-per-yard holds",
    "zoom_in": "scale grows; pixels-per-yard changes, recalibrate per frame",
    "zoom_out": "scale shrinks; pixels-per-yard changes, recalibrate per frame",
    "pan+zoom": "translation and scale change together; hardest case",
    "rotate": "camera rolls with little translation; line slopes change",
    "cut": "correspondence lost; hard segment boundary, do not track across",
    "unknown": "estimation failed (too few tracks)",
}


@dataclass
class Motion:
    label: str = "unknown"
    dx: float = 0.0
    dy: float = 0.0
    zoom: float = 1.0        # >1 moving in
    rot_deg: float = 0.0
    n_tracks: int = 0
    speed: float = 0.0       # median displacement magnitude, px/frame
    speed_s: float = 0.0     # smoothed, the value actually classified
    zoom_s: float = 1.0
    rot_s: float = 0.0
    #: 3x3 similarity taking previous-frame pixels to this frame; identity when
    #: estimation failed.  Geometry tracking propagates through this.
    H: object = None


def classify(speed: float, zoom: float, rot_deg: float, n_tracks: int) -> str:
    """Label one (already smoothed) motion estimate."""
    # A cut needs *both* a huge apparent jump and a collapse in correspondence.
    # Speed alone misfires on a genuinely fast pan, which still tracks well.
    if speed > CUT_PX and n_tracks < CUT_TRACKS:
        return "cut"
    moving = speed > STATIC_PX
    zooming = abs(zoom - 1.0) > ZOOM_EPS
    rotating = abs(rot_deg) > ROT_EPS_DEG
    if not (moving or zooming or rotating):
        return "static"
    if zooming and moving:
        return "pan+zoom"
    if zooming:
        return "zoom_in" if zoom > 1.0 else "zoom_out"
    if rotating and not moving:
        return "rotate"
    return "pan"


class MotionEstimator:
    """Frame-to-frame similarity transform, classified from smoothed signals."""

    def __init__(self, smooth: int = 5, max_corners: int = 400):
        self.prev = None
        self.max_corners = max_corners
        self.hist: deque = deque(maxlen=max(1, smooth))

    def update(self, frame: np.ndarray, mask: np.ndarray | None = None) -> Motion:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        m = None
        if mask is not None:
            m = cv2.erode(mask.astype(np.uint8),
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
        if self.prev is None:
            self.prev = gray
            return Motion()

        p0 = cv2.goodFeaturesToTrack(self.prev, self.max_corners, 0.01, 8, mask=m)
        if p0 is None or len(p0) < 12:
            self.prev = gray
            return self._finish(Motion())

        p1, stt, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, p0, None,
                                              winSize=(21, 21), maxLevel=3)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(gray, self.prev, p1, None,
                                              winSize=(21, 21), maxLevel=3)
        ok = (stt.ravel() == 1) & (stb.ravel() == 1)
        ok &= np.linalg.norm((p0 - pb).reshape(-1, 2), axis=1) < 1.5
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        self.prev = gray
        if len(a) < 12:
            return self._finish(Motion())

        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC,
                                             ransacReprojThreshold=2.5)
        if M is None:
            return self._finish(Motion())
        H = np.eye(3)
        H[:2, :] = M
        mo = Motion(
            H=H,
            dx=float(M[0, 2]), dy=float(M[1, 2]),
            zoom=float(np.hypot(M[0, 0], M[0, 1])),
            rot_deg=float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))),
            n_tracks=int(inl.sum()) if inl is not None else 0,
            speed=float(np.median(np.linalg.norm(b - a, axis=1))),
        )
        return self._finish(mo)

    def _finish(self, mo: Motion) -> Motion:
        """Median-filter the parameters, then classify from the smoothed values."""
        self.hist.append(mo)
        live = [h for h in self.hist if h.n_tracks > 0]
        if not live:
            mo.label = "unknown"
            return mo
        mo.speed_s = float(np.median([h.speed for h in live]))
        mo.zoom_s = float(np.median([h.zoom for h in live]))
        mo.rot_s = float(np.median([h.rot_deg for h in live]))
        tracks = int(np.median([h.n_tracks for h in live]))
        mo.label = ("unknown" if mo.n_tracks == 0
                    else classify(mo.speed_s, mo.zoom_s, mo.rot_s, tracks))
        return mo


# --------------------------------------------------------------------------- #
# segments
# --------------------------------------------------------------------------- #
def segment(labels, frames=None, min_len: int = 5) -> list:
    """Merge per-frame labels into contiguous runs.

    Runs shorter than ``min_len`` are absorbed into the preceding segment: they
    are almost always single-frame estimator noise rather than real changes in
    what the camera is doing.
    """
    if not labels:
        return []
    lab = list(labels)
    i = 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        if j - i < min_len and i > 0:
            fill = lab[i - 1]
            for k in range(i, j):
                lab[k] = fill
            i = 0        # re-scan: absorbing may have created a longer run
            continue
        i = j

    out, i = [], 0
    while i < len(lab):
        j = i
        while j < len(lab) and lab[j] == lab[i]:
            j += 1
        out.append({
            "label": lab[i],
            "start": int(frames[i]) if frames is not None else i,
            "end": int(frames[j - 1]) if frames is not None else j - 1,
            "n_frames": j - i,
            "meaning": MEANING.get(lab[i], ""),
        })
        i = j
    return out
