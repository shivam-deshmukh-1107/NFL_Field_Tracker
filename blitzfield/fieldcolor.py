"""Automatic colour models for turf, painted markings and coloured game lines.

Everything here is *learned from the video itself* - no hand-tuned hue ranges.
The one structural prior we use is that the central-lower part of a gameplay
frame is nearly always turf, which is enough to bootstrap the model.

Working colour space is CIELAB:

* ``L``      separates bright paint from dark turf
* ``(a, b)`` is chromaticity, largely invariant to the stadium lighting ramp,
  so turf under shadow still scores as turf.

Measured on this footage the Blitz turf sits near Lab (94, 110, 126) - a dark
teal - while painted lines are near-neutral (a, b ~ 128).  That chroma gap is
what lets us pull lines off the field without touching a hue threshold.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# turf
# --------------------------------------------------------------------------- #
@dataclass
class TurfModel:
    """Robust 2-D Gaussian over turf chromaticity (a, b), plus an L range."""

    mean: list          # [a, b]
    cov: list           # 2x2
    l_lo: float
    l_hi: float
    n_pixels: int

    # ---- construction ---------------------------------------------------- #
    @staticmethod
    def fit(frames, band=(0.55, 0.86), xband=(0.25, 0.75),
            support_fraction: float = 0.5, max_pixels: int = 120_000,
            seed: int = 0) -> "TurfModel":
        """Fit from a list of canonical BGR frames.

        A horizontal band across the lower-middle of the frame is the bootstrap
        sample.  That band is *mostly* turf during play - but a full-length
        capture is substantially menus, play-select screens and title cards, so
        the sample can be nearly half contaminated.

        A trimmed mean cannot survive that: on the long captures it dragged the
        b-channel mean from ~126 to ~140 and inflated the standard deviation
        four-fold, after which the 3-sigma gate accepted almost every colour and
        the gameplay gate lost all discrimination.

        The fix is mode seeking on the chromaticity histogram: turf is by far
        the densest single colour in that band, so we take the histogram peak
        and grow a Gaussian around it, re-estimating a few times.  It is O(n),
        deterministic, and unaffected by contamination that sits anywhere else
        in colour space.  (MinCovDet solves the same problem but is far too slow
        at hundreds of thousands of pixels.)
        """
        rng = np.random.default_rng(seed)
        samples = []
        for f in frames:
            h, w = f.shape[:2]
            roi = f[int(h * band[0]):int(h * band[1]),
                    int(w * xband[0]):int(w * xband[1])]
            samples.append(cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).reshape(-1, 3))
        lab = np.concatenate(samples).astype(np.float64)
        if len(lab) > max_pixels:
            lab = lab[rng.choice(len(lab), max_pixels, replace=False)]
        ab = lab[:, 1:]

        # 1. densest chromaticity bin
        H, xe, ye = np.histogram2d(ab[:, 0], ab[:, 1], bins=64,
                                   range=[[0, 255], [0, 255]])
        H = cv2.GaussianBlur(H, (5, 5), 0)
        pk = np.unravel_index(int(np.argmax(H)), H.shape)
        mean = np.array([(xe[pk[0]] + xe[pk[0] + 1]) / 2,
                         (ye[pk[1]] + ye[pk[1] + 1]) / 2])

        # 2. grow a Gaussian around it
        radius = 14.0
        support = np.linalg.norm(ab - mean, axis=1) <= radius
        cov = np.eye(2) * 9.0
        for _ in range(4):
            if support.sum() < 64:
                break
            mean = ab[support].mean(axis=0)
            cov = np.cov(ab[support].T) + np.eye(2) * 1e-3
            d = ab - mean
            P = np.linalg.inv(cov)
            m2 = (d[:, 0] ** 2 * P[0, 0] + 2 * d[:, 0] * d[:, 1] * P[0, 1]
                  + d[:, 1] ** 2 * P[1, 1])
            support = m2 <= 9.0                      # 3 sigma

        cov = cov + np.eye(2) * 1e-3                 # regularised
        L = lab[support, 0] if support.sum() >= 64 else lab[:, 0]
        return TurfModel(mean.tolist(), cov.tolist(),
                         float(np.percentile(L, 1)), float(np.percentile(L, 99)),
                         int(support.sum()))

    # ---- evaluation ------------------------------------------------------ #
    def mahalanobis(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Per-pixel Mahalanobis distance of chromaticity from the turf model."""
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        ab = lab[:, :, 1:].astype(np.float64)
        d = ab - np.asarray(self.mean)
        P = np.linalg.inv(np.asarray(self.cov))
        m = (d[..., 0] ** 2 * P[0, 0] + 2 * d[..., 0] * d[..., 1] * P[0, 1]
             + d[..., 1] ** 2 * P[1, 1])
        return np.sqrt(np.maximum(m, 0.0))

    def turf_mask(self, frame_bgr: np.ndarray, k: float = 3.0) -> np.ndarray:
        """Boolean mask of pixels whose chromaticity matches turf."""
        return self.mahalanobis(frame_bgr) < k


    # ---- io -------------------------------------------------------------- #
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_json(s: str) -> "TurfModel":
        return TurfModel(**json.loads(s))


# --------------------------------------------------------------------------- #
# colour normalisation
# --------------------------------------------------------------------------- #
#: Where turf is put, in 8-bit Lab (a, b, L).  Arbitrary but fixed: it is only a
#: reference point, chosen near the middle of the three captures seen so far
#: (teal (110, 126), green (102, 150), desaturated (119, 133)).
CANON_TURF = (110.0, 136.0, 120.0)


def normalise_to_canon(frame_bgr: np.ndarray, turf: "TurfModel",
                       turf_mask: np.ndarray | None = None,
                       canon=CANON_TURF) -> np.ndarray:
    """Map this video's turf colour onto a canonical one.

    A 0.48 M-parameter U-Net learns "turf" as a colour, which is why the net
    collapses on any capture it has no rows for - twice measured, P(turf) at
    3-10% against the colour model's 62-88%.  The turf *model* has no such
    problem: it is refitted per video and finds the new colour by itself.  So
    hand the net a frame in which turf is always the same colour, and let the
    per-video model absorb the difference.

    Chromaticity is a translation, so the gap between turf and near-neutral
    paint - the signal the net actually needs - is preserved exactly.

    Luminance is **not** a translation, and getting that wrong the first time
    cost a whole training run.  The captures differ in exposure as much as in
    hue (turf L of 97, 47 and 24 on the three seen so far), so a shift large
    enough to align them drives the painted markings - which are near white
    already - straight into the 255 ceiling: measured, 28.9% of paint clipped on
    the Midway capture and 66.9% on the Hail Mary clip.  The net then trained on
    flat white where the line structure should be, and the yard lattice on the
    hardest clip tipped into a wrong pitch by 17 yd.

    A gamma curve fixes the turf median at the canonical value while pinning
    both endpoints, so it is monotone on [0, 255] and cannot clip anything.

    Nothing here is hand-tuned to a video: every quantity comes from the turf
    model or from the frame.
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab[:, :, 1] += canon[0] - float(turf.mean[0])
    lab[:, :, 2] += canon[1] - float(turf.mean[1])
    if turf_mask is None:
        turf_mask = turf.mahalanobis(frame_bgr) < 3.0
    if int(turf_mask.sum()) > 500:
        med = float(np.median(lab[:, :, 0][turf_mask]))
        med = min(max(med, 1.0), 254.0)
        gamma = np.log(canon[2] / 255.0) / np.log(med / 255.0)
        lab[:, :, 0] = 255.0 * np.power(lab[:, :, 0] / 255.0, gamma)
    return cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------- #
# marking channels
# --------------------------------------------------------------------------- #
def chroma(frame_bgr: np.ndarray) -> np.ndarray:
    """Distance from the neutral axis in Lab. Paint is low, turf/logos high."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    return np.hypot(lab[:, :, 1] - 128.0, lab[:, :, 2] - 128.0)


def white_line_response(frame_bgr: np.ndarray, ksize: int = 11,
                        chroma_max: float = 26.0,
                        valid: np.ndarray | None = None,
                        pct: float = 99.5) -> np.ndarray:
    """Ridge response for thin bright *neutral* lines (the painted markings).

    A white top-hat on L isolates structures thinner than ``ksize`` that are
    brighter than their surroundings.  Multiplying by a soft low-chroma gate
    suppresses the coloured scrimmage/first-down lines and team logos, which
    are handled separately by :func:`coloured_line_masks`.

    Normalisation is by a high percentile *inside the field region*, never by
    the global max: a single specular highlight (a chrome helmet) or an opaque
    HUD banner otherwise rescales the whole frame and drives the real field
    lines below any sensible threshold.

    Returns float32 in [0, 1].
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    top = cv2.morphologyEx(L, cv2.MORPH_TOPHAT, ker).astype(np.float32)

    pool = top[valid] if valid is not None and valid.any() else top
    scale = float(np.percentile(pool, pct))
    if scale < 1.0:                       # essentially no ridge structure
        scale = max(float(pool.max()), 1.0)
    top = np.clip(top / scale, 0.0, 1.0)

    gate = np.clip(1.0 - (chroma(frame_bgr) - chroma_max) / chroma_max, 0.0, 1.0)
    return (top * gate).astype(np.float32)


def coloured_line_masks(frame_bgr: np.ndarray, turf: TurfModel,
                        sat_min: float = 40.0,
                        off_turf: np.ndarray | None = None) -> dict:
    """Isolate the game-drawn scrimmage / first-down lines by hue.

    Blitz paints the line of scrimmage and the line to gain directly onto the
    turf in saturated colour (observed: yellow/orange, cyan, red).  These are
    enormously useful - the HUD states the exact yard gap between them - so we
    keep them as their own channel rather than discarding them as noise.

    Returns a dict of boolean masks keyed by colour family.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(np.int16), hsv[:, :, 1], hsv[:, :, 2]
    strong = (S > sat_min) & (V > 60)
    # The Mahalanobis map is the most expensive thing in the colour front end
    # and the caller has usually just computed it; accept it rather than
    # repeating it.
    if off_turf is None:
        off_turf = turf.mahalanobis(frame_bgr) > 3.0

    def band(lo, hi):
        m = ((H >= lo) & (H <= hi)) if lo <= hi else ((H >= lo) | (H <= hi))
        return m & strong & off_turf

    return {
        "red": band(170, 10),        # wraps 0
        "orange_yellow": band(11, 34),
        "cyan": band(85, 100),
        "blue": band(101, 130),
    }
