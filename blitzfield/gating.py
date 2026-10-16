"""Stage 1 - shot segmentation and gameplay gating.

The long captures are mostly *not* gameplay: attract mode, team select, replays,
cutscenes and "STRATEGY TIPS" screens.  Feeding those to the geometry stages
produces confident nonsense, so we gate first.

The gate is unsupervised.  Per sampled frame we measure the fraction of pixels
matching the learned turf model; on a real capture that statistic is strongly
bimodal (field vs. everything else) and a 2-component 1-D Gaussian mixture
finds the split by itself.  Absolute floors keep it sane on inputs that are
*all* gameplay - the four hand-cut clips - where no split exists to find.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

import cv2
import numpy as np
from sklearn.mixture import GaussianMixture

from .fieldcolor import TurfModel
from .ingest import ActiveArea, FrameSource
from .config import VideoSpec

# A frame with less turf than this is never gameplay; more than this always is.
TURF_FLOOR = 0.18
TURF_CEIL = 0.55


@dataclass
class Segment:
    """A contiguous run of frames of one kind."""
    start: int
    stop: int          # exclusive, in source frame indices
    kind: str          # "gameplay" | "other"
    mean_turf: float


def frame_features(frame: np.ndarray, turf: TurfModel) -> dict:
    """Cheap per-frame descriptors used for gating and cut detection."""
    m = turf.turf_mask(frame, k=3.0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    h = frame.shape[0]
    return {
        "turf_frac": float(m.mean()),
        "turf_frac_lower": float(m[int(h * 0.4):].mean()),
        "mean_v": float(gray.mean()),
        "edge_density": float((cv2.Canny(gray, 60, 160) > 0).mean()),
        "hist": hist.flatten(),
    }


def _auto_threshold(values: np.ndarray) -> float:
    """Split a 1-D distribution into low/high modes, with guard rails.

    If the mixture is degenerate (one true mode, i.e. the whole video is
    gameplay) the components collapse together and we fall back to the floor.
    """
    v = values.reshape(-1, 1)
    if len(v) < 16 or np.ptp(values) < 0.05:
        return TURF_FLOOR
    gm = GaussianMixture(2, n_init=4, random_state=0).fit(v)
    lo, hi = np.sort(gm.means_.ravel())
    if hi - lo < 0.12:                      # poorly separated -> no real split
        return TURF_FLOOR
    grid = np.linspace(lo, hi, 256).reshape(-1, 1)
    p = gm.predict_proba(grid)
    lo_comp = int(np.argmin(gm.means_.ravel()))
    cross = np.argmin(np.abs(p[:, lo_comp] - 0.5))
    return float(np.clip(grid[cross, 0], TURF_FLOOR, TURF_CEIL))


def _smooth_binary(flags: np.ndarray, min_run: int) -> np.ndarray:
    """Remove runs shorter than ``min_run`` so segments are usable shots."""
    out = flags.copy()
    n = len(out)
    i = 0
    while i < n:
        j = i
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < min_run and (i > 0 or j < n):
            out[i:j] = out[i - 1] if i > 0 else out[j]
        i = j
    return out


def run(spec: VideoSpec, area: ActiveArea, stride: int = 5,
        cut_threshold: float = 0.45, min_run_seconds: float = 0.5) -> dict:
    """Execute stage 1 and persist ``stage1/gating.json``."""
    # ---- pass 1: learn the turf model from a spread of frames ------------- #
    boot = []
    with FrameSource(spec, area, stride=max(1, stride * 8)) as src:
        for _, f in src:
            boot.append(f)
    if not boot:
        raise RuntimeError("no frames for " + spec.vid)
    turf = TurfModel.fit(boot)

    # ---- pass 2: features over the whole video ---------------------------- #
    idx, feats, hists = [], [], []
    with FrameSource(spec, area, stride=stride) as src:
        fps = src.fps or 30.0
        for i, f in src:
            ft = frame_features(f, turf)
            hists.append(ft.pop("hist"))
            idx.append(i)
            feats.append(ft)
    idx = np.asarray(idx)
    turf_frac = np.asarray([f["turf_frac_lower"] for f in feats])

    # ---- scene cuts ------------------------------------------------------- #
    hists = np.asarray(hists)
    d = np.zeros(len(hists))
    if len(hists) > 1:
        num = (hists[1:] * hists[:-1]).sum(1)
        den = np.linalg.norm(hists[1:], axis=1) * np.linalg.norm(hists[:-1], axis=1)
        d[1:] = 1.0 - num / np.maximum(den, 1e-9)
    cuts = idx[d > cut_threshold].tolist()

    # ---- gameplay decision ------------------------------------------------ #
    thr = _auto_threshold(turf_frac)
    flags = turf_frac >= thr
    min_run = max(1, int(round(min_run_seconds * fps / stride)))
    flags = _smooth_binary(flags, min_run)

    segments = []
    i = 0
    while i < len(flags):
        j = i
        while j < len(flags) and flags[j] == flags[i]:
            j += 1
        stop = int(idx[j - 1] + stride) if j < len(flags) else int(idx[-1] + stride)
        segments.append(Segment(int(idx[i]), stop,
                                "gameplay" if flags[i] else "other",
                                float(turf_frac[i:j].mean())))
        i = j

    play = [s for s in segments if s.kind == "gameplay"]
    total_play = sum(s.stop - s.start for s in play)

    d_out = spec.workdir / "stage1"
    d_out.mkdir(parents=True, exist_ok=True)
    (d_out / "turf_model.json").write_text(turf.to_json())
    payload = {
        "stride": stride,
        "fps": fps,
        "turf_threshold": thr,
        "n_sampled": len(idx),
        "n_cuts": len(cuts),
        "cuts": cuts[:2000],
        "gameplay_frames": int(total_play),
        "gameplay_fraction": float(total_play / max(1, idx[-1] + stride)),
        "n_segments": len(segments),
        "n_gameplay_segments": len(play),
        "segments": [asdict(s) for s in segments],
    }
    (d_out / "gating.json").write_text(json.dumps(payload, indent=2))
    np.savez_compressed(d_out / "features.npz", idx=idx, turf_frac=turf_frac,
                        cut_dist=d,
                        mean_v=np.asarray([f["mean_v"] for f in feats]),
                        edge_density=np.asarray([f["edge_density"] for f in feats]))
    return payload
