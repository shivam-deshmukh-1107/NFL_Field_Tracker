"""Stage 0 - ingest and normalisation.

Responsibilities
----------------
1. Probe container/stream metadata (ffprobe).
2. Find the *active picture area* - these captures are 4:3 arcade output letter-
   or pillar-boxed into 16:9, and one upload is additionally stretched.
3. Emit an exact, invertible transform between canonical frame coordinates and
   original source pixels, so every later measurement can be mapped back.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import CANON_H, CANON_W, VideoSpec


# --------------------------------------------------------------------------- #
# metadata
# --------------------------------------------------------------------------- #
def ffprobe(path: Path) -> dict:
    """Return the video stream key fields as plain Python types."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,"
        "codec_name,pix_fmt,sample_aspect_ratio,display_aspect_ratio",
        "-of", "json", str(path),
    ]
    raw = json.loads(subprocess.run(cmd, capture_output=True, text=True,
                                    check=True).stdout)
    s = raw["streams"][0]

    def _rate(v):
        if not v or "/" not in str(v):
            return float(v) if v else 0.0
        n, d = str(v).split("/")
        return float(n) / float(d) if float(d) else 0.0

    return {
        "width": int(s["width"]),
        "height": int(s["height"]),
        "fps": _rate(s.get("r_frame_rate")),
        "avg_fps": _rate(s.get("avg_frame_rate")),
        "nb_frames": int(s["nb_frames"]) if s.get("nb_frames") else None,
        "duration": float(s["duration"]) if s.get("duration") else None,
        "codec": s.get("codec_name"),
        "pix_fmt": s.get("pix_fmt"),
        "sar": s.get("sample_aspect_ratio"),
        "dar": s.get("display_aspect_ratio"),
    }


# --------------------------------------------------------------------------- #
# active picture area
# --------------------------------------------------------------------------- #
@dataclass
class ActiveArea:
    x: int
    y: int
    w: int
    h: int
    aspect: float          # w / h of the active area
    anisotropy: float      # aspect / (4/3); 1.0 == undistorted 4:3
    n_samples: int
    luma_threshold: int


def detect_active_area(path: Path, n_samples: int = 64,
                       luma_threshold: int = 18,
                       coverage: float = 0.004) -> ActiveArea:
    """Locate the non-black picture region.

    Rather than trusting a single frame (which may be a fade or a dark menu) we
    take the per-pixel *maximum* luma over frames spread across the whole video.
    Any row/column that is part of the picture will contain bright content in at
    least one sample; true padding stays black in all of them.

    ``coverage`` guards against stray codec noise in the bars: a row/column must
    have at least this fraction of pixels above threshold to count as active.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("cannot open " + str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skip the first/last 5% - intros and outros are often full black.
    idxs = np.linspace(int(total * 0.05), int(total * 0.95), n_samples).astype(int)
    acc = np.zeros((H, W), np.uint8)
    got = 0
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        y = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        np.maximum(acc, y, out=acc)
        got += 1
    cap.release()
    if got == 0:
        raise RuntimeError("no frames decoded from " + str(path))

    hot = acc > luma_threshold
    row_ok = hot.mean(axis=1) > coverage
    col_ok = hot.mean(axis=0) > coverage
    if not row_ok.any() or not col_ok.any():
        x, y, w, h = 0, 0, W, H
    else:
        ys = np.flatnonzero(row_ok)
        xs = np.flatnonzero(col_ok)
        x, y = int(xs[0]), int(ys[0])
        w, h = int(xs[-1] - xs[0] + 1), int(ys[-1] - ys[0] + 1)

    aspect = w / h
    return ActiveArea(x, y, w, h, aspect, aspect / (4 / 3), got, luma_threshold)


# --------------------------------------------------------------------------- #
# canonical transform
# --------------------------------------------------------------------------- #
def canon_to_source_matrix(a: ActiveArea) -> np.ndarray:
    """3x3 mapping canonical frame coords -> original source pixel coords."""
    sx = a.w / CANON_W
    sy = a.h / CANON_H
    return np.array([[sx, 0.0, a.x],
                     [0.0, sy, a.y],
                     [0.0, 0.0, 1.0]], np.float64)


class FrameSource:
    """Iterate a video, yielding canonical-size BGR frames.

    Parameters
    ----------
    stride : take every Nth frame.
    start, stop : frame index bounds in the source video.
    """

    SEEK_THRESHOLD = 12
    """Stride above which random access beats decoding every frame.

    Measured on this footage: sequential decode runs at ~220 fps while a seek
    plus read costs about the same as decoding 10 frames.  Below the threshold
    sequential wins; above it, seeking avoids decoding frames we discard.
    """

    def __init__(self, spec: VideoSpec, area: ActiveArea, stride: int = 1,
                 start: int = 0, stop=None, seek: bool | None = None,
                 with_source: bool = False):
        self.spec, self.area, self.stride = spec, area, stride
        self.start, self.stop = start, stop
        self.with_source = with_source
        self.seek = (stride >= self.SEEK_THRESHOLD) if seek is None else seek
        self.cap = cv2.VideoCapture(str(spec.path))
        if not self.cap.isOpened():
            raise RuntimeError("cannot open " + str(spec.path))
        self.total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cap.release()

    def _canon(self, frame: np.ndarray):
        """Canonical frame, plus the un-resized crop when asked for.

        Analysis always runs on the canonical buffer so behaviour is identical
        across sources, but rendering wants the full-resolution crop.
        """
        a = self.area
        roi = frame[a.y:a.y + a.h, a.x:a.x + a.w]
        canon = cv2.resize(roi, (CANON_W, CANON_H), interpolation=cv2.INTER_AREA)
        return (canon, roi) if self.with_source else canon

    def __iter__(self):
        stop = self.stop if self.stop is not None else self.total
        if self.seek:
            for i in range(self.start, stop, self.stride):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = self.cap.read()
                if not ok:
                    break
                yield i, self._canon(frame)
            return
        if self.start:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start)
        i = self.start
        while i < stop:
            ok, frame = self.cap.read()
            if not ok:
                break
            if (i - self.start) % self.stride == 0:
                yield i, self._canon(frame)
            i += 1


def load(spec: VideoSpec) -> tuple[dict, ActiveArea]:
    """Read back a persisted stage 0 result."""
    p = spec.workdir / "stage0" / "source.json"
    payload = json.loads(p.read_text())
    area = ActiveArea(**payload["active_area"])
    return payload, area


def run(spec: VideoSpec, n_samples: int = 64) -> dict:
    """Execute stage 0 for one video and persist ``stage0/source.json``."""
    meta = ffprobe(spec.path)
    area = detect_active_area(spec.path, n_samples=n_samples)
    M = canon_to_source_matrix(area)

    d = spec.workdir / "stage0"
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "video": {"order": spec.order, "vid": spec.vid,
                  "filename": spec.filename, "note": spec.note},
        "meta": meta,
        "active_area": asdict(area),
        "canon": {"w": CANON_W, "h": CANON_H},
        "canon_to_source": M.tolist(),
        "source_to_canon": np.linalg.inv(M).tolist(),
    }
    (d / "source.json").write_text(json.dumps(payload, indent=2))

    return payload
