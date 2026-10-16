"""Project-wide paths, constants and the video registry.

Canonical frame
---------------
Every video is normalised into a fixed CANON_W x CANON_H buffer.  A homography
absorbs any affine distortion introduced by that resize, so the only thing that
matters is that the transform is recorded exactly and is invertible.  Downstream
stages therefore never need to know the source resolution.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "Data"
OUT = ROOT / "out"      # deliverables only: the rendered videos
WORK = ROOT / "work"    # everything the stages need to talk to each other

# Canonical working frame (4:3).  Large enough that 1-2 px field lines survive.
CANON_W = 960
CANON_H = 720

# px: separates full yard lines (5-yard, span the field) from 1-yard ticks.
# The length distribution is strongly bimodal, so the exact cut is not critical.
LONG_CUT = 110.0
YARDS_PER_CELL = 5.0   # validated against HUD down-and-distance


@dataclass(frozen=True)
class VideoSpec:
    """One input video.  ``order`` is the user-specified processing priority."""

    order: int
    vid: str          # short slug used for output directories
    filename: str
    note: str

    @property
    def path(self) -> Path:
        return DATA / self.filename

    @property
    def workdir(self) -> Path:
        """Per-video scratch: learned models, gating, HUD map, diagnostics.

        Deliberately separate from ``OUT``.  The output folder is meant to hold
        finished videos and nothing else, but the stages still need somewhere
        durable to hand results to each other so a run can be resumed or a
        single stage re-run.
        """
        return WORK / f"{self.order:02d}_{self.vid}"

    @property
    def video_out(self) -> Path:
        return OUT / f"{self.vid}_field.mp4"


# Processing order is significant: clips first (short, dense, hand-picked by the
# user to exercise specific camera behaviour), then the long captures.
VIDEOS: list[VideoSpec] = [
    VideoSpec(1, "clip1_kickoff", "Clip_1 Kickoff return not QB.mp4",
              "kickoff return, wide camera, non-QB view"),
    VideoSpec(2, "clip2_congestion", "Clip_2 Congestion Contact Test.mp4",
              "heavy player congestion / contact"),
    VideoSpec(3, "clip3_qb", "Clip_3 Easiest QB Test.mp4",
              "clean QB view, easiest case"),
    VideoSpec(4, "clip4_swing", "Clip_4 Camera Swing and Distance Change Test.mp4",
              "camera yaw swing + zoom change"),
    VideoSpec(5, "arcade_full", "NFL-Blitz-Midway-1997-Arcade-Gameplay-Co_Media.mp4",
              "37 min arcade capture, 1440x1080 pillarboxed"),
    VideoSpec(6, "yt_midway", "YouTube_NFL-Blitz-MIDWAY-1997_Media.mp4",
              "18 min, 60 fps, 1440x1080 pillarboxed"),
    VideoSpec(7, "yt_part3", "YouTube_NFL-Blitz-1997-Part-3-Threading-The-Need_Media.mp4",
              "19 min, NON-4:3 active area (1746x982) - anisotropically stretched"),
    VideoSpec(8, "hail_mary_left", "1_Hail_Mary_Left_Side.mp4",
              "7.4 s single play, 30 fps, pillarboxed; dark desaturated grade"),
]

BY_ORDER = {v.order: v for v in VIDEOS}
BY_VID = {v.vid: v for v in VIDEOS}


def get_video(key: int | str) -> VideoSpec:
    """Look a video up by processing order (1-7) or by slug."""
    if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
        return BY_ORDER[int(key)]
    return BY_VID[key]
