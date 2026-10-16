"""Per-video driver: preparation, then the measurement self-check.

Deliberately one video at a time: each stage persists to ``out/<NN>_<vid>/`` and
later stages read what earlier ones wrote, so a run can be resumed or a single
stage re-run without redoing the rest.
"""
from __future__ import annotations

import json
import time

import cv2
import numpy as np

from . import fieldcolor, gating, geometry as G, hudmask, ingest, lattice, markings
from .config import LONG_CUT, YARDS_PER_CELL, VideoSpec


def colour_lines(frame, turf, prep, vp_cross, min_area: int = 400,
                 elong: float = 6.0, ang_tol: float = 3.0) -> dict:
    """Fit the game-drawn scrimmage / first-down lines.

    Each candidate must be a large, strongly elongated blob that is *parallel to
    the yard lines*.  Without that last test the blue channel locks onto team
    jerseys instead of the line.
    """
    out = {}
    masks = (prep.colour_masks if prep.colour_masks is not None
             else fieldcolor.coloured_line_masks(frame, turf))
    for name, m in masks.items():
        m = (m & prep.region & ~prep.hud).astype(np.uint8)
        n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
        best = None
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] < min_area:
                continue
            ys, xs = np.nonzero(lbl == i)
            P = np.stack([xs, ys], 1).astype(float)
            mu = P.mean(0)
            _, S, Vt = np.linalg.svd(P - mu, full_matrices=False)
            if S[1] < 1e-6 or S[0] / S[1] < elong:
                continue
            d = Vt[0]
            t = (P - mu) @ d
            seg = np.array([[*(mu + t.min() * d), *(mu + t.max() * d)]])
            if vp_cross is not None and \
                    G.vp_angular_error(seg, np.asarray(vp_cross, float))[0] > ang_tol:
                continue
            if best is None or S[0] > best[0]:
                best = (S[0], seg[0])
        if best is not None:
            out[name] = best[1]
    return out


def analyse(spec: VideoSpec, stride: int = 5, limit: int | None = None) -> dict:
    """Stages 3-5a over the gameplay frames of one video."""
    _, area = ingest.load(spec)
    turf = fieldcolor.TurfModel.from_json(
        (spec.workdir / "stage1" / "turf_model.json").read_text())
    hud = hudmask.load(spec)
    gate = json.loads((spec.workdir / "stage1" / "gating.json").read_text())
    play = [(s["start"], s["stop"]) for s in gate["segments"]
            if s["kind"] == "gameplay"]

    rng = np.random.default_rng(0)
    rows, n_done = [], 0
    for a, b in play:
        with ingest.FrameSource(spec, area, stride=stride, start=a, stop=b) as src:
            for i, fr in src:
                fm = markings.analyse_frame(fr, turf, hud=hud, frame_index=i, rng=rng)
                pen = fm.segs[fm.label == 0]
                row = {"frame": int(i), "n_pencil": int(len(pen)),
                       "n_boundary": int(len(fm.boundary))}
                if len(pen) >= 5:
                    Ls = G.seg_length(pen)
                    ell = np.array([1.0, 0.0, -fr.shape[1] / 2])
                    s, k = lattice.transversal_positions(pen[Ls >= LONG_CUT], ell)
                    if k.any():
                        sc = lattice.cluster_1d(s[k])
                        f5 = lattice.fit(sc)
                        if f5 is not None:
                            f5 = lattice.refine(sc, f5)
                            row.update(n_long=int(len(sc)), v=f5.v,
                                       alpha=f5.alpha, beta=f5.beta,
                                       score=f5.score, n_in=f5.n_in,
                                       cells=int(np.ptp(f5.index)) + 1)
                            prep = markings.FramePrep.build(fr, turf, hud=hud)
                            cl = colour_lines(fr, turf, prep, fm.vp_cross)
                            idx = {}
                            for nm, seg in cl.items():
                                ss, kk = lattice.transversal_positions(seg[None], ell)
                                if kk.any():
                                    u = 1.0 / (ss[0] - f5.v)
                                    idx[nm] = float((u - f5.beta) / f5.alpha)
                            if "orange_yellow" in idx and "red" in idx:
                                row["los_to_gain_yd"] = abs(
                                    idx["orange_yellow"] - idx["red"]) * YARDS_PER_CELL
                rows.append(row)
                n_done += 1
                if limit and n_done >= limit:
                    break
        if limit and n_done >= limit:
            break

    d = spec.workdir / "stage5"
    d.mkdir(parents=True, exist_ok=True)
    (d / "frames.json").write_text(json.dumps(rows, indent=1))

    fit = [r for r in rows if "score" in r]
    dist = [r["los_to_gain_yd"] for r in rows if "los_to_gain_yd" in r]
    summary = {
        "video": spec.filename,
        "frames_analysed": len(rows),
        "frames_with_lattice": len(fit),
        "lattice_yield": len(fit) / max(1, len(rows)),
        "median_pencil": float(np.median([r["n_pencil"] for r in rows])) if rows else 0,
        "median_long_lines": float(np.median([r["n_long"] for r in fit])) if fit else 0,
        "median_score": float(np.median([r["score"] for r in fit])) if fit else 0,
        "median_boundary": float(np.median([r["n_boundary"] for r in rows])) if rows else 0,
        "los_to_gain_n": len(dist),
        "los_to_gain_median_yd": float(np.median(dist)) if dist else None,
        "los_to_gain_iqr_yd": (float(np.percentile(dist, 75) - np.percentile(dist, 25))
                               if len(dist) > 3 else None),
    }
    (d / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def prepare(spec: VideoSpec, gate_stride: int | None = None) -> dict:
    """Stages 0-2 for one video.

    The gating stride adapts to length: the hand-cut clips are only a few
    seconds long and need dense sampling to give the mixture model anything to
    work with, while the 20-40 minute captures only need ~2 samples/second to
    segment shots - and at that stride the frame source switches to seeking,
    which avoids decoding tens of thousands of frames we would throw away.
    """
    t0 = time.time()
    ingest.run(spec)
    payload, area = ingest.load(spec)
    fps = payload["meta"]["fps"] or 30.0
    duration = payload["meta"]["duration"] or 0.0
    if gate_stride is None:
        gate_stride = max(1, round(fps / 2)) if duration > 60 else 3
    min_run = 1.0 if duration > 60 else 0.5
    g = gating.run(spec, area, stride=gate_stride, min_run_seconds=min_run)
    turf = fieldcolor.TurfModel.from_json(
        (spec.workdir / "stage1" / "turf_model.json").read_text())
    play = [(x["start"], x["stop"]) for x in g["segments"]
            if x["kind"] == "gameplay"]
    h = hudmask.run(spec, area, turf, segments=play)
    return {"gate_stride": gate_stride,
            "gating": {k: g[k] for k in
                       ("turf_threshold", "gameplay_fraction",
                        "n_gameplay_segments", "n_cuts", "n_sampled")},
            "hud": h, "seconds": round(time.time() - t0, 1)}
