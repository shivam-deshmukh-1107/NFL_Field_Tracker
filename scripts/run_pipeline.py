"""Field-detection pipeline: video in, annotated video out.

    python scripts/run_pipeline.py            # all four clips
    python scripts/run_pipeline.py 1 2 3 4    # by priority order
    python scripts/run_pipeline.py 5 --canon  # 960x720 instead of source res

The only thing written to ``out/`` is the rendered video, one per input, plus
``field_template.json`` which is kept as a standing reference. Everything the
stages need to talk to each other - learned turf model, gating, HUD map -
lives under ``work/`` and is regenerated automatically when missing, so the
output folder can be deleted at any time without breaking a re-run.

Analysis runs on the canonical 960x720 buffer so behaviour is identical across
sources; the overlay is scaled onto the full-resolution frame for drawing,
which is sharper than upscaling an already-drawn canonical image.
"""
import argparse
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from blitzfield import (fieldcolor, hudmask, ingest, motion as mo, pipeline,
                        render, tracker as tk)
from blitzfield.config import CANON_H, CANON_W, OUT, get_video

def load_net(enabled=True):
    """The learned front end, when it has been trained and torch is present.

    Optional on purpose: the colour pipeline is the fallback, so a checkout with
    no weights - or no torch - still renders, just with the older front end.
    """
    if not enabled:
        return None
    try:
        from blitzfield.fieldnet import FieldSegmenter
    except Exception as e:
        print(f"    fieldnet unavailable ({e.__class__.__name__}); "
              "using the colour model", flush=True)
        return None
    if not FieldSegmenter.available():
        print("    no fieldnet weights; using the colour model", flush=True)
        return None
    net = FieldSegmenter()
    print(f"    fieldnet on {net.device}", flush=True)
    return net


def ensure_prepared(spec, force=False):
    """Run stages 0-2 if their outputs are not already on disk."""
    needed = [spec.workdir / "stage0" / "source.json",
              spec.workdir / "stage1" / "turf_model.json",
              spec.workdir / "stage1" / "gating.json",
              spec.workdir / "stage2" / "hud_mask.png"]
    if force or not all(p.exists() for p in needed):
        print("    preparing (learning turf model, gating, HUD map)...", flush=True)
        info = pipeline.prepare(spec)
        g = info["gating"]
        print(f"    gameplay {g['gameplay_fraction']:.0%} in "
              f"{g['n_gameplay_segments']} segment(s); "
              f"HUD {info['hud']['hud_fraction']:.1%} of frame "
              f"({info['seconds']:.0f}s)", flush=True)


def run(vid, full_res=True, force_prep=False, net=None):
    spec = get_video(vid)
    print(f"[{spec.order}] {spec.filename}", flush=True)
    ensure_prepared(spec, force=force_prep)

    _, area = ingest.load(spec)
    turf = fieldcolor.TurfModel.from_json(
        (spec.workdir / "stage1" / "turf_model.json").read_text())
    hud = hudmask.load(spec)

    OUT.mkdir(parents=True, exist_ok=True)
    out_path = spec.video_out

    cap = cv2.VideoCapture(str(spec.path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    size = (area.w, area.h) if full_res else (CANON_W, CANON_H)
    sx, sy = size[0] / CANON_W, size[1] / CANON_H
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

    est = mo.MotionEstimator()
    track = tk.FieldTracker()
    rng = np.random.default_rng(0)
    last, nok, n = render.Overlay(), 0, 0
    labels, counts = [], {}
    n_pred = 0
    t0 = time.time()
    with ingest.FrameSource(spec, area, stride=1, with_source=True) as src:
        for i, (canon, srcroi) in src:
            # motion first: the tracker propagates geometry through it
            prep_region = last.region if last.region is not None else None
            m = est.update(canon, prep_region)
            if m.label == "cut":
                track.reset()
            ov = render.analyse(canon, turf, hud, rng=rng, track=track,
                                motion=m, net=net)
            labels.append(m.label)
            counts[m.label] = counts.get(m.label, 0) + 1
            if ov.ok:
                last, nok = ov, nok + 1
            else:
                last = ov
            if full_res:
                draw_ov, canvas = render.scale_overlay(ov, sx, sy, srcroi.shape), srcroi
            else:
                draw_ov, canvas = ov, canon
            vw.write(render.draw(canvas, draw_ov,
                                 label=f"{spec.vid}  f{i}", motion=m))
            n_pred += 1 if (ov.ok and ov.tracked) else 0
            n += 1
    vw.release()

    dt = time.time() - t0
    segs = mo.segment(labels)
    order = sorted(counts.items(), key=lambda kv: -kv[1])
    print(f"    {size[0]}x{size[1]}  {n} frames  field present in {nok} "
          f"({nok / max(n, 1):.0%}; {n_pred} from prediction)  {dt:.0f}s")
    print("    camera: " + "  ".join(f"{k} {v / max(n, 1):.0%}" for k, v in order))
    for sg in segs:
        print(f"      f{sg['start']:>4}-{sg['end']:<4} {sg['label']:<9} "
              f"({sg['n_frames']:>3}f)")
    print(f"    -> {out_path}", flush=True)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="*", default=["1", "2", "3", "4"],
                    help="priority order 1-7 (default: the four clips)")
    ap.add_argument("--canon", action="store_true",
                    help="render at 960x720 instead of source resolution")
    ap.add_argument("--force-prep", action="store_true",
                    help="re-learn turf/gating/HUD even if already cached")
    ap.add_argument("--no-net", action="store_true",
                    help="ignore trained fieldnet weights, use the colour model")
    a = ap.parse_args()
    net = load_net(enabled=not a.no_net)
    for v in a.videos:
        run(int(v), full_res=not a.canon, force_prep=a.force_prep, net=net)
