"""Render one video's field annotation, in parallel across processes.

    python scripts/render_video.py 6                 # source resolution
    python scripts/render_video.py 6 --workers 8
    python scripts/render_video.py 6 --canon         # 960x720 instead

``run_pipeline.py`` renders a clip in one process, which is the right shape for
a few hundred frames.  The full captures are 40-70 thousand frames at ~0.3 s of
analysis each - eleven hours before the hot paths were vectorised, six after -
and none of that work is shared between distant frames.

Splitting it
------------
The video is cut into contiguous chunks, one per worker, and each chunk is
encoded on its own.  The only state that crosses a chunk boundary is the
tracker's, so each worker starts decoding ``LEAD`` frames early and runs the
full analysis over them without writing anything.  By the time it reaches its
own first frame the tracker has locked exactly as it would have in a single
pass.  Two seconds of lead costs ~2% and removes the visible re-acquisition a
cold chunk start would otherwise show.

Encoding is handed to ffmpeg rather than ``cv2.VideoWriter``: the chunks have to
be concatenated afterwards, and x264 at a fixed CRF gives a far smaller and
sharper file than the MPEG-4 Part 2 OpenCV defaults to.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from blitzfield.config import CANON_H, CANON_W, WORK, get_video

#: Frames analysed before a chunk's first written frame, to warm the tracker.
LEAD = 120
CRF = "18"


def _encoder(path: pathlib.Path, w: int, h: int, fps: float) -> subprocess.Popen:
    """A raw-BGR sink that writes an H.264 file."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
           "-r", str(fps), "-i", "pipe:0",
           "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", CRF,
           "-pix_fmt", "yuv420p", str(path)]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE)


def render_chunk(job: dict) -> dict:
    """Analyse and encode ``[start, stop)`` of one video."""
    from blitzfield import (fieldcolor, hudmask, ingest, lattice, motion as mo,
                            render, tracker as tk)

    spec = get_video(job["video"])
    _, area = ingest.load(spec)
    turf = fieldcolor.TurfModel.from_json(
        (spec.workdir / "stage1" / "turf_model.json").read_text())
    hud = hudmask.load(spec)

    net = None
    if job["net"]:
        from blitzfield.fieldnet import FieldSegmenter, WEIGHTS
        w = job.get("weights") or WEIGHTS
        if FieldSegmenter.available(w):
            net = FieldSegmenter(w)

    start, stop = job["start"], job["stop"]
    warm = max(0, start - LEAD)
    size = (area.w, area.h) if job["full_res"] else (CANON_W, CANON_H)
    sx, sy = size[0] / CANON_W, size[1] / CANON_H

    enc = _encoder(pathlib.Path(job["path"]), size[0], size[1], job["fps"])
    est, track = mo.MotionEstimator(), tk.FieldTracker()
    rng = np.random.default_rng(0)
    last = render.Overlay()
    n = nok = n_pred = 0
    counts: dict = {}
    stats = []          # per-frame, so the render can be measured afterwards
    try:
        with ingest.FrameSource(spec, area, stride=1, start=warm, stop=stop,
                                seek=False, with_source=True) as src:
            for i, (canon, srcroi) in src:
                m = est.update(canon,
                               last.region if last.region is not None else None)
                if m.label == "cut":
                    track.reset()
                ov = render.analyse(canon, turf, hud, rng=rng, track=track,
                                    motion=m, net=net)
                last = ov
                if i < start:                  # lead-in: analysed, not written
                    continue
                counts[m.label] = counts.get(m.label, 0) + 1
                nok += 1 if ov.ok else 0
                n_pred += 1 if (ov.ok and ov.tracked) else 0
                nl = len(ov.yard_lines) if (ov.ok and ov.yard_lines is not None) else 0
                sp = None
                if nl >= 2:
                    ell = np.array([1.0, 0.0, -480.0])
                    sv = np.sort(lattice.transversal_positions(ov.yard_lines, ell)[0])
                    sv = sv[np.isfinite(sv)]
                    if len(sv) >= 2:
                        sp = round(float(np.median(np.diff(sv))), 2)
                stats.append({"f": int(i), "ok": bool(ov.ok),
                              "fok": bool(ov.field_ok), "n": nl,
                              "det": int(ov.n_detected), "sp": sp,
                              "fld": round(float(ov.field.mean()), 4)
                              if ov.field is not None else 0.0,
                              "cam": m.label})
                if job["full_res"]:
                    draw_ov = render.scale_overlay(ov, sx, sy, srcroi.shape)
                    canvas = srcroi
                else:
                    draw_ov, canvas = ov, canon
                vis = render.draw(canvas, draw_ov,
                                  label="%s  f%d" % (spec.vid, i), motion=m)
                enc.stdin.write(np.ascontiguousarray(vis).tobytes())
                n += 1
    finally:
        enc.stdin.close()
        enc.wait()
    sp = pathlib.Path(job["path"]).with_suffix(".json")
    sp.write_text(json.dumps(stats))
    return {"path": job["path"], "start": start, "stop": stop, "n": n,
            "ok": nok, "pred": n_pred, "camera": counts, "stats": str(sp)}


def plan(total: int, workers: int) -> list:
    """Contiguous, near-equal frame ranges."""
    edges = np.linspace(0, total, workers + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def concat(parts: list, out: pathlib.Path, source: pathlib.Path,
           with_audio: bool = True) -> None:
    """Join the chunks without re-encoding, carrying the original audio over."""
    lst = out.with_suffix(".parts.txt")
    lst.write_text("".join("file '%s'\n" % p.as_posix() for p in parts))
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", str(lst)]
    if with_audio:
        cmd += ["-i", str(source), "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-shortest"]
    else:
        cmd += ["-c", "copy"]
    cmd.append(str(out))
    subprocess.run(cmd, check=True)
    lst.unlink(missing_ok=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("video", help="processing order 1-7, or slug")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--canon", action="store_true",
                    help="render at 960x720 instead of source resolution")
    ap.add_argument("--no-net", action="store_true")
    ap.add_argument("--weights", default=None,
                    help="fieldnet checkpoint (default: the shipped one)")
    ap.add_argument("--no-audio", action="store_true")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N frames (for a quick look)")
    ap.add_argument("--out", default=None, help="output path override")
    a = ap.parse_args()

    spec = get_video(a.video)
    src = json.loads((spec.workdir / "stage0" / "source.json").read_text())
    fps = src["meta"]["fps"] or 30.0
    total = a.limit or src["meta"]["nb_frames"] or 0
    if not total:
        cap = cv2.VideoCapture(str(spec.path))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    tmp = WORK / ("%02d_%s" % (spec.order, spec.vid)) / "chunks"
    tmp.mkdir(parents=True, exist_ok=True)
    for old in list(tmp.glob("*.mp4")) + list(tmp.glob("*.json")):
        old.unlink()

    ranges = plan(total, a.workers)
    jobs = [{"video": spec.order, "start": s, "stop": e, "fps": fps,
             "full_res": not a.canon, "net": not a.no_net,
             "weights": a.weights,
             "path": str(tmp / ("part%03d.mp4" % k))}
            for k, (s, e) in enumerate(ranges)]
    print("[%d] %s" % (spec.order, spec.filename))
    print("    %d frames  %d chunks x ~%d  (+%d lead each)"
          % (total, len(jobs), total // len(jobs), LEAD), flush=True)

    t0 = time.time()
    res = []
    with Pool(len(jobs)) as pool:
        for r in pool.imap_unordered(render_chunk, jobs):
            res.append(r)
            done = sum(x["n"] for x in res)
            print("    chunk f%d-%d done (%d/%d; %d/%d frames, %.0fs)"
                  % (r["start"], r["stop"], len(res), len(jobs), done, total,
                     time.time() - t0), flush=True)
    res.sort(key=lambda r: r["start"])
    dt = time.time() - t0

    out_path = pathlib.Path(a.out) if a.out else spec.video_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concat([pathlib.Path(r["path"]) for r in res], out_path, spec.path,
           with_audio=not a.no_audio)

    n = sum(r["n"] for r in res)
    nok = sum(r["ok"] for r in res)
    npred = sum(r["pred"] for r in res)
    cam: dict = {}
    for r in res:
        for k, v in r["camera"].items():
            cam[k] = cam.get(k, 0) + v
    print("    %d frames  field present in %d (%.0f%%; %d from prediction)  "
          "%.0fs  (%.1f fps)" % (n, nok, 100.0 * nok / max(n, 1), npred, dt,
                                 n / max(dt, 1e-9)))
    print("    camera: " + "  ".join(
        "%s %.0f%%" % (k, 100.0 * v / max(n, 1))
        for k, v in sorted(cam.items(), key=lambda kv: -kv[1])))
    # One per-frame record for the whole video, so the result can be measured
    # rather than eyeballed.
    allstats = []
    for r in res:
        allstats.extend(json.loads(pathlib.Path(r["stats"]).read_text()))
    sp = spec.workdir / "render_stats.json"
    sp.write_text(json.dumps(allstats))
    nfok = sum(1 for s in allstats if s.get("fok"))
    print("    field highlighted in %d (%.0f%%)"
          % (nfok, 100.0 * nfok / max(len(allstats), 1)))
    solved = [s for s in allstats if s["ok"]]
    if solved:
        nl = np.array([s["n"] for s in solved])
        print("    solved frames: lines median %.0f  p10 %.0f  p90 %.0f  max %d"
              % (np.median(nl), np.percentile(nl, 10), np.percentile(nl, 90), nl.max()))
    print("    per-frame stats -> %s" % sp)
    print("    -> %s" % out_path, flush=True)


if __name__ == "__main__":
    main()
