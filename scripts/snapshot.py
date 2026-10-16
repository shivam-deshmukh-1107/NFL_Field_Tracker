"""Freeze the exact per-frame geometry of the clips, so a change can be checked
for behaviour rather than just for speed.

An optimisation is only safe if the output is identical; a model change is only
an improvement if you can say what moved.  Both need the same thing: a record
of what the pipeline produced, frame by frame, that can be compared later.

    python scripts/snapshot.py take before
    ... make the change ...
    python scripts/snapshot.py take after
    python scripts/snapshot.py diff before after

``diff`` is exact on the flags and the counts, and reports the largest
numerical movement in the yard lines and the field coverage.  Byte-identical
output means a refactor was behaviour-neutral; anything else is a change you
have to justify.

The companion gate is ``valid_scale.py``, which checks the yard scale against
the HUD.  This one checks that *nothing else* moved.

Compare snapshots taken in the *same* environment.  The net's two channels are
thresholded at 0.625 and 0.925, so a handful of pixels sitting on either side
of a cut will land differently between a CUDA and a CPU kernel, or between
torch builds - enough to move a few line detections without anything in this
repository having changed.  Across the four clips that shows up as a drift of
well under 1% in the mean line count.  A real regression is larger than that,
and shows in the flags rather than only in the counts.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from blitzfield import (fieldcolor, hudmask, ingest, motion as mo, render,
                        tracker as tk)
from blitzfield.config import WORK, get_video

#: Snapshots live under work/ with the other intermediates; out/ is for
#: deliverables only.
SNAPDIR = WORK / "regress"

#: The four hand-picked clips are the regression suite: short, dense, and each
#: chosen to exercise a different camera behaviour.
DEFAULT_VIDEOS = [1, 2, 3, 4]


#: Frames analysed before the first recorded one when starting at an offset, so
#: the tracker is locked rather than mid-acquisition.  Same reasoning, and the
#: same figure, as ``render_video.LEAD``.
LEAD = 120


def take(tag: str, n_per_clip: int = 60, use_net: bool = True,
         weights: str | None = None, videos=DEFAULT_VIDEOS,
         start: int = 0) -> pathlib.Path:
    net = None
    if use_net:
        from blitzfield.fieldnet import FieldSegmenter, WEIGHTS
        net = FieldSegmenter(weights or WEIGHTS)
        print("  weights %s  thr %s %s  normalise %s"
              % (pathlib.Path(weights or WEIGHTS).name, net.turf_thr,
                 net.line_thr, getattr(net, "normalise", False)))

    out = {}
    for v in videos:
        spec = get_video(v)
        _, area = ingest.load(spec)
        turf = fieldcolor.TurfModel.from_json(
            (spec.workdir / "stage1" / "turf_model.json").read_text())
        hud = hudmask.load(spec)
        est, track = mo.MotionEstimator(), tk.FieldTracker()
        rng = np.random.default_rng(0)      # fixed: RANSAC must be replayable
        last, rows = render.Overlay(), []
        warm = max(0, start - LEAD) if start else 0
        with ingest.FrameSource(spec, area, stride=1, start=warm,
                                seek=bool(start)) as src:
            for i, fr in src:
                m = est.update(fr, last.region if last.region is not None else None)
                if m.label == "cut":
                    track.reset()
                ov = render.analyse(fr, turf, hud, rng=rng, track=track,
                                    motion=m, net=net)
                last = ov
                if i < start:            # lead-in: analysed, not recorded
                    continue
                rows.append(dict(
                    i=int(i), ok=bool(ov.ok), fok=bool(ov.field_ok),
                    tracked=bool(ov.tracked), nd=int(ov.n_detected),
                    cam=m.label,
                    field=round(float(ov.field.mean()), 6)
                    if ov.field is not None else 0.0,
                    yl=(np.round(np.asarray(ov.yard_lines, float), 4).tolist()
                        if ov.yard_lines is not None and len(ov.yard_lines) else []),
                    yi=(np.round(np.asarray(ov.yard_index, float), 4).tolist()
                        if ov.yard_index is not None and len(ov.yard_index) else []),
                ))
                if len(rows) >= n_per_clip:
                    break
        out[spec.vid] = rows
        print("  %-18s %d frames  solved %d  highlighted %d  mean lines %.2f"
              % (spec.vid, len(rows), sum(r["ok"] for r in rows),
                 sum(r["fok"] for r in rows),
                 float(np.mean([len(r["yl"]) for r in rows]))))

    SNAPDIR.mkdir(parents=True, exist_ok=True)
    p = SNAPDIR / ("snap_%s.json" % tag)
    p.write_text(json.dumps(out))
    print("->", p)
    return p


def _load(tag: str) -> dict:
    p = SNAPDIR / ("snap_%s.json" % tag)
    if not p.exists():
        sys.exit("no snapshot %r (expected %s)" % (tag, p))
    return json.loads(p.read_text())


def diff(tag_a: str, tag_b: str) -> int:
    """0 if the two snapshots are identical, 1 otherwise."""
    A, B = _load(tag_a), _load(tag_b)
    if set(A) != set(B):
        print("different videos: %s vs %s" % (sorted(A), sorted(B)))
        return 1

    changed = 0
    for vid in sorted(A):
        ra, rb = A[vid], B[vid]
        if len(ra) != len(rb):
            print("  %-18s FRAME COUNT %d -> %d" % (vid, len(ra), len(rb)))
            changed += 1
            continue
        flags = [f for f in ("ok", "fok", "tracked", "nd", "cam")
                 if any(x[f] != y[f] for x, y in zip(ra, rb))]
        dfield = max((abs(x["field"] - y["field"]) for x, y in zip(ra, rb)),
                     default=0.0)
        dlines = max((abs(len(x["yl"]) - len(y["yl"])) for x, y in zip(ra, rb)),
                     default=0)
        # Endpoints are where the drawn line was clipped to the field mask,
        # so a line that barely moved can still report a large endpoint shift
        # if its extent changed.  Treat this as a tripwire - nonzero means
        # something moved - and not as a measure of how far the geometry moved.
        dgeom = 0.0
        for x, y in zip(ra, rb):
            if len(x["yl"]) == len(y["yl"]) and x["yl"]:
                dgeom = max(dgeom, float(np.abs(np.asarray(x["yl"])
                                                - np.asarray(y["yl"])).max()))
        if flags or dfield or dlines or dgeom:
            changed += 1
            print("  %-18s CHANGED" % vid)
            if flags:
                print("     flags differing: %s" % ", ".join(flags))
                for f in flags:
                    n = sum(1 for x, y in zip(ra, rb) if x[f] != y[f])
                    print("       %-8s on %d/%d frames" % (f, n, len(ra)))
            if dlines:
                print("     line count moves by up to %d" % dlines)
            if dgeom:
                print("     yard-line endpoints differ (max %.1f px; extent-"
                      "sensitive, treat as a tripwire not a magnitude)" % dgeom)
            if dfield:
                print("     field coverage moves by up to %.6f" % dfield)
        else:
            print("  %-18s identical (%d frames)" % (vid, len(ra)))

    print("\n%s" % ("IDENTICAL - the change was behaviour-neutral" if not changed
                    else "%d of %d videos changed" % (changed, len(A))))
    return 1 if changed else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("take", help="record a snapshot")
    t.add_argument("tag")
    t.add_argument("--weights", default=None,
                   help="fieldnet checkpoint (default: the shipped one)")
    t.add_argument("--no-net", action="store_true")
    t.add_argument("--n", type=int, default=60, help="frames per clip")
    t.add_argument("--videos", type=int, nargs="*", default=DEFAULT_VIDEOS)
    t.add_argument("--start", type=int, default=0,
                   help="first frame to record; the long captures open on "
                        "menus, so a spot check has to seek into gameplay. "
                        f"{LEAD} frames before it are analysed to lock the "
                        "tracker and then discarded")

    d = sub.add_parser("diff", help="compare two snapshots")
    d.add_argument("a")
    d.add_argument("b")

    a = ap.parse_args()
    if a.cmd == "take":
        take(a.tag, a.n, not a.no_net, a.weights, a.videos, a.start)
    else:
        sys.exit(diff(a.a, a.b))


if __name__ == "__main__":
    main()
