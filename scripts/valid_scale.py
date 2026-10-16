"""Ground-truth check of the tracked lattice against the game's own HUD.

Blitz prints down-and-distance, so the yard gap between the scrimmage line and
the line to gain is known.  Measuring that gap *through the fitted lattice*
tests the geometry end to end against something the pipeline never sees: the
HUD is masked out before any geometry runs, and the two coloured lines are
found by colour, independently of the painted yard lines the lattice is fitted
to.

This is the promotion gate.  A new fieldnet checkpoint only ships if the error
here does not move backwards:

    python scripts/valid_scale.py 2 4                      # the shipped net
    python scripts/valid_scale.py 8 --no-net               # colour front end only
    python scripts/valid_scale.py 2 4 --tol 0.35 \
        --weights work/fieldnet/fieldnet_v7b_tightjit.pt   # a candidate

One run of this is not a verdict.  Replicate a candidate over several
``--seed`` values before concluding anything: clip 4's line count has a
measured sd of 4.6% across identical configurations, and a single run of a
rejected configuration passed every criterion.

Expect |error| <= 0.25 yd.  The IQR matters as much as the median: a wide
spread means the scale is being recovered differently on different frames, even
when the middle happens to land in the right place.
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from blitzfield import (fieldcolor, geometry as G, hudmask, ingest,
                        lattice as L, markings, motion as mo, pipeline,
                        tracker as tk)
from blitzfield.config import LONG_CUT, YARDS_PER_CELL, get_video

#: Down-and-distance read off the HUD by hand, once, per capture.  These are
#: the only hand-entered numbers anywhere in the project, and they are here
#: precisely so the automated geometry has something external to be wrong
#: against.
TRUTH = {"clip2_congestion": 30.0, "clip4_swing": 29.0, "hail_mary_left": 30.0}


def measure(spec, net, stride: int = 2) -> np.ndarray:
    """LOS-to-first-down gap, in yards, for every frame that yields one."""
    _, area = ingest.load(spec)
    turf = fieldcolor.TurfModel.from_json(
        (spec.workdir / "stage1" / "turf_model.json").read_text())
    hud = hudmask.load(spec)
    est, track = mo.MotionEstimator(), tk.FieldTracker()
    rng = np.random.default_rng(0)
    last_region, gaps = None, []
    with ingest.FrameSource(spec, area, stride=stride, seek=False) as src:
        for i, canon in src:
            m = est.update(canon, last_region)
            if m.label == "cut":
                track.reset()
            h, w = canon.shape[:2]
            ell = np.array([1.0, 0.0, -w / 2])
            prep = markings.FramePrep.build(canon, turf, hud=hud, net=net)
            last_region = prep.region
            fm = markings.analyse_frame(canon, turf, hud=hud, rng=rng, prep=prep)
            pen = fm.segs[fm.label == 0]
            lng = pen[G.seg_length(pen) >= LONG_CUT] if len(pen) else np.zeros((0, 4))

            # Cold fit, offered to the tracker as a candidate.  The tracker is
            # free to keep its own lock instead, which is the point: the scale
            # being tested is the one the render would actually draw.
            cold = None
            if len(lng) >= 5:
                s, k = L.transversal_positions(lng, ell)
                if k.any():
                    sc = L.cluster_1d(s[k])
                    f0 = L.fit(sc)
                    if f0 is not None:
                        cold = L.refine(sc, f0)
            st = track.step(lng, fm.vp_cross, cold, w, h, motion=m)
            if not (st.ok and st.fit is not None):
                continue

            cl = pipeline.colour_lines(canon, turf, prep, fm.vp_cross)
            idx = {}
            for nm, seg in cl.items():
                ss, kk = L.transversal_positions(np.asarray(seg)[None], ell)
                if kk.any():
                    u = 1.0 / (ss[0] - st.fit.v)
                    idx[nm] = float((u - st.fit.beta) / st.fit.alpha)
            if "orange_yellow" in idx and "red" in idx:
                gaps.append(abs(idx["orange_yellow"] - idx["red"]) * YARDS_PER_CELL)
    return np.asarray(gaps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("videos", nargs="+", help="processing order 1-8, or slug")
    ap.add_argument("--no-net", action="store_true",
                    help="colour front end only, ignore the trained weights")
    ap.add_argument("--weights", default=None,
                    help="fieldnet checkpoint (default: the shipped one)")
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--tol", type=float, default=0.25,
                    help="|error| bar in yards for the verdict (default 0.25, "
                         "the standing expectation; an individual experiment "
                         "may have been agreed against a looser one, so the "
                         "bar is printed with the verdict)")
    a = ap.parse_args()

    net = None
    if not a.no_net:
        from blitzfield.fieldnet import FieldSegmenter, WEIGHTS
        net = FieldSegmenter(a.weights or WEIGHTS)
        print("net %s  thr %s %s  normalise %s"
              % (net.device, net.turf_thr, net.line_thr,
                 getattr(net, "normalise", False)))

    worst = 0.0
    for v in a.videos:
        spec = get_video(v)
        g = measure(spec, net, stride=a.stride)
        if not len(g):
            print("%-18s no frames with both coloured lines" % spec.vid)
            continue
        med = float(np.median(g))
        iqr = float(np.percentile(g, 75) - np.percentile(g, 25))
        msg = ("%-18s n=%4d  LOS->first-down median %6.2f yd  IQR %5.2f"
               % (spec.vid, len(g), med, iqr))
        t = TRUTH.get(spec.vid)
        if t:
            err = med - t
            worst = max(worst, abs(err))
            msg += "   HUD says %.0f  err %+.2f" % (t, err)
        print(msg)
    if worst:
        print("\nworst |error| %.2f yd  vs bar %.2f  ->  %s"
              % (worst, a.tol, "PASS" if worst <= a.tol else "REGRESSION"))


if __name__ == "__main__":
    main()
