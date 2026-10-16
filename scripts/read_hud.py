"""Pool HUD glyphs across videos, decode the alphabet, read field position."""
import sys, json, pathlib, collections
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import numpy as np
from blitzfield import (ingest, fieldcolor, hudmask, markings, hudtext,
                        lattice as L, geometry as G, pipeline)
from blitzfield.config import get_video, LONG_CUT, YARDS_PER_CELL

VIDEOS = [int(v) for v in (sys.argv[1:] or ["5", "6", "7"])]
STRIDE = 7

all_glyphs, all_lines, geo = [], [], {}
for vid in VIDEOS:
    spec = get_video(vid)
    _, area = ingest.load(spec)
    turf = fieldcolor.TurfModel.from_json(
        (spec.workdir / "stage1" / "turf_model.json").read_text())
    hud = hudmask.load(spec)
    gate = json.loads((spec.workdir / "stage1" / "gating.json").read_text())
    play = [(s["start"], s["stop"]) for s in gate["segments"]
            if s["kind"] == "gameplay"]
    rng = np.random.default_rng(0)
    n_lines0, n_geo0 = len(all_lines), len(geo)
    for a, b in play:
        with ingest.FrameSource(spec, area, stride=STRIDE, start=a, stop=b) as src:
            for i, fr in src:
                tls = hudtext.text_lines(fr)
                if not tls:
                    continue
                key = (vid, i)
                for gl in tls:
                    ids = []
                    for g in gl:
                        all_glyphs.append(g)
                        ids.append(len(all_glyphs) - 1)
                    all_lines.append((key, ids))
                # geometric distance between the two coloured lines
                fm = markings.analyse_frame(fr, turf, hud=hud, frame_index=i, rng=rng)
                pen = fm.segs[fm.label == 0]
                if len(pen) < 5 or fm.vp_cross is None:
                    continue
                Ls = G.seg_length(pen)
                ell = np.array([1.0, 0.0, -fr.shape[1] / 2])
                s, k = L.transversal_positions(pen[Ls >= LONG_CUT], ell)
                if not k.any():
                    continue
                sc = L.cluster_1d(s[k])
                fit = L.fit(sc)
                if fit is None:
                    continue
                fit = L.refine(sc, fit)
                prep = markings.FramePrep.build(fr, turf, hud=hud)
                cl = pipeline.colour_lines(fr, turf, prep, fm.vp_cross)
                idx = {}
                for nm, seg in cl.items():
                    ss, kk = L.transversal_positions(seg[None], ell)
                    if kk.any():
                        idx[nm] = float((1.0 / (ss[0] - fit.v) - fit.beta) / fit.alpha)
                if "orange_yellow" in idx and "red" in idx:
                    geo[key] = abs(idx["orange_yellow"] - idx["red"]) * YARDS_PER_CELL
    print(f"  video {vid}: +{len(all_lines)-n_lines0} lines, +{len(geo)-n_geo0} geo-frames",
          flush=True)

print(f"pooled: glyphs={len(all_glyphs)} lines={len(all_lines)} geo={len(geo)}")
labels, K = hudtext.cluster_glyphs(all_glyphs)
print(f"clusters={K}")
seqs = [(key, [int(labels[k]) for k in ids]) for key, ids in all_lines]
book = hudtext.build_book(seqs, geo, labels)
print("AND block:", book.and_block, " ON THE prefix:", book.onthe_prefix)
print(f"digits labelled: {len(book.digit)}/10 ->",
      dict(sorted((int(k), v) for k, v in book.digit.items())))
pos, unread = hudtext.read_positions(seqs, book)
vals = np.array([p for _, p in pos])
print(f"ON THE readings: {len(vals)}  (unreadable: {unread})")
if len(vals):
    print("  distribution:", dict(sorted(collections.Counter(vals.tolist()).items())))
    print(f"  min={vals.min()}  max={vals.max()}")
    print(f"  => half-field = {vals.max()} yd  => TOTAL LENGTH = {2*vals.max()} yd")
