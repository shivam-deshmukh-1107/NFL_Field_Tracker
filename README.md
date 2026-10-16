# Field Tracker — NFL Blitz 1997 Field Detection

Detects the playing field in NFL Blitz (1997 arcade) gameplay video, draws every
visible yard line, and measures the field in real yards — fully automatically,
with **no hand-labelled data anywhere in the system**. Every colour model,
threshold, training label and glyph alphabet is learned from the footage itself.

The output is one annotated video per input, with the field highlighted, yard
lines drawn and labelled, and the camera state reported per frame.

> **Scope:** field detection, measurement and tracking. Player tracking is the
> eventual downstream goal but is explicitly **not** part of this project.

---

## Table of contents

- [Quick start](#quick-start)
- [The approach](#the-approach)
- [Implementation](#implementation)
- [Tech stack and libraries](#tech-stack-and-libraries)
- [Output](#output)
- [Accuracy — detection](#accuracy--detection)
- [Accuracy — measurement](#accuracy--measurement)
- [Accuracy — tracking stability](#accuracy--tracking-stability)
- [Measured field constants](#measured-field-constants)
- [Limitations](#limitations)
- [Repository layout](#repository-layout)
- [Regression testing](#regression-testing)

---

## Quick start

```bash
python scripts/render_video.py 6                 # render a full capture, all cores
python scripts/run_pipeline.py 1 2 3 4           # prepare + render the four clips
python scripts/render_video.py 6 --no-net        # colour front end only, no neural net
```

```bash
python scripts/valid_scale.py 2 4                # accuracy gate vs the game's scoreboard
python scripts/snapshot.py take before           # freeze per-frame geometry
python scripts/snapshot.py diff before after     # prove a change is behaviour-neutral
```

Videos are referenced by processing order (1–8) or by slug; see
`blitzfield/config.py`. `out/` holds only the rendered videos plus
`field_template.json`; everything the stages hand each other lives under
`work/<NN>_<slug>/` and is rebuilt on demand, so `out/` can be deleted at any
time.

**Interpreter matters.** This needs Python 3.13 with `opencv-contrib-python`
(for `ximgproc.thinning` and `LSD_REFINE_STD`). A plain `opencv-python` install
cannot run the pipeline at all.

---

## The approach

### The problem

Two difficulties compound:

1. **Every yard line looks identical.** A white stripe carries no information
   about whether it is the 20 or the 45. There is no appearance feature to key on.
2. **Perspective spaces them unevenly.** The camera sits low and looks along the
   field, so near lines are far apart in the image and distant ones crowd toward
   the horizon. "They are evenly spaced, so count them" does not work.

On top of that: players occlude the lines, a permanent HUD overlays the picture,
the camera pans and zooms continuously, and menus and replays cut in between plays.

### Core idea — cancel the perspective

Find the point where the yard lines converge on the horizon (as railway tracks
appear to meet in the distance). For each detected line, measure its position
relative to that convergence point, and use **one over that distance** instead
of the distance itself:

```
u = 1 / (s − v)
```

where `s` is the line's position along a fixed transversal and `v` is the horizon
position. Under this transform the perspective cancels: positions that are evenly
spaced on the real ground become evenly spaced again in `u`, and `u` is **linear
in yard index**.

The problem then reduces to **finding the pitch of a regular ruler** — scored like
finding the beat in a repeating rhythm. That is robust in exactly the ways this
footage demands:

- a line hidden behind a player simply does not contribute
- a spurious detection only dilutes the score, it does not corrupt the fit
- not all lines need to be visible, only enough of them

Two details are load-bearing rather than incidental:

- **The branch is decided by physics, not by score.** Lines crowd toward the
  horizon, so the horizon lies beyond the end where the gaps shrink. Left to
  score freely, the fitter sometimes picks the inverted branch, putting the
  horizon below the image and silently reversing the yard axis.
- **Once the lines are labelled, stop searching.** Pushing the horizon further
  away and stretching the pitch to match leaves the predicted lines almost where
  they were, so the cold objective is nearly flat along that ridge and hops along
  it frame to frame — measured at up to 111 px of horizon movement between
  neighbours while the lattice itself did not move. With labels in hand,
  `lattice.fit_labelled` solves the pitch in closed form and anchors the horizon
  to where the camera motion puts it. Horizon jitter fell 3.2×, pitch jitter 4×.

### Core idea — the flank test

**A painted line has turf on both sides of it.** A player's silhouette, a banner
border and a shadow edge do not.

One local test, no temporal model, and it removes almost all structural false
positives. It is the cheapest high-value component in the system.

### Core idea — tracking is geometry, not learning

Detection alone is memoryless, and the cost was measured: up to 63 consecutive
frames with no lattice, 10–20% of the line set turning over between neighbouring
frames (58% at the 90th percentile), and ~28 px of horizon jitter on a camera
that was in fact moving smoothly.

The motion estimator already recovers a similarity transform between consecutive
frames, so the previous line set is propagated through it and detection is
demoted from "solve the frame" to "correct the prediction". A **line** transforms
by the inverse transpose of the point transform, so propagation is exact for the
estimated motion; error enters only through the motion estimate, which is why
prediction is trusted for a bounded number of frames and then dropped.

The asymmetry buys the robustness: establishing the lattice cold needs **five or
more** lines, but *correcting* a tracked one needs **a single** matched line.
Frames that fail cold detection outright usually still carry two or three usable
segments.

---

## Implementation

### Pipeline stages

Each stage persists to disk, so any one can be re-run without redoing the rest.

| Stage | Module | Writes | What it establishes |
|---|---|---|---|
| **0 · Ingest** | `ingest.py` | `stage0/source.json` | `ffprobe` metadata; the active picture inside the pillarbox, from 64 sampled frames; an invertible `canon_to_source` homography. Everything downstream works on a canonical 960×720 buffer and never knows the source resolution |
| **1 · Colour + gating** | `fieldcolor.py`, `gating.py` | `stage1/turf_model.json`, `gating.json`, `features.npz` | Turf colour as a robust 2-D Gaussian over Lab (a, b) chromaticity, found by mode-seeking. Gameplay vs. menus from a 1-D `GaussianMixture(2)` over turf coverage, auto-thresholded and smoothed into runs |
| **2 · HUD mask** | `hudmask.py` | `stage2/hud_mask.png`, `hud.json` | Which pixels are permanent HUD, via persistence × stability over gameplay frames. Excluded from everything after |
| **3 · Markings** | `markings.py`, `fieldnet.py` | per frame | Turf/occluder/line masks → thinning → line segments → collinear merge → **flank test** → the yard-line "pencil", plus the field boundary |
| **4 · Geometry** | `geometry.py`, `lattice.py` | per frame | Vanishing point by RANSAC; the yard lattice fitted in horizon-relative inverse coordinates |
| **5 · Tracking** | `tracker.py`, `motion.py` | per frame | The labelled lattice carried through camera motion; re-fitted against the *tracked* set rather than raw detections |
| **6 · Render** | `render.py`, `render_video.py` | `out/*.mp4`, `render_stats.json` | The annotated overlay, scaled to source resolution, encoded in parallel chunks |

### Stage 3 in detail — from pixels to a line pencil

`markings.FramePrep.build` does the colour-domain preprocessing:

1. **Turf mask** — either Mahalanobis distance < 3.0 from the turf model, or
   `P(turf) ≥ 0.625` from the neural net. Morphological close (25×25) then open
   (7×7), then a connected-component area filter → the field `region`.
2. **A second, extended region** for boundary tracing only. The game-drawn
   scrimmage line reads as "not turf" and carves a channel across the field,
   which otherwise surfaces as a cross-field edge and is mistaken for a goal
   line. Only *line-like* coloured components are folded in (area ≥ 300,
   elongation ≥ 4) — taking every saturated pixel also swallows the
   out-of-bounds apron and erases the sideline.
3. **Occluder mask** — large off-turf blobs, via an opening wider than any
   painted line, so markings drop out while solid objects survive. Deliberately
   *not* taken from the net: players are marked `ignore` when the training set is
   built, so the net has no reason to carve them out.
4. **Line response** — a top-hat ridge filter for thin bright *neutral* lines,
   with an **adaptive Otsu threshold** clamped to [0.15, 0.60]. A fixed cut
   either floods the mask with speckle (which makes the LSD density test reject
   every rectangle and silently return zero segments) or misses faint distant
   lines. With the net, `P(line) ≥ 0.925` replaces this.

Then:

- `detect_segments` — `cv2.ximgproc.thinning` (Guo-Hall) to one-pixel skeletons,
  then `cv2.createLineSegmentDetector(LSD_REFINE_STD)`
- `merge_collinear` — vectorised grouping by angle tolerance (2°) and
  perpendicular distance, with a running group extent
- `classify_flanks` — sample ±5 px perpendicular to each segment; keep it only
  if turf is present on **both** sides
- `boundary_segments` — trace the turf region contour for the sidelines and goal
  lines

### Stage 4 in detail — the lattice fit

- `geometry.ransac_vp` — RANSAC over segment pairs for the cross-field vanishing
  point, scored by angular error. Hypothesis generation is kept as a loop so the
  sequence is reproducible; scoring is batched over all segments at once.
- `lattice.transversal_positions` — intersect each line with a fixed vertical
  transversal `ell` to get a scalar `s` per line.
- `lattice.cluster_1d` — merge near-duplicates (7 px tolerance).
- `lattice.fit` — search over horizon `v` (220 values) and pitch `alpha` (900,
  geomspace) scoring `u = 1/(s − v)` with a comb score built from `cos`/`sin`
  phase sums. `YARDS_PER_CELL = 5.0`; `LONG_CUT = 110.0` px separates full yard
  lines from 1-yard ticks.
- `lattice.refine` — least squares on the inliers.
- `lattice.fit_labelled` — closed-form solve once the tracker supplies labels.

### Stage 5 in detail — the tracker

`tracker.FieldTracker` holds a dict of `yard index → line`, plus per-line ages.

- `_predict` warps every tracked line through the motion homography (inverse
  transpose, exact for the estimated motion)
- `step` matches detections to predictions within `match_px = 12`, updates
  matched lines, evicts lines older than `max_age = 12` frames, and re-fits the
  lattice against the tracked set
- `MAX_CELLS = 30` guards at three levels. A football field is 120 yd including
  both end zones — 24 five-yard cells — so a wider span is a broken fit, not a
  field. Without this, a collapsed pitch once drew **17,550 lines on one frame**,
  because every generated line landed *inside* the field and clipping could not
  bound it.
- `index_origin` is re-set on **every** cold start. A re-lock without a reset
  kept a stale origin and produced yard labels reading "+1585" and "−3560".

### Stage 6 in detail — parallel rendering

The full captures are 40–70 thousand frames at roughly 0.3 s of analysis each —
eleven hours before the hot paths were vectorised, six after — and none of that
work is shared between distant frames.

`render_video.py` cuts the video into contiguous chunks, one per worker, and
encodes each separately. **The only state that crosses a chunk boundary is the
tracker's**, so each worker starts decoding `LEAD = 120` frames early and runs
the full analysis over them without writing anything. By the time it reaches its
own first frame the tracker has locked exactly as it would have in a single pass.
Two seconds of lead costs ~2% and removes the visible re-acquisition a cold chunk
start would otherwise show.

Encoding goes to `ffmpeg` (raw BGR pipe → libx264, CRF 18) rather than
`cv2.VideoWriter`: the chunks must be concatenated afterwards, and x264 at fixed
CRF gives a far smaller and sharper file than OpenCV's MPEG-4 Part 2 default.
Audio is muxed back from the source.

A per-frame record is written to `work/<NN>_<slug>/render_stats.json` so a render
can be *measured* rather than eyeballed.

### Where machine learning fits — and where it does not

| Component | Learned? | What it is |
|---|---|---|
| `fieldnet.py` | **Yes — neural net** | U-Net, **484,176 parameters**, predicts P(turf) and P(painted line) per pixel |
| `gating.py` | Yes — classical | `sklearn.mixture.GaussianMixture(2)` over turf coverage |
| `hudtext.py` | Yes — classical | `AgglomerativeClustering` over glyph shapes to decode the HUD alphabet |
| `fieldcolor.py` | Statistical fit | 2-D Gaussian over Lab (a, b), refit per video |
| `markings.py` | **No** | Ridge filter → thinning → LSD → flank test |
| `geometry.py` | **No** | Projective helpers, vanishing-point RANSAC |
| `lattice.py` | **No** | Comb score in horizon-relative inverse coordinates |
| `tracker.py` | **No** | Propagation through the motion homography |
| `motion.py` | **No** | Lucas–Kanade optical flow + similarity estimate |

**The net is optional by design.** With no weights, or no torch installed,
`markings` falls back to the colour pipeline and still renders — on the Hail Mary
capture the colour front end alone solved 100% of frames.

**Why a net only there.** Everything downstream of the colour decision is already
stable; the colour decision itself was the measured weak link — ~4% of every
frame sits within ±15% of the Mahalanobis cut, and the drawn field mask moved
2.5–5% between neighbouring frames *after* camera motion was compensated for.
A per-pixel Gaussian has one Lab coordinate to go on; a convolutional net has
context — it can use that a dark patch is surrounded by turf, that a line
continues behind a player, that the apron beyond the sideline is a different
surface even where its colour nearly matches.

**Architecture choice:** three pooling levels, not four. The features that matter
are thin — a painted line is 1–3 px at canonical scale — and each extra level
halves the resolution the decoder must reconstruct them from. Three still gives
the bottleneck a ~100 px receptive field, wider than a player.

**Where the labels come from.** Nothing is hand-drawn. The classical pipeline is
run over the footage and its per-frame masks are warped into a common frame
through the estimated camera motion and voted on, in 9-frame windows. Pixels the
vote cannot settle are marked `ignore` and contribute no gradient, so the
teacher's uncertainty is not learned as fact. This matters for more than
convenience: temporal consensus is strictly more evidence than any single frame
carries, so the net is trained to reproduce from *one* frame what the colour model
only gets right when allowed to see nine.

**Training details:** 1,170 training rows / 159 validation, 30 epochs, AdamW +
OneCycleLR, class-weighted loss (12:1 toward "line", since lines are ~2% of a
frame and an unweighted loss just predicts "no line" everywhere). Because the
loss is weighted, the decision threshold is nowhere near 0.5 — it is **swept**
over [0.20, 0.95] each epoch and the best value stored *in the checkpoint*
alongside the weights. Shipped net: `turf_thr = 0.625`, `line_thr = 0.925`,
validation IoU **0.958 turf / 0.818 line**. Fixing the threshold at 0.5 cost
roughly half the line IoU (0.481 vs 0.831).

---

## Tech stack and libraries

| Component | Version | Used for |
|---|---|---|
| **Python** | 3.13.9 (Anaconda) | Everything |
| **OpenCV** (`opencv-contrib-python`) | 5.0.0 | Image processing; `ximgproc.thinning` (Guo-Hall), `createLineSegmentDetector(LSD_REFINE_STD)`, morphology, connected components, colour conversion, optical flow |
| **NumPy** | 2.1.3 | All numerics. Hot paths are vectorised, not looped — this took per-frame analysis from 612 ms to 324 ms, byte-identical on 240 test frames |
| **scikit-learn** | 1.6.1 | `GaussianMixture` (gameplay vs. menu), `AgglomerativeClustering` (HUD glyph alphabet) |
| **PyTorch** | 2.14.1+cu130 | The learned front end. Optional |
| **FFmpeg / FFprobe** | 4.3.1 | Source metadata, decode, and libx264 encode at CRF 18 |
| **multiprocessing** | stdlib | Parallel chunked rendering, 10 workers |

No pretrained backbones, no model zoo, no object-detection framework. The net is
hand-written in ~170 lines.

**GPU note.** The net must be built for your card's architecture. A stock `cu126`
wheel reports `cuda.is_available() == True` on an RTX 50-series card and then
fails with *no kernel image is available for execution on the device*, so
`fieldnet._best_device()` probes with a real kernel and falls back to CPU rather
than crashing. For an sm_120 card, install a CUDA 12.8+ build:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**Code size:** 5,179 lines across 22 files — a 14-module library (`blitzfield/`)
plus 8 command-line scripts (`scripts/`).

| Module | Lines | Role |
|---|---|---|
| `render.py` | 573 | Annotated overlay: field mask, line clipping, drawing, HUD chrome |
| `tracker.py` | 479 | Temporal tracking of the field geometry |
| `markings.py` | 422 | Ridge → segments → flank test → yard-line pencil, field boundary |
| `lattice.py` | 336 | **The yard lattice** |
| `fieldcolor.py` | 253 | Turf model, ridge response, coloured-line masks |
| `ingest.py` | 234 | Active-picture detection, canonical normalisation, frame iteration |
| `motion.py` | 202 | Camera motion: static / pan / zoom / rotate / cut |
| `yardmap.py` | 188 | Image point → yards; tick columns |
| `fieldnet.py` | 171 | Learned per-pixel front end |
| `hudtext.py` | 168 | Reads down / distance / field position from the HUD |
| `gating.py` | 167 | Gameplay vs. menus, unsupervised |
| `pipeline.py` | 157 | Per-video driver + measurement self-check |
| `geometry.py` | 153 | Projective helpers, vanishing-point RANSAC |
| `hudmask.py` | 121 | Static HUD overlay map |
| `config.py` | 88 | Video registry, canonical frame size, shared constants |

---

## Output

Six annotated videos in `out/`, **66,702 frames** total, rendered at source
resolution (1442×1080 for the captures) with the original audio muxed back:

| File | Source | Frames | Duration |
|---|---|---|---|
| `clip1_kickoff_field.mp4` | kickoff return, wide camera | 253 | 8 s |
| `clip2_congestion_field.mp4` | heavy player congestion | 174 | 6 s |
| `clip3_qb_field.mp4` | clean QB view | 186 | 6 s |
| `clip4_swing_field.mp4` | camera yaw swing + zoom change | 260 | 9 s |
| `hail_mary_left_field.mp4` | single play, dark colour grade | 221 | 7 s |
| `yt_midway_field.mp4` | 18 min capture, 60 fps | 65,608 | 18 min |

Plus `field_template.json` — the standing reference for what was measured and
what proved unmeasurable.

Each frame's overlay shows: the field highlighted (everything off-field dimmed to
45%), yard lines drawn and labelled in yards, 1-yard ticks, tick columns,
sidelines and goal lines, the game-drawn scrimmage and first-down lines, and a
status line with the camera state and detected/predicted line counts.

**Render cost:** the 18-minute capture takes ~59 min across 10 workers (18.5
fps). GPU utilisation during a render is only 16–40% — the pipeline is **CPU-bound
in the classical stages**, not in the net.

---

## Accuracy — detection

### Short gameplay clips

| Capture | Frames | Field highlighted | Lattice solved | Median lines | p90 |
|---|---|---|---|---|---|
| clip1_kickoff | 253 | **100%** | **100%** | 10 | 11 |
| clip2_congestion | 174 | **100%** | **100%** | 12 | 15 |
| clip3_qb | 186 | **100%** | **100%** | 9 | 16 |
| clip4_swing | 260 | **100%** | **100%** | 14 | 17 |
| hail_mary_left | 221 | **100%** | **100%** | 10 | 13 |

### The full 18-minute capture

Headline figures: field highlighted on 34,222 frames (52%), lattice solved on
25,817 (39%).

**"39% of all frames" is the wrong denominator.** Roughly a quarter of this
capture has no turf on screen at all (attract mode, team select, crowd cutaways)
and another quarter is the play-select menu. Split by how much field is actually
showing:

| Turf mask coverage | Frames | Lattice solved |
|---|---|---|
| < 0.15 (no field) | 14,690 | **0.1%** ← correctly detects nothing |
| 0.15 – 0.45 | 6,549 | 17.3% |
| 0.45 – 0.60 (menu backdrop) | 17,706 | 22.4% |
| **> 0.60 (a field on screen)** | 26,663 | **77.6%** |
| **> 0.75 (wide shot)** | 13,072 | **88.2%** |

So when a field is genuinely in view the geometry solves **77.6%** of the time,
rising to **88.2%** on clean wide shots. The 0.1% on no-field frames is as
important as the high numbers: the system is not hallucinating fields.

The remaining unsolved on-field frames were examined rather than assumed: they
are overwhelmingly quarter-end and celebration screens and deep-zoom replays,
occurring in sustained runs rather than scattered dropouts.

---

## Accuracy — measurement

This is the strongest result, because it is validated against **ground truth the
pipeline never sees**.

NFL Blitz prints down-and-distance on screen, so the yard gap between the
scrimmage line and the line to gain is *known*. The system measures that gap
*through its own fitted lattice* and the two are compared. The check is genuinely
independent: the HUD is masked out before any geometry runs, and the two coloured
lines are located by colour, separately from the white yard lines the lattice was
fitted to.

| Clip | HUD says | Measured (median) | Error | IQR |
|---|---|---|---|---|
| clip2_congestion | 30 yd | 30.17 yd | **+0.17 yd** | 0.16 |
| clip4_swing | 29 yd | 29.24 yd | **+0.24 yd** | 0.12 |

**Worst error 0.24 yd — roughly 0.8%.** The IQR matters as much as the median: a
tight spread means the scale is recovered the same way on every frame, not
averaging out from frames that disagree.

This is the promotion gate. No change to the front end ships unless this error
does not move backwards — run `scripts/valid_scale.py 2 4`.

---

## Accuracy — tracking stability

Measured over all 65,608 frames of the Midway capture:

| Measure | Result |
|---|---|
| Brief in-play dropouts (runs < 10 frames) | **111 frames — 0.17%** across 44 runs |
| Longest sustained solved run | **660 consecutive frames** |
| Median spacing between adjacent yard lines | **42.5 px** |
| Frames with near-degenerate spacing (< 6 px) | 46 (0.07%) |
| Maximum lines drawn on any frame | **22** |
| Frames exceeding the 25-line sanity bound | **0** |

That last row is the `MAX_CELLS` guard working. Before it existed, one frame drew
17,550 lines when the fitted pitch collapsed.

Line counts are bounded and sane throughout — median 9, p90 14, max 22 — which is
the cheap numeric signal that no frame has a degenerate lattice.

---

## Measured field constants

Everything here was measured from the footage, not assumed:

| Quantity | Value | Evidence |
|---|---|---|
| Field length, goal to goal | **100 yd** | 1,814 HUD readings, hard ceiling at 50 |
| Yard line spacing | **5 yd** | validated ±0.1 yd against the HUD |
| Tick mark spacing | **1 yd** | pitch / 5 relative to yard lines |
| First-down distance | **30 yd** | measured 30.01 |
| Field **width** | *not recoverable* | see Limitations |

---

## Limitations

**The net does not generalise across captures.** Twice a new source has broken it
completely — not degraded, collapsed:

| Capture | Turf Lab (a, b) | Net P(turf) | Colour model |
|---|---|---|---|
| Midway vs. clips-only net | (101.9, 150.3) | 3–5% | 62–81% |
| Hail Mary vs. clips+Midway net | (119.3, 132.9) | 4–10% | 65–88% |

A 0.48 M-parameter U-Net learns "turf" as a *colour*. The turf *model* refits per
video and handles new colour fine; the net does not. Each new capture needs its
own auto-labelled training rows. `--no-net` is the fallback and works well.
Colour-normalising the input to remove this tax was tried across five training
runs; it works in principle (held-out turf IoU 0.120 → 0.889) but costs line
recall on the hardest clip, with a median that fails the acceptance bar — see
`ENGINEERING_NOTES.md`.

**Field width is not measurable from this footage.** Length works; width does not.
Blitz uses a low, near-ground camera, so the cross-field vanishing point sits
~200 image widths from centre while the down-field one sits at ~0.6. Being
near-perpendicular from the principal point makes any focal-length estimate
hypersensitive — 0 of 20 sampled frames had both vanishing points usably placed.
This was tested and documented rather than shipped as a bad number.

**The auto-labelled dataset is not in the tree.** It was 345 MB of generated
jpg/png and was deleted once the normalisation question was settled. `train`
exits with instructions until it is rebuilt (~1 hour, single-threaded). Nothing
else needs it — inference loads only the checkpoint.

**Videos 5 and 7 are unrendered.** Video 7 has a non-4:3 active area (1746×982,
anisotropically stretched) and contributes little.

**Output is not bit-reproducible across runs.** `--workers` sets the chunk
boundaries, and a chunk's 120-frame lead-in converges the tracker to a very
similar but not identical state, so file sizes vary a few percent between runs
with different worker counts.

---

## Repository layout

```
Field_Tracker/
├── blitzfield/              the library — 14 modules, no entry points
├── scripts/
│   ├── render_video.py      parallel chunked render (full captures)
│   ├── run_pipeline.py      prepare stages 0–2, then render serially
│   ├── run_video.py         measure one video + self-check
│   ├── train_fieldnet.py    auto-label a dataset, then train
│   ├── read_hud.py          decode the HUD alphabet → field position
│   ├── valid_scale.py       accuracy gate vs the HUD
│   └── snapshot.py          freeze / diff per-frame geometry
├── Data/                    source videos (8 registered)
├── work/                    all intermediates, per video, rebuildable
│   ├── <NN>_<slug>/stage0..2/
│   ├── fieldnet/            trained checkpoints
│   └── regress/             reference snapshots
├── out/                     deliverables: 6 videos + field_template.json
├── README.md                this file
└── ENGINEERING_NOTES.md     the full engineering log — failed experiments,
                             dead ends, and why each decision was made
```

---

## Regression testing

Two harnesses, and a new front end ships only if both agree nothing moved
backwards:

```bash
python scripts/valid_scale.py 2 4 --tol 0.25     # yard scale vs the HUD
python scripts/snapshot.py take before
python scripts/snapshot.py diff before after     # per-frame geometry, exact
```

`valid_scale.py` measures the lattice against down-and-distance printed by the
game, which the pipeline never sees, and prints `PASS`/`REGRESSION` against the
bar. `snapshot.py` freezes the per-frame geometry of the four clips so a refactor
can be *shown* behaviour-neutral rather than merely faster;
`work/regress/snap_baseline.json` is the reference for the shipped net.

Two cautions, both learned the hard way:

- **One run is not a verdict.** Clip 4's line count has a measured standard
  deviation of 4.6% across *identical* training configurations, and a rejected
  configuration once passed every criterion on a single run. Replicate a
  candidate over several `--seed` values before concluding anything.
- **Compare snapshots within one environment.** The net's thresholds sit at
  0.625/0.925, so a handful of pixels on either side of a cut land differently
  between CUDA and CPU kernels — enough to move the mean line count by under 1%
  with nothing in the repository having changed.

---

For the full engineering history — every failed experiment, the measurements that
turned out to be invalid, and the reasoning behind each design decision — see
[`ENGINEERING_NOTES.md`](ENGINEERING_NOTES.md).
