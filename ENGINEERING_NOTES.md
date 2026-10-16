# Field Tracker — NFL Blitz 1997 field detection

Detects the playing field in NFL Blitz gameplay video and measures it in yards,
with no manual annotation anywhere. Every colour model, threshold and glyph
label is learned from the video itself.

## Run it

```bash
python scripts/run_pipeline.py 1 2 3 4      # annotated overlay videos (clips)
```

```bash
python scripts/render_video.py 6            # a full capture, across all cores
```

```bash
python scripts/run_video.py 1 --prep        # measure one video + self-check
```

```bash
python scripts/read_hud.py 5 6 7            # read field position from the HUD
```

```bash
python scripts/train_fieldnet.py dataset 1 2 3 4   # auto-label from the footage
python scripts/train_fieldnet.py train             # fit the learned front end
python scripts/train_fieldnet.py train --workers 0 # load in-process instead
```

**The auto-labelled dataset is not in the tree** — it was 345 MB of generated
jpg/png under `work/fieldnet/{img,lbl}` and was deleted once the normalisation
question was settled. `train` exits with *no dataset* until it is rebuilt.
Nothing else needs it: inference loads only the checkpoint, so rendering and the
regression harnesses are unaffected. The 1329 rows it held were:

```bash
python scripts/train_fieldnet.py dataset 1 2 3 4              # 421 rows, the clips
python scripts/train_fieldnet.py dataset 6 --windows 700 --stride 1   # 695, Midway
python scripts/train_fieldnet.py dataset 8 --stride 1         # 213, Hail Mary
```

The labels come from the classical pipeline voting over 9-frame windows, so this
re-runs the full per-frame analysis several times per row — budget an hour or so,
and note it is single-threaded, unlike training. It depends only on the
per-video `work/<NN>_<slug>/stage1` and `stage2` artifacts, which are kept.

Training loads through a `DataLoader`; the per-sample cost is two `imread`s plus,
when normalising, a full-frame Mahalanobis pass — 51 ms measured, most of an
epoch with the GPU idle. Six workers take that from 103 to 23 s/epoch (a
30-epoch run, 53 min → 12), and twelve are no faster, so ~23 s is the compute
floor. Each worker runs cv2 single-threaded: cv2 defaults to a thread per core,
and six workers × 24 threads is slower than no workers at all. Augmentation
draws per-worker RNG streams, so runs are reproducible for a fixed `--workers`
but not bit-comparable across different values, nor with runs made before this.

```bash
python scripts/valid_scale.py 2 4           # yard scale vs. the HUD (the gate)
python scripts/snapshot.py take before      # freeze per-frame geometry
python scripts/snapshot.py diff before after
```

Those two are the regression suite. `valid_scale.py` measures the lattice
against down-and-distance printed by the game, which the pipeline never sees —
expect |error| ≤ 0.25 yd, and it reports `REGRESSION` past that. `snapshot.py`
freezes the per-frame geometry of the four clips so a refactor can be shown to
be behaviour-neutral rather than merely faster. **A new fieldnet checkpoint
ships only if both agree nothing moved backwards.**

Use `run_pipeline.py` for the clips and `render_video.py` for the full captures:
the latter splits the video into one chunk per worker and encodes with x264,
which takes video 6 from eleven hours to about forty minutes. Each worker
analyses 120 frames before its own first frame so the tracker is already locked
at a chunk boundary; nothing else crosses one.

Videos are referenced by priority order 1–7 (see `blitzfield/config.py`).
`out/` holds only the rendered videos and `field_template.json`; everything the
stages hand each other lives under `work/<NN>_<slug>/` and is rebuilt on demand,
so `out/` can be deleted at any time. `run_pipeline.py --canon` renders at
960×720 instead of source resolution.

## Results

Measured from the footage, not assumed:

| Quantity | Value | Evidence |
|---|---|---|
| Field length, goal to goal | **100 yd** | 1814 HUD readings, hard ceiling at 50 |
| Yard line spacing | **5 yd** | validated ±0.1 yd against the HUD |
| Tick mark spacing | **1 yd** | pitch/5 relative to yard lines |
| First-down distance | **30 yd** | measured 30.01 |
| Field **width** | *not recoverable* | see below |

Full evidence and caveats: `out/field_template.json`.

The yard scale is validated against the game's own down-and-distance display —
independent ground truth needing no annotation:

| Source | Game says | Measured |
|---|---|---|
| Clip 2 Congestion | 30 yd | 30.24 |
| Clip 4 Swing | 29 yd | **29.16** |
| Midway capture, f50220 | "1ST AND 30" | **30.3** |

Clip 4 is the camera-swing-and-zoom test — the hardest case, and among the most
accurate. If the geometry were subtly wrong, that is where it would break. The
Midway row is a different capture of a different build, measured with the same
lattice and no retuning.

### What the full capture actually yields

`render_video.py` writes a per-frame record to `work/<NN>_<slug>/render_stats.json`,
so the render can be measured rather than eyeballed. Over all 65,608 frames of
the Midway capture:

| | frames | share |
|---|---|---|
| field highlighted | 35,795 | **55%** |
| yard lattice solved | 25,920 | 40% |
| of highlighted frames, lattice solved | | **71%** |

"40% of all frames" is the wrong denominator: roughly a quarter of this capture
has no turf on screen at all (attract mode, team select, crowd cutaways) and
another quarter is the play-select menu. Split by how much field is actually
showing:

| turf mask coverage | frames | lattice solved |
|---|---|---|
| < 0.15 (no field) | 14,822 | 0.1% |
| 0.45-0.60 (menu backdrop) | 17,575 | 20% |
| > 0.60 (a field on screen) | 27,589 | **77%** |
| > 0.75 | 13,982 | **87%** |

Line counts are bounded and sane - median 9, p90 14, max 19 - and adjacent
lines sit a median 42.8 px apart, with 6 frames in the whole capture under 6 px.
Both clip runs and the capture render at 1442x1080; the four clips highlight
the field on **100%** of frames and solve the lattice on 92 / 100 / 100 / 100%,
clip 1's shortfall being the end-of-play close-ups discussed above.

## Modules

| Module | Role |
|---|---|
| `config.py` | video registry, canonical frame size, shared constants |
| `ingest.py` | active-picture detection, canonical normalisation, frame iteration |
| `fieldcolor.py` | turf model (mode-seeking), ridge response, coloured-line masks |
| `gating.py` | gameplay vs. menus, unsupervised via a 1-D Gaussian mixture |
| `hudmask.py` | static HUD overlay map (persistence × stability) |
| `geometry.py` | projective helpers, vanishing-point RANSAC |
| `markings.py` | ridge → segments → flank test → yard-line pencil, field boundary |
| `lattice.py` | **the yard lattice** — see below |
| `yardmap.py` | image point → yards; tick columns |
| `hudtext.py` | reads down / distance / field position from the HUD |
| `motion.py` | camera motion: static / pan / zoom / cut |
| `tracker.py` | carries the labelled lattice across frames through camera motion |
| `fieldnet.py` | **learned front end** — turf and painted markings per pixel |
| `render.py` | annotated overlay |
| `pipeline.py` | per-video driver + the HUD measurement self-check |


## The core idea

Every yard line looks identical, so seeing one tells you nothing about *which*
yard it is — and perspective means they are not evenly spaced in the image.

Take the point where the lines converge on the horizon and measure distance
*relative to it*, as one-over-distance. The perspective cancels: evenly-spaced
ground positions become an evenly-spaced lattice again. Finding the yard
spacing then reduces to finding the pitch of a regular ruler, scored like a
repeating rhythm. That is robust in exactly the ways this footage needs — lines
hidden behind players simply do not contribute, and stray detections only
dilute the score.

Two details that are load-bearing rather than incidental:

- **The branch is decided by physics, not by scoring.** Lines crowd toward the
  horizon, so the horizon lies beyond the end where gaps shrink. Left to choose
  freely, the fitter sometimes picks the inverted branch, putting the horizon
  below the image and silently reversing the yard axis.
- **The flank test** — a painted line has turf on *both* sides. Player
  silhouettes, banner borders and shadow edges do not. One local test, no
  temporal model, and it removes almost all structural false positives.
- **Once the lines are labelled, stop searching.** Pushing the horizon further
  away and stretching the pitch to match leaves the predicted lines almost
  where they were, so the cold objective is nearly flat along that ridge and
  hops along it frame to frame — measured at up to 111 px of horizon movement
  between neighbours while the lattice itself did not move at all. The tracker
  already knows which yard line is which, so `lattice.fit_labelled` solves the
  pitch in closed form and anchors the horizon to where the camera motion puts
  it. Horizon jitter fell 3.2× and pitch jitter 4×, with the yard scale
  unchanged against the HUD.

## The learned front end

Everything downstream of the colour decision was already stable; the colour
decision was not. ~4% of every frame sits within ±15% of the Mahalanobis cut,
and the drawn field mask moved 2.5–5% per frame *after* camera motion was
compensated for. A per-pixel Gaussian has one Lab coordinate to go on; a
convolutional net has context.

`fieldnet.py` is a 0.48 M-parameter U-Net predicting P(turf) and P(painted
line). **No frame is labelled by hand.** The colour pipeline is run over a
9-frame window, every mask is warped into the centre frame through the estimated
camera motion, and the per-pixel vote is the target — so the net learns from one
frame what the colour model only gets right when allowed to see nine. Pixels the
window cannot agree on are marked *ignore* and contribute no gradient.

It generalises across the clips rather than memorising them. Trained on clips
1–3 only and evaluated on clip 4, which it never saw: **turf IoU 0.975, line
IoU 0.904**. It does *not* generalise across captures — see "A second source"
below.

### The decision threshold belongs to the weights

The line loss is weighted 12:1 towards the positive class, because lines are
~2% of a frame and an unweighted loss just predicts "no line" everywhere. That
weighting also moves the logit at which precision and recall balance a long way
from zero, and it drifts as training sharpens — so scoring at a fixed 0.5 was
measuring the drift, not the model. Validation line IoU read 0.21 at one epoch
and 0.68 at the next while the training loss fell smoothly through both.

Swept properly on the same checkpoint:

| threshold | 0.50 | 0.65 | 0.75 | 0.85 | **0.875** | 0.90 | 0.95 |
|---|---|---|---|---|---|---|---|
| line IoU | 0.481 | 0.673 | 0.785 | 0.828 | **0.831** | 0.831 | 0.822 |

Nearly a factor of two, from one number. Training now sweeps 0.20–0.95, keeps
the **best epoch rather than the last** (they differ: best was epoch 26 of 30),
and stores the chosen thresholds in the checkpoint; `FieldSegmenter` reads them
back, so a threshold is a property of a model and not a constant in
`markings.py`. The grid has to be wide enough that the optimum is interior — a
first attempt stopped at 0.85 and simply reported its own top end.

End to end, colour front end → learned front end:

| Clip | cold lattice fit | long lines/frame | tint IoU (p10) | LOS error |
|---|---|---|---|---|
| 1 Kickoff | 72% → **75%** | 10 → 11 | 0.924 → **0.931** | — |
| 2 Congestion | 100% → 100% | 9 → 9 | 0.938 → **0.942** | +0.16 → +0.21 |
| 3 QB | 91% → **100%** | 8 → 8 | 0.947 → 0.945 | — |
| 4 Swing | 100% → 100% | 14 → **18** | 0.932 → **0.944** | +0.33 → **+0.20** |

The gains are real but bounded, and that is the honest summary: detection was
not the bottleneck, so a better front end mostly buys the frames that used to be
solvable only by the tracker (clip 3's last 9%) and more evidence per frame
where the camera is moving hardest (clip 4, +29% long lines).

Weights are optional — with none trained, or no torch installed, the colour
pipeline runs unchanged and `run_pipeline.py --no-net` forces it.

### Colour normalisation: a real benefit that did not ship

Two captures in a row broke the net completely, and the second one made the
pattern plain: it is **not learning turf, it is learning a colour**. The turf
*model* has no such problem - it is refitted per video and finds the new colour
by itself - so the fix is to let it absorb the difference. `normalise_to_canon`
maps every frame's turf chromaticity and luminance onto one fixed reference
before the net sees it. Nothing is hand-tuned: every quantity comes from the
per-video turf model or the frame.

It aligns the three captures, luminance included - which matters, because they
differ in exposure as much as in hue:

| capture | turf Lab (a, b, L) before | after |
|---|---|---|
| clips | (110, 126, **97**) | (110, 136, 120) |
| Midway | (101, 152, **47**) | (109, 137, 120) |
| Hail Mary | (118, 135, **24**) | (108, 138, 120) |

**Leave-one-capture-out**, both arms trained with every Hail Mary row withheld,
so the validation score is literally "footage the net has never seen":

| turf IoU on the held-out capture | min | median | max | best checkpoint |
|---|---|---|---|---|
| raw colour | 0.001 | 0.120 | 0.530 | 0.291 |
| **normalised** | **0.809** | **0.889** | **0.930** | **0.942** |

The normalised arm's *worst* epoch beats the raw arm's *best*. In the practical
measure that diagnosed the original failure - turf fraction on real frames, with
no rows from that capture anywhere in training:

| frame | colour model | raw net | normalised net |
|---|---|---|---|
| f5 | 0.78 | 0.29 | **0.79** |
| f90 | 0.88 | 0.28 | **0.90** |
| f215 | 0.65 | 0.12 | **0.65** |

The normalised net tracks the colour model to within a couple of points on
unseen footage.

Two details worth keeping. **Line IoU is unaffected** (0.875 raw vs 0.872
normalised): painted markings are near-neutral, so the colour grade barely
touches that channel - the damage was only ever in turf, which is also why the
line channel was useless as a symptom. And inference costs one extra colour
conversion per frame, about 15 ms against a ~320 ms frame.

#### Three attempts to ship it, none promoted

The leave-one-out benefit above is real and reproduces. What defeats it every
time is the production gate on the clips.

| attempt | luminance | jitter | val line IoU | clip4 lines vs baseline | HUD worst |
|---|---|---|---|---|---|
| v5 | additive shift | 1.0× | 0.811 | −7.6% | **+11.0 yd** |
| v6 | gamma curve | 1.0× | 0.811 | **−14.0%** | 0.20 yd |
| v7 | gamma curve | 0.2× | 0.816 | **−12.3%** | 0.26 yd |

**v5 failed on scale.** The additive L shift drove near-white paint into the 255
ceiling — 0.4% / 0.2% / 28.9% / 66.9% of painted pixels clipped across the four
captures — so the net trained on flat white where line structure should be and
clip 4's lattice tipped into a wrong pitch. The gamma curve fixed that: 0.0%
clipping, verified, and v6's yard scale came out the best of any run.

**v6 and v7 failed on line count.** Clip 4 lost 14.0% and 12.3% of its yard
lines against a 10% bar. v7 narrowed the photometric jitter because at full
width it fights the normaliser, which is measurable rather than speculative: the
normaliser pins turf luminance to a relative spread of 0.007 where the raw
cross-capture spread is 0.405, so a ±27% exposure jitter swings turf L across
~106–136 when the real normalised range is 116–121. Narrowing it to 0.2× brings
that to 114–127 and recovered 1.7 of the 4.0 percentage points needed — real,
reproducible, and not enough.

#### Then five replicates, because two runs are not a trend

The first reading of the table above was that the trade is inherent. That was
inferred from one run per configuration, and it was the wrong way to reach it.
Once a run cost 13 minutes instead of 53 the question became answerable: five
narrow-jitter runs, varying `--seed`, which moves both the train/val split and
the augmentation stream.

| run | clip1 | clip2 | clip3 | clip4 | |
|---|---|---|---|---|---|
| v7, pre-DataLoader | +0.0 | +5.6 | −3.6 | **−12.3** | fail |
| seed 0 | +1.4 | +5.5 | −6.6 | **−2.7** | pass |
| seed 1 | +0.9 | +5.5 | −3.7 | **−5.7** | pass |
| seed 2 | +0.3 | +3.4 | −5.6 | **−13.5** | fail |
| seed 3 | −0.2 | +5.5 | −6.8 | **−10.3** | fail |

Clip 4: **median −10.3%, mean −8.9%, range −13.5 to −2.7, sd 4.6, 2 of 5 inside
the 10% bar.** The distribution straddles the bar, so the honest statement is
not "normalisation costs 12% of clip 4's lines" but "normalisation is a coin
flip on clip 4", which is a stronger reason not to ship it than a consistent
loss would be — a consistent loss could be budgeted for.

The narrow jitter did help: the median moved from v6's −14.0% to −10.3%. Real,
grounded in the measured 58× over-wide envelope, and not enough.

The instability is specific to one clip. Clip 1 holds within ±1.4% across all
five, clip 2 is consistently *better* (+3.4 to +5.6), clip 3 consistently
−3.6 to −6.8 and always inside the bar. Only clip 4 swings, at sd 4.6 against
roughly 0.5 elsewhere: absolute colour is genuinely useful evidence *within* a
capture, and removing it bites hardest where the margin is already thinnest.

Note what would have happened without the replicates. Seed 0 passed all four
criteria cleanly — clips, HUD, Hail Mary and a Midway spot check — and was the
best of five samples from a distribution whose median fails. Promoting on it
would have shipped a lucky draw.

The HUD scale, by contrast, is unaffected and slightly better than shipped:
worst error 0.21 yd on every one of the five, against the shipped net's 0.24.

Worth noting against the table above: the leave-one-out arms showed line IoU
unaffected (0.875 vs 0.872), and that measure did not predict the production
loss, because held-out crops do not reproduce the swing case. A fourth instance
of the recurring lesson below — a measure validated on a sample that did not
span the space.

The machinery stays wired in and inert. `normalise_to_canon` runs only when the
loaded checkpoint carries `normalise: True`, and the shipped one does not, so
the colour path is unchanged.

`work/fieldnet/fieldnet_v7b_tightjit.pt` is kept as the one surviving normalised
checkpoint — the seed-0 row, the best of the five. The others were pruned; any
of them is ~13 minutes to retrain with
`train --epochs 30 --normalise --seed <n>`.

```bash
python scripts/valid_scale.py 2 4 --tol 0.35 \
    --weights work/fieldnet/fieldnet_v7b_tightjit.pt
python scripts/snapshot.py take v7b \
    --weights work/fieldnet/fieldnet_v7b_tightjit.pt
python scripts/snapshot.py diff baseline v7b
```

## A second source: what did not transfer

The four clips are one capture of one game. `YouTube_NFL-Blitz-MIDWAY-1997`
(video 6, 18 min, 65,608 frames at 60 fps) is a different build with a
different pitch: **green grass at Lab (a, b) = (102, 151)** against the clips'
**teal (110, 126)**.

Almost everything survived the move. The turf model is refitted per video and
landed on the new colour by itself; the gate found 77% gameplay in 13 segments;
the HUD map, the lattice, the tracker and the yard scale all carried over
untouched.

The **net did not**. Trained on clips 1-4 it scored P(turf) above threshold on
**3-5%** of this capture's pixels against the colour model's 62-81% - not a
degradation, a collapse. A 0.48 M-parameter net with three pooling levels has
enough context to learn "turf" as a colour, and that is evidently what it did.

The failure is in the **turf** channel alone. The line channel looked broken too
- firing on 18-30% of the frame - but that figure was never diagnostic: a net
working perfectly well on this capture produces 18-23% on the same statistic,
because the learned line channel is deliberately thick and covers yard lines,
hash marks, painted numbers and the apron together. Reading it as evidence was a
mistake; the control was only run later, on video 8.

The fix is the design working as intended rather than a change to it:
`train_fieldnet.py dataset 6 --windows 700 --stride 1` labelled 695 new samples
from this capture by temporal consensus, with no annotation, and retraining on
all 1116 cost the clips nothing:

| | clips-only net | + Midway capture |
|---|---|---|
| clip 2 LOS error vs HUD | +0.20 yd | +0.24 yd |
| clip 4 LOS error vs HUD | +0.20 yd | +0.16 yd |
| clips solved (240 frames) | 100% | 100% |
| clip 1 / 2 / 3 lines per frame | 10.7 / 14.2 / 15.3 | 10.8 / 14.9 / 15.4 |

On the Midway capture itself the same ground-truth test passes: on a frame whose
own banner reads **"1ST AND 30 ON THE 37"**, the scrimmage-to-first-down gap
measured through the fitted lattice is **30.3 yd**.

Two things made the labelling practical. The dataset builder was constructing a
`FramePrep` twice per frame - once directly and once inside `analyse_frame`,
which accepts a prepared one - and that is the bulk of the per-frame cost across
a nine-frame voting window. And `--stride`, which exists to stop a *short* clip
contributing near-duplicate neighbours, was discarding two of every three
windows from a long capture whose windows are already spread across 18 minutes.

## Rendering a full capture

A clip is a few hundred frames; this capture is 65,608. At the original
612 ms/frame that is eleven hours, which is too slow to iterate against, so the
hot paths were vectorised - **612 -> 324 ms/frame, with byte-identical output on
all 240 clip frames** at every step:

| change | why it was slow |
|---|---|
| `merge_collinear` | recomputed a direction and a length per *pair* of a few hundred segments |
| `ransac_vp` | 700 separate `vp_angular_error` calls; now one batched evaluation, with the draws left in their original order so the result is unchanged |
| `np.cross` in RANSAC | generic dispatch costing more than three multiply-subtracts |
| `TurfModel.mahalanobis` | computed three times per frame; now once, shared on the `FramePrep` |
| `lattice._comb_score` | `exp(1j*phase)` over ~4 M elements per fit, for values `cos`/`sin` produce directly |

The rest is parallelism: `render_video.py` splits the video into one chunk per
worker, each encoding its own x264 file, concatenated without re-encoding and
muxed back with the original audio. The tracker is the only state that crosses a
chunk boundary, so every worker analyses 120 frames before its own first frame
and starts already locked.

## Highlighting the field

The overlay used to tint the field green at 12%. On teal turf that reads; on
green grass it is invisible. The field now keeps its own colour and **everything
outside it is dimmed to 45%**, which reads on any pitch colour and makes a wrong
mask obvious instead of subtle.

**Extent and geometry are separate claims.** The turf mask is reliable at zoom
levels where the yard lattice is simply underdetermined - clip 1's end-of-play
close-ups show a plain field with one yard line on it - so gating the highlight
on the lattice threw away the claim that was still good. The field is now
highlighted whenever the mask is a credible playing surface, and yard lines are
drawn only when the lattice solves; the status panel says `FIELD ONLY` when
those two disagree.

That needs a test for "is this a field", and Blitz makes it awkward by drawing
its play-select menu over a turf backdrop - a menu and a game both sit near half
the frame.

**The first answer was wrong, and the way it was wrong is the point.** Asking
whether the turf is *in one piece* looked decisive: the menu grid leaves 16-18
fragments with the largest holding 6-7% of the mask, against 2-3 fragments
holding 57-97% for gameplay. An order of magnitude of clearance - measured
entirely on wide shots. At close range a painted yard line is tens of pixels
across, so **the lines themselves cut the field into bands**, and zoomed
gameplay scores 0.25 against menus reaching 0.54. The statistic inverts exactly
where it matters, and it cost 1,447 frames of the capture their highlight.

Coverage over the whole capture is trimodal, and the valley is unambiguous:

| coverage | frames | what it is | lattice solves |
|---|---|---|---|
| 0.000-0.025 | 14,078 | no turf (attract, splash, crowd) | 0.1% |
| 0.425-0.500 | 12,300 | the play-select menu | 6.5% |
| **0.500-0.525** | **820** | **valley** | 33% |
| 0.525 and up | | the field | 46-69% |

So the test is coverage >= 0.52 **or a solved lattice** - two independent kinds
of evidence, because a lattice needs two yard lines lying on turf and so proves
a field by itself. That second clause is not a nicety: without it 441 frames
had yard lines drawn across an unhighlighted frame, which is a contradiction on
its face.

Between 0.15 and 0.50 coverage with no lattice the answer is genuinely
unavailable - a corner-of-the-field view and turf behind a menu occupy the same
range - and those 3,418 frames are left unhighlighted rather than guessed.

Two further honesty fixes came out of watching this capture:

- **Nothing geometric is drawn on an unsolved frame.** A failed frame still
  carries raw detections, and on a replay wipe or a team-select screen those are
  turf texture and banner edges. One frame was drawing 73 "yard lines" across a
  zoomed shot.
- **Two lines minimum.** One line does not fix a lattice, and a single survivor
  coasting through a play-select screen was being drawn with the field
  highlighted behind it. No clip frame is lost to this - the smallest solved
  frame there carries ten.

Yard labels are also now relative to the first lock. The lattice's index origin
falls out of wherever `beta` landed in the cold fit, so the overlay was
labelling real yard lines `+1585`. Lines are still never renamed once assigned -
that is what keeps them stable - but the *displayed* number now counts from the
track's own origin.

## Completing the yard lattice

Detection alone leaves holes: on one clip-3 frame the tracked set was yard
indices 6-11 and 14, with 12 and 13 simply absent. The lattice knows where they
belong, so the obvious move is to draw the whole lattice.

**Measured, that does not work.** Sweeping each drawn line sideways over ±12 px
against the colour ridge response — which neither the net nor the lattice uses:

| | median peak / local floor | median offset from the paint |
|---|---|---|
| detected | 10.9× | 1.0 px |
| lattice-placed | 2.9–4.3× | 7.5 px |

Rebuilding the line through two transversals instead of the vanishing point made
it slightly worse, so the vanishing point was not the cause — the lattice simply
cannot place an unseen line better than ~8 px.

What does work is treating the prediction as a **search seed** rather than an
answer. `render.snap_to_evidence` sweeps each predicted line across a third of
the local yard spacing, moves it to the strongest ridge response, and drops it
entirely unless that peak is 3× the local floor. Inference becomes detection:

| | median peak / floor | median offset | within 2 px |
|---|---|---|---|
| detected | 10.9× | 1.0 px | 70% |
| recovered | 3.7× | **2.0 px** | 52% |

That adds 1.5–3.0 lines per frame on top of 9–12 detected, and the acceptance
test discards roughly four out of five candidates — a gap stays a gap unless the
image supports filling it. Recovered lines are drawn dimmer, because weaker
evidence should look like weaker evidence.

**The threshold needs hysteresis.** Deciding each line afresh every frame makes
anything near the cut toggle, and that is what reads as flicker. A line already
recovered last frame only has to clear 1.8× the local floor; a new one has to
clear 3×. Churn of the drawn label set between neighbouring frames:

| Clip | churn p90 | lines/frame |
|---|---|---|
| 1 Kickoff | 33% → **22%** | 10 |
| 2 Congestion | 25% → **17%** | 12 |
| 3 QB | 17% → **12%** | 9 → 10 |
| 4 Swing | 20% → **12%** | 14 → 15 |

Broken out by camera motion, clip 4 under pan+zoom went from 11% median / 22%
p90 to **6% / 14%** — the flicker was roughly twice as bad while the camera was
moving, which is where it is most visible, and hysteresis roughly halved it.

### The lattice needs a bound it cannot argue its way out of

`all_lines` used to reason that it did not need one: anything it placed off the
field gets clipped away, so the extension is self-limiting. That is true right
up until the fitted pitch collapses, at which point **every** generated line
lands inside the field and clipping removes none of them.

That happened on the full capture, and only a per-frame trace found it:

| frame | lines drawn | image spacing |
|---|---|---|
| 1700 | 11 | 23.8 px |
| 1800 | 138 | 0.4 px |
| 1900 | **17,550** | 0.0 px |

A near-zero pitch makes the lattice's `alpha` tiny, so `_index_of` hands a line
a few pixels away an index thousands of cells from the set, and `all_lines`
then iterates that whole range. The `max_lines` cap did not help — it bounds how
many lines are *tracked*, not how far apart their labels are. It also quietly
inflated the "field present" rate with frames that were nonsense, and it is why
the first chunk of that render took 5772 s against ~3400 s for the others.

The fix is a bound from the sport rather than from the arithmetic: a football
field is 120 yd including both end zones, so **30 five-yard cells is wider than
any camera can be looking at**. Past that the fit is not a field, and the
tracker abandons it and re-acquires instead of drawing from it. The bound never
trips on the clips — output there is byte-identical.

The general lesson is the one this project keeps relearning: a guard that
depends on the thing it is guarding is not a guard. Clipping to the field cannot
bound a lattice whose failure mode is putting everything *on* the field.

## The fit has to be robust, and the labels have to come from detection

The lattice was misfitting its own data — in-sample residuals of 5–6 px on
clips 1 and 2, meaning the fitted lattice could not reproduce the positions of
the very lines it was fitted to. The cause was a plain least-squares fit over a
set containing the occasional **mislabelled** line: a 1-cell error where every
other point is off by hundredths, and a squared cost lets it drag everything.

Two changes, both in `fit_labelled`: score candidate horizons by *median*
absolute residual rather than RMS, then re-fit pitch and phase on the lines that
agree. `FieldTracker._evict` drops the single worst-explained line per frame.

| clip | in-sample residual (med / p90) | leave-one-out (med / p90) |
|---|---|---|
| 1 Kickoff | 5.07 / 31.6 → **3.03 / 12.5** px | 6.38 / 49.2 → **3.86 / 18.5** px |
| 2 Congestion | 6.20 / 30.1 → **1.42 / 6.3** px | 8.15 / 41.5 → **2.31 / 10.2** px |
| 3 QB | 2.95 / 18.2 → **1.70 / 9.1** px | 4.07 / 28.2 → **2.96 / 16.5** px |
| 4 Swing | 1.17 / 10.8 → **0.97 / 6.2** px | 1.55 / 13.8 → **1.64 / 9.6** px |

The p90 column is the one that matters for what a viewer sees — 2–5× tighter is
where the flicker lived.

**Labels are only ever removed, never renamed.** Relabelling a line from the
fitted lattice was tried and is unsafe: re-indexing lets the fit move the labels
to suit itself, the next refit agrees with the labels it just created, and the
pitch walks away. Measured, clip 4's scrimmage-to-first-down drifted from 29 yd
to **75.4** against a HUD still reading 29. The HUD check is the only reason that
was caught, which is the argument for keeping it.

### A dead end worth recording: merging fragments in the pencil

`merge_collinear` runs before the vanishing point is known, so it refuses to
join fragments more than 80 px apart — and a yard line crossed by three players
breaks further apart than that. Merging instead by shared transversal crossing,
once the pencil is known, looked like the obvious fix.

It is not. It recovered **no** extra lines (it only consolidated duplicates of
lines already found: clip 4's pencil went 36 → 24 segments with the tracked
count unchanged at 15), and a fixed 6 px tolerance fuses genuinely distinct yard
lines where they crowd near the horizon — which broke clip 4's yard scale to
75.4 yd. Removed. A scale-aware tolerance could be made safe, but with no
measured recall benefit there is nothing to buy.

## The down-field direction: measured, and it does not close

The yard lattice pins one direction of the field completely. The other — the
sidelines and hash columns running *down* the field — has no lattice, because
they sit at fixed but unequal insets. The published sports-registration work
([BHITK](https://arxiv.org/abs/2311.10361),
[PnLCalib](https://arxiv.org/pdf/2404.08401),
[Homayounfar et al.](https://arxiv.org/pdf/1604.02715)) closes this by fitting a
homography to a **known metric template**. We do not have one — field width is
not observable here — so the down-field vanishing point is the only cross-field
constraint available. It is not strong enough.

**Finding the vanishing point works.** A sideline and the horizon both pass
through it, and the horizon is free (it runs through the yard-line vanishing
point and the lattice's own horizon crossing). Checked against the hash columns,
which come from an unrelated detector, the error is **1.5–4.6 deg** on all four
clips, available on 52–83% of frames.

Voting on tick marks instead — constraining the point to the horizon and scoring
by how tightly hash marks collapse into columns — is *biased*: 2.7 deg on one
clip, 20.4 deg on another, and temporal filtering cut its jitter 3–4x
(100.8→24.8 px) while leaving the agreement unchanged, which is exactly what
averaging a biased estimator does.

**Using it does not work.** Forcing each column through the tracked vanishing
point, instead of fitting it locally to the tick marks, measured against the
tick points themselves:

| Clip | ticks on the line, local fit | via the tracked VP |
|---|---|---|
| 1 Kickoff | 8 (16× a displaced control) | **0** |
| 2 Congestion | 10 (20×) | **1** |
| 3 QB | 14 (28×) | 13 |
| 4 Swing | 13 (26×) | 9 |

The locally fitted columns were already right — 16–28× the tick support of a
displaced control, 1–3 px placement. A few degrees of vanishing-point error
rotates them off the marks they were fitted to. Reverted.

Note the metric trap: averaging the ridge response *along* a column says nothing,
because a hash column is dashed and most of its length is bare turf. That
measure scored the tracked columns at 1.1–1.6× the floor and the local ones the
same, which reads as "both bad" and is simply the wrong instrument. Counting
tick points near the line is the right one, and it separates them cleanly.

**What this leaves.** Column *placement* was never the problem; column
*presence* is — 25% / 94% / 43% / 88% of frames across the four clips. The
vanishing point does not improve presence either (18% / 82% / 41% / 88%).

## Why width is not measured

Width needs focal length; length does not. Focal length is not observable in
this footage: Blitz uses a low, near-ground camera that always looks along the
field axis, so the cross-field vanishing point sits ~200 image widths from
centre while the down-field one sits at ~0.6. Being near-perpendicular from the
principal point makes any focal estimate hypersensitive — a 0.5° error changes
it by 100%. **Zero of 20 sampled frames** had both vanishing points usable.

Five approaches were tried and the metric-rectification code removed.

### The selection ring, measured properly

Blitz draws a blue ring on the turf under the active player. It is a circle on
the ground plane, so its image meets the horizon at the two circular points, and
those fix the plane's metric structure with no focal length at all
(Hartley & Zisserman 8.3.2) — the same role a centre circle plays in soccer.

Detection is the easy part: **89–100% of frames**, far better availability than
the sidelines (51–84%) or the goal lines (4–31%).

The test is whether the yard-line direction and the down-field direction come
out perpendicular after rectification. Validated first on a synthetic camera,
where the method returns **90.000°** at every pitch angle and still 89.36° with
the horizon displaced 20 px. On the footage, with a robust conic fit:

| Clip | camera | angle (90° = correct) | IQR |
|---|---|---|---|
| 1 Kickoff | wide | **82.1°** | 19.2 |
| 4 Swing | wide, swinging | **82.0°** | 17.7 |
| 2 Congestion | zoomed in | 54.4° | 23.9 |
| 3 QB | zoomed in | 50.0° | 28.1 |

So the ring **is** real ground-plane geometry — 82° on the wide clips is not
something a flat billboard sprite would produce. But it splits cleanly by camera:
the wide shots land 8° short, the zoomed shots 36–40° short. That is a
conditioning failure, not a bug: under a longer lens the perspective weakens, the
ring's image approaches a fixed shape, and the circular points — which live
exactly in how the conic *departs* from that shape — become ill-determined. It is
the same sensitivity that makes the focal length unrecoverable.

Ruled out along the way: ring occlusion is *not* the cause. Visible arc coverage
is 68–81% on all four clips and restricting to well-exposed rings changes the
answer by under 6°.

Also recorded because it cost a full round: the first version of this test
returned 3.5–10.5° and looked like a flat refutation. It was a bug — the
C*inf quadratic form measures the angle between two *lines*, and it was being
fed two vanishing *points*. The synthetic check caught it; nothing else would
have. Build the instrument a known answer before trusting it on data.

The other remaining route is the cross-ratio of the four tick columns, which
gives hash-inset-to-width as a calibration-free *ratio*.

## Known weak points

- **Painted yard numbers** — not drawn. The size window in
  `render.yard_numbers` assumes a digit is ~2 yd tall (0.40 × the 5-yd line
  spacing), but measured on these clips the blobs sit at 0.81 × spacing
  (median, n=200) and a directly measured "2" came out at 0.92 × — Blitz
  numbers are nearer 4 yd than 2, so the window is centred below the digits and
  admits hash marks instead. Fixing the window is necessary but probably not
  sufficient; doing it properly means rectifying the turf flat so digits become
  upright, then template-matching as `hudtext` does for the HUD font.
- **Occasional spurious tick columns** — a density constraint removed most, not
  all, and they inherit whatever jitter is left in the lattice fit. Typically
  1–2 of the field's columns are found per frame, sometimes none. Requiring
  mutual consistency between columns in a frame would help; fitting them all to
  a single down-field vanishing point would help more.
- **Hash marks need the thin mask, not the net.** A hash mark is identified by
  its size — a few dozen pixels, no dimension over ~26 px — and the learned line
  channel is trained to be thick and continuous. On clip 4 that was 84 tick
  candidates and 2 columns from the top-hat mask against 27 and 0 from the net,
  so `FramePrep` carries both and `yardmap` uses the thin one.
- **Line-set churn on clip 4 went 0% → 8% at the 90th percentile** with the
  learned front end. It finds ~29% more long lines there, and the extra ones at
  the edge of the set turn over between frames. More evidence for a cost in
  steadiness; the tracked lattice itself did not move.
- **A net is fitted to the captures it has seen.** The clips-only net scored
  P(turf) above threshold on 3–5% of the Midway capture's pixels — a complete
  failure, not a degradation. **This has now happened twice.** A third capture
  (`1_Hail_Mary_Left_Side`, a dark desaturated grade, turf at Lab (a,b) =
  (119.3, 132.9)) broke the clips+Midway net exactly the same way: 4–10% turf
  against the colour model's 65–88%, field mask 0.00. Adding 213 auto-labelled
  rows from it fixed that and cost the other sources nothing. Treat it as a
  standing rule: **a new capture needs its own dataset rows before the net is
  trusted on it**, and the colour front end (`--no-net`) is the fallback that
  works meanwhile. See "A second source".
- **Down-field camera angles** — observed on one frame, never quantified.
  Looking along the field rather than across it, the sidelines and hash columns
  are long and numerous while the yard lines are short and nearly edge-on, and
  RANSAC appeared to take the down-field pencil as the dominant one. An attempt
  to measure how often this happens failed: the referee used was "is the pencil
  parallel to the longest boundary segment", and the longest boundary segment is
  frequently a goal line or the far turf edge, which runs *with* the yard lines.
  It flagged 37% of frames, but those frames still solved at 79% with a median
  of 7 lines — which a genuinely wrong pencil cannot do. **The rate is unknown,
  and the bound below says it cannot be large.**
- **What is actually left unsolved.** Of the 27,589 frames with a field on
  screen, 6,339 (23%) get no yard lattice. 6,020 of those sit in 41 *sustained*
  runs — quarter-end and "LIONS WIN!" celebration screens, and deep-zoom
  replays, all with a giant text overlay across the field and few yard lines in
  view. Only 319 frames, **0.49% of the capture**, are brief dropouts during
  play. The field is highlighted on all of them either way. There is no large
  addressable category of failure remaining in this footage.
- **Play-select screens** are recognised by mask fragmentation, not by anything
  that understands what they are. A future screen that leaves the turf in one
  piece would defeat the test.
- **End zone depth** — unmeasured; goal lines appear in only ~9% of frames.
- Video 7 is anisotropically stretched and contributes little.

## Requirements

Python 3.13, `opencv-contrib-python` (needs `ximgproc.thinning` and
`LSD_REFINE_STD`), numpy, scikit-learn, ffmpeg on PATH.

`torch` is optional and only for the learned front end. It must be built for
your GPU's architecture: a stock `cu126` wheel reports `cuda.is_available()`
True on an RTX 50-series card and then fails with *no kernel image is available
for execution on the device*, so `fieldnet` probes with a real kernel and falls
back to CPU. For training, install a matching build:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu130
```
