"""Build the auto-labelled dataset and train ``blitzfield.fieldnet``.

    python scripts/train_fieldnet.py dataset            # label from the footage
    python scripts/train_fieldnet.py train              # fit the net

Both stages are offline: nothing here runs during a render, and ``out/`` is
untouched.  The dataset lands in ``work/fieldnet/`` and the weights beside it.

No frame is labelled by hand.  The existing colour pipeline is run over a window
of consecutive frames, every mask in the window is warped into the centre
frame's view through the estimated camera motion, and the per-pixel vote becomes
the target.  A pixel the window cannot agree on is marked *ignore* and never
contributes a gradient - so the net learns what nine frames establish, not what
one frame guessed.
"""
import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from blitzfield.config import WORK, VIDEOS, get_video

DATA = WORK / "fieldnet"
IMG, LBL = DATA / "img", DATA / "lbl"
IGNORE = 255

#: Half-width of the voting window, in frames.  Wide enough that a marginal
#: pixel is outvoted, short enough that the chained motion estimate has not
#: drifted: error accumulates over the chain, and past ~5 frames the warp itself
#: starts to blur the very boundaries the vote is meant to sharpen.
HALF = 4

# Vote thresholds.  Deliberately asymmetric: turf has to be nearly unanimous to
# count as positive because the tint is what over-reaches, while a line only
# needs a clear majority because thin structures lose votes to resampling.
TURF_POS, TURF_NEG = 0.85, 0.15
LINE_POS, LINE_NEG = 0.65, 0.05
OCC_MAX = 0.10          # do not call "no line" where a player kept getting in the way
OCC_POS = 0.30          # nor call "line" where one mostly was
MAYBE_MAX = 0.20        # nor anywhere a lenient test still sees something bright
SUPPORT = 0.6           # fraction of the window that must actually see a pixel
DRAW_PX = 3             # thickness a detected segment is painted into the target


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
def _to_centre(Hs: list, c: int, j: int) -> np.ndarray:
    """Transform carrying frame ``j``'s pixels into frame ``c``'s view.

    ``Hs[i]`` maps entry ``i-1`` to entry ``i``, so the composition telescopes
    in one direction and is inverted in the other.
    """
    M = np.eye(3)
    if j < c:
        for i in range(j + 1, c + 1):
            M = Hs[i] @ M
    elif j > c:
        for i in range(c + 1, j + 1):
            M = Hs[i] @ M
        M = np.linalg.inv(M)
    return M


def _vote(buf: list, c: int, shape: tuple) -> tuple:
    """Per-pixel mean of the window's masks, warped into the centre frame.

    The weight map is not a detail: warping leaves the incoming border empty,
    and without dividing by the count of frames that actually *saw* each pixel
    the frame edges would be systematically voted "not turf".
    """
    h, w = shape
    Hs = [e["H"] for e in buf]
    keys = ("turf", "line", "occ", "maybe")
    acc = np.zeros((len(keys), h, w), np.float32)
    wt = np.zeros((h, w), np.float32)
    ones = np.ones((h, w), np.float32)
    for j, e in enumerate(buf):
        stack = np.stack([e[k] for k in keys]).astype(np.float32)
        if j == c:
            acc += stack
            wt += ones
            continue
        M = _to_centre(Hs, c, j)
        for ch in range(len(keys)):
            acc[ch] += cv2.warpPerspective(stack[ch], M, (w, h),
                                           flags=cv2.INTER_LINEAR)
        wt += cv2.warpPerspective(ones, M, (w, h), flags=cv2.INTER_LINEAR)
    ok = wt >= SUPPORT * len(buf)
    safe = np.where(wt > 1e-6, wt, 1.0)
    return dict(zip(keys, acc / safe)), ok


def _labels(c, ok) -> np.ndarray:
    """Two label planes, values 0 / 1 / IGNORE.

    The negative rule for lines is the delicate one.  The ridge mask alone is
    patchy - near the camera a painted line is wide and a thin-line top-hat only
    answers at its two edges - so a target built from it labels the *middle of a
    real yard line* as "no line", which is worse than no supervision at all.
    The fitted segments are voted on alongside the mask precisely to close those
    gaps, and anything still undecided is left ignored rather than guessed.
    """
    ct, cl, co, cm = c["turf"], c["line"], c["occ"], c["maybe"]
    turf = np.full(ct.shape, IGNORE, np.uint8)
    turf[ct <= TURF_NEG] = 0
    turf[ct >= TURF_POS] = 1
    line = np.full(cl.shape, IGNORE, np.uint8)
    line[(cl <= LINE_NEG) & (co <= OCC_MAX) & (cm <= MAYBE_MAX)] = 0
    line[(cl >= LINE_POS) & (co <= OCC_POS)] = 1
    turf[~ok] = IGNORE
    line[~ok] = IGNORE
    # Written as an ordinary BGR PNG so it can be opened in any viewer:
    # red plane = turf, green plane = line.
    return np.stack([np.zeros_like(turf), line, turf], axis=-1)


def _runs(spec, n_windows: int, span: int):
    """Frame ranges to label: whole video when short, sampled when long.

    A 37-minute capture has ~67k frames and labelling one costs a full colour
    pass, so long videos are sampled as short bursts of consecutive frames -
    consecutive because the vote needs neighbours, scattered because 400 bursts
    spread over the whole capture carry far more variety than 3600 frames of one
    drive.
    """
    gate = json.loads((spec.workdir / "stage1" / "gating.json").read_text())
    play = [(s["start"], s["stop"]) for s in gate["segments"]
            if s["kind"] == "gameplay" and s["stop"] - s["start"] > span]
    if not play:
        return []
    total = sum(b - a for a, b in play)
    if total <= 2500:
        return play
    starts, acc = [], 0.0
    step = total / n_windows
    for a, b in play:
        t = a
        while t + span <= b:
            if acc >= step:
                starts.append((t, t + span))
                acc -= step
            acc += 1.0
            t += 1
    return starts


def _line_evidence(prep, fm) -> np.ndarray:
    """Everything this frame says is a painted marking.

    Two sources, because they fail in opposite places: the ridge mask is dense
    but only answers where a line is thin, while the fitted segments are clean
    and span the wide near-camera lines the mask hollows out.  Boundaries are
    included - a sideline is painted too, and the net has no reason to treat it
    differently from a yard line.
    """
    m = cv2.dilate(prep.mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    segs = [fm.segs[fm.label == 0], fm.boundary]
    for arr in segs:
        for s in (np.asarray(arr, float).reshape(-1, 4) if arr is not None
                  and len(arr) else []):
            cv2.line(m, (int(round(s[0])), int(round(s[1]))),
                     (int(round(s[2])), int(round(s[3]))), 1, DRAW_PX)
    return m


def _maybe_marking(frame, prep, pct: float = 88.0, chroma_max: float = 26.0):
    """A deliberately permissive "this could be paint" test.

    Used only to *withhold* the negative label.  The strict evidence misses the
    wide, washed-out lines nearest the camera - a thin-line top-hat answers at
    their two edges and LSD often declines them altogether - and calling the
    middle of a real yard line "no line" teaches the net the one thing it must
    not learn.  Anything bright and colour-neutral on the field is left
    unsupervised instead.
    """
    from blitzfield import fieldcolor
    L = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32)
    reg = prep.region & ~prep.hud
    if reg.sum() < 500:
        return np.zeros(frame.shape[:2], np.uint8)
    thr = float(np.percentile(L[reg], pct))
    m = (reg & (L >= thr) & (fieldcolor.chroma(frame) < chroma_max))
    return cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))


def build(videos, n_windows: int, stride: int, limit=None):
    from blitzfield import fieldcolor, hudmask, ingest, markings, motion as mo

    IMG.mkdir(parents=True, exist_ok=True)
    LBL.mkdir(parents=True, exist_ok=True)
    span = 2 * HALF + 1
    # Merge rather than overwrite: the long captures each take a while to label
    # and are prepared at different times, so the dataset is built up video by
    # video.  Re-running one video replaces just its own rows.
    prev = (json.loads((DATA / "index.json").read_text())
            if (DATA / "index.json").exists() else [])
    vids = {get_video(v).vid for v in videos}
    index = [r for r in prev if r["vid"] not in vids]
    for v in videos:
        spec = get_video(v)
        if not (spec.workdir / "stage1" / "turf_model.json").exists():
            print(f"  [{spec.vid}] not prepared - skipping")
            continue
        turf = fieldcolor.TurfModel.from_json(
            (spec.workdir / "stage1" / "turf_model.json").read_text())
        hud = hudmask.load(spec)
        _, area = ingest.load(spec)
        n, t0 = 0, time.time()
        rng = np.random.default_rng(0)
        for a, b in _runs(spec, n_windows, span):
            est = mo.MotionEstimator()
            buf = []
            with ingest.FrameSource(spec, area, stride=1, start=a, stop=b) as src:
                for i, canon in src:
                    m = est.update(canon, buf[-1]["region"] if buf else None)
                    if m.label == "cut":
                        buf = []
                        continue
                    prep = markings.FramePrep.build(canon, turf, hud=hud)
                    # Hand the prep over rather than letting analyse_frame build
                    # its own: it is the bulk of the per-frame cost, and the
                    # window needs nine of them for every sample kept.
                    fm = markings.analyse_frame(canon, turf, hud=hud, rng=rng,
                                                prep=prep)
                    buf.append(dict(
                        i=i, frame=canon, region=prep.region,
                        H=(np.asarray(m.H, float) if m.H is not None else np.eye(3)),
                        turf=prep.turf_mask.astype(np.uint8),
                        line=_line_evidence(prep, fm),
                        occ=(prep.occluder | prep.hud).astype(np.uint8),
                        maybe=_maybe_marking(canon, prep)))
                    if len(buf) > span:
                        buf.pop(0)
                    if len(buf) < span:
                        continue
                    c = HALF
                    if buf[c]["i"] % stride:
                        continue
                    cons, ok = _vote(buf, c, canon.shape[:2])
                    lbl = _labels(cons, ok)
                    if (lbl[:, :, 1] == 1).sum() < 200:      # nothing to learn
                        continue
                    name = f"{spec.vid}_{buf[c]['i']:06d}"
                    cv2.imwrite(str(IMG / f"{name}.jpg"), buf[c]["frame"],
                                [cv2.IMWRITE_JPEG_QUALITY, 95])
                    cv2.imwrite(str(LBL / f"{name}.png"), lbl)
                    index.append({"name": name, "vid": spec.vid,
                                  "frame": int(buf[c]["i"])})
                    n += 1
                    if limit and n >= limit:
                        break
            if limit and n >= limit:
                break
        print(f"  [{spec.vid}] {n} samples  ({time.time() - t0:.0f}s)", flush=True)
    (DATA / "index.json").write_text(json.dumps(index, indent=1))
    print(f"{len(index)} samples -> {DATA}")


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
#: Photometric jitter when the frames have been normalised, as a fraction of
#: the un-normalised width.  Measured, not guessed: the normaliser pins turf
#: luminance to a relative spread of 0.007 across the six captures, down from
#: 0.405 raw, so the full-width jitter swings turf L over ~106-136 where the
#: real normalised range is 116-121.  Augmentation is meant to cover the spread
#: inference will present, slightly broadened; at full width it instead destroys
#: the canonical colour the normaliser exists to establish, and the net cannot
#: learn to use it.  Paint luminance still varies ~12% after normalisation, but
#: that variation is already carried by the six real captures and does not need
#: synthesising.  0.2 covers the measured residual with margin.
NORM_JITTER = 0.2


def _augment(img, lbl, rng, crop=(384, 512), jitter: float = 1.0):
    """Random crop, flip and photometric jitter.

    Geometry is left alone beyond the flip.  The camera only ever looks along
    the field from a low angle, so rotating or shearing the crop would teach the
    net line orientations this game never produces.

    ``jitter`` scales the photometric deviation from identity.  1.0 is the width
    used for raw frames, which have to cover a 40% cross-capture luminance
    spread; see :data:`NORM_JITTER` for why normalised frames want far less.
    """
    ch, cw = crop
    h, w = img.shape[:2]
    y = int(rng.integers(0, max(1, h - ch + 1)))
    x = int(rng.integers(0, max(1, w - cw + 1)))
    img, lbl = img[y:y + ch, x:x + cw], lbl[y:y + ch, x:x + cw]
    if rng.random() < 0.5:
        img, lbl = img[:, ::-1], lbl[:, ::-1]
    f = np.float32(img)
    f *= 1.0 + jitter * float(rng.uniform(-0.25, 0.30))     # exposure
    f = ((f - f.mean()) * (1.0 + jitter * float(rng.uniform(-0.15, 0.20)))
         + f.mean())                                        # contrast
    f += rng.normal(0.0, 4.0 * jitter, (1, 1, 3)).astype(np.float32)  # colour cast
    return np.clip(f, 0, 255).astype(np.uint8), np.ascontiguousarray(lbl)


def _worker_init(_wid: int) -> None:
    """One cv2 thread per loader worker.

    cv2 defaults to a thread per core - 24 on this machine - so N workers each
    opening 24 threads oversubscribes it badly and runs slower than no workers
    at all.  The parallelism we want is across samples, not inside one imread.
    """
    cv2.setNumThreads(1)


class _Rows:
    """One split's samples, as a map-style dataset.

    Deliberately a dataset rather than the inline loop it replaces, because the
    per-sample cost is dominated by work that parallelises cleanly: two imreads
    and, when normalising, a full-frame Mahalanobis pass.  Measured at 51 ms a
    sample - 2.8 jpg, 2.7 png, 45.6 normalise, of which 25.1 is the Mahalanobis
    - which is most of an epoch spent with the GPU idle.

    Steady-state seconds per epoch on the 1170-row set, normalising, on this
    machine (24 cores, one RTX 5070 Ti):

    ====  ====  ====  ====
    w=0   w=3   w=6   w=12
    ====  ====  ====  ====
    103    37    23    23
    ====  ====  ====  ====

    Six saturates it; twelve buys nothing, so ~23 s is the compute floor and the
    data path is fully hidden behind it.  A 30-epoch run went from 53 min to 12.

    Note the per-worker RNG below: augmentation draws from a different stream
    than the single shared generator this replaced, so **results are not
    bit-comparable with runs made before this change**, and reproducing a run
    means reusing its ``--workers`` value.  The train/val split is unaffected,
    being drawn before any augmentation.

    Not a ``torch.utils.data.Dataset`` subclass on purpose: this module is
    importable and the ``dataset`` stage runnable without torch installed, so
    torch is imported lazily.  ``DataLoader`` treats any object with
    ``__len__``/``__getitem__`` as map-style, which is all it needs.
    """

    def __init__(self, rows, turfs, normalise, augment, jitter):
        self.rows = rows
        self.turfs = turfs            # vid -> TurfModel, pickled to workers
        self.normalise = normalise
        self.augment = augment
        self.jitter = jitter
        self._g = None                # per-worker, created on first use

    def __len__(self):
        return len(self.rows)

    def _rng(self):
        # Seeded from torch's per-worker seed, which the DataLoader derives from
        # the generator it was given, so a run is reproducible for a fixed
        # worker count.  Each worker therefore draws its own stream and no two
        # workers augment a sample identically.
        if self._g is None:
            import torch
            self._g = np.random.default_rng(torch.initial_seed() % (2 ** 32))
        return self._g

    def __getitem__(self, i):
        import torch
        from blitzfield.fieldnet import to_tensor
        from blitzfield import fieldcolor

        rec = self.rows[i]
        img = cv2.imread(str(IMG / f"{rec['name']}.jpg"), cv2.IMREAD_COLOR)
        lbl = cv2.imread(str(LBL / f"{rec['name']}.png"), cv2.IMREAD_COLOR)
        if self.normalise:
            img = fieldcolor.normalise_to_canon(img, self.turfs[rec["vid"]])
        lbl = lbl[:, :, [2, 1]]                # PNG is BGR: R=turf, G=line
        if self.augment:
            img, lbl = _augment(img, lbl, self._rng(), jitter=self.jitter)
        y = np.where(lbl == IGNORE, 0, lbl).astype(np.float32)
        m = (lbl != IGNORE).astype(np.float32)
        return (to_tensor(img)[0],
                torch.from_numpy(y.transpose(2, 0, 1)),
                torch.from_numpy(m.transpose(2, 0, 1)))


def train(epochs: int, batch: int, base: int, lr: float, val_frac: float,
          seed: int = 0, holdout: str | None = None, out=None,
          normalise: bool = False, workers: int = 6):
    import torch
    from torch import nn
    from blitzfield.fieldnet import FieldNet, _best_device
    from blitzfield import fieldcolor

    # Applied on load rather than baked into the dataset: the stored jpg is the
    # canonical frame, and normalisation is a pure function of it and the
    # video's own turf model, so the same rows serve both experiments.
    turf_of: dict = {}

    def _turf(vid):
        if vid not in turf_of:
            spec = get_video(vid)
            turf_of[vid] = fieldcolor.TurfModel.from_json(
                (spec.workdir / "stage1" / "turf_model.json").read_text())
        return turf_of[vid]

    jit = NORM_JITTER if normalise else 1.0
    if normalise:
        print(f"normalising on load; photometric jitter narrowed to {jit:g}x")

    # Absent, not merely empty: the generated dataset is large and gets deleted
    # between campaigns, so a missing index is the normal case to report well
    # rather than a FileNotFoundError traceback.
    idx_path = DATA / "index.json"
    if not idx_path.exists():
        raise SystemExit(
            f"no dataset at {DATA} - run the `dataset` stage first, e.g.\n"
            "  train_fieldnet.py dataset 1 2 3 4\n"
            "  train_fieldnet.py dataset 6 --windows 700 --stride 1\n"
            "  train_fieldnet.py dataset 8 --stride 1")
    index = json.loads(idx_path.read_text())
    if not index:
        raise SystemExit(f"dataset index at {idx_path} is empty")
    rng = np.random.default_rng(seed)
    if holdout:
        # The honest test with only four clips.  A random split leaves frames
        # two apart from each other on both sides of it, so a net that simply
        # memorised the footage would still score well; holding out a whole clip
        # asks whether it learned the game instead.
        val = [r for r in index if r["vid"] == holdout]
        trn = [r for r in index if r["vid"] != holdout]
        if not val:
            raise SystemExit(f"no samples for holdout {holdout!r}")
        print(f"holding out {holdout}: {len(trn)} train / {len(val)} val")
    else:
        order = rng.permutation(len(index))
        n_val = max(1, int(len(index) * val_frac))
        val = [index[i] for i in order[:n_val]]
        trn = [index[i] for i in order[n_val:]]
        print(f"{len(trn)} train / {len(val)} val samples")

    # Not `cuda.is_available()`: it answers "is there a driver and a device",
    # not "was this build compiled for it".  On this machine it says True and
    # then dies on the first BatchNorm.  _best_device probes with a real kernel.
    dev = torch.device(_best_device())
    net = FieldNet(base=base).to(dev)
    n_par = sum(p.numel() for p in net.parameters())
    print(f"FieldNet base={base}  {n_par/1e6:.2f} M parameters on {dev}")
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    # Lines cover ~2% of a frame, so an unweighted loss is minimised by
    # predicting "no line" everywhere.
    posw = torch.tensor([1.0, 12.0], device=dev).view(1, 2, 1, 1)
    lossf = nn.BCEWithLogitsLoss(reduction="none", pos_weight=posw)

    # Turf models are resolved up front so the workers receive them by pickle
    # rather than each re-reading and re-parsing the json.
    turfs = {v: _turf(v) for v in {r["vid"] for r in index}} if normalise else {}
    pin = dev.type == "cuda"
    g = torch.Generator()
    g.manual_seed(seed)
    loader = torch.utils.data.DataLoader
    # persistent_workers because Windows spawns rather than forks: re-creating
    # six interpreters every epoch costs more than the epoch saves.
    tl = loader(_Rows(trn, turfs, normalise, True, jit),
                batch_size=batch, shuffle=True, drop_last=True,
                num_workers=workers, pin_memory=pin, generator=g,
                worker_init_fn=_worker_init,
                persistent_workers=workers > 0)
    vl = loader(_Rows(val, turfs, normalise, False, jit),
                batch_size=batch, shuffle=False, drop_last=False,
                num_workers=workers, pin_memory=pin,
                worker_init_fn=_worker_init,
                persistent_workers=workers > 0)
    print(f"data loading on {workers} worker(s); "
          f"{len(tl)} train batches/epoch, {len(vl)} val")

    # From the loader, not recomputed: OneCycleLR raises if it is stepped past
    # total_steps, so the batch count has to come from the thing that decides it.
    steps = epochs * max(1, len(tl))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)

    def _to_dev(x, y, m):
        return (x.to(dev, non_blocking=pin), y.to(dev, non_blocking=pin),
                m.to(dev, non_blocking=pin))

    @torch.no_grad()
    def evaluate(thresholds=(0.5,)):
        """Val IoU per class, at each threshold.

        Swept rather than fixed at 0.5 because the loss is class-weighted 12:1
        towards "line".  That is necessary - lines are ~2% of a frame and an
        unweighted loss just predicts "no line" - but it also means the logit at
        which precision and recall balance is nowhere near zero, and it drifts
        as training sharpens.  Scoring only at 0.5 measured the drift, not the
        model: line IoU read 0.21 at one epoch and 0.68 at the next while the
        training loss fell smoothly through both.
        """
        net.eval()
        th = np.asarray(thresholds, np.float32)
        inter = np.zeros((len(th), 2)); union = np.zeros((len(th), 2))
        for bx, by, bm in vl:
            x, y, m = _to_dev(bx, by, bm)
            prob = torch.sigmoid(net(x))
            for ti, tv in enumerate(th):
                p = (prob > float(tv)).float()
                for c in range(2):
                    pc, yc, mc = p[:, c] * m[:, c], y[:, c] * m[:, c], m[:, c]
                    inter[ti, c] += float((pc * yc).sum())
                    union[ti, c] += float(((pc + yc) > 0).float().mul(mc).sum())
        net.train()
        return inter / np.maximum(union, 1)

    # Wide enough that the optimum is interior.  Measured on this dataset the
    # line threshold sits at 0.87-0.90 - a grid that stops at 0.85 reports its
    # own top end and silently understates the model.
    SWEEP = np.round(np.arange(0.20, 0.96, 0.025), 3)
    best = {"score": -1.0}
    from blitzfield.fieldnet import WEIGHTS as _W
    dest = pathlib.Path(out) if out else _W
    dest.parent.mkdir(parents=True, exist_ok=True)

    def _save():
        """Write the best-so-far.  Called on every improvement rather than once
        at the end: a 50-minute run killed at epoch 16 leaves nothing behind
        otherwise, which is exactly what happened."""
        torch.save({"model": best["state"], "base": base, "holdout": holdout,
                    "val_iou": [best["turf_iou"], best["score"]],
                    "turf_thr": best["turf_thr"], "line_thr": best["line_thr"],
                    "normalise": normalise,
                    "epoch": best["epoch"], "n_train": len(trn)}, str(dest))
    t0, step = time.time(), 0
    for ep in range(epochs):
        run = 0.0
        for bx, by, bm in tl:        # shuffling is the loader's job now
            x, y, m = _to_dev(bx, by, bm)
            loss = (lossf(net(x), y) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step < steps - 1:
                sched.step()
            run += float(loss.detach())
            step += 1
        iou = evaluate(SWEEP)
        bi = int(np.argmax(iou[:, 1]))
        # The checkpoint kept is the best epoch at its own best threshold, not
        # the last.  Line IoU does not improve monotonically here and the final
        # epoch is routinely well short of the peak.
        if iou[bi, 1] > best["score"]:
            best = {"score": float(iou[bi, 1]),
                    "epoch": ep + 1,
                    "line_thr": float(SWEEP[bi]),
                    "turf_thr": float(SWEEP[int(np.argmax(iou[:, 0]))]),
                    "turf_iou": float(iou[:, 0].max()),
                    "state": {k: v.detach().cpu().clone()
                              for k, v in net.state_dict().items()}}
            _save()
        i50 = int(np.argmin(np.abs(SWEEP - 0.5)))
        print(f"  epoch {ep+1:2d}/{epochs}  loss {run/max(1,len(tl)):.4f}"
              f"   val IoU@0.5  turf {iou[i50,0]:.3f}  line {iou[i50,1]:.3f}"
              f"   best line {iou[bi,1]:.3f} @{SWEEP[bi]:.2f}"
              f"   ({time.time()-t0:.0f}s)", flush=True)

    _save()
    print(f"best epoch {best['epoch']}: line IoU {best['score']:.3f} at "
          f"threshold {best['line_thr']:.2f}, turf IoU {best['turf_iou']:.3f} "
          f"at {best['turf_thr']:.2f}")
    print(f"-> {dest}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("videos", nargs="*", default=[v.order for v in VIDEOS])
    d.add_argument("--windows", type=int, default=300,
                   help="bursts to sample from a long video")
    d.add_argument("--stride", type=int, default=3,
                   help="keep every Nth centre frame (short videos only; a "
                        "sampled long video is already spread out, so pass 1)")
    d.add_argument("--limit", type=int, default=None)
    t = sub.add_parser("train")
    t.add_argument("--epochs", type=int, default=24)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--base", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-3)
    t.add_argument("--val-frac", type=float, default=0.12)
    t.add_argument("--holdout", default=None,
                   help="validate on this clip's slug, train on the others")
    t.add_argument("--out", default=None, help="weights path override")
    t.add_argument("--normalise", action="store_true",
                   help="map every frame's turf colour onto a canonical one "
                        "before the net sees it")
    t.add_argument("--seed", type=int, default=0,
                   help="controls BOTH the train/val split and the "
                        "augmentation stream, so varying it measures total "
                        "run-to-run variance - which is what a ship decision "
                        "needs, since production does not get to pick a "
                        "lucky split")
    t.add_argument("--workers", type=int, default=6,
                   help="data-loading worker processes; 0 loads in-process. "
                        "Default 6, which saturates the data path here "
                        "(103 -> 23 s/epoch; 12 is no faster)")
    a = ap.parse_args()
    if a.cmd == "dataset":
        build([int(v) for v in a.videos], a.windows, a.stride, a.limit)
    else:
        train(a.epochs, a.batch, a.base, a.lr, a.val_frac, seed=a.seed,
              holdout=a.holdout, out=a.out, normalise=a.normalise,
              workers=a.workers)
