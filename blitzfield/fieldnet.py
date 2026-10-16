"""Stage 3b - a learned front end: turf and painted markings, per pixel.

Why a network here and nowhere else
-----------------------------------
Everything *downstream* of the colour decision is already stable.  The flank
test rejects player silhouettes without a temporal model, the tracker carries
lines through occlusion, and the lattice is anchored in time.  The colour
decision itself is the weak link, and measurably so: ~4% of every frame sits
within +-15% of the Mahalanobis cut, and the drawn field mask moves by 2.5-5%
between neighbouring frames *after* camera motion is compensated for.  A
per-pixel Gaussian has one Lab coordinate to go on, so mottled turf, cast
shadows and the out-of-bounds apron are all one bad pixel away from flipping.

A convolutional net has context.  It can use that a dark patch is surrounded by
turf, that a line continues behind a player, that the apron beyond the sideline
is a different surface even where its colour nearly matches.

Where the labels come from
--------------------------
Nothing is hand-drawn - the project's no-manual-input rule still holds.  The
existing pipeline is run over the footage and its per-frame masks are warped
into a common frame through the estimated camera motion and voted on.  That
matters for more than convenience: temporal consensus is strictly more evidence
than any single frame carries, so the network is not merely distilling the
colour model.  It is trained to reproduce from *one* frame what the colour model
only gets right when allowed to see nine.  Pixels the vote cannot settle are
marked ignore and contribute no gradient, so the teacher's uncertainty is not
learned as fact.

The network is deliberately small (~0.5 M parameters).  The domain is one game
with one renderer; capacity buys nothing here and costs inference time on every
frame of every render.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .config import WORK

#: Trained weights.  Shared across videos: the point of the net is that one
#: model covers footage the per-video colour models had to be refitted for.
WEIGHTS = WORK / "fieldnet" / "fieldnet.pt"

TURF, LINE = 0, 1


def _block(cin: int, cout: int) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False),
        nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class FieldNet(nn.Module):
    """U-Net predicting P(turf) and P(painted line) per pixel.

    Three pooling levels, not four.  The features that matter are thin - a
    painted line is 1-3 px at canonical scale - and each extra level halves the
    resolution the decoder has to reconstruct them from.  Three still gives the
    bottleneck a ~100 px receptive field, which is wider than a player.
    """

    def __init__(self, base: int = 16):
        super().__init__()
        b = base
        self.e1, self.e2, self.e3 = _block(3, b), _block(b, 2 * b), _block(2 * b, 4 * b)
        self.bottleneck = _block(4 * b, 8 * b)
        self.u3 = nn.ConvTranspose2d(8 * b, 4 * b, 2, 2)
        self.d3 = _block(8 * b, 4 * b)
        self.u2 = nn.ConvTranspose2d(4 * b, 2 * b, 2, 2)
        self.d2 = _block(4 * b, 2 * b)
        self.u1 = nn.ConvTranspose2d(2 * b, b, 2, 2)
        self.d1 = _block(2 * b, b)
        self.head = nn.Conv2d(b, 2, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        z = self.bottleneck(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(z), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def _best_device() -> str:
    """CUDA only if it actually runs.

    ``torch.cuda.is_available()`` answers "is there a driver and a device", not
    "was this build compiled for it".  A torch built for an older architecture
    reports True on a newer card and then raises
    ``no kernel image is available for execution on the device`` on the first
    real kernel - which is exactly what a stock install does on this machine.
    One tiny matmul settles it.
    """
    if not torch.cuda.is_available():
        return "cpu"
    try:
        (torch.zeros(8, 8, device="cuda") @ torch.zeros(8, 8, device="cuda")).sum()
        torch.cuda.synchronize()
        return "cuda"
    except Exception:
        return "cpu"


def to_tensor(frame_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HxWx3 -> normalised NCHW float tensor."""
    x = frame_bgr[:, :, ::-1].astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))[None]


class FieldSegmenter:
    """Run :class:`FieldNet` on single frames.

    Inference is fully convolutional, so the net trains on crops and runs on
    whole frames.  The only constraint is that the input side is a multiple of
    8 (three poolings); frames are padded by replication and cropped back, which
    keeps the border free of the dark halo zero-padding would introduce.
    """

    #: Fallbacks for a checkpoint that predates threshold selection.
    DEFAULT_TURF_THR = 0.5
    DEFAULT_LINE_THR = 0.5

    def __init__(self, weights: Path | str = WEIGHTS, device: str | None = None,
                 base: int = 16):
        self.device = torch.device(device or _best_device())
        state = torch.load(str(weights), map_location=self.device)
        self.turf_thr = self.DEFAULT_TURF_THR
        self.line_thr = self.DEFAULT_LINE_THR
        #: Whether this net expects colour-normalised input.  A property of the
        #: weights, like the thresholds: feeding a normalised frame to a net
        #: trained on raw ones is as wrong as the reverse.
        self.normalise = False
        if isinstance(state, dict) and "model" in state:
            base = int(state.get("base", base))
            # Decision thresholds belong to the weights, not to the caller: the
            # loss is class-weighted, so the logit where precision and recall
            # balance is model-specific and nowhere near zero.  Training picks
            # them on held-out frames and stores them here.
            self.turf_thr = float(state.get("turf_thr", self.DEFAULT_TURF_THR))
            self.line_thr = float(state.get("line_thr", self.DEFAULT_LINE_THR))
            self.normalise = bool(state.get("normalise", False))
            state = state["model"]
        self.net = FieldNet(base=base).to(self.device).eval()
        self.net.load_state_dict(state)

    @staticmethod
    def available(weights: Path | str = WEIGHTS) -> bool:
        return Path(weights).exists()

    @torch.no_grad()
    def predict(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(turf_prob, line_prob)`` float32 maps the size of the frame."""
        h, w = frame_bgr.shape[:2]
        ph, pw = (-h) % 8, (-w) % 8
        img = (cv2.copyMakeBorder(frame_bgr, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
               if (ph or pw) else frame_bgr)
        x = to_tensor(img).to(self.device)
        p = torch.sigmoid(self.net(x))[0].float().cpu().numpy()
        return p[TURF, :h, :w], p[LINE, :h, :w]
