"""Stage 5d - read the HUD's down, distance and field position.

Blitz prints the ball's absolute position ("ON THE 23") in a fixed bitmap font,
which is the absolute reference the lattice lacks: the lattice measures distance
along the field beautifully but its origin is arbitrary per frame.

Nothing here is hand-labelled.  Glyphs are clustered by appearance, and the
*text structure* names them:

* every down-and-distance line shares a three-glyph block - that is ``AND``;
* the three glyphs before it are the ordinal, giving ``1ST``/``2ND``/``3RD``/
  ``4TH`` and hence the digits 1-4;
* short lines share a five-glyph prefix - that is ``ON THE``;
* trailing glyphs are digits, and their *values* come from geometry: where the
  scrimmage and first-down lines are both visible, the yard gap between them is
  measured directly (validated to ~0.1 yd), and the trailing glyphs must spell
  that number.

Digit coverage is the limiting factor.  Geometry can only teach digits that
actually occur in the distance field, and because Blitz needs 30 yards for a
first down, "AND 4" through "AND 9" are rare - so glyphs are pooled across
videos to see enough of them.
"""
from __future__ import annotations

import collections
from dataclasses import dataclass

import cv2
import numpy as np

GLYPH_W, GLYPH_H = 20, 28


# --------------------------------------------------------------------------- #
# glyph extraction
# --------------------------------------------------------------------------- #
def yellow_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H, S, V = hsv[:, :, 0].astype(int), hsv[:, :, 1], hsv[:, :, 2]
    return ((H >= 18) & (H <= 38) & (S > 110) & (V > 150)).astype(np.uint8)


def normalise_glyph(g: np.ndarray) -> np.ndarray:
    """Scale to a fixed box, centre by mass, and blur.

    Resizing the bounding box to a fixed size also removes the anisotropic
    stretch of the one upload that is not 4:3, so glyphs from every source are
    directly comparable.  The blur absorbs sub-pixel misalignment - without it
    the same character fragments into many clusters.
    """
    g = cv2.resize(g, (GLYPH_W, GLYPH_H),
                   interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    M = cv2.moments(g)
    if M["m00"] > 1e-6:
        T = np.float32([[1, 0, GLYPH_W / 2 - M["m10"] / M["m00"]],
                        [0, 1, GLYPH_H / 2 - M["m01"] / M["m00"]]])
        g = cv2.warpAffine(g, T, (GLYPH_W, GLYPH_H))
    return cv2.GaussianBlur(g, (5, 5), 1.2)


def text_lines(frame: np.ndarray, top: float = 0.62, min_glyphs: int = 4):
    """Rows of large yellow glyphs in the lower banner.

    The height filter matters: play-select screens fill the same area with small
    yellow X markers, which outnumber the banner text several to one.
    """
    h, w = frame.shape[:2]
    m = yellow_mask(frame)
    m[:int(h * top)] = 0
    n, lbl, st, _ = cv2.connectedComponentsWithStats(m, 8)
    cand = [(st[j, 0], st[j, 1], st[j, 2], st[j, 3], j) for j in range(1, n)
            if st[j, 4] >= 90 and 20 <= st[j, 3] <= 80 and 5 <= st[j, 2] <= 70
            and st[j, 2] / max(st[j, 3], 1) <= 2.2]
    if len(cand) < min_glyphs:
        return []
    cand.sort(key=lambda c: c[1] + c[3] / 2)
    lines, cur = [], [cand[0]]
    for c in cand[1:]:
        if abs((c[1] + c[3] / 2) - (cur[-1][1] + cur[-1][3] / 2)) <= 0.55 * cur[-1][3]:
            cur.append(c)
        else:
            lines.append(cur)
            cur = [c]
    lines.append(cur)

    out = []
    for ln in lines:
        if len(ln) < min_glyphs:
            continue
        hh = np.array([c[3] for c in ln], float)
        if hh.std() / max(hh.mean(), 1) > 0.30:
            continue
        ln.sort(key=lambda c: c[0])
        out.append([normalise_glyph((lbl[y:y + h_, x:x + w_] == j).astype(np.uint8) * 255)
                    for (x, y, w_, h_, j) in ln])
    return out


# --------------------------------------------------------------------------- #
# glyph book
# --------------------------------------------------------------------------- #
@dataclass
class GlyphBook:
    labels: np.ndarray           # cluster id per glyph
    digit: dict                  # cluster id -> character
    and_block: tuple
    onthe_prefix: tuple
    n_clusters: int


def cluster_glyphs(glyphs, threshold: float = 0.25) -> tuple[np.ndarray, int]:
    from sklearn.cluster import AgglomerativeClustering
    X = np.array([g.reshape(-1) for g in glyphs], np.float32)
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-9)
    lab = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold,
                                  linkage="average").fit(X).labels_
    return lab, int(lab.max()) + 1


def build_book(seqs, geo: dict, labels: np.ndarray,
               min_votes: int = 3, purity: float = 0.6) -> GlyphBook:
    """Decode the alphabet from text structure plus geometric distances."""
    mid = collections.Counter(tuple(s[3:6]) for _, s in seqs if len(s) >= 8)
    and_block = mid.most_common(1)[0][0] if mid else ()

    votes = collections.defaultdict(collections.Counter)

    # geometry: trailing glyphs of a down line spell the measured distance
    for fi, s in seqs:
        if len(s) < 8 or tuple(s[3:6]) != and_block or fi not in geo:
            continue
        val = geo[fi]
        if abs(val - round(val)) > 0.45 or not (1 <= round(val) <= 40):
            continue
        txt = str(int(round(val)))
        tail = s[6:6 + len(txt)]
        if len(tail) != len(txt):
            continue
        for cl, ch in zip(tail, txt):
            votes[cl][ch] += 1

    digit = {}
    for cl, cnt in votes.items():
        ch, n = cnt.most_common(1)[0]
        if n >= min_votes and n / sum(cnt.values()) >= purity:
            digit[cl] = ch

    pre = collections.Counter(tuple(s[:5]) for _, s in seqs if 6 <= len(s) <= 7)
    onthe = pre.most_common(1)[0][0] if pre else ()
    return GlyphBook(labels, digit, and_block, onthe, int(labels.max()) + 1)


def read_positions(seqs, book: GlyphBook):
    """Field position readings, ``(frame_index, yard)``, plus how many failed."""
    out, unread = [], 0
    for fi, s in seqs:
        if not (6 <= len(s) <= 7) or tuple(s[:5]) != book.onthe_prefix:
            continue
        tail = s[5:]
        if any(t not in book.digit for t in tail):
            unread += 1
            continue
        try:
            out.append((fi, int("".join(book.digit[t] for t in tail))))
        except ValueError:
            unread += 1
    return out, unread
