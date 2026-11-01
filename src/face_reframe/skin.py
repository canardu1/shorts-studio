"""Skin-tone normalisation: a targeted "skin secondary", done automatically.

1. *Measure* — sample cheeks and forehead (from YuNet landmarks, so eyes,
   lips and hair are excluded) on a few dozen frames and take the median
   colour in CIELAB.
2. *Compare* with the skin tone line: healthy skin of any complexion sits on
   nearly the same Lab hue (~55°, the vectorscope's ~123° line). Cameras that
   render skin "lobster red" push it towards red/magenta and over-saturate it.
3. *Correct* — rotate the hue towards the line and trim excess chroma, only
   on pixels whose colour is close to the measured skin (soft Gaussian mask
   in the a/b plane, faded out in deep shadows and highlights). Red clothes,
   walls and logos sit far from skin in a/b and stay as they are.

The correction is computed once per video (stable, no flicker) and can be
saved as a .cube LUT for Resolve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

TARGET_HUE = 55.0
"""Lab hue angle (degrees) of the skin tone line."""
NATURAL_HUE = (50.0, 64.0)
"""Hue range already accepted as natural: only skin outside it is rotated
(to the nearest edge), so well-exposed skin is left alone."""
TARGET_CHROMA = (14.0, 24.0)
"""Chroma range natural skin is pulled into."""


@dataclass
class SkinCorrection:
    a: float
    b: float
    L: float
    sigma: float
    """Width of the skin mask in the a/b plane."""
    rot: float
    """Hue rotation to apply, degrees (positive = towards yellow)."""
    scale: float
    """Chroma scale on skin."""
    strength: float = 0.7
    samples: int = 0

    @property
    def hue(self) -> float:
        return math.degrees(math.atan2(self.b, self.a))

    @property
    def chroma(self) -> float:
        return math.hypot(self.a, self.b)

    @property
    def needed(self) -> bool:
        return abs(self.rot) > 0.5 or abs(self.scale - 1.0) > 0.02

    def describe(self) -> str:
        if not self.needed:
            return ("pelle misurata: tinta %.0f°, saturazione %.0f — già naturale, nessuna correzione"
                    % (self.hue, self.chroma))
        return ("pelle misurata: tinta %.0f° → %.0f° (naturale %.0f–%.0f°), saturazione %.0f → %.0f"
                % (self.hue, self.hue + self.rot, NATURAL_HUE[0], NATURAL_HUE[1],
                   self.chroma, self.chroma * self.scale))

    def with_strength(self, s: float) -> "SkinCorrection":
        return SkinCorrection(self.a, self.b, self.L, self.sigma, self.rot, self.scale, s, self.samples)


def _patches(face):
    """Cheek and forehead sample circles (cx, cy, r) from the 5 landmarks."""
    rex, rey, lex, ley, nx, ny, rmx, rmy, lmx, lmy = face.landmarks
    ed = math.hypot(lex - rex, ley - rey)
    if ed < 8:
        return []
    r = 0.14 * ed
    # cheeks: below each eye, level with the nose, a bit towards the outside
    c1 = (0.55 * rex + 0.45 * rmx - 0.12 * ed, 0.5 * (rey + rmy))
    c2 = (0.55 * lex + 0.45 * lmx + 0.12 * ed, 0.5 * (ley + lmy))
    fh = (0.5 * (rex + lex), 0.5 * (rey + ley) - 0.45 * ed)
    return [(c1[0], c1[1], r), (c2[0], c2[1], r), (fh[0], fh[1], r * 0.9)]


def _lab(bgr: np.ndarray) -> np.ndarray:
    import cv2
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def sample_pixels(frame: np.ndarray, faces) -> np.ndarray:
    """Lab pixels (N, 3) from the skin patches of ``faces`` in ``frame``."""
    h, w = frame.shape[:2]
    out = []
    for f in faces:
        if f.landmarks is None:
            continue
        for cx, cy, r in _patches(f):
            x0, x1 = int(max(0, cx - r)), int(min(w, cx + r))
            y0, y1 = int(max(0, cy - r)), int(min(h, cy + r))
            if x1 - x0 < 3 or y1 - y0 < 3:
                continue
            lab = _lab(frame[y0:y1, x0:x1]).reshape(-1, 3)
            out.append(lab)
    if not out:
        return np.zeros((0, 3), np.float32)
    px = np.concatenate(out)
    # drop specular highlights, deep shadows and grey (beard, glasses)
    keep = (px[:, 0] > 20) & (px[:, 0] < 92) & (np.hypot(px[:, 1], px[:, 2]) > 6)
    return px[keep]


def measure(frames_and_faces, strength: float = 0.7) -> Optional[SkinCorrection]:
    """Correction from an iterable of ``(frame_bgr, faces)``; None without skin."""
    pix: List[np.ndarray] = []
    n = 0
    for frame, faces in frames_and_faces:
        p = sample_pixels(frame, faces)
        if len(p):
            pix.append(p)
            n += 1
    if not pix:
        return None
    px = np.concatenate(pix)
    if len(px) < 200:
        return None
    L, a, b = (float(np.median(px[:, k])) for k in range(3))
    d = np.hypot(px[:, 1] - a, px[:, 2] - b)
    sigma = float(np.clip(2.2 * np.percentile(d, 68), 7.0, 16.0))
    hue = math.degrees(math.atan2(b, a))
    chroma = math.hypot(a, b)
    rot = float(np.clip(np.clip(hue, *NATURAL_HUE) - hue, -25.0, 25.0))
    target_c = float(np.clip(chroma, *TARGET_CHROMA))
    scale = target_c / max(chroma, 1e-6)
    return SkinCorrection(a, b, L, sigma, rot, scale, strength, n)


def measure_video(path: str, analysis, max_frames: int = 30, strength: float = 0.7) -> Optional[SkinCorrection]:
    """Sample up to ``max_frames`` analysed frames that have faces with landmarks."""
    import cv2
    cand = [s for s in analysis.samples if any(f.landmarks is not None and f.h >= 40 for f in s.faces)]
    if not cand:
        return None
    pick = [cand[int(i)] for i in np.linspace(0, len(cand) - 1, min(max_frames, len(cand)))]
    cap = cv2.VideoCapture(path)

    def gen():
        for s in pick:
            cap.set(cv2.CAP_PROP_POS_FRAMES, analysis.start_frame + s.frame)
            ok, frame = cap.read()
            if ok:
                yield frame, [f for f in s.faces if f.landmarks is not None]
    try:
        return measure(gen(), strength)
    finally:
        cap.release()


def _correct_ab(a: np.ndarray, b: np.ndarray, L: np.ndarray, c: SkinCorrection):
    d2 = ((a - c.a) ** 2 + (b - c.b) ** 2) / (c.sigma ** 2)
    w = np.exp(-0.5 * d2)
    # fade out in deep shadows / near white, where "skin-coloured" is noise
    w *= np.clip((L - 8.0) / 12.0, 0, 1) * np.clip((97.0 - L) / 8.0, 0, 1)
    w *= c.strength
    th = math.radians(c.rot)
    cos, sin = math.cos(th) * c.scale, math.sin(th) * c.scale
    na = a * cos - b * sin
    nb = a * sin + b * cos
    return a + w * (na - a), b + w * (nb - b)


def apply_exact(frame: np.ndarray, c: Optional[SkinCorrection]) -> np.ndarray:
    """Corrected copy of a BGR uint8 frame, computed per pixel (slow, reference)."""
    import cv2
    if c is None or c.strength <= 0:
        return frame
    lab = _lab(frame)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    lab[..., 1], lab[..., 2] = _correct_ab(a, b, L, c)
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


_LUT_BITS = 6  # 64 levels per channel


def _delta_lut(c: SkinCorrection) -> np.ndarray:
    """Per-colour BGR offset on a 64^3 grid (the correction is smooth, so the
    offset barely changes within a cell; adding it to the exact pixel avoids
    the banding a plain quantised LUT would give)."""
    n = 1 << _LUT_BITS
    step = 256 // n
    g = (np.arange(n, dtype=np.uint8) * step + step // 2)
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")
    grid = np.stack([bb, gg, rr], axis=-1).reshape(1, -1, 3)
    fixed = apply_exact(grid, c)
    return (fixed.astype(np.int16) - grid.astype(np.int16)).reshape(-1, 3)


def apply(frame: np.ndarray, c: Optional[SkinCorrection]) -> np.ndarray:
    """Corrected copy of a BGR uint8 frame (fast: ~5x quicker than per-pixel Lab)."""
    if c is None or c.strength <= 0 or not c.needed:
        return frame
    lut = getattr(c, "_lut", None)
    if lut is None:
        lut = _delta_lut(c)
        object.__setattr__(c, "_lut", lut)
    sh = 8 - _LUT_BITS
    q = (frame >> sh).astype(np.int32)
    idx = (q[..., 0] << (2 * _LUT_BITS)) | (q[..., 1] << _LUT_BITS) | q[..., 2]
    out = frame.astype(np.int16) + lut[idx]
    return np.clip(out, 0, 255).astype(np.uint8)


def write_cube(c: SkinCorrection, path: str, size: int = 33) -> str:
    """The same correction as a 3D LUT (.cube, Resolve-compatible)."""
    import cv2
    g = np.linspace(0.0, 1.0, size, dtype=np.float32)
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")          # red varies fastest
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 1, 3)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    lab[..., 1], lab[..., 2] = _correct_ab(lab[..., 1], lab[..., 2], lab[..., 0], c)
    out = np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1).reshape(-1, 3)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write('TITLE "Shorts Studio skin"\nLUT_3D_SIZE %d\n' % size)
        for r, gr, bl in out:
            fh.write("%.6f %.6f %.6f\n" % (r, gr, bl))
    return path
