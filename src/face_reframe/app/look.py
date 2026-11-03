"""Colour looks and automatic white balance / exposure, as one fast 3D
transform per video (64^3 offset table, like the skin correction)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOOKS = [("Natural", "none"), ("Warm", "warm"), ("Cool", "cool"), ("Cinematic (teal & orange)", "cine"),
         ("Pop (contrast & colour)", "pop"), ("Vintage", "vintage"), ("B&W", "bw")]


def _smoothstep(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


def _luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def _sat(rgb, k):
    y = _luma(rgb)[..., None]
    return y + (rgb - y) * k


def _contrast(rgb, k, pivot=0.45):
    return pivot + (rgb - pivot) * k


@dataclass
class Grade:
    """Hand grading on top of the preset look. All per-pixel ops so it folds
    into the same fast transform as the look — except ``vignette``, a spatial
    multiply applied after the LUT."""
    temp: float = 0.0        # -1 cool … +1 warm
    tint: float = 0.0        # -1 green … +1 magenta
    exposure: float = 0.0    # EV stops, -1 … +1
    contrast: float = 1.0    # 0 … 2, pivot 0.45
    saturation: float = 1.0  # 0 … 2
    shadows: float = 0.0     # -1 crush … +1 lift
    highlights: float = 0.0  # -1 tame … +1 push
    vignette: float = 0.0    # 0 … 1 edge darkening

    @property
    def active(self) -> bool:
        """LUT-bakeable ops differ from neutral (vignette excluded: post-pass)."""
        return (self.temp != 0 or self.tint != 0 or self.exposure != 0
                or self.contrast != 1 or self.saturation != 1
                or self.shadows != 0 or self.highlights != 0)


def _grade_apply(x: np.ndarray, g: Grade) -> np.ndarray:
    if g.temp:
        x = x * np.array([1 + 0.09 * g.temp, 1 + 0.02 * g.temp, 1 - 0.09 * g.temp], np.float32)
    if g.tint:
        x = x * np.array([1 + 0.05 * g.tint, 1 - 0.08 * g.tint, 1 + 0.05 * g.tint], np.float32)
    if g.exposure:
        x = x * np.float32(2.0 ** g.exposure)
    if g.contrast != 1:
        x = _contrast(x, g.contrast)
    if g.saturation != 1:
        x = _sat(x, g.saturation)
    if g.shadows or g.highlights:
        y = _luma(x)[..., None]
        x = x + (1 - _smoothstep(y / 0.55)) * np.float32(0.18 * g.shadows) \
              + _smoothstep((y - 0.45) / 0.55) * np.float32(0.15 * g.highlights)
    return x


def transform(rgb: np.ndarray, look: str, gains: Optional[Tuple[float, float, float]], strength: float,
              grade: Optional[Grade] = None) -> np.ndarray:
    """RGB float (0-1) -> graded RGB (same shape).

    Order: auto gains -> manual grade (absolute) -> preset look mixed in at
    ``strength``. With ``grade=None`` this is exactly the old transform."""
    x = rgb.copy()
    if gains is not None:
        x = x * np.asarray(gains, np.float32)
    if grade is not None and grade.active:
        x = _grade_apply(x, grade)
    base = x
    if look == "warm":
        x = x * np.array([1.06, 1.0, 0.92], np.float32)
        x = _sat(x, 1.05)
    elif look == "cool":
        x = x * np.array([0.93, 1.0, 1.07], np.float32)
    elif look == "cine":
        y = _luma(x)[..., None]
        shadows = 1 - _smoothstep(y / 0.55)
        highs = _smoothstep((y - 0.45) / 0.55)
        x = x + shadows * np.array([-0.03, 0.015, 0.045], np.float32) + highs * np.array([0.05, 0.015, -0.04], np.float32)
        x = _contrast(x, 1.08)
        x = _sat(x, 0.92)
    elif look == "pop":
        x = _contrast(x, 1.15)
        x = _sat(x, 1.25)
    elif look == "vintage":
        x = 0.06 + x * 0.88                     # lifted blacks, softer whites
        x = _sat(x, 0.78) * np.array([1.04, 1.0, 0.9], np.float32)
    elif look == "bw":
        y = _luma(x)[..., None]
        x = np.repeat(_contrast(y, 1.12), 3, axis=-1)
    x = base + (x - base) * strength
    return np.clip(x, 0, 1)


@dataclass
class Look:
    look: str = "none"
    strength: float = 1.0
    gains: Optional[Tuple[float, float, float]] = None
    """Auto white balance / exposure gains (r, g, b), or None."""
    grade: Optional[Grade] = None

    @property
    def active(self) -> bool:
        g = self.grade
        return ((self.look != "none" and self.strength > 0) or self.gains is not None
                or (g is not None and (g.active or g.vignette > 0)))


_BITS = 6


def _table(lk: Look) -> np.ndarray:
    n = 1 << _BITS
    step = 256 // n
    g = (np.arange(n, dtype=np.float32) * step + step / 2.0) / 255.0
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    out = transform(rgb, lk.look, lk.gains, lk.strength, lk.grade)
    d = (out - rgb) * 255.0
    return np.round(d[:, ::-1]).astype(np.int16)   # BGR offsets


def _lut(frame: np.ndarray, lk: Look) -> np.ndarray:
    t = getattr(lk, "_t", None)
    if t is None:
        t = _table(lk)
        object.__setattr__(lk, "_t", t)
    sh = 8 - _BITS
    q = (frame >> sh).astype(np.int32)
    idx = (q[..., 0] << (2 * _BITS)) | (q[..., 1] << _BITS) | q[..., 2]
    return np.clip(frame.astype(np.int16) + t[idx], 0, 255).astype(np.uint8)


def _vignette(frame: np.ndarray, v: float, lk: Look) -> np.ndarray:
    """Multiply by a radial falloff, strongest towards the corners. The mask
    depends only on frame size and strength, cached on the Look."""
    key = (frame.shape[0], frame.shape[1], round(v, 3))
    m = getattr(lk, "_vm", None)
    if m is None or getattr(lk, "_vm_key", None) != key:
        h, w = key[0], key[1]
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.sqrt(((xx / w - 0.5) * (w / h)) ** 2 + (yy / h - 0.5) ** 2) * 2
        m = 1.0 - v * np.clip((r - 0.62) / 0.7, 0, 1) ** 1.5
        object.__setattr__(lk, "_vm", m[..., None].astype(np.float32))
        object.__setattr__(lk, "_vm_key", key)
        m = lk._vm
    return np.clip(frame.astype(np.float32) * m, 0, 255).astype(np.uint8)


def apply(frame: np.ndarray, lk: Optional[Look]) -> np.ndarray:
    if lk is None or not lk.active:
        return frame
    g = lk.grade
    lut_on = ((lk.look != "none" and lk.strength > 0) or lk.gains is not None
            or (g is not None and g.active))
    out = frame
    if lut_on:
        if lk.look == "bw":
            # exact grey (an offset table leaves a faint colour residue)
            import cv2
            base = _lut(frame, Look("none", 1.0, lk.gains, g))
            gy = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
            gy = np.clip(0.45 * 255 + (gy - 0.45 * 255) * 1.12, 0, 255)
            bw = np.repeat(gy[..., None], 3, axis=2)
            out = np.clip(base + (bw - base) * lk.strength, 0, 255).astype(np.uint8)
        else:
            out = _lut(frame, lk)
    if g is not None and g.vignette > 0:
        out = _vignette(out, g.vignette, lk)
    return out


def auto_gains(path: str, start: int = 0, count: Optional[int] = None):
    """Gray-world white balance + exposure (the Normalize Color measurement)."""
    import cv2
    from ..color import measure_gains
    if count is None:
        cap = cv2.VideoCapture(path)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        cap.release()
    g = measure_gains(path, start, count, samples=12)
    if g is None:
        return None
    # soften: full gray-world can over-correct deliberately warm scenes
    return tuple(1.0 + 0.7 * (x - 1.0) for x in g)
