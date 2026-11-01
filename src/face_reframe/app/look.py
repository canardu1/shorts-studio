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


def transform(rgb: np.ndarray, look: str, gains: Optional[Tuple[float, float, float]], strength: float) -> np.ndarray:
    """RGB float (0-1) -> graded RGB (same shape)."""
    x = rgb.copy()
    if gains is not None:
        x = x * np.asarray(gains, np.float32)
    base = x.copy()
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

    @property
    def active(self) -> bool:
        return (self.look != "none" and self.strength > 0) or self.gains is not None


_BITS = 6


def _table(lk: Look) -> np.ndarray:
    n = 1 << _BITS
    step = 256 // n
    g = (np.arange(n, dtype=np.float32) * step + step / 2.0) / 255.0
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    out = transform(rgb, lk.look, lk.gains, lk.strength)
    d = (out - rgb) * 255.0
    return np.round(d[:, ::-1]).astype(np.int16)   # BGR offsets


def apply(frame: np.ndarray, lk: Optional[Look]) -> np.ndarray:
    if lk is None or not lk.active:
        return frame
    if lk.look == "bw":
        # exact grey (an offset table leaves a faint colour residue)
        import cv2
        base = apply(frame, Look("none", 1.0, lk.gains)) if lk.gains is not None else frame
        g = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY).astype(np.float32)
        g = np.clip(0.45 * 255 + (g - 0.45 * 255) * 1.12, 0, 255)
        bw = np.repeat(g[..., None], 3, axis=2)
        return np.clip(base + (bw - base) * lk.strength, 0, 255).astype(np.uint8)
    t = getattr(lk, "_t", None)
    if t is None:
        t = _table(lk)
        object.__setattr__(lk, "_t", t)
    sh = 8 - _BITS
    q = (frame >> sh).astype(np.int32)
    idx = (q[..., 0] << (2 * _BITS)) | (q[..., 1] << _BITS) | q[..., 2]
    return np.clip(frame.astype(np.int16) + t[idx], 0, 255).astype(np.uint8)


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
