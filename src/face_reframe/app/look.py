"""Colour looks and automatic white balance / exposure, as one fast 3D
transform per video (64^3 offset table, like the skin correction)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

LOOKS = [("Natural", "none"), ("Warm", "warm"), ("Cool", "cool"), ("Cinematic (teal & orange)", "cine"),
         ("Pop (contrast & colour)", "pop"), ("Vintage", "vintage"), ("B&W", "bw"),
         ("Urban", "urban"), ("Teal & orange", "tealorange"), ("Neon cyberpunk", "neon"),
         ("Dreamy glow", "glow"), ("Y2K digicam", "y2k"), ("Faded film", "faded"),
         ("Midnight blue", "midnight"),
         ("Custom .cube file…", "cube")]


def parse_cube(path: str) -> Optional[np.ndarray]:
    """Parse an Adobe .cube 3D LUT -> float32 (n, n, n, 3) indexed [b, g, r]
    (red varies fastest in the file). DOMAIN_MIN/MAX rescale inputs;
    1D LUTs and malformed files return None."""
    size, rows = 0, []
    lo, hi = np.zeros(3, np.float32), np.ones(3, np.float32)
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith(("#", "TITLE", "LUT_1D", "COMMENT")):
                    continue
                if s.startswith("LUT_3D_SIZE"):
                    size = int(s.split()[-1])
                elif s.startswith("DOMAIN_MIN"):
                    lo = np.array([float(x) for x in s.split()[1:4]], np.float32)
                elif s.startswith("DOMAIN_MAX"):
                    hi = np.array([float(x) for x in s.split()[1:4]], np.float32)
                else:
                    try:
                        v = [float(x) for x in s.split()[:3]]
                    except ValueError:
                        continue
                    if len(v) == 3:
                        rows.append(v)
    except (OSError, ValueError):
        return None
    if not size or len(rows) < size ** 3:
        return None
    cube = np.array(rows[: size ** 3], np.float32).reshape(size, size, size, 3)
    return cube, lo, hi


def _cube_apply(x: np.ndarray, lut) -> np.ndarray:
    """Trilinear-interpolated .cube lookup on RGB float 0-1."""
    cube, lo, hi = lut
    n = cube.shape[0]
    c = np.clip((x - lo) / np.maximum(hi - lo, 1e-6), 0.0, 1.0) * (n - 1)
    lo_i = np.floor(c).astype(np.int32)
    hi_i = np.minimum(lo_i + 1, n - 1)
    f = c - lo_i                                    # (..., 3) per-channel fraction
    r0, g0, b0 = lo_i[..., 0], lo_i[..., 1], lo_i[..., 2]
    r1, g1, b1 = hi_i[..., 0], hi_i[..., 1], hi_i[..., 2]
    c00 = cube[b0, g0, r0] + (cube[b0, g0, r1] - cube[b0, g0, r0]) * f[..., 0:1]
    c01 = cube[b0, g1, r0] + (cube[b0, g1, r1] - cube[b0, g1, r0]) * f[..., 0:1]
    c10 = cube[b1, g0, r0] + (cube[b1, g0, r1] - cube[b1, g0, r0]) * f[..., 0:1]
    c11 = cube[b1, g1, r0] + (cube[b1, g1, r1] - cube[b1, g1, r0]) * f[..., 0:1]
    c0 = c00 + (c01 - c00) * f[..., 1:2]
    c1 = c10 + (c11 - c10) * f[..., 1:2]
    return c0 + (c1 - c0) * f[..., 2:3]


_CUBES: dict = {}


def load_cube(path: str):
    """mtime-cached .cube loader (preview and export share it)."""
    try:
        mt = os.path.getmtime(path)
    except OSError:
        return None
    ent = _CUBES.get(path)
    if ent is None or ent[0] != mt:
        _CUBES[path] = (mt, parse_cube(path))
    return _CUBES[path][1]


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
              grade: Optional[Grade] = None, cube=None) -> np.ndarray:
    """RGB float (0-1) -> graded RGB (same shape).

    Order: auto gains -> manual grade (absolute) -> preset look or .cube LUT
    mixed in at ``strength``. With ``grade=None`` this is exactly the old transform."""
    x = rgb.copy()
    if gains is not None:
        x = x * np.asarray(gains, np.float32)
    if grade is not None and grade.active:
        x = _grade_apply(x, grade)
    base = x
    if look == "cube":
        if cube is not None:
            x = _cube_apply(x, cube)
    elif look == "warm":
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
    elif look == "urban":
        y = _luma(x)[..., None]
        shadows = 1 - _smoothstep(y / 0.6)
        x = x + shadows * np.array([-0.02, 0.025, 0.055], np.float32)   # teal-leaning shadows
        x = _sat(x, 0.78)
        x = _contrast(x, 1.1)
        x = x * np.array([0.96, 1.0, 1.04], np.float32)                 # cold cast
    elif look == "tealorange":
        y = _luma(x)[..., None]
        shadows = 1 - _smoothstep(y / 0.55)
        highs = _smoothstep((y - 0.45) / 0.55)
        x = x + shadows * np.array([-0.06, 0.02, 0.075], np.float32) \
              + highs * np.array([0.075, 0.01, -0.08], np.float32)      # the punchy version of "cine"
        x = _contrast(x, 1.1)
        x = _sat(x, 1.05)
    elif look == "neon":
        y = _luma(x)[..., None]
        shadows = 1 - _smoothstep(y / 0.55)
        highs = _smoothstep((y - 0.45) / 0.55)
        x = x + shadows * np.array([0.05, -0.02, 0.1], np.float32) \
              + highs * np.array([-0.05, 0.04, 0.08], np.float32)        # magenta lows / cyan highs
        x = _sat(x, 1.18)
    elif look == "glow":
        y = _luma(x)[..., None]
        highs = _smoothstep((y - 0.4) / 0.6)
        x = x + highs * np.float32(0.09)                                # pushed whites, bloom feel
        x = _contrast(x, 0.92)
        x = _sat(x, 0.95)
        x = x * np.array([1.02, 0.99, 1.02], np.float32)                # soft warm-magenta
    elif look == "y2k":
        x = x * np.array([1.0, 1.03, 0.97], np.float32)                 # digicam green-cyan cast
        x = _contrast(x, 0.92) + 0.04                                   # washed-out blacks
        x = _sat(x, 0.9)
    elif look == "faded":
        x = 0.075 + x * 0.86                                            # matte: lifted blacks, soft whites
        x = _sat(x, 0.72)
        x = x * np.array([1.03, 1.0, 0.96], np.float32)
    elif look == "midnight":
        x = _sat(x, 0.55)
        y = _luma(x)[..., None]
        shadows = 1 - _smoothstep(y / 0.5)
        x = x + shadows * np.array([-0.03, 0.01, 0.09], np.float32)     # deep navy shadows
        x = _contrast(x, 1.18)
    x = base + (x - base) * strength
    return np.clip(x, 0, 1)


@dataclass
class Look:
    look: str = "none"
    strength: float = 1.0
    gains: Optional[Tuple[float, float, float]] = None
    """Auto white balance / exposure gains (r, g, b), or None."""
    grade: Optional[Grade] = None
    cube: Optional[tuple] = None
    """Parsed .cube LUT (cube, domain_lo, domain_hi) — used when look="cube"."""

    @property
    def active(self) -> bool:
        g = self.grade
        return ((self.look != "none" and self.look != "cube" and self.strength > 0)
                or (self.look == "cube" and self.cube is not None and self.strength > 0)
                or self.gains is not None
                or (g is not None and (g.active or g.vignette > 0)))


_BITS = 6


def _table(lk: Look) -> np.ndarray:
    n = 1 << _BITS
    step = 256 // n
    g = (np.arange(n, dtype=np.float32) * step + step / 2.0) / 255.0
    bb, gg, rr = np.meshgrid(g, g, g, indexing="ij")
    rgb = np.stack([rr, gg, bb], axis=-1).reshape(-1, 3)
    out = transform(rgb, lk.look, lk.gains, lk.strength, lk.grade, lk.cube)
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
