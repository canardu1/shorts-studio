"""Per-frame zoom factors for the punch-in effect.

The classic shorts retention trick: alternate between the normal frame and a
slightly zoomed-in frame every few seconds, with a quick ease between them.
Zooming is around the frame centre, so it composes fine on top of the face
reframe (which keeps the face centred).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class PunchParams:
    interval: float = 3.5
    """Seconds between zoom alternations."""
    zoom: float = 1.18
    """Zoom factor of the "in" state."""
    ease: float = 0.18
    """Seconds the in/out transition takes (smoothstep)."""
    start_zoomed: bool = False
    """Start on the zoomed state instead of the plain one."""


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _ease(target: np.ndarray, ease_frames: int) -> np.ndarray:
    """Step function -> smoothed curve: transitions take ``ease_frames``."""
    n_frames = len(target)
    out = np.empty(n_frames)
    out[0] = target[0]
    i = 1
    while i < n_frames:
        if target[i] == target[i - 1]:
            out[i] = out[i - 1]
            i += 1
            continue
        j = i
        while j < n_frames and target[j] == target[i]:
            j += 1
        k = min(ease_frames, j - i)
        t = _smoothstep(np.arange(k) / float(k))
        out[i:i + k] = out[i - 1] + (target[i] - out[i - 1]) * t
        out[i + k:j] = target[i]
        i = j
    return out


def punch_curve(n_frames: int, fps: float, params: PunchParams = None) -> np.ndarray:
    """Zoom factor for every frame: 1.0 <-> params.zoom, alternating."""
    p = params or PunchParams()
    if n_frames <= 0:
        return np.zeros(0)
    zoomed = (np.arange(n_frames) / fps / max(p.interval, 0.1)).astype(int) % 2 == 1
    if p.start_zoomed:
        zoomed = ~zoomed
    target = np.where(zoomed, p.zoom, 1.0).astype(float)
    return _ease(target, max(1, int(round(p.ease * fps))))


def punch_curve_synced(speech_starts_s, n_frames: int, fps: float,
                       params: PunchParams = None) -> np.ndarray:
    """Zoom toggles at each speech-segment start (emphasis on phrases)."""
    p = params or PunchParams()
    if n_frames <= 0:
        return np.zeros(0)
    zoomed = np.full(n_frames, p.start_zoomed, dtype=bool)
    for s in speech_starts_s:
        f = int(round(s * fps))
        if 0 <= f < n_frames:
            zoomed[f:] = ~zoomed[f]
    target = np.where(zoomed, p.zoom, 1.0).astype(float)
    return _ease(target, max(1, int(round(p.ease * fps))))


def curve_keys(curve: np.ndarray, step: int = 3) -> List[Tuple[int, float]]:
    """(frame, value) keyframes approximating the curve, every ``step`` frames."""
    if len(curve) == 0:
        return []
    frames = list(range(0, len(curve), max(1, step)))
    if frames[-1] != len(curve) - 1:
        frames.append(len(curve) - 1)
    return [(f, float(curve[f])) for f in frames]
