"""Stabilisation of shaky footage, applied through the crop we already make.

Global camera motion between frames is measured on a small greyscale copy
(features + optical flow + similarity fit), the trajectory is smoothed per
shot, and the difference becomes a per-frame correction: a crop offset (free,
since a vertical crop has room to move sideways) plus a small rotation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass
class Stabilization:
    dx: np.ndarray    # source pixels to move the crop by
    dy: np.ndarray
    rot: np.ndarray   # degrees to rotate the output by

    def scaled(self, amount: float) -> "Stabilization":
        return Stabilization(self.dx * amount, self.dy * amount, self.rot * amount)


def _gauss(v: np.ndarray, sigma: float) -> np.ndarray:
    if sigma < 0.5 or len(v) < 3:
        return v.copy()
    r = int(3 * sigma)
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    pad = np.pad(v, r, mode="edge")
    return np.convolve(pad, k, mode="valid")


def estimate(path: str, cuts: Sequence[int] = (), smooth_s: float = 1.0,
             progress: Optional[Callable[[float], None]] = None) -> Optional[Stabilization]:
    import cv2
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1
    scale = min(1.0, 480.0 / W)
    cutset = set(int(c) for c in cuts)
    moves = []      # per-frame (dx, dy, da) relative to the previous frame, in small-frame px
    prev = None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
        m = (0.0, 0.0, 0.0)
        if prev is not None and i not in cutset:
            pts = cv2.goodFeaturesToTrack(prev, maxCorners=200, qualityLevel=0.01, minDistance=15, blockSize=3)
            if pts is not None and len(pts) >= 10:
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, pts, None)
                good = st.reshape(-1) == 1
                if good.sum() >= 10:
                    M, inl = cv2.estimateAffinePartial2D(pts[good], nxt[good], method=cv2.RANSAC,
                                                         ransacReprojThreshold=2.0)
                    if M is not None and inl is not None and inl.sum() >= 8:
                        m = (float(M[0, 2]), float(M[1, 2]), float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
        moves.append(m)
        prev = g
        i += 1
        if progress and total and i % 25 == 0:
            progress(min(1.0, i / total))
    cap.release()
    n = len(moves)
    if n == 0:
        return None
    mv = np.asarray(moves)
    # trajectory per shot (a hard cut restarts it), smoothed, correction = smooth - raw
    bounds = [0] + sorted(c for c in cutset if 0 < c < n) + [n]
    corr = np.zeros((n, 3))
    sigma = smooth_s * fps
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < 2:
            continue
        traj = np.cumsum(mv[a:b], axis=0)
        traj[0] = 0
        sm = np.stack([_gauss(traj[:, k], sigma) for k in range(3)], axis=1)
        corr[a:b] = sm - traj
    # move the crop with the camera and rotate the output against it (measured:
    # a frame turned by +a needs the output turned by -a, i.e. -correction)
    return Stabilization(-corr[:, 0] / scale, -corr[:, 1] / scale, -corr[:, 2])
