"""Auto white-balance / exposure: sample frames, compute channel gains,
bake them into a small .cube LUT that Resolve applies non-destructively."""

from __future__ import annotations

import os
from typing import Optional, Tuple

import cv2
import numpy as np

TARGET_LUMA = 0.42
"""Mean luminance (0-1) aimed for — broadcast mid-grey, safe for faces."""
GAIN_LIMIT = (0.4, 2.5)


def measure_gains(
    path: str, start: int, count: int, samples: int = 9
) -> Optional[Tuple[float, float, float]]:
    """(r, g, b) gains: gray-world white balance + exposure to TARGET_LUMA."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video: %s" % path)
    picks = np.linspace(start, start + max(0, count - 1), max(1, samples)).astype(int)
    means, lumas = [], []
    for f in picks:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
        ok, frame = cap.read()
        if not ok:
            continue
        small = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        b, g, r = cv2.split(small)
        means.append((float(r.mean()), float(g.mean()), float(b.mean())))
        lumas.append(float((0.2126 * r + 0.7152 * g + 0.0722 * b).mean()))
    cap.release()
    if not means:
        return None
    r_m, g_m, b_m = np.median(np.asarray(means), axis=0)
    luma = float(np.median(lumas))
    if luma <= 0.0:
        return None
    expo = TARGET_LUMA / luma
    eps = 1e-4
    gains = (
        expo * (g_m + eps) / (r_m + eps),
        expo,
        expo * (g_m + eps) / (b_m + eps),
    )
    lo, hi = GAIN_LIMIT
    return tuple(float(np.clip(g, lo, hi)) for g in gains)


def write_lut(gains: Tuple[float, float, float], out_path: str, size: int = 17) -> str:
    """Write a 3D .cube LUT that applies per-channel gains."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    gr, gg, gb = gains
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write('TITLE "face_reframe auto color"\n')
        fh.write("LUT_3D_SIZE %d\n" % size)
        fh.write("DOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
        n = size - 1
        for b in range(size):
            for g in range(size):
                for r in range(size):
                    fh.write("%.6f %.6f %.6f\n" % (r / n * gr, g / n * gg, b / n * gb))
    return out_path
