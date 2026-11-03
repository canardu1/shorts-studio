"""Salient-object localisation with U2-Net-p — the same 4.6 MB model the
thumbnail cut-out already downloads, reused here as a saliency map.

When a shot has no face (product demos, pets, cars, B-roll) the reframer can
follow the salient blob instead of staring at the frame centre.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from . import models
from .i18n import tr


def available() -> bool:
    return models.available("u2netp")


class Saliency:
    """``frame -> (cx, cy, w, h)`` of the most salient blob in source pixels,
    or None when nothing stands out."""

    def __init__(self, min_mass: float = 0.015):
        self.sess = models.session("u2netp")
        self.inp = self.sess.get_inputs()[0].name
        self.min_mass = min_mass

    def map(self, img: np.ndarray) -> np.ndarray:
        """Float 0..1 saliency at the input resolution (320x320)."""
        import cv2
        x = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)[..., ::-1].astype(np.float32) / 255.0
        x = (x - (0.485, 0.456, 0.406)) / (0.229, 0.224, 0.225)
        out = self.sess.run(None, {self.inp: x.transpose(2, 0, 1)[None].astype(np.float32)})[0][0, 0]
        out = (out - out.min()) / max(1e-6, float(out.max() - out.min()))
        return out.astype(np.float32)

    def __call__(self, img: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        import cv2
        m = self.map(img)
        hard = (m > 0.5).astype(np.uint8)
        n, lab, stats, cents = cv2.connectedComponentsWithStats(hard, 8)
        if n <= 1:
            return None
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        area = float(stats[k, cv2.CC_STAT_AREA]) / hard.size
        if area < self.min_mass:
            return None
        sx, sy = img.shape[1] / 320.0, img.shape[0] / 320.0
        bw, bh = stats[k, cv2.CC_STAT_WIDTH] * sx, stats[k, cv2.CC_STAT_HEIGHT] * sy
        return float(cents[k][0] * sx), float(cents[k][1] * sy), float(bw), float(bh)


def build(log=print, progress=None) -> Optional[Saliency]:
    """Download U2-Net-p when needed and return the detector, else None."""
    if models.ensure("u2netp", progress, log) is None:
        return None
    try:
        return Saliency()
    except Exception as exc:
        log(tr("Salient-object detection unavailable (%s)") % exc)
        return None


class LazySaliency:
    """Builds the model on the first faceless frame — videos full of faces
    never trigger the download at all. ``ok`` reports whether the model
    actually came up (False keeps the analysis re-fillable later)."""

    def __init__(self, log=print, progress=None):
        self.log, self.progress = log, progress
        self._sal: Optional[Saliency] = None
        self.ok = True

    def __call__(self, img: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
        if self._sal is None:
            self._sal = build(self.log, self.progress)
            if self._sal is None:
                self.ok = False
                return None
        return self._sal(img)


def backfill(path: str, analysis, sal: Saliency, progress=None) -> int:
    """Fill ``FrameSample.salient`` on the faceless samples of an existing
    (e.g. cached) analysis — only those frames are re-read from the video."""
    import cv2
    todo = [s for s in analysis.samples if not s.faces and s.salient is None]
    if not todo:
        analysis.salient_pass = True
        return 0
    cap = cv2.VideoCapture(path)
    done = 0
    for k, smp in enumerate(todo):
        cap.set(cv2.CAP_PROP_POS_FRAMES, analysis.start_frame + smp.frame)
        ok, frame = cap.read()
        if ok:
            smp.salient = sal(frame)
            done += 1
        if progress:
            progress((k + 1) / len(todo))
    cap.release()
    analysis.salient_pass = True
    return done
