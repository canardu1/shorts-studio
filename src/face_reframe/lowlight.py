"""Neural low-light enhancement (URetinex-Net / SG-LLIE).

Tiny fixed-input nets (~0.05-1.6 MB, tens of ms on CPU) that brighten
underexposed shots without blowing the highlights. The models are downloaded
on first use through :mod:`models`; the strength blends the lifted frame with
the original, and URetinex's own exposure input scales it further.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from . import models
from .i18n import tr

MODEL_KEYS = {"uretinex": "uretinex", "sg_llie": "sg_llie"}
"""UI value -> registry key (kept separate so the UI can rename freely)."""


def available(key: str) -> bool:
    return models.available(MODEL_KEYS[key])


class Lifter:
    """Callable ``frame -> frame``: neural low-light lift at ``strength`` (0-1)."""

    def __init__(self, key: str = "uretinex", strength: float = 0.6):
        self.key = MODEL_KEYS[key]
        self.strength = float(np.clip(strength, 0.0, 1.0))
        self.sess = models.session(self.key)
        inp = self.sess.get_inputs()
        self.inp_name = inp[0].name
        sh = inp[0].shape            # [1, 3, H, W], fixed
        self.mh = int(sh[2]) if isinstance(sh[2], int) else 180
        self.mw = int(sh[3]) if isinstance(sh[3], int) else 320
        self.extra = inp[1].name if len(inp) > 1 else None

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        import cv2
        h, w = frame.shape[:2]
        x = cv2.resize(frame, (self.mw, self.mh), interpolation=cv2.INTER_AREA)
        x = x[..., ::-1].astype(np.float32) / 255.0
        feed = {self.inp_name: x.transpose(2, 0, 1)[None]}
        if self.extra is not None:
            # URetinex's target-exposure input: a bit more gain at full strength
            feed[self.extra] = np.array([3.0 + 2.0 * self.strength], np.float32)
        y = self.sess.run(None, feed)[0][0].transpose(1, 2, 0)
        y = cv2.resize(np.clip(y, 0.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        y = (y[..., ::-1] * 255.0).astype(np.float32)
        if self.strength >= 0.999:
            return y.astype(np.uint8)
        return np.clip(frame.astype(np.float32) * (1.0 - self.strength) + y * self.strength,
                       0, 255).astype(np.uint8)


def build(key: str, strength: float, log=print, progress=None) -> Optional[Lifter]:
    """Download the model when needed and return a lifter, else None."""
    if key not in MODEL_KEYS:
        return None
    if models.ensure(MODEL_KEYS[key], progress, log) is None:
        return None
    try:
        return Lifter(key, strength)
    except Exception as exc:
        log(tr("Low-light boost unavailable (%s)") % exc)
        return None
