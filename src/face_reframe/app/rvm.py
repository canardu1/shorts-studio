"""Blurred background behind the person, with a real matte.

Robust Video Matting (PeterL1n, MIT) gives a per-frame alpha matte of the
person; it is recurrent, so the matte is temporally coherent and doesn't
flicker. The official MobileNetV3 fp32 ONNX (14 MB) is downloaded once into
the shared models dir. The session is module-cached like the VAD one — each
:class:`Matter` only carries the recurrent state, so previews and renders can
run their own without paying session setup every time.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np

from ..i18n import tr

RVM_URL = "https://github.com/PeterL1n/RobustVideoMatting/releases/download/v1.0.0/rvm_mobilenetv3_fp32.onnx"
RVM_FILE = "rvm_mobilenetv3_fp32.onnx"


def rvm_model_path() -> str:
    from ..restore import models_dir
    return os.path.join(models_dir(), RVM_FILE)


def rvm_available() -> bool:
    p = rvm_model_path()
    return os.path.exists(p) and os.path.getsize(p) > 10_000_000


def download_rvm(progress=None) -> str:
    path = rvm_model_path()
    if rvm_available():
        return path
    from ..restore import fetch
    return fetch(RVM_URL, path, progress)


_SESS = None


def _sess():
    global _SESS
    if _SESS is None:
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        _SESS = ort.InferenceSession(rvm_model_path(), so, providers=["CPUExecutionProvider"])
    return _SESS


class Matter:
    """Person alpha matte (float32 0..1, frame size), temporally coherent.

    ``downsample_ratio`` feeds the model's own internal downscale; the input
    is additionally capped at ``max_side`` for CPU speed.
    """

    def __init__(self, downsample: float = 0.25, max_side: int = 480):
        self.max_side = max_side
        self.dsr = np.array([downsample], np.float32)
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]

    def reset(self) -> None:
        self.rec = [np.zeros((1, 1, 1, 1), np.float32) for _ in range(4)]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2
        h, w = img.shape[:2]
        sc = min(1.0, self.max_side / max(h, w))
        x = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) \
            if sc < 1.0 else img
        x = x[..., ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = _sess().run(["pha", "r1o", "r2o", "r3o", "r4o"], {
            "src": x, "r1i": self.rec[0], "r2i": self.rec[1],
            "r3i": self.rec[2], "r4i": self.rec[3], "downsample_ratio": self.dsr})
        self.rec = out[1:]
        return cv2.resize(out[0][0, 0], (w, h), interpolation=cv2.INTER_LINEAR)


class BlurBackground:
    """``post_frame`` hook: keep the person, blur everything outside the matte.

    ``strength`` scales the blur radius. The matte state resets when frames
    restart or seek backwards, so one hook survives seeks and multiple
    sequential renders.
    """

    def __init__(self, matter, strength: float = 0.6):
        self.matter, self.strength = matter, strength
        self._last = -1

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        import cv2
        if self._last < 0 or i <= self._last:
            self.matter.reset()
        self._last = i
        a = np.clip(self.matter(frame), 0.0, 1.0).astype(np.float32)
        a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]        # soft edges
        sigma = 4.0 + 26.0 * self.strength
        bg = cv2.GaussianBlur(frame, (0, 0), sigma).astype(np.float32)
        return (frame.astype(np.float32) * a + bg * (1 - a)).astype(np.uint8)


def build(strength: float, progress=None, log=print) -> Optional[BlurBackground]:
    """Blur hook, downloading the model on first use. ``None`` on failure."""
    if strength <= 0:
        return None
    try:
        if not rvm_available():
            log(tr("Background blur: downloading the matting model (14 MB, once)…"))
            download_rvm(progress)
        return BlurBackground(Matter(), strength)
    except Exception as exc:
        log(tr("  background blur unavailable (%s)") % exc)
        return None
