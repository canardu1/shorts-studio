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

    def __init__(self, downsample: float = 0.25, max_side: int = 640):
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


class ModnetMatter:
    """MODNet photographic portrait matte — noticeably finer hair/edges than
    RVM-MobileNet. Per-frame (not recurrent): a light temporal blend damps
    flicker between consecutive frames. Input is padded to a multiple of 32.
    """

    PAD = 32

    def __init__(self, max_side: int = 640, temporal: float = 0.35):
        from .. import models
        self.sess = models.session("modnet")
        self.max_side, self.temporal = max_side, temporal
        self._prev: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, img: np.ndarray) -> np.ndarray:
        import cv2
        h, w = img.shape[:2]
        sc = min(1.0, self.max_side / max(h, w))
        mh = int(h * sc) // self.PAD * self.PAD or self.PAD
        mw = int(w * sc) // self.PAD * self.PAD or self.PAD
        x = cv2.resize(img, (mw, mh), interpolation=cv2.INTER_AREA)[..., ::-1]
        x = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        m = self.sess.run(["output"], {"input": x})[0][0, 0]
        m = cv2.resize(np.clip(m, 0.0, 1.0), (w, h), interpolation=cv2.INTER_LINEAR)
        if self._prev is not None and self.temporal > 0:
            m = m * (1.0 - self.temporal) + self._prev * self.temporal
        self._prev = m
        return m


def refine_matte(img: np.ndarray, matte: np.ndarray,
                 contract: int = 2, feather: float = 1.2) -> np.ndarray:
    """Snap a (possibly upscaled) matte to the real pixel edges.

    A guided filter transfers the frame's own luminance edges onto the
    matte — this is what restores hair and outlines lost by the model's
    small input size. ``contract`` then shrinks a few px so the background
    cannot halo into the subject."""
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p = matte.astype(np.float32)
    r = max(2, int(round(min(img.shape[:2]) * 0.02)))
    mean_I = cv2.boxFilter(g, -1, (r, r))
    mean_p = cv2.boxFilter(p, -1, (r, r))
    cov = cv2.boxFilter(g * p, -1, (r, r)) - mean_I * mean_p
    var = cv2.boxFilter(g * g, -1, (r, r)) - mean_I * mean_I
    a = cov / (var + 1e-3)
    b = mean_p - a * mean_I
    q = cv2.boxFilter(a, -1, (r, r)) * g + cv2.boxFilter(b, -1, (r, r))
    q = np.clip(q, 0.0, 1.0)
    if contract > 0:
        q = cv2.erode(q, np.ones((contract * 2 + 1,) * 2, np.uint8))
    return cv2.GaussianBlur(q, (0, 0), feather)


class BlurBackground:
    """``post_frame`` hook: keep the person, blur everything outside the matte.

    ``strength`` scales the blur radius. The matte state resets when frames
    restart or seek backwards, so one hook survives seeks and multiple
    sequential renders.
    """

    def __init__(self, matter, strength: float = 0.6, refine: bool = True):
        self.matter, self.strength, self.refine = matter, strength, refine
        self._last = -1

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        import cv2
        if self._last < 0 or i <= self._last:
            self.matter.reset()
        self._last = i
        a = np.clip(self.matter(frame), 0.0, 1.0).astype(np.float32)
        if self.refine:
            a = refine_matte(frame, a)
        else:
            a = cv2.GaussianBlur(a, (0, 0), 1.5)
        sigma = 4.0 + 26.0 * self.strength
        bg = cv2.GaussianBlur(frame, (0, 0), sigma).astype(np.float32)
        return (frame.astype(np.float32) * a[..., None] + bg * (1 - a[..., None])).astype(np.uint8)


def build(strength: float, engine: str = "rvm", progress=None, log=print) -> Optional[BlurBackground]:
    """Blur hook, downloading the model on first use. ``None`` on failure."""
    if strength <= 0:
        return None
    try:
        if engine == "modnet":
            from .. import models
            if models.ensure("modnet", progress, log) is None:
                return None
            return BlurBackground(ModnetMatter(), strength)
        if not rvm_available():
            log(tr("Background blur: downloading the matting model (14 MB, once)…"))
            download_rvm(progress)
        return BlurBackground(Matter(), strength)
    except Exception as exc:
        log(tr("  background blur unavailable (%s)") % exc)
        return None
