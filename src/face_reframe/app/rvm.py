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
        from .. import models
        _SESS = ort.InferenceSession(rvm_model_path(), so, providers=models.providers())
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

    def __init__(self, matter, strength: float = 0.6, refine: bool = True,
                 alt_matter=None, use_alt=None):
        self.matter, self.strength, self.refine = matter, strength, refine
        self.alt_matter, self.use_alt = alt_matter, use_alt
        self._last = -1

    def _bg(self, i: int, frame: np.ndarray) -> np.ndarray:
        """What replaces the background (float32 frame); blurred by default."""
        import cv2
        sigma = 4.0 + 26.0 * self.strength
        return cv2.GaussianBlur(frame, (0, 0), sigma).astype(np.float32)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        import cv2
        if self._last < 0 or i <= self._last:
            self.matter.reset()
            if self.alt_matter is not None:
                self.alt_matter.reset()
        self._last = i
        m = self.alt_matter if (self.alt_matter is not None
                                and self.use_alt is not None and self.use_alt()) \
            else self.matter
        a = np.clip(m(frame), 0.0, 1.0).astype(np.float32)
        if self.refine:
            a = refine_matte(frame, a)
        else:
            a = cv2.GaussianBlur(a, (0, 0), 1.5)
        bg = self._bg(i, frame)
        return (frame.astype(np.float32) * a[..., None] + bg * (1 - a[..., None])).astype(np.uint8)


class FillBackground(BlurBackground):
    """Same matte pipeline, but the background is a fill provider (colour,
    image or looping video) instead of a blur — a digital green screen."""

    def __init__(self, matter, fill_fn, refine: bool = True, alt_matter=None, use_alt=None):
        super().__init__(matter, 0.0, refine, alt_matter, use_alt)
        self.fill_fn = fill_fn

    def _bg(self, i: int, frame: np.ndarray) -> np.ndarray:
        return self.fill_fn(i, frame).astype(np.float32)


def _cover(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Scale to cover the output and centre-crop — no letterboxing."""
    import cv2
    sc = max(out_w / img.shape[1], out_h / img.shape[0])
    img = cv2.resize(img, (int(img.shape[1] * sc + 0.5), int(img.shape[0] * sc + 0.5)),
                     interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LANCZOS4)
    x0 = max(0, (img.shape[1] - out_w) // 2)
    y0 = max(0, (img.shape[0] - out_h) // 2)
    return img[y0:y0 + out_h, x0:x0 + out_w]


class ColorFill:
    """Solid colour backdrop — pick #00b140 for a classic green screen."""

    def __init__(self, hex_color: str, out_w: int, out_h: int):
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        self.bg = np.full((out_h, out_w, 3), (b, g, r), np.uint8)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        return self.bg


class ImageFill:
    """Still image backdrop, cover-fitted once."""

    def __init__(self, path: str, out_w: int, out_h: int):
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise IOError(tr("unreadable background image: %s") % path)
        self.bg = _cover(img, out_w, out_h)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        return self.bg


class VideoFill:
    """Looping background video, decoded per frame.

    Source fps is honoured (``rate``), playback loops. Reads are strictly
    sequential: a repeated index (background slower than the output) reuses
    the last decoded frame instead of seeking, and only an actual jump
    (preview scrubs, a new render pass, the loop wrap) seeks at all — on a
    long-GOP source every seek costs a re-decode from the previous keyframe."""

    def __init__(self, path: str, out_w: int, out_h: int, fps: float = 25.0):
        import cv2
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(tr("unreadable background video: %s") % path)
        self.out_w, self.out_h = out_w, out_h
        self.n = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1)
        self.rate = (self.cap.get(cv2.CAP_PROP_FPS) or 25.0) / fps
        self._last = -1
        self._img: Optional[np.ndarray] = None

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        import cv2
        src_i = int(i * self.rate) % self.n
        if src_i == self._last and self._img is not None:
            return self._img
        if src_i != (self._last + 1) % self.n:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, src_i)
        ok, img = self.cap.read()
        if not ok:  # loop wrap or corrupt tail
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, img = self.cap.read()
        self._last = src_i
        self._img = _cover(img, self.out_w, self.out_h) if ok \
            else np.zeros((self.out_h, self.out_w, 3), np.uint8)
        return self._img


def make_fill(mode: str, out_w: int, out_h: int, fps: float = 25.0,
              color: str = "#00b140", image=None, video=None, log=print):
    """Fill provider for ``mode`` — ``None`` means fall back to blur."""
    try:
        if mode == "color":
            return ColorFill(color, out_w, out_h)
        if mode == "image" and image and os.path.exists(image):
            return ImageFill(image, out_w, out_h)
        if mode == "video" and video and os.path.exists(video):
            return VideoFill(video, out_w, out_h, fps)
    except IOError as exc:
        log("  %s" % exc)
    return None


def build(strength: float, engine: str = "rvm", progress=None, log=print,
          size=None, fill: str = "blur", color: str = "#00b140",
          image=None, video=None, fps: float = 25.0) -> Optional[BlurBackground]:
    """Matte hook, downloading the model on first use. ``None`` on failure.

    ``fill != "blur"`` composites the person over a colour/image/video
    (``size`` is the (w, h) of the finished frame) — the green-screen path."""
    if strength <= 0 and fill in ("blur", "off"):
        return None
    try:
        matter = None
        if engine == "modnet":
            from .. import models
            if models.ensure("modnet", progress, log) is None:
                return None
            matter = ModnetMatter()
        else:
            if not rvm_available():
                log(tr("Background blur: downloading the matting model (14 MB, once)…"))
                download_rvm(progress)
            matter = Matter()
        if fill not in ("blur", "off") and size is not None:
            fn = make_fill(fill, size[0], size[1], fps, color, image, video, log)
            if fn is not None:
                return FillBackground(matter, fn)
        return BlurBackground(matter, strength)
    except Exception as exc:
        log(tr("  background blur unavailable (%s)") % exc)
        return None
