"""Per-frame enhancement for the render path — only the crop is touched.

Three modes:

* ``off`` — plain lanczos upscale (the existing behaviour);
* ``fast`` — lanczos + unsharp mask, plus extra sharpening inside the face
  region (the tracked face position comes for free from the crop plan);
* ``ai`` — Real-ESRGAN compact (``realesr-general-x4v3``, SRVGGNetCompact)
  through onnxruntime. The model is x4, so it is run on a quarter-size input
  and its internal denoise/deblur does the enhancement. On Apple Silicon the
  CoreML provider keeps it around 100-150 ms per frame.

numpy and cv2 are imported lazily: ``import face_reframe.enhance`` must stay
safe on interpreters without the vendored deps (e.g. bare ``python3``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple

MODEL_NAME = "realesr_x4v3.onnx"


@dataclass
class EnhanceOptions:
    mode: str = "off"
    """``off`` | ``fast`` | ``ai``."""
    sharpen: float = 0.5
    """Unsharp-mask amount applied to the upscaled frame."""
    sharpen_sigma: float = 2.0
    face_boost: float = 0.35
    """Extra sharpening inside the face region (added to ``sharpen``)."""
    ai_fraction: float = 1.0
    """Fraction of the output size the model itself produces (1.0: the x4
    network gets output/4 input). Never more input than the crop really has."""
    ai_max_pixels: int = 140_000
    """Input pixel budget per frame (~270x480): keeps the network around
    1-2 s/frame on an M1 CPU. The input is never upscaled to reach it."""
    ai_native: bool = False
    """Standalone restore pass (same size in and out): run the network even
    when the crop isn't upscaled, and ignore the pixel budget."""
    ai_strength: float = 0.55
    """Mix of the network output with a plain lanczos upscale of the real
    crop. 1.0 = network only (can look painted on real faces)."""


def model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "models", MODEL_NAME)


def _cv2():
    import cv2
    return cv2


def _unsharp(img, amount: float, sigma: float):
    cv2 = _cv2()
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _lock_colour(ai, ref):
    """Match the AI output's coarse colour to the plain-upscaled source, per
    Y/Cr/Cb channel. The network keeps the detail it adds but can't drift the
    palette (or flicker it frame to frame — the reference is stable)."""
    import numpy as np
    cv2 = _cv2()
    ya = cv2.cvtColor(ai, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    yr = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb).astype(np.float32)
    for c in range(3):
        a, r = ya[..., c], yr[..., c]
        sa, sr = float(a.std()), float(r.std())
        ya[..., c] = (a - float(a.mean())) * min(2.0, sr / max(1e-3, sa)) + float(r.mean())
    return cv2.cvtColor(np.clip(ya, 0, 255).astype(np.uint8), cv2.COLOR_YCrCb2BGR)


def _face_rect(face: Tuple[float, float], out_w: int, out_h: int
               ) -> Tuple[int, int, int, int]:
    """Soft box around the face: generous region (face + hair + shoulders)."""
    fx, fy = face
    fw, fh = 0.6 * out_w, 0.55 * out_h
    x0 = int(max(0, fx - fw / 2))
    y0 = int(max(0, fy - fh / 2))
    return x0, y0, int(min(out_w, fx + fw / 2)), int(min(out_h, fy + fh / 2))


class Enhancer:
    """Callable on each source crop: ``(crop_bgr, out_w, out_h, face) -> frame``.

    ``face`` is the tracked face centre ``(cx, cy)`` in output pixels;
    ``None`` sharpens uniformly.
    """

    def __init__(self, opts: EnhanceOptions, log=print):
        try:
            _cv2()
            import numpy  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "enhance needs numpy+opencv — missing %s on this interpreter "
                "(inside Resolve the vendored deps provide them)" % exc) from exc
        self.opts = opts
        self.mode = opts.mode
        self._sess = None
        self._inp = ""
        if self.mode == "ai":
            try:
                import onnxruntime as ort
            except ImportError:
                log("onnxruntime not installed -> enhance falls back to 'fast' "
                    "(run install.py --deps again to vendor it)")
                self.mode = "fast"
            else:
                path = model_path()
                if not os.path.exists(path):
                    log("model %s missing -> enhance falls back to 'fast'"
                        % MODEL_NAME)
                    self.mode = "fast"
                else:
                    # CoreML EP sends the model to the ANE compiler, which can
                    # stall the caller for tens of seconds (or hang on low
                    # disk) — not worth it at our capped input size.
                    have = set(ort.get_available_providers())
                    providers = [p for p in ("CUDAExecutionProvider",
                                             "CPUExecutionProvider")
                                 if p in have]
                    self._sess = ort.InferenceSession(path, providers=providers)
                    self._inp = self._sess.get_inputs()[0].name
                    log("AI enhance: %s via %s" % (MODEL_NAME, providers[0]))

    def needs_ai(self, crop, out_w: int, out_h: int) -> bool:
        """Super-resolution only helps when the crop is really upscaled; on a
        crop that already has the output's pixels it just repaints it."""
        if self.opts.ai_native:
            return True
        return crop.shape[1] < 0.8 * out_w and crop.shape[0] < 0.8 * out_h

    def _ai(self, crop, out_w: int, out_h: int):
        import numpy as np
        cv2 = _cv2()
        o = self.opts
        ch, cw = crop.shape[:2]
        iw, ih = out_w * o.ai_fraction / 4.0, out_h * o.ai_fraction / 4.0
        k = 1.0 if o.ai_native else min(1.0, (o.ai_max_pixels / max(1.0, iw * ih)) ** 0.5)
        iw, ih = iw * k, ih * k
        if iw > cw or ih > ch:            # never invent input pixels
            iw, ih = cw, ch
        iw, ih = max(32, int(round(iw))), max(32, int(round(ih)))
        inp = crop if (iw, ih) == (cw, ch) else cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[None]
        res = self._sess.run(None, {self._inp: blob})[0][0].transpose(1, 2, 0)
        bgr = cv2.cvtColor(np.clip(res * 255.0, 0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        down = bgr.shape[1] > out_w
        ai = cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_AREA if down else cv2.INTER_LANCZOS4)
        plain = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)
        ai = _lock_colour(ai, plain)
        if o.ai_strength >= 1.0:
            return ai
        return cv2.addWeighted(ai, o.ai_strength, plain, 1.0 - o.ai_strength, 0)

    def __call__(self, crop, out_w: int, out_h: int,
                 face: Optional[Tuple[float, float]] = None):
        import numpy as np
        cv2 = _cv2()
        ai = self.mode == "ai" and self._sess is not None and self.needs_ai(crop, out_w, out_h)
        if ai:
            out = self._ai(crop, out_w, out_h)
        else:
            upscaling = crop.shape[1] < out_w
            out = cv2.resize(crop, (out_w, out_h),
                             interpolation=cv2.INTER_LANCZOS4 if upscaling
                             else cv2.INTER_AREA)
        o = self.opts
        if ai:   # the network already sharpens: only a touch on top
            import dataclasses
            o = dataclasses.replace(o, sharpen=o.sharpen * 0.3, face_boost=o.face_boost * 0.3)
        if self.mode == "off" or (o.sharpen <= 0.0
                                  and (o.face_boost <= 0.0 or face is None)):
            return out
        if o.face_boost > 0.0 and face is not None:
            x0, y0, x1, y1 = _face_rect(face, out_w, out_h)
            mask = np.zeros((out_h, out_w), dtype=np.float32)
            mask[y0:y1, x0:x1] = 1.0
            mask = cv2.GaussianBlur(mask, (0, 0), 15.0)[..., None]
            sharp = _unsharp(out, o.sharpen + o.face_boost, o.sharpen_sigma)
            return (sharp * mask + _unsharp(out, o.sharpen, o.sharpen_sigma)
                    * (1.0 - mask)).astype(np.uint8)
        return _unsharp(out, o.sharpen, o.sharpen_sigma)


def bake_enhanced(video: str, out_path: str, enhancer: Enhancer,
                  progress=None) -> str:
    """Enhance a whole media file in place-resolution: every frame through
    ``enhancer`` (no crop, no scale change), encoded to h264 with the source
    audio stream-copied in. Used by the standalone Enhance tool."""
    import subprocess
    cv2 = _cv2()
    from .silence import find_ffmpeg

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("cannot open %s" % video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    cmd = [
        find_ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h),
        "-r", repr(fps), "-i", "-",
        "-i", video,
        "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "18",
        "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-b:a", "192k", "-shortest", out_path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            proc.stdin.write(enhancer(frame, w, h).tobytes())
            i += 1
            if progress and total and i % 25 == 0:
                progress(i / float(total))
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        code = proc.wait()
    if code != 0:
        raise RuntimeError("ffmpeg failed with exit code %d" % code)
    if progress:
        progress(1.0)
    return out_path
