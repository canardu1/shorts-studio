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
    ai_fraction: float = 0.5
    """Fraction of the output size the model itself produces; 0.5 runs the
    x4 network on output/8 input — the fastest setting still worth using."""


def model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "models", MODEL_NAME)


def _cv2():
    import cv2
    return cv2


def _unsharp(img, amount: float, sigma: float):
    cv2 = _cv2()
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


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
                    have = set(ort.get_available_providers())
                    providers = [p for p in ("CoreMLExecutionProvider",
                                             "CUDAExecutionProvider",
                                             "CPUExecutionProvider")
                                 if p in have]
                    self._sess = ort.InferenceSession(path, providers=providers)
                    self._inp = self._sess.get_inputs()[0].name
                    log("AI enhance: %s via %s" % (MODEL_NAME, providers[0]))

    def _ai(self, crop, out_w: int, out_h: int):
        import numpy as np
        cv2 = _cv2()
        f = self.opts.ai_fraction
        iw = max(32, int(round(out_w * f / 4.0)))
        ih = max(32, int(round(out_h * f / 4.0)))
        inp = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = rgb.transpose(2, 0, 1)[None]
        res = self._sess.run(None, {self._inp: blob})[0][0].transpose(1, 2, 0)
        bgr = cv2.cvtColor(np.clip(res * 255.0, 0, 255).astype(np.uint8),
                           cv2.COLOR_RGB2BGR)
        return cv2.resize(bgr, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

    def __call__(self, crop, out_w: int, out_h: int,
                 face: Optional[Tuple[float, float]] = None):
        import numpy as np
        cv2 = _cv2()
        if self.mode == "ai" and self._sess is not None:
            out = self._ai(crop, out_w, out_h)
        else:
            upscaling = crop.shape[1] < out_w
            out = cv2.resize(crop, (out_w, out_h),
                             interpolation=cv2.INTER_LANCZOS4 if upscaling
                             else cv2.INTER_AREA)
        o = self.opts
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
