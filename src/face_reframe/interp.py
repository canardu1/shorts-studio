"""Frame interpolation with RIFE 4.8 (slow motion / speed ramps).

The ONNX model (~21 MB, downloaded on first use) generates the in-between
frames of a pair, so ``factor`` = 2 or 4 slows the video by that much while
staying smooth — an alternative to ffmpeg's blend-based ``minterpolate``.
Export-only: at ~90 ms per 320×180 frame on CPU it can't play live.
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable, Optional

import numpy as np

from . import models
from .i18n import tr


def available() -> bool:
    return models.available("rife")


class Rife:
    """``(img0, img1, t) -> interpolated middle frame(s)``, BGR uint8 in/out."""

    PAD = 32

    def __init__(self):
        self.sess = models.session("rife")

    def _run(self, a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
        import cv2
        h, w = a.shape[:2]
        ph = (self.PAD - h % self.PAD) % self.PAD
        pw = (self.PAD - w % self.PAD) % self.PAD

        def to_x(img):
            x = cv2.copyMakeBorder(img, 0, ph, 0, pw, cv2.BORDER_REPLICATE)
            return x[..., ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        y = self.sess.run(None, {"img0": to_x(a), "img1": to_x(b),
                                 "timestep": np.array([t], np.float32)})[0][0]
        y = (np.clip(y.transpose(1, 2, 0), 0.0, 1.0) * 255.0).astype(np.uint8)[:h, :w]
        return y[..., ::-1]

    def between(self, a: np.ndarray, b: np.ndarray, factor: int) -> list:
        """The ``factor - 1`` frames between a and b (a excluded, b excluded):
        factor=2 → [mid], factor=4 → recursive quarter points."""
        if factor <= 1:
            return []
        mid = self._run(a, b, 0.5)
        if factor == 2:
            return [mid]
        q1 = self._run(a, mid, 0.5)
        q3 = self._run(mid, b, 0.5)
        return [q1, mid, q3] if factor >= 4 else [q1, mid]


def slowmo(src: str, dst: str, factor: int, engine: str = "rife",
           progress: Optional[Callable[[float], None]] = None,
           log=print) -> bool:
    """Slow ``src`` by ``factor`` (2/4) into ``dst`` — video only slows,
    audio is retimed with atempo. ``engine`` = ``rife`` (neural) or
    ``ffmpeg`` (minterpolate blend); returns False when the engine fails."""
    from .silence import find_ffmpeg
    if factor <= 1:
        return False
    work_v = None
    if engine == "rife":
        work_v = _rife_video(src, factor, progress, log)
        if work_v is None:
            log(tr("RIFE failed, using ffmpeg interpolation"))
            engine = "ffmpeg"
    else:
        work_v = _ffmpeg_video(src, factor, progress, log)
    if work_v is None:
        return False
    # retime the audio over the slowed video
    atempo = ",".join("atempo=%.3f" % t for t in _atempo_chain(1.0 / factor))
    r = subprocess.run([find_ffmpeg(), "-y", "-v", "error", "-i", work_v, "-i", src,
                        "-map", "0:v", "-map", "1:a?", "-af", atempo,
                        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-movflags", "+faststart", dst],
                       capture_output=True, text=True)
    if work_v != src and os.path.exists(work_v):
        os.unlink(work_v)
    if r.returncode != 0:
        log(tr("Slow-motion audio failed: %s") % r.stderr[-200:])
        return False
    return True


def _atempo_chain(f: float):
    """``atempo`` accepts 0.5-2.0: chain factors to reach f."""
    out = []
    while f < 0.5:
        out.append(0.5)
        f /= 0.5
    while f > 2.0:
        out.append(2.0)
        f /= 2.0
    out.append(f)
    return out


def _rife_video(src: str, factor: int, progress, log) -> Optional[str]:
    """Slowed, silent video via RIFE next to ``src`` (temp file)."""
    import cv2
    try:
        rife = Rife()
    except Exception as exc:
        log(tr("RIFE unavailable (%s)") % exc)
        return None
    from .silence import find_ffmpeg
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # RIFE on 1080p costs ~1 s/frame: cap the working size, quality keeps up
    scale = 1.0
    while w * scale * h * scale > 1280 * 720:
        scale *= 0.5
    if scale < 1.0:
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    tmp = src + ".rife.mp4"
    proc = subprocess.Popen([find_ffmpeg(), "-y", "-v", "error", "-f", "rawvideo",
                             "-pix_fmt", "bgr24", "-s", "%dx%d" % (w, h), "-r", "%.5f" % fps,
                             "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "16",
                             "-pix_fmt", "yuv420p", tmp],
                            stdin=subprocess.PIPE)
    ok, prev = cap.read()
    n = 0
    try:
        while ok:
            frame = cv2.resize(prev, (w, h)) if scale < 1.0 else prev
            proc.stdin.write(frame.tobytes())
            ok, nxt = cap.read()
            if ok:
                mid = cv2.resize(nxt, (w, h)) if scale < 1.0 else nxt
                for m in rife.between(frame, mid, factor):
                    proc.stdin.write(np.ascontiguousarray(m).tobytes())
            prev = nxt
            n += 1
            if progress and total and n % 20 == 0:
                progress(min(0.95, n / float(total)))
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()
    return tmp if proc.returncode == 0 else None


def _ffmpeg_video(src: str, factor: int, progress, log) -> Optional[str]:
    """Slowed, silent video via ffmpeg minterpolate."""
    from .silence import find_ffmpeg
    tmp = src + ".slomo.mp4"
    # double the fps then slow timestamps: minterpolate works on the fps grid
    r = subprocess.run([find_ffmpeg(), "-y", "-v", "error", "-i", src,
                        "-vf", "minterpolate=fps=%d:tmi=blend:mi_mode=mci,setpts=%.3f*PTS"
                               % (factor * 30000 // 1001, factor),
                        "-an", "-c:v", "libx264", "-preset", "fast", "-crf", "16", tmp],
                       capture_output=True, text=True)
    return tmp if r.returncode == 0 and os.path.exists(tmp) else None


def ensure(progress=None, log=print) -> bool:
    """Download the RIFE model; True when usable."""
    return models.ensure("rife", progress, log) is not None
