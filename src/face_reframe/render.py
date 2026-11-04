"""Bake a reframed video with OpenCV + ffmpeg (audio is copied from the source)."""

from __future__ import annotations

import subprocess
import tempfile
from typing import Callable, Optional, Tuple

import cv2
import numpy as np

from .detect import iter_frames
from .geometry import CropPlan
from .silence import find_ffmpeg  # noqa: F401  (re-export)


def render(
    video: str,
    plan: CropPlan,
    out_path: str,
    out_size: Tuple[int, int],
    fps: float,
    start_frame: int = 0,
    crf: int = 18,
    debug_overlay: Optional[np.ndarray] = None,
    progress: Optional[Callable[[float], None]] = None,
    enhancer=None,
    preset: str = "medium",
    post_frame: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    frame_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None,
    video_args: Optional[list] = None,
) -> str:
    """``post_frame(i, frame)`` runs on each finished output frame (e.g. to
    burn captions); ``preset`` is the x264 speed preset. ``frame_fn(i, src)``
    replaces the crop/scale step entirely (custom layouts, e.g. split screen).
    ``video_args`` replaces the x264 encoder arguments (e.g. hardware encoding)."""
    out_w, out_h = out_size
    n = len(plan.cx)
    cmd = [
        find_ffmpeg(), "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "%dx%d" % (out_w, out_h), "-r", repr(fps), "-i", "-",
        "-ss", "%.6f" % (start_frame / fps), "-t", "%.6f" % (n / fps), "-i", video,
        "-map", "0:v:0", "-map", "1:a?"] + (
        video_args or ["-c:v", "libx264", "-crf", str(crf), "-preset", preset]) + [
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", out_path,
    ]
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=err)
    cap = cv2.VideoCapture(video)
    try:
        for i, frame in enumerate(iter_frames(cap, start_frame, n)):
            x, y, w, h = plan.box(i)
            crop = frame[y:y + h, x:x + w]
            if frame_fn is not None:
                out = frame_fn(i, frame)
            elif enhancer is not None:
                face = ((plan.cx[i] - x) * out_w / w,
                        (plan.cy[i] - y) * out_h / h)
                out = enhancer(crop, out_w, out_h, face)
            else:
                out = cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4 if w < out_w else cv2.INTER_AREA)
            if post_frame is not None:
                out = post_frame(i, out)
            if debug_overlay is not None:
                fx = int(round((debug_overlay[i] - x) * out_w / float(w)))  # tracked face x
                cv2.line(out, (fx, 0), (fx, out_h), (0, 255, 0), 2)
            try:
                proc.stdin.write(out.tobytes())
            except BrokenPipeError:
                break  # ffmpeg died early; the exit code carries the real error
            if progress and i % 25 == 0:
                progress(i / float(n))
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except BrokenPipeError:
            pass
        code = proc.wait()
    if code != 0:
        err.seek(0)
        detail = err.read().decode("utf-8", "replace").strip()
        raise RuntimeError("ffmpeg failed with exit code %d%s" % (code, ": " + detail[-400:] if detail else ""))
    if progress:
        progress(1.0)
    return out_path
