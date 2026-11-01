"""Loudness measurement via ffmpeg's loudnorm filter (EBU R128)."""

from __future__ import annotations

import json
import re
import subprocess
from typing import Optional

from .silence import find_ffmpeg

TARGET_I = -14.0
"""Integrated loudness target for shorts (LUFS)."""


def measure_integrated(path: str, start: float = 0.0, duration: Optional[float] = None) -> float:
    """Integrated loudness of the audio in LUFS."""
    cmd = [find_ffmpeg(), "-v", "info"]
    if start > 0:
        cmd += ["-ss", "%.6f" % start]
    cmd += ["-i", path, "-map", "a:0"]
    if duration and duration > 0:
        cmd += ["-t", "%.6f" % duration]
    cmd += ["-af", "loudnorm=I=%.1f:print_format=json" % TARGET_I, "-f", "null", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    text = raw.stderr.decode(errors="replace")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", text)
    if not m:
        raise RuntimeError("ffmpeg loudnorm failed: %s" % text[-300:])
    return float(json.loads(m.group(0))["input_i"])


def gain_db(measured_i: float, target: float = TARGET_I, limit: float = 20.0) -> float:
    """Clip gain (dB) to bring ``measured_i`` to ``target``, safety-clamped."""
    return max(-limit, min(limit, target - measured_i))


def bake_normalized(path: str, out_path: str, gain: float) -> str:
    """Copy ``path`` to ``out_path`` with ``gain`` dB applied to the audio.

    Video is stream-copied (lossless, fast); audio is re-encoded with the
    gain plus a safety limiter. Used on Resolve versions with no clip-gain
    API (everything before 21.1).
    """
    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", path,
           "-map", "0:v:0?", "-map", "0:a:0?", "-c:v", "copy", "-af",
           "volume=%.2fdB,alimiter=limit=0.95" % gain,
           "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    if raw.returncode != 0:
        raise RuntimeError("ffmpeg gain bake failed: %s" % raw.stderr.decode(errors="replace")[:400])
    return out_path
