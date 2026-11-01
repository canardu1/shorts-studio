"""Sidechain ducking: dip the music bed under the voice.

The voice track is rebuilt in numpy at timeline positions and fed to
ffmpeg's ``sidechaincompress`` as the key — the music ducks wherever the
camera audio is active. No Resolve audio-FX needed (none are scriptable).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .silence import SAMPLE_RATE, extract_audio, find_ffmpeg


@dataclass
class DuckParams:
    threshold_db: float = -33.0
    """Voice level (in the key) that engages ducking."""
    ratio: float = 8.0
    """Ducking depth: roughly -20*log10(ratio) dB on loud voice parts."""
    attack_ms: float = 15.0
    release_ms: float = 400.0
    makeup_db: float = 0.0
    """Extra trim applied to the music after ducking."""


def build_voice_key(segments: List[Tuple[str, float, float, float]],
                    length_s: float, log=print) -> np.ndarray:
    """Sum of ``segments`` audio placed at timeline positions.

    Each segment is ``(path, source_start_s, dur_s, timeline_offset_s)`` —
    the key is 16 kHz mono float32, ``length_s`` long."""
    key = np.zeros(int(max(0.0, length_s) * SAMPLE_RATE), dtype=np.float32)
    for path, s0, dur, off in segments:
        try:
            pcm = extract_audio(path, s0, dur)
        except Exception as exc:
            log("  key decode failed on %s (%s)" % (os.path.basename(path), exc))
            continue
        at = int(off * SAMPLE_RATE)
        if at < len(key):
            key[at:at + len(pcm)] += pcm[:max(0, len(key) - at)]
    return key


def slice_key(key: np.ndarray, off_s: float, dur_s: float) -> np.ndarray:
    """Window of the key in its own time axis, zero-padded at the edges."""
    n = int(max(0.0, dur_s) * SAMPLE_RATE)
    out = np.zeros(n, dtype=np.float32)
    src0 = int(off_s * SAMPLE_RATE)
    a, b = max(0, src0), min(len(key), src0 + n)
    if a < b:
        out[a - src0:b - src0] = key[a:b]
    return out


def bake_ducked(music_path: str, key_pcm: np.ndarray, out_path: str,
                p: DuckParams, progress: Optional[Callable[[float], None]] = None,
                total_s: float = 0.0) -> str:
    """Duck ``music_path`` under ``key_pcm`` into ``out_path``.

    The key is written to a temp wav (it is only the control signal);
    the music keeps its own sample rate and channel layout.
    """
    fd, key_path = tempfile.mkstemp(suffix=".wav", prefix="face_reframe_key_")
    os.close(fd)
    with wave.open(key_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes((np.clip(key_pcm, -1, 1) * 32767).astype(np.int16).tobytes())
    thr = 10.0 ** (p.threshold_db / 20.0)
    filt = ("[1:a]asplit=2[k][kk];[0:a][k]sidechaincompress=threshold=%.5f"
            ":ratio=%.1f:attack=%.0f:release=%.0f:makeup=%.4f[d]"
            % (thr, p.ratio, p.attack_ms, p.release_ms, 10.0 ** (p.makeup_db / 20.0)))
    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", music_path, "-i", key_path,
           "-filter_complex", filt, "-map", "[d]", out_path]
    try:
        from .cinematic import _run_ffmpeg
        _run_ffmpeg(cmd, total_s, progress)
    finally:
        try:
            os.unlink(key_path)
        except OSError:
            pass
    return out_path
