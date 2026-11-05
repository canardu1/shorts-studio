"""Find the spoken segments of a clip so pauses can be cut out.

Audio is decoded to mono 16 kHz PCM with ffmpeg (a vendored imageio-ffmpeg
binary is used when ffmpeg isn't on PATH). Speech is detected with the
Silero VAD ONNX shipped inside faster-whisper when available (set
SHORTS_STUDIO_NO_VAD=1 to disable); otherwise short-window RMS energy is
thresholded adaptively against the clip's own noise floor.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000

VAD_CHUNK = 512       # 32 ms at 16 kHz — the model's step
VAD_CONTEXT = 64      # tail of the previous chunk the model also sees


def _vad_model_path() -> Optional[str]:
    """The silero_vad ONNX inside faster-whisper, or a copy in models_dir."""
    try:
        import faster_whisper
        p = os.path.join(os.path.dirname(faster_whisper.__file__), "assets", "silero_vad_v6.onnx")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    try:
        from .restore import models_dir
        p = os.path.join(models_dir(), "silero_vad_v6.onnx")
        if os.path.exists(p):
            return p
    except Exception:
        pass
    return None


_vad_session = None
_vad_failed = False


def vad_probs(pcm: np.ndarray) -> Optional[np.ndarray]:
    """Per-chunk speech probabilities (one per 32 ms), or None if the VAD
    model can't be loaded. Stateless wrapper around the bundled Silero net."""
    global _vad_session, _vad_failed
    if _vad_failed or os.environ.get("SHORTS_STUDIO_NO_VAD"):
        return None
    path = _vad_model_path()
    if path is None:
        _vad_failed = True
        return None
    n = len(pcm) // VAD_CHUNK
    if n == 0:
        return np.zeros(0, np.float32)
    try:
        if _vad_session is None:
            import onnxruntime as ort
            from . import models
            so = ort.SessionOptions()
            so.inter_op_num_threads = so.intra_op_num_threads = 1
            so.enable_cpu_mem_arena = False
            so.log_severity_level = 4
            _vad_session = ort.InferenceSession(path, sess_options=so,
                                                providers=models.providers())
        chunks = np.ascontiguousarray(pcm[: n * VAD_CHUNK]).reshape(n, VAD_CHUNK)
        ctx = chunks[:, -VAD_CONTEXT:].copy()   # copy: don't scribble on caller's pcm
        ctx[-1] = 0
        ctx = np.roll(ctx, 1, 0)
        out = _vad_session.run(None, {
            "input": np.concatenate([ctx, chunks], axis=1),
            "h": np.zeros((1, 1, 128), np.float32),
            "c": np.zeros((1, 1, 128), np.float32),
        })[0]
        return np.asarray(out, np.float32).reshape(-1)[:n]
    except Exception:
        _vad_failed = True
        return None


def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise RuntimeError(
            "ffmpeg not found: install it, or run install.py --deps (vendors a binary)"
        ) from exc


def extract_audio(path: str, start: float = 0.0, duration: Optional[float] = None) -> np.ndarray:
    """Mono 16 kHz float32 PCM for ``duration`` seconds starting at ``start``."""
    cmd = [find_ffmpeg(), "-v", "error"]
    if start > 0:
        cmd += ["-ss", "%.6f" % start]
    cmd += ["-i", path, "-map", "a:0", "-ac", "1", "-ar", str(SAMPLE_RATE)]
    if duration and duration > 0:
        cmd += ["-t", "%.6f" % duration]  # output option: must precede the "-" output url
    cmd += ["-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    if raw.returncode != 0:
        raise RuntimeError("ffmpeg audio decode failed: %s" % raw.stderr.decode(errors="replace")[:400])
    if not raw.stdout:
        raise RuntimeError("no audio stream in %s" % path)
    return np.frombuffer(raw.stdout, dtype=np.float32)


@dataclass
class SpeechParams:
    vad: bool = True
    """Neural VAD when the model is available; RMS levels otherwise."""
    vad_on: float = 0.5
    """Chunk probability that starts speech."""
    vad_off: float = 0.35
    """Speech keeps going until probability falls below this (hysteresis)."""
    window_ms: float = 25.0
    """RMS window length."""
    floor_percentile: float = 25.0
    """Percentile of window energies treated as the noise floor."""
    margin_db: float = 8.0
    """Speech must start this many dB above the noise floor."""
    release_db: float = 4.0
    """Speech keeps going until energy falls below floor + this; the gap
    between margin_db and release_db is what stops word tails being clipped."""
    min_pause: float = 0.35
    """Pauses shorter than this are kept (speech stays continuous)."""
    padding: float = 0.16
    """Seconds kept around each speech interval."""
    min_speech: float = 0.15
    """Speech blips shorter than this are dropped."""


def _window_db(pcm: np.ndarray, window: int) -> Tuple[np.ndarray, float]:
    n = len(pcm) // window
    if n == 0:
        return np.zeros(0), window
    rms = np.sqrt(np.mean(pcm[: n * window].reshape(n, window) ** 2, axis=1) + 1e-12)
    return 20.0 * np.log10(rms), float(window)


def _intervals(v: np.ndarray, step: float, hi: float, lo: float, p: SpeechParams
               ) -> List[Tuple[float, float]]:
    """Hysteresis runs over a per-window signal, merged and padded to seconds."""
    # enter speech at the high threshold, leave it at the low one (hysteresis)
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(v)
    while i < n:
        if v[i] <= hi:
            i += 1
            continue
        a = i
        while a > 0 and v[a - 1] > lo:
            a -= 1
        j = i
        while j < n and v[j] > lo:
            j += 1
        runs.append((a, j))
        i = j
    if not runs:
        return []
    gap = int(round(p.min_pause / step))
    merged: List[List[int]] = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    pad = int(round(p.padding / step))
    out = []
    for a, b in merged:
        if (b - a) * step < p.min_speech:
            continue
        out.append((max(0.0, (a - pad) * step), min(n * step, (b + pad) * step)))
    return out


def speech_intervals(
    pcm: np.ndarray, params: Optional[SpeechParams] = None
) -> List[Tuple[float, float]]:
    """(start, end) seconds of speech in the decoded audio."""
    p = params or SpeechParams()
    if p.vad:
        probs = vad_probs(pcm)
        if probs is not None:
            return _intervals(probs, VAD_CHUNK / SAMPLE_RATE, p.vad_on, p.vad_off, p)
    win = max(1, int(round(p.window_ms * SAMPLE_RATE / 1000.0)))
    db, win = _window_db(pcm, win)
    if len(db) == 0:
        return []
    floor = float(np.percentile(db, p.floor_percentile))
    loud = float(np.percentile(db, 90))
    if loud < -50.0:
        return []  # true silence / no audio content
    if loud - floor < p.margin_db:
        hi = lo = float(db.min()) - 1.0  # no real dynamic range -> treat everything as speech
    else:
        hi = max(-50.0, floor + p.margin_db)
        lo = max(-50.0, floor + min(p.release_db, p.margin_db))
    return _intervals(db, win / SAMPLE_RATE, hi, lo, p)


@dataclass
class BreathParams:
    window_ms: float = 25.0
    floor_percentile: float = 25.0
    margin_db: float = 8.0
    """Must match SpeechParams.margin_db: breaths are the bursts that never
    crossed the speech-entry threshold."""
    breath_db: float = 2.0
    """A pause window counts as breathing when it sits this far above the floor."""
    min_dur: float = 0.08
    max_dur: float = 0.9
    """Audible inhales are short; longer bursts are left alone."""


def breath_intervals(
    pcm: np.ndarray, params: Optional[BreathParams] = None
) -> List[Tuple[float, float]]:
    """(start, end) seconds of audible breaths inside the pauses.

    A breath is a run of windows louder than the noise floor (inhale, lip
    smack, sigh) that never reached the speech-entry threshold — so it sits
    inside a kept pause or between merged speech runs.
    """
    p = params or BreathParams()
    win = max(1, int(round(p.window_ms * SAMPLE_RATE / 1000.0)))
    db, win = _window_db(pcm, win)
    if len(db) == 0:
        return []
    floor = float(np.percentile(db, p.floor_percentile))
    loud = float(np.percentile(db, 90))
    if loud < -50.0 or loud - floor < p.margin_db:
        return []
    hi = max(-50.0, floor + p.margin_db)
    lo = floor + p.breath_db
    step = win / SAMPLE_RATE
    out: List[Tuple[float, float]] = []
    i = 0
    n = len(db)
    while i < n:
        if db[i] <= lo or db[i] > hi:
            i += 1
            continue
        j = i
        while j < n and lo < db[j] <= hi:
            j += 1
        dur = (j - i) * step
        if p.min_dur <= dur <= p.max_dur:
            out.append((i * step, j * step))
        i = j
    return out


def clip_intervals(
    path: str, start_frame: int, frame_count: int, fps: float,
    params: Optional[SpeechParams] = None,
) -> List[Tuple[int, int]]:
    """Speech intervals as source-frame (start, end] ranges within the item."""
    start_s = start_frame / fps
    pcm = extract_audio(path, start_s, frame_count / fps)
    spans = speech_intervals(pcm, params)
    if not spans:
        return []
    return [
        (start_frame + int(round(a * fps)), start_frame + min(frame_count, int(round(b * fps))))
        for a, b in spans if b - a > 0
    ]
