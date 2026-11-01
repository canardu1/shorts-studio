"""Find the spoken segments of a clip so pauses can be cut out.

Audio is decoded to mono 16 kHz PCM with ffmpeg (a vendored imageio-ffmpeg
binary is used when ffmpeg isn't on PATH), then short-window RMS energy is
thresholded adaptively against the clip's own noise floor.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

SAMPLE_RATE = 16000


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


def speech_intervals(
    pcm: np.ndarray, params: Optional[SpeechParams] = None
) -> List[Tuple[float, float]]:
    """(start, end) seconds of speech in the decoded audio."""
    p = params or SpeechParams()
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
    voiced = db > lo
    step = win / SAMPLE_RATE
    # enter speech at the high threshold, leave it at the low one (hysteresis)
    runs: List[Tuple[int, int]] = []
    i = 0
    n = len(db)
    while i < n:
        if db[i] <= hi:
            i += 1
            continue
        a = i
        while a > 0 and db[a - 1] > lo:
            a -= 1
        j = i
        while j < n and db[j] > lo:
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
    total = len(db)
    pad = int(round(p.padding / step))
    out = []
    for a, b in merged:
        if (b - a) * step < p.min_speech:
            continue
        out.append((max(0.0, (a - pad) * step), min(total * step, (b + pad) * step)))
    return out


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
