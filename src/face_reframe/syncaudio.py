"""Align an external recorder's audio to the camera track (GCC-PHAT).

``find_offset`` searches where a short reference (the camera audio over the
item's timeline span) best aligns inside a possibly very long external file.
Sliding windows of 2x the reference keep memory bounded on hour-long takes.
"""

from __future__ import annotations

import subprocess
from typing import Tuple

import numpy as np

from .silence import SAMPLE_RATE, find_ffmpeg


def _xcorr(ref: np.ndarray, win: np.ndarray) -> np.ndarray:
    """Circular GCC-PHAT: r[k] = sum_n ref[n] * win[n - k]."""
    n = len(ref) + len(win) - 1
    nfft = 1 << (n - 1).bit_length()
    a = np.fft.rfft(ref, nfft)
    b = np.fft.rfft(win, nfft)
    return np.fft.irfft(a * np.conj(b) / (np.abs(a * np.conj(b)) + 1e-15), nfft)


def find_offset(ref: np.ndarray, sig: np.ndarray,
                progress=None) -> Tuple[float, float]:
    """(lag_seconds, score): position inside ``sig`` where ``ref`` starts.

    Positive = ref sits that far into sig (the recorder started earlier);
    negative = ref precedes sig's head (the recorder started later).
    Score is the GCC-PHAT peak (~0-1): real alignments score ~0.05+, noise
    stays flat and low.
    """
    nref = len(ref)
    if nref == 0 or len(sig) < nref // 2:
        return 0.0, 0.0
    win_len = 2 * nref
    step = nref  # windows overlap by half, so a match can't be skipped
    best_score, best_lag = 0.0, 0
    pos = 0
    while pos < len(sig):
        w = sig[pos:pos + win_len]
        if len(w) < nref:
            break
        r = _xcorr(ref, w)
        nfft = len(r)
        L = len(w) - nref          # full-overlap lags k in [-L, -1] live at r[nfft-L:]
        # plus positive lags k in [0, nref): ref starting before the window
        # (recorder started later than the camera)
        cand = np.concatenate([r[nfft - L:], r[:nref]])
        j = int(np.argmax(np.abs(cand)))
        score = float(cand[j])
        k = j - L                  # signed: ref[n] ~= w[n - k]
        start = pos - k            # sig position where ref[0] aligns
        if score > best_score:
            best_score, best_lag = score, start
        pos += step
        if progress:
            progress(min(1.0, pos / float(len(sig))))
    return best_lag / float(SAMPLE_RATE), best_score


def aligned_copy(path: str, out_path: str, shift_s: float) -> str:
    """Write ``out_path``: ``path`` shifted by ``shift_s`` seconds — the head
    trimmed when positive, silence padded when negative. Stream properties
    (rate, channels) are preserved; the container stays WAV."""
    if shift_s >= 0:
        af = "atrim=%.4f,asetpts=PTS-STARTPTS" % shift_s
    else:
        af = "adelay=%d" % int(round(-shift_s * 1000))
    cmd = [find_ffmpeg(), "-y", "-v", "error", "-i", path, "-af", af,
           "-vn", "-c:a", "pcm_s16le", out_path]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    if raw.returncode != 0:
        raise RuntimeError("ffmpeg align bake failed: %s"
                           % raw.stderr.decode(errors="replace")[:300])
    return out_path
