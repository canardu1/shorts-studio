"""Social-style camera moves and cut transitions, as per-frame curves.

Everything is a small change to the crop (offset, zoom, rotation) plus a few
image effects on the transition frames, so it costs next to nothing to render
and the preview shows exactly what the export will do.

Moves:      handheld drift, slow push-in per phrase, crash zoom on strong
            words / loud peaks, impact shake on peaks.
Transitions (on shot changes and jump cuts): whip pan, zoom-through, glitch.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

STRONG = {
    "mai", "nessuno", "sempre", "tutto", "niente", "incredibile", "pazzesco", "assurdo", "segreto",
    "attenzione", "errore", "verità", "importante", "subito", "basta", "davvero", "enorme", "gratis",
    "never", "nobody", "always", "everything", "nothing", "crazy", "insane", "secret", "huge", "free",
}


@dataclass
class MotionSettings:
    handheld: bool = False
    handheld_amount: float = 0.5
    push: bool = False
    push_amount: float = 0.5
    crash: bool = False
    crash_amount: float = 0.5
    shake: bool = False
    shake_amount: float = 0.5
    transition: str = "off"
    """``off`` | ``whip`` | ``zoom`` | ``glitch`` | ``random``."""
    transition_every: str = "all"
    """``all`` cuts, ``alternate`` (every other one) or ``shots`` (camera changes only)."""
    min_gap: float = 1.5
    """Seconds between two punctual effects (crash, shake, transitions)."""
    beat_sync: bool = False
    """Snap crash/shake triggers to the nearest music beat (see ``beats``)."""

    @property
    def active(self) -> bool:
        return self.handheld or self.push or self.crash or self.shake or self.transition != "off"

    @property
    def needs_audio(self) -> bool:
        return self.push or self.crash or self.shake


@dataclass
class MotionCurves:
    dx: np.ndarray       # crop offset, fraction of the crop width
    dy: np.ndarray       # crop offset, fraction of the crop height
    zoom: np.ndarray     # extra zoom (>= 1)
    rot: np.ndarray      # degrees
    whip: np.ndarray     # signed motion-blur strength (-1..1, sign = direction)
    zblur: np.ndarray    # radial blur strength (0..1)
    glitch: np.ndarray   # 0..1

    @classmethod
    def empty(cls, n: int) -> "MotionCurves":
        z = np.zeros(n)
        return cls(z.copy(), z.copy(), np.ones(n), z.copy(), z.copy(), z.copy(), z.copy())


Cut = Tuple[int, int]
"""(last frame + 1 of the outgoing piece, first frame of the incoming piece).
Equal on a finished timeline; different in the source-timeline preview,
where a jump cut skips the removed range [out, in)."""


# ------------------------------------------------------------------ triggers

def loud_peaks(pcm: Optional[np.ndarray], sr: int = 16000, min_gap: float = 1.5) -> List[float]:
    """Seconds where the voice jumps well above its recent level."""
    if pcm is None or len(pcm) < sr:
        return []
    win = sr // 20  # 50 ms
    n = len(pcm) // win
    db = 20 * np.log10(np.sqrt(np.mean(pcm[:n * win].reshape(n, win) ** 2, axis=1)) + 1e-9)
    if db.max() < -45:
        return []
    k = 20  # 1 s of context
    base = np.convolve(db, np.ones(k) / k, mode="full")[:n]  # trailing mean
    thr = max(np.percentile(db, 92), np.percentile(db, 50) + 6)
    out, last = [], -1e9
    for i in range(k, n - 1):
        if db[i] >= thr and db[i] - base[i] > 7 and db[i] >= db[i + 1] and i * 0.05 - last >= min_gap:
            out.append(i * 0.05)
            last = i * 0.05
    return out


def strong_words(words: Optional[Sequence]) -> List[float]:
    """Start times of punchy words: questions, exclamations, numbers, STRONG."""
    out = []
    for w in words or []:
        t = w.text.strip()
        n = re.sub(r"[^\wàèéìòù]+", "", t.lower())
        if n in STRONG or t.endswith(("!", "?")) or re.search(r"\d", t):
            out.append(float(w.start))
    return out


def _thin(times: Sequence[float], gap: float) -> List[float]:
    out: List[float] = []
    for t in sorted(times):
        if not out or t - out[-1] >= gap:
            out.append(t)
    return out


def beats(pcm: Optional[np.ndarray], sr: int = 16000, min_gap: float = 0.28) -> List[float]:
    """Music-beat times via spectral-flux onset detection (numpy only):
    strong transients (kick, snare, strum) above the local energy trend."""
    if pcm is None or len(pcm) < sr:
        return []
    win, hop = 1024, 512
    n = 1 + (len(pcm) - win) // hop
    if n < 8:
        return []
    idx = np.arange(win)
    S = np.stack([pcm[i * hop + idx] for i in range(n)]).astype(np.float32) \
        * np.hanning(win).astype(np.float32)
    mag = np.abs(np.fft.rfft(S, axis=1))
    flux = np.maximum(mag[1:] - mag[:-1], 0).sum(axis=1)
    if flux.max() <= 0:
        return []
    k = max(4, int(round(sr / hop * 1.0)))              # ~1 s of context
    loc = np.convolve(flux, np.ones(k) / k, mode="same")
    thr = np.percentile(flux, 75)
    out, last = [], -1e9
    for i in range(1, len(flux) - 1):
        t = i * hop / sr
        if (flux[i] > thr and flux[i] > loc[i] * 1.3
                and flux[i] >= flux[i - 1] and flux[i] >= flux[i + 1]
                and t - last >= min_gap):
            out.append(t)
            last = t
    return out


def _snap_to_beat(times: Sequence[float], beat_times: Sequence[float],
                  tol: float = 0.14) -> List[float]:
    """Pull each trigger to the nearest beat inside ``tol``; drop the rest —
    when beat-sync is on, effects should land *on* the beat or not fire."""
    if not beat_times:
        return list(times)
    out = []
    for t in sorted(times):
        k = int(np.searchsorted(beat_times, t))
        cand = [b for b in beat_times[max(0, k - 1):k + 1] if abs(b - t) <= tol]
        if cand:
            out.append(min(cand, key=lambda b: abs(b - t)))
    return sorted(set(out))


# ------------------------------------------------------------------ builder

def _smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def _noise(n: int, fps: float, rng, freqs=(0.23, 0.41, 0.77, 1.13)) -> np.ndarray:
    """Smooth pseudo-random wobble in [-1, 1] (sum of incommensurate sines)."""
    t = np.arange(n) / fps
    v = np.zeros(n)
    for k, f in enumerate(freqs):
        v += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi)) / (k + 1)
    return v / sum(1.0 / (k + 1) for k in range(len(freqs)))


def build_motion(n: int, fps: float, m: MotionSettings, cuts: Sequence[Cut] = (),
                 shot_cuts: Sequence[int] = (), phrases: Sequence[Tuple[float, float]] = (),
                 words: Optional[Sequence] = None, pcm: Optional[np.ndarray] = None,
                 seed: int = 0, skip: Sequence[Tuple[int, int]] = (),
                 beat_times: Sequence[float] = ()) -> MotionCurves:
    """Per-frame curves for ``n`` frames.

    ``cuts`` are all edit points (jump cuts + shot changes), ``shot_cuts`` the
    frames among them that are real camera changes. ``skip`` are frame ranges
    that won't be shown (preview of removed parts): no effect starts there.
    """
    c = MotionCurves.empty(n)
    if n == 0 or not m.active:
        return c
    rng = np.random.default_rng(seed)
    fr = lambda s: int(round(s * fps))  # noqa: E731

    def hidden(f: int) -> bool:
        return any(a <= f < b for a, b in skip)

    if m.handheld:
        a = 0.35 + 1.3 * m.handheld_amount
        c.dx += 0.010 * a * _noise(n, fps, rng)
        c.dy += 0.008 * a * _noise(n, fps, rng)
        c.rot += 0.45 * a * _noise(n, fps, rng, (0.17, 0.31, 0.53))
        c.zoom *= 1.0 + 0.018 * a  # margin so the drift never shows an edge

    if m.push and phrases:
        top = 0.03 + 0.09 * m.push_amount
        ease = max(1, fr(0.35))
        z = np.ones(n)
        for a_s, b_s in phrases:
            a, b = max(0, fr(a_s)), min(n, fr(b_s))
            if b - a < 3:
                continue
            z[a:b] = 1.0 + top * (np.arange(b - a) / float(b - a))
            # ease back to 1 after the phrase instead of popping
            e = min(n, b + ease)
            if e > b:
                z[b:e] = np.maximum(z[b:e], 1.0 + top * (1 - _smoothstep(np.arange(e - b) / float(ease))))
        c.zoom *= z

    punctual: List[float] = []  # times already used by an effect (min_gap)

    def free(t: float) -> bool:
        return all(abs(t - u) >= m.min_gap for u in punctual)

    def sync(trig: List[float]) -> List[float]:
        if not (m.beat_sync and beat_times):
            return trig
        merged = list(trig) + list(beat_times)   # beats themselves trigger too
        return _snap_to_beat(merged, beat_times)

    if m.crash:
        trig = sync(strong_words(words) or loud_peaks(pcm, min_gap=m.min_gap))
        top = 0.12 + 0.28 * m.crash_amount
        z = np.ones(n)
        att, hold, rel = max(2, fr(0.12)), fr(0.6), max(3, fr(0.3))
        for t in _thin(trig, max(m.min_gap, 2.0)):
            f = fr(t)
            if f >= n or hidden(f) or not free(t):
                continue
            punctual.append(t)
            seg = np.concatenate([1 + top * _smoothstep(np.arange(1, att + 1) / att),
                                  np.full(hold, 1 + top),
                                  1 + top * (1 - _smoothstep(np.arange(1, rel + 1) / rel))])
            e = min(n, f + len(seg))
            z[f:e] = np.maximum(z[f:e], seg[:e - f])
        c.zoom *= z

    if m.shake:
        dur = fr(0.4)
        amp = 0.012 + 0.03 * m.shake_amount
        for t in sync(loud_peaks(pcm, min_gap=m.min_gap)):
            f = fr(t)
            if f >= n or hidden(f) or not free(t):
                continue
            punctual.append(t)
            k = np.arange(min(dur, n - f))
            env = np.exp(-k / max(1.0, dur / 4.0))
            ph = rng.uniform(0, 2 * np.pi)
            c.dx[f:f + len(k)] += amp * env * np.sin(2 * np.pi * 11 * k / fps + ph)
            c.dy[f:f + len(k)] += 0.7 * amp * env * np.sin(2 * np.pi * 13 * k / fps + ph * 1.7)
            c.rot[f:f + len(k)] += 1.6 * m.shake_amount * env * np.sin(2 * np.pi * 9 * k / fps + ph)
            c.zoom[f:f + len(k)] *= 1.0 + 1.2 * amp

    if m.transition != "off" and cuts:
        shots = set(shot_cuts)
        styles = ["whip", "zoom", "glitch"]
        k = 0
        for out_f, in_f in sorted(cuts):
            if m.transition_every == "shots" and in_f not in shots and out_f not in shots:
                continue
            k += 1
            if m.transition_every == "alternate" and k % 2 == 0:
                continue
            t = in_f / fps
            if not free(t):
                continue
            punctual.append(t)
            style = styles[int(rng.integers(3))] if m.transition == "random" else m.transition
            _transition(c, style, out_f, in_f, n, fps, rng)
    return c


def _transition(c: MotionCurves, style: str, out_f: int, in_f: int, n: int, fps: float, rng) -> None:
    half = max(3, int(round(0.14 * fps)))  # ~3.5 frames each side at 25 fps
    before = np.arange(max(0, out_f - half), out_f)
    after = np.arange(in_f, min(n, in_f + half))
    tb = (before - (out_f - half)) / float(half)       # 0 -> 1 into the cut
    ta = 1.0 - (after - in_f) / float(half)            # 1 -> 0 out of the cut
    if style == "whip":
        d = 1.0 if rng.random() < 0.5 else -1.0
        c.dx[before] += d * 0.45 * tb ** 2
        c.dx[after] -= d * 0.45 * ta ** 2
        c.whip[before] = d * np.maximum(c.whip[before] * d, tb)
        c.whip[after] = d * np.maximum(c.whip[after] * d, ta)
    elif style == "zoom":
        c.zoom[before] *= 1.0 + 0.45 * tb ** 2
        c.zoom[after] *= 1.0 + 0.45 * ta ** 2
        c.zblur[before] = np.maximum(c.zblur[before], tb)
        c.zblur[after] = np.maximum(c.zblur[after], ta)
    elif style == "glitch":
        g_b = np.arange(max(0, out_f - 2), out_f)
        g_a = np.arange(in_f, min(n, in_f + 3))
        for idx, val in ((g_b, 0.7), (g_a, 1.0)):
            c.glitch[idx] = np.maximum(c.glitch[idx], val)
            c.dx[idx] += rng.uniform(-0.03, 0.03, len(idx))


# ------------------------------------------------------------------ image effects

def apply_effects(img: np.ndarray, c: MotionCurves, i: int, rng_seed: int = 0) -> np.ndarray:
    """Rotation, whip blur, zoom blur, glitch on one output frame."""
    import cv2
    if i >= len(c.zoom):
        return img
    h, w = img.shape[:2]
    rot = float(c.rot[i])
    if abs(rot) > 0.01:
        th = np.radians(abs(rot))
        scale = np.cos(th) + (max(w, h) / float(min(w, h))) * np.sin(th)  # cover the corners
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), rot, scale)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    s = float(c.whip[i])
    if abs(s) > 0.05:
        k = max(3, int(abs(s) * w * 0.09)) | 1
        kernel = np.zeros((1, k), np.float32)
        kernel[0, :] = 1.0 / k
        img = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)
    zb = float(c.zblur[i])
    if zb > 0.05:
        acc = img.astype(np.float32)
        steps = 4
        for j in range(1, steps + 1):
            sc = 1.0 + 0.035 * zb * j
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), 0, sc)
            acc += cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        img = (acc / (steps + 1)).astype(np.uint8)
    g = float(c.glitch[i])
    if g > 0.05:
        rng = np.random.default_rng(rng_seed + i)
        sh = max(2, int(w * 0.012 * g))
        out = img.copy()
        out[:, :, 2] = np.roll(img[:, :, 2], sh, axis=1)
        out[:, :, 0] = np.roll(img[:, :, 0], -sh, axis=1)
        for _ in range(int(3 + 5 * g)):  # displaced horizontal slices
            y0 = int(rng.integers(0, h - 8))
            hh = int(rng.integers(4, max(6, h // 25)))
            out[y0:y0 + hh] = np.roll(out[y0:y0 + hh], int(rng.integers(-3 * sh, 3 * sh)), axis=1)
        img = out
    return img


def output_cuts(spans: Optional[Sequence[Tuple[float, float]]], segments: Sequence[Tuple[int, int]],
                fps: float) -> Tuple[List[Cut], List[int]]:
    """Edit points of a finished (cut) timeline: jump cuts + shot changes."""
    shot = [a for a, _ in segments[1:]]
    pts = set(shot)
    if spans:
        acc = 0.0
        for a, b in spans[:-1]:
            acc += b - a
            pts.add(int(round(acc * fps)))
    merged: List[int] = []
    for p in sorted(pts):  # a jump cut and a shot change 2 frames apart are one edit
        if not merged or p - merged[-1] > 2:
            merged.append(p)
    return [(p, p) for p in merged], shot
