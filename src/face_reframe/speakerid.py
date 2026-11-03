"""Speaker voice-prints with ECAPA-TDNN (84 MB, downloaded on first use).

Each speech span gets a 192-d embedding; spans are clustered by cosine
similarity into speakers — so "whoever speaks" can cut on actual voice
changes instead of mouth motion alone, and speaker switches can be marked.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from . import models
from .i18n import tr

SR = 16000
MIN_SEG = 0.8        # shorter speech gives unreliable voice-prints
SAME = 0.45          # cosine similarity >= this ~ the same speaker


class SpeakerID:
    """``(batch of pcm spans) -> [n, 192] L2-normalised voice-prints``."""

    def __init__(self):
        self.sess = models.session("ecapa")

    def __call__(self, spans: Sequence[np.ndarray]) -> np.ndarray:
        n = max(1, max(len(s) for s in spans))
        wav = np.zeros((len(spans), n), np.float32)
        lens = np.zeros(len(spans), np.float32)
        for i, s in enumerate(spans):
            wav[i, :len(s)] = s
            lens[i] = min(1.0, len(s) / float(n))
        emb = self.sess.run(None, {"wav": wav, "lens": lens})[0][:, 0]
        return emb / np.maximum(1e-9, np.linalg.norm(emb, axis=1, keepdims=True))


def cluster(embs: np.ndarray, same: float = SAME) -> np.ndarray:
    """Greedy first-seen clustering by cosine similarity (deterministic)."""
    labels = np.full(len(embs), -1, int)
    cents: List[np.ndarray] = []
    for i, e in enumerate(embs):
        if not cents:
            cents.append(e.copy())
            labels[i] = 0
            continue
        sims = np.array([float(np.dot(e, c)) for c in cents])
        k = int(np.argmax(sims))
        if sims[k] >= same:
            labels[i] = k
            cents[k] = 0.9 * cents[k] + 0.1 * e          # slow drift with the voice
            cents[k] /= max(1e-9, float(np.linalg.norm(cents[k])))
        else:
            labels[i] = len(cents)
            cents.append(e.copy())
    return labels


def voice_segments(pcm: np.ndarray, speech: Sequence[Tuple[float, float]],
                   sid: SpeakerID, min_seg: float = MIN_SEG) -> List[Tuple[float, float, int]]:
    """``[(start_s, end_s, speaker_id)]`` — every speech span tagged by voice.

    Spans too short for a reliable print inherit the previous label."""
    spans = [(a, b, pcm[int(a * SR):int(b * SR)]) for a, b in speech if b - a >= 0.2]
    long_idx = [i for i, (a, b, s) in enumerate(spans) if len(s) >= min_seg * SR]
    labels = np.full(len(spans), -1, int)
    if long_idx:
        embs = sid([spans[i][2] for i in long_idx])
        labs = cluster(embs)
        for i, lab in zip(long_idx, labs):
            labels[i] = lab
    last = -1
    out = []
    for i, (a, b, _s) in enumerate(spans):
        lab = labels[i] if labels[i] >= 0 else last
        out.append((a, b, int(lab)))
        last = lab
    return out


def analyze(pcm: np.ndarray, speech: Sequence[Tuple[float, float]],
            log=print, progress=None) -> Optional[List[Tuple[float, float, int]]]:
    """Voice-tagged speech spans, or None when the model is unavailable."""
    if models.ensure("ecapa", progress, log) is None:
        return None
    try:
        out = voice_segments(pcm, speech, SpeakerID())
    except Exception as exc:
        log(tr("Voice identification unavailable (%s)") % exc)
        return None
    n = len({lab for _, _, lab in out if lab >= 0})
    log(tr("Voices heard: %d speaker%s") % (n, "s" if n != 1 else ""))
    return out
