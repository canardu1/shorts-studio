"""Audio events with YAMNet (AudioSet, 521 classes) on 16 kHz mono PCM.

Used to enrich "Find the best moments": laughter, applause and music inside
a candidate clip are real engagement signals the transcript can't see.
The 16 MB ONNX model downloads on first use; the class list ships in
``models/yamnet_class_map.csv``.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from . import models
from .i18n import tr

CLASS_CSV = os.path.join(os.path.dirname(__file__), "models", "yamnet_class_map.csv")
SR = 16000
WINDOW = int(0.975 * SR)      # YAMNet's patch: 0.975 s
HOP = int(0.48 * SR)          # ~2 scores a second is plenty

def class_names() -> List[str]:
    with open(CLASS_CSV, newline="", encoding="utf-8") as fh:
        return [r["display_name"] for r in csv.DictReader(fh)]


def group_indices() -> Dict[str, List[int]]:
    """Resolve the wanted classes by name from the bundled CSV (robust to
    index drift between YAMNet variants)."""
    names = [n.lower() for n in class_names()]
    want = {
        "laughter": ["laughter", "baby laughter", "giggle", "snicker", "belly laugh", "chuckle, chortle"],
        "applause": ["applause"],
        "music": ["music"],
        "cheer": ["cheering", "crowd"],
    }
    out = {}
    for g, pats in want.items():
        out[g] = sorted({i for i, n in enumerate(names) if n in pats})
    return out


@dataclass
class Events:
    """Per-window max scores for the interesting classes."""
    times: np.ndarray            # centre time (s) of each scored window
    scores: Dict[str, np.ndarray]
    by: str = "yamnet"

    def boost(self, start: float, end: float) -> float:
        """Engagement bonus for a clip spanning [start, end): laughter and
        applause weigh most, music a little."""
        if len(self.times) == 0:
            return 0.0
        m = (self.times >= start) & (self.times < end)
        if not m.any():
            return 0.0
        laugh = float(self.scores["laughter"][m].max()) if "laughter" in self.scores else 0.0
        appl = float(self.scores["applause"][m].max()) if "applause" in self.scores else 0.0
        cheer = float(self.scores["cheer"][m].max()) if "cheer" in self.scores else 0.0
        music = float(self.scores["music"][m].mean()) if "music" in self.scores else 0.0
        return 0.9 * max(laugh, 0.8 * cheer, appl) + 0.15 * min(1.0, music)


class YAMNet:
    """Callable: ``pcm16k -> scores [n_windows, 521]``."""

    def __init__(self):
        self.sess = models.session("yamnet")
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, pcm: np.ndarray) -> np.ndarray:
        pcm = np.asarray(pcm, np.float32)
        if len(pcm) < WINDOW:
            pcm = np.pad(pcm, (0, WINDOW - len(pcm)))
        starts = np.arange(0, len(pcm) - WINDOW + 1, HOP)
        outs = []
        for a in starts:
            y = self.sess.run(None, {self.inp: pcm[a:a + WINDOW][None]})[0]
            outs.append(np.asarray(y).reshape(-1, y.shape[-1])[-1])
        return np.stack(outs) if outs else np.zeros((0, 521), np.float32)


def analyze(pcm: np.ndarray, log=print, progress=None) -> Optional[Events]:
    """Event scores for the whole clip, or None when the model is missing."""
    if models.ensure("yamnet", progress, log) is None:
        return None
    try:
        y = YAMNet()(pcm)
    except Exception as exc:
        log(tr("Audio events unavailable (%s)") % exc)
        return None
    groups = group_indices()
    times = (np.arange(len(y)) * HOP + WINDOW / 2.0) / SR
    scores = {g: y[:, idx].max(axis=1) for g, idx in groups.items() if idx}
    return Events(times, scores)
