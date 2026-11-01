"""Speech-to-text via faster-whisper, shared by subtitles and filler cutting."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

MODEL_NAME = os.environ.get("FACE_REFRAME_WHISPER_MODEL", "small")

_model = None


def _model_cached(name: str) -> bool:
    """True when the model snapshot is already in the HuggingFace cache."""
    cache = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
    root = os.path.join(cache, "hub", "models--Systran--faster-whisper-%s" % name)
    snaps = os.path.join(root, "snapshots")
    return os.path.isdir(snaps) and any(os.listdir(snaps))


def get_model(name: Optional[str] = None, log=print):
    global _model
    if _model is None or (name and name != MODEL_NAME):
        from faster_whisper import WhisperModel
        want = name or MODEL_NAME
        if _model_cached(want):
            log("  whisper model '%s' already cached, loading..." % want)
        else:
            log("  downloading whisper model '%s' (~140 MB, one time only)..." % want)
        _model = WhisperModel(want, device="cpu", compute_type="int8")
    return _model


@dataclass
class Word:
    start: float
    end: float
    text: str


def transcribe(pcm: np.ndarray, language: Optional[str] = None,
               progress=None, log=print) -> Tuple[List[Word], str]:
    """Word-level transcription of 16 kHz mono float32 audio.

    Times are relative to the start of ``pcm``. Returns (words, language).
    ``progress`` is called with a 0-1 fraction every ~5%%.
    """
    model = get_model(log=log)
    segments, info = model.transcribe(pcm, language=language, word_timestamps=True,
                                    vad_filter=True, beam_size=1)
    duration = len(pcm) / 16000.0 if len(pcm) else 0.0
    words: List[Word] = []
    last_mark = -1
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(float(w.start), float(w.end), w.word))
        if progress and duration > 0:
            mark = int(min(1.0, seg.end / duration) * 20)
            if mark > last_mark:
                last_mark = mark
                progress(seg.end / duration)
    return words, getattr(info, "language", "") or (language or "")


FILLERS = {
    "ehm", "em", "emm", "eh", "uhm", "um", "uh", "er", "erm", "ah", "ahm",
    "hmm", "hm", "mh", "mhm", "mm",
}
# sub-word leading filler syllables worth cutting too
FILLER_PREFIXES = ("ehm", "uhm", "emm")


def is_filler(text: str) -> bool:
    return re.sub(r"[^\w'àèéìòù]", "", text.lower()) in FILLERS


def filler_spans(words: List[Word], pad: float = 0.05, max_dur: float = 1.5) -> List[Tuple[float, float]]:
    """Intervals to cut: isolated filler words only."""
    out = []
    for w in words:
        if is_filler(w.text) and w.end - w.start <= max_dur:
            out.append((max(0.0, w.start - pad), w.end + pad))
    return out


def group_captions(words: List[Word], max_words: int = 3, max_dur: float = 1.6) -> List[Tuple[float, float, str]]:
    """Pack words into short TikTok-style caption events."""
    events: List[Tuple[float, float, str]] = []
    i = 0
    while i < len(words):
        j, texts = i, []
        start = words[i].start
        while j < len(words) and len(texts) < max_words and (
                not texts or words[j].end - start <= max_dur):
            texts.append(words[j].text.strip())
            j += 1
        events.append((start, words[j - 1].end, " ".join(t for t in texts if t)))
        i = j
    return events


def _ts(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return "%02d:%02d:%02d,%03d" % (ms // 3600000, (ms % 3600000) // 60000, (ms % 60000) // 1000, ms % 1000)


def to_srt(events: List[Tuple[float, float, str]]) -> str:
    lines = []
    for i, (a, b, text) in enumerate(events, 1):
        lines += [str(i), "%s --> %s" % (_ts(a), _ts(b)), text, ""]
    return "\n".join(lines)
