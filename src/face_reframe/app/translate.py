"""English subtitles with Whisper's built-in translation (offline, any
source language -> English), word-timed like the transcript."""

from __future__ import annotations

from typing import List, Optional


def translate_to_english(pcm, progress=None, log=print):
    from ..transcribe import Word, get_model
    model = get_model(log=log)
    segments, _info = model.transcribe(pcm, task="translate", word_timestamps=True,
                                       vad_filter=True, beam_size=1)
    dur = len(pcm) / 16000.0 if len(pcm) else 0.0
    words: List = []
    for seg in segments:
        for w in (seg.words or []):
            words.append(Word(float(w.start), float(w.end), w.word))
        if progress and dur > 0:
            progress(min(1.0, seg.end / dur))
    return words


def english_words(path: str, cache, progress=None, log=print, pcm=None):
    """Translated words for a whole file, cached on disk like the transcript."""
    from . import diskcache
    w = diskcache.load(path, "words_en", "translate")
    if w is not None:
        return w
    from ..silence import extract_audio
    if pcm is None:
        pcm = extract_audio(path)
    w = translate_to_english(pcm, progress, log)
    diskcache.save(path, "words_en", w, "translate")
    return w


def cached_english(path: str):
    from . import diskcache
    return diskcache.load(path, "words_en", "translate")


def overlap_text(words, a: float, b: float) -> str:
    """Words of another track overlapping [a, b) (by their midpoint)."""
    return " ".join(w.text.strip() for w in words if a <= 0.5 * (w.start + w.end) < b).strip()
