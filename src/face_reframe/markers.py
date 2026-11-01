"""Timeline markers from the transcript: keyword hits and topic breaks."""

from __future__ import annotations

import re
from typing import List, Tuple

from .transcribe import Word


def _norm(text: str) -> str:
    return re.sub(r"[^\w'àèéìòù]+", "", text.lower())


def keyword_hits(words: List[Word], keywords: List[str]) -> List[Tuple[float, str]]:
    """[(seconds, keyword)] — each transcribed word matching a keyword.

    Keywords like "taglia" or "rifacciamo" are how editors mark a retake
    or a cut point while recording; matching is on the normalized word.
    """
    keys = {k.strip().lower() for k in keywords if k.strip()}
    return [(w.start, _norm(w.text)) for w in words if _norm(w.text) in keys]


def topic_marks(words: List[Word], min_pause: float = 4.0,
                title_words: int = 6) -> List[Tuple[float, str]]:
    """[(seconds, title)] — a marker where speech resumes after a gap of
    at least ``min_pause``; the title is the block's first words."""
    blocks = []  # [(start, [title words])]
    prev_end = None
    for w in words:
        if prev_end is None or w.start - prev_end >= min_pause:
            blocks.append((w.start, []))
        if len(blocks[-1][1]) < title_words:
            blocks[-1][1].append(w.text.strip())
        prev_end = w.end
    return [(t, " ".join(title)) for t, title in blocks]
