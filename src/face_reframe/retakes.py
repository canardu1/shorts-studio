"""Retake detection: phrases repeated in the transcript (same line, new take).

A failed take is usually said again right away, sometimes with a comment in
between ("ok, again"). The keeper is the LAST take, so every earlier similar
block plus everything up to the keeper's start becomes a cut span.
"""

from __future__ import annotations

import difflib
import re
from typing import List, Tuple

from .transcribe import Word


def _norm(text: str) -> str:
    return re.sub(r"[^\w'àèéìòù]+", "", text.lower())


def utterances(words: List[Word], gap: float = 1.2) -> List[Tuple[float, float, List[str]]]:
    """Words grouped into spoken blocks [(start_s, end_s, [normalized words])]
    split wherever the pause between words exceeds ``gap`` seconds."""
    blocks = []
    prev_end = None
    for w in words:
        if prev_end is None or w.start - prev_end > gap:
            blocks.append((w.start, w.end, []))
        blocks[-1][2].append(_norm(w.text))
        prev_end = w.end
    return blocks


def _similar(a: Tuple, b: Tuple) -> float:
    return difflib.SequenceMatcher(None, a[2], b[2]).ratio() if a[2] and b[2] else 0.0


def retake_spans(words: List[Word], min_words: int = 4, similarity: float = 0.65,
                 lookahead: int = 3) -> List[Tuple[float, float]]:
    """(start, end) second ranges to cut: earlier takes of repeated phrases.

    A block counts as a retake when a later block is similar; short blocks
    in between ("ok", "dai") ride along — anything else breaks the chain so
    unrelated content is never cut.
    """
    blocks = utterances(words)
    holes, i = [], 0
    while i < len(blocks):
        bi = blocks[i]
        hit = False
        if len(bi[2]) >= min_words:
            for j in range(i + 1, min(i + lookahead + 1, len(blocks))):
                bj = blocks[j]
                if len(bj[2]) < min_words or _similar(bi, bj) < similarity:
                    continue
                middle_ok = all(
                    len(blocks[k][2]) < min_words or _similar(blocks[k], bj) >= similarity
                    for k in range(i + 1, j))
                if middle_ok:
                    holes.append((bi[0], bj[0]))
                    i = j
                    hit = True
                    break
        if not hit:
            i += 1
    return holes
