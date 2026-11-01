"""Editing the transcript behind the captions (Whisper mishears names,
jargon, dialect...). Qt-free: the dialog in ``subedit_qt`` calls these.

Captions are groups of words; editing a line replaces that group's words
with the new text, spreading the group's time span over the new words in
proportion to their length, so word-by-word highlighting stays in sync.
"""

from __future__ import annotations

import re
from typing import List, Sequence, Tuple

from ..transcribe import Word


def lines(words: Sequence[Word], max_words: int, max_dur: float):
    """Caption lines as ``(start, end, text, first_word_index, n_words)``."""
    from .jobs import group_word_events
    out, i = [], 0
    for a, b, grp in group_word_events(list(words), max_words, max_dur):
        out.append((a, b, " ".join(w.text.strip() for w in grp), i, len(grp)))
        i += len(grp)
    return out


def retime(tokens: Sequence[str], start: float, end: float) -> List[Word]:
    """Words for ``tokens`` spread over [start, end] by character length."""
    if not tokens:
        return []
    weights = [len(t) + 1 for t in tokens]
    total = float(sum(weights))
    span = max(0.05 * len(tokens), end - start)
    out, t = [], start
    for tok, wgt in zip(tokens, weights):
        d = span * wgt / total
        out.append(Word(t, t + d * 0.92, tok))   # a hair of gap between words
        t += d
    return out


def edit_line(words: Sequence[Word], first: int, count: int, text: str) -> List[Word]:
    """Replace ``words[first:first+count]`` with ``text`` (empty = delete the line)."""
    words = list(words)
    grp = words[first:first + count]
    if not grp:
        return words
    new = retime(text.split(), grp[0].start, grp[-1].end)
    return words[:first] + new + words[first + count:]


def replace_all(words: Sequence[Word], find: str, repl: str, max_words: int, max_dur: float,
                whole_word: bool = True) -> Tuple[List[Word], int]:
    """Find/replace in every caption line (case-insensitive); returns (words, lines changed)."""
    if not find.strip():
        return list(words), 0
    pat = re.escape(find.strip())
    if whole_word:
        pat = r"(?<!\w)" + pat + r"(?!\w)"
    rx = re.compile(pat, re.IGNORECASE)
    words = list(words)
    changed = 0
    # edit from the end so earlier word indices stay valid
    for a, b, text, first, n in reversed(lines(words, max_words, max_dur)):
        if rx.search(text):
            words = edit_line(words, first, n, rx.sub(repl, text))
            changed += 1
    return words, changed
