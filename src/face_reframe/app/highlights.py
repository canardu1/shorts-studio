"""Find the best 30-60 s moments of a long video for shorts.

Works on the word-level transcript: a candidate clip starts at the start of
a sentence and ends at the end of one; it scores on how dense and lively the
speech is, whether it opens with a hook (a question, a strong word, a
number) and whether it ends on a complete sentence."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Sequence

HOOKS = {
    # it
    "perché", "come", "segreto", "errore", "errori", "mai", "nessuno", "sempre", "attenzione",
    "incredibile", "verità", "problema", "soluzione", "trucco", "consiglio", "importante",
    "devi", "dovete", "smetti", "ecco", "immagina", "pensate", "pensa", "quanto", "cosa",
    # en
    "why", "how", "secret", "mistake", "never", "nobody", "always", "truth", "problem",
    "trick", "tip", "important", "imagine", "what",
}
FILLER = {"ehm", "uhm", "um", "uh", "eh", "ah", "mm", "hmm", "em", "emm"}


@dataclass
class Highlight:
    start: float
    end: float
    score: float
    title: str


def _norm(t: str) -> str:
    return re.sub(r"[^\wàèéìòù]+", "", t.lower())


def _ends_sentence(t: str) -> bool:
    return t.strip().endswith((".", "?", "!", "…"))


def find_highlights(words: Sequence, duration: float, count: int = 5,
                    min_len: float = 30.0, max_len: float = 60.0,
                    gap: float = 0.6, events=None) -> List[Highlight]:
    """Top ``count`` non-overlapping clips, best first.

    ``events`` is an :class:`audioevents.Events` (YAMNet) — laughter and
    applause inside a clip raise its score, as an extra on top of the
    transcript heuristics (they keep working without it)."""
    w = list(words)
    if len(w) < 10:
        return []
    starts = [0] + [i for i in range(1, len(w))
                    if w[i].start - w[i - 1].end >= gap or _ends_sentence(w[i - 1].text)]
    ends = [i for i in range(len(w) - 1)
            if w[i + 1].start - w[i].end >= gap * 0.8 or _ends_sentence(w[i].text)] + [len(w) - 1]
    cands = []
    for si in starts:
        t0 = max(0.0, w[si].start - 0.15)
        for ei in ends:
            if ei <= si:
                continue
            t1 = min(duration, w[ei].end + 0.25)
            length = t1 - t0
            if length < min_len:
                continue
            if length > max_len:
                break
            seg = w[si:ei + 1]
            n = len(seg)
            spoken = sum(x.end - x.start for x in seg)
            density = n / length                      # words per second
            talk_ratio = spoken / length              # how little dead air
            fillers = sum(1 for x in seg if _norm(x.text) in FILLER) / n
            opening = seg[:min(n, 14)]
            hook = sum(1 for x in opening if _norm(x.text) in HOOKS)
            hook += 2 if any("?" in x.text for x in opening) else 0
            hook += 1 if any(re.search(r"\d", x.text) for x in opening) else 0
            complete = 1.0 if _ends_sentence(seg[-1].text) else 0.0
            score = (min(density, 3.5) / 3.5) * 2.0 + talk_ratio * 1.5 + min(hook, 4) * 0.35 \
                + complete * 0.5 - fillers * 3.0
            # mild preference for ~45 s
            score -= abs(length - 0.5 * (min_len + max_len)) / max_len * 0.3
            if events is not None:
                score += events.boost(t0, t1)
            title = " ".join(x.text.strip() for x in seg[:8]).strip()
            cands.append(Highlight(t0, t1, score, title))
    cands.sort(key=lambda h: -h.score)
    picked: List[Highlight] = []
    for h in cands:
        if all(h.end <= p.start or h.start >= p.end for p in picked):
            picked.append(h)
        if len(picked) >= count:
            break
    return picked
