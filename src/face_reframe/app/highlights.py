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


# Opening on a discourse connector means the thought started earlier — the
# clip would feel like it begins mid-sentence.
CONTINUATION = {
    # it
    "e", "ed", "o", "ma", "però", "pero", "allora", "quindi", "infatti",
    "invece", "perché", "perche", "anche", "comunque", "cioè", "cioe",
    "dunque", "eppure", "mentre",
    # en
    "and", "but", "so", "then", "because", "also", "though", "while", "or",
    "yet", "plus",
}
# A bare demonstrative/subject pronoun opener usually refers to the sentence
# before the clip — the viewer is missing its referent.
ANAPHORA = {
    # it
    "questo", "questa", "questi", "queste", "lui", "lei", "loro", "ciò", "cio",
    "esso", "essa", "essi", "esse",
    # en
    "this", "that", "these", "those", "it", "he", "she", "they",
}
# A clip can't end on a function word — that would be a fragment.
TAIL_FUNC = {
    # it
    "il", "lo", "la", "i", "gli", "le", "un", "una", "uno", "di", "del",
    "della", "dei", "degli", "delle", "a", "da", "in", "con", "su", "per",
    "tra", "fra", "e", "che", "nel", "nella", "sul", "alla", "al",
    # en
    "the", "a", "an", "of", "to", "and", "in", "on", "for", "with", "that",
    "which",
}


def _sentences(words, gap: float):
    """Group words into (i0, i1, text) sentences on terminal punctuation or a
    pause — the units the optional sentence embedder scores."""
    sents, s = [], 0
    for i, x in enumerate(words):
        nxt = words[i + 1] if i + 1 < len(words) else None
        if _ends_sentence(x.text) or (nxt is not None and nxt.start - x.end >= gap):
            sents.append((s, i, " ".join(y.text.strip() for y in words[s:i + 1])))
            s = i + 1
    if s < len(words):
        sents.append((s, len(words) - 1, " ".join(y.text.strip() for y in words[s:])))
    return sents


def find_highlights(words: Sequence, duration: float, count: int = 5,
                    min_len: float = 30.0, max_len: float = 60.0,
                    gap: float = 0.6, events=None, embedder=None) -> List[Highlight]:
    """Top ``count`` non-overlapping clips, best first.

    ``events`` is an :class:`audioevents.Events` (YAMNet) — laughter and
    applause inside a clip raise its score, as an extra on top of the
    transcript heuristics (they keep working without it).

    ``embedder`` — optional callable mapping a list of sentence strings to an
    L2-normalised (n, d) embedding matrix (or None when unavailable).  It adds
    two semantic penalties: a clip whose first sentence closely resembles the
    one just before it starts mid-discussion, and a clip containing an abrupt
    similarity drop between consecutive sentences switches topic halfway."""
    w = list(words)
    if len(w) < 10:
        return []
    starts = [0] + [i for i in range(1, len(w))
                    if w[i].start - w[i - 1].end >= gap or _ends_sentence(w[i - 1].text)]
    ends = [i for i in range(len(w) - 1)
            if w[i + 1].start - w[i].end >= gap * 0.8 or _ends_sentence(w[i].text)] + [len(w) - 1]
    # optional semantic layer — embed the video's sentences once, share them
    # across every candidate clip
    emb = wsent = None
    if embedder is not None:
        sents = _sentences(w, gap)
        try:
            emb = embedder([s[2] for s in sents])
        except Exception:
            emb = None
        if emb is not None and len(emb) == len(sents):
            import numpy as np
            wsent = np.zeros(len(w), dtype=int)
            for k, (a, b, _t) in enumerate(sents):
                wsent[a:b + 1] = k
        else:
            emb = None
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
            # a "?" is a hook only when enough words follow it inside the clip
            # — an unanswered question ending the clip hooks nobody
            qidx = max((i for i, x in enumerate(opening) if "?" in x.text), default=-1)
            hook += 2 if 0 <= qidx < n - 8 else (0.5 if qidx >= 0 else 0)
            hook += 1 if any(re.search(r"\d", x.text) for x in opening) else 0
            complete = 1.0 if _ends_sentence(seg[-1].text) else 0.0
            first, last = _norm(seg[0].text), _norm(seg[-1].text)
            edge = (-0.6 if first in CONTINUATION else 0.0) \
                + (-0.35 if first in ANAPHORA else 0.0) \
                + (-0.7 if last in TAIL_FUNC else 0.0)
            score = (min(density, 3.5) / 3.5) * 2.0 + talk_ratio * 1.5 + min(hook, 4) * 0.35 \
                + complete * 0.5 - fillers * 3.0 + edge
            if emb is not None:
                s0, s1 = int(wsent[si]), int(wsent[ei])
                if s0 > 0:  # too similar to the previous sentence = mid-thought start
                    score -= max(0.0, float((emb[s0] * emb[s0 - 1]).sum()) - 0.55) * 2.5
                if s1 > s0:  # abrupt similarity drop inside = topic switch
                    cs = emb[s0:s1 + 1]
                    ms = float((cs[:-1] * cs[1:]).sum(1).min())
                    score -= max(0.0, 0.50 - ms) * 2.0
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
