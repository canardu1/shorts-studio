"""Title, description and hashtags for posting, drafted from the transcript
(offline: first sentence + keyword frequency; meant to be edited)."""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Optional, Sequence

STOP = set("""
a ad al alla alle allo ai agli anche ancora che chi ci come con cosa così cui da dal dalla dai dei del della
delle dello di dove e ed è era essere fa fare fra gli ha hai hanno ho i il in io la le lei li lo loro lui ma
me mi mia mio molto ne nel nella noi non nostro o per perché però più poi può quando quello questa questo
quindi se sei si sia siamo sono sta stato su sua suo sul sulla te ti tra tu tutto un una uno vi voi già
proprio cioè allora ecco ok va bene poco tanto tutti tutte qui lì là ehm uhm eh ah dice detto fatto
the a an and or of to in on for with is are was were be been it this that you i we they he she at by from
as but not so do does did have has had just like really very can will would there their what which who
""".split())


def keywords(words: Sequence, top: int = 8):
    toks = [re.sub(r"[^\wàèéìòù]+", "", w.text.lower()) for w in words or []]
    toks = [t for t in toks if len(t) > 3 and t not in STOP and not t.isdigit()]
    return [w for w, _ in Counter(toks).most_common(top)]


def make_texts(words: Sequence, hook: str = "", lang: Optional[str] = None) -> Dict[str, str]:
    from .overlays import first_sentence
    title = (hook or first_sentence(words, 12)).strip()
    if len(title) > 90:
        title = title[:88].rsplit(" ", 1)[0] + "…"
    sent, cur = [], []
    for w in words or []:
        cur.append(w.text.strip())
        if w.text.strip().endswith((".", "?", "!")):
            sent.append(" ".join(cur))
            cur = []
        if len(sent) >= 3:
            break
    if cur and len(sent) < 3:
        sent.append(" ".join(cur[:30]))
    en = (lang or "").startswith("en")
    cta = "Follow for more!" if en else "Seguimi per altri video!"
    desc = (" ".join(sent).strip() + "\n\n" + cta).strip()
    tags = ["#" + k for k in keywords(words, 6)] + (["#shorts", "#reels", "#fyp"] if en else
                                                     ["#shorts", "#reels", "#perte"])
    return {"title": title, "description": desc, "hashtags": " ".join(dict.fromkeys(tags))}


def to_text(t: Dict[str, str]) -> str:
    return "TITOLO\n%s\n\nDESCRIZIONE\n%s\n\nHASHTAG\n%s\n" % (t["title"], t["description"], t["hashtags"])
