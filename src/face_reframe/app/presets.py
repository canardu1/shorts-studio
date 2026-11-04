"""Built-in and user presets: a preset is a dict of setting key -> value
(combo boxes by their data value), applied on top of the current settings."""

from __future__ import annotations

import json
from typing import Dict

BUILTIN: Dict[str, dict] = {
    "Interview": dict(
        subject="main", layout="single", smoothing=0.6, dead_zone=0.05, follow_vertical=False,
        punch=False, cut_pauses=True, min_pause=0.4, cut_breaths=False, cut_fillers=True,
        cut_retakes=False, cinematic=True, enhance="off", restore="gpen256", restore_blend=0.8,
        subtitles="burn", caption_style="word",
    ),
    "2-person podcast": dict(
        subject="speaker", layout="single", smoothing=0.5, dead_zone=0.06, punch=True,
        cut_pauses=True, min_pause=0.6, cut_breaths=False, cut_fillers=True, cut_retakes=False,
        cinematic=True, enhance="fast", subtitles="burn", caption_style="word",
        mv_handheld=False, mv_push=True, mv_push_amount=0.4, mv_crash=True, mv_crash_amount=0.4,
        mv_shake=False, mv_transition="off",
    ),
    "Split-screen podcast": dict(
        subject="main", layout="split", smoothing=0.5, punch=False, cut_pauses=True,
        min_pause=0.6, cut_fillers=True, cinematic=True, enhance="fast", subtitles="burn",
        caption_style="word", caption_pos=0.5,
    ),
    "Vlog": dict(
        subject="main", layout="single", smoothing=0.4, follow_vertical=True, punch=True,
        cut_pauses=True, min_pause=0.3, cut_breaths=True, cut_fillers=True, cut_retakes=True,
        cinematic=True, enhance="fast", subtitles="burn", caption_style="word",
        mv_handheld=True, mv_handheld_amount=0.4, mv_push=False, mv_crash=True, mv_crash_amount=0.5,
        mv_shake=True, mv_shake_amount=0.4, mv_transition="random", mv_every="alternate", mv_gap=2.0,
    ),
    "Slow PC": dict(
        resolution=1280, preset="veryfast", stride=3, enhance="off", cinematic=False, restore="off",
    ),
}


# Caption-only style templates: same dict shape (key -> value) but limited to
# the caption controls; the font family travels under "caption_font".
CAPTION_KEYS = ("caption_style", "caption_words", "caption_max_dur", "caption_pos",
                "caption_color", "caption_hl", "caption_upper", "caption_size",
                "caption_box", "caption_anim")

CAPTION_BUILTIN: Dict[str, dict] = {
    "Default": dict(caption_style="word", caption_words=3, caption_max_dur=1.6,
                    caption_pos=0.7, caption_color="#FFFFFF", caption_hl="#FFD400",
                    caption_upper=True, caption_size=0.075, caption_box="none",
                    caption_anim="off", caption_font="Arial Black"),
    "Hormozi": dict(caption_style="word", caption_words=3, caption_max_dur=1.4,
                    caption_pos=0.78, caption_color="#FFFFFF", caption_hl="#7CFF6B",
                    caption_upper=True, caption_size=0.1, caption_box="none",
                    caption_anim="pop", caption_font="Arial Black"),
    "Minimal": dict(caption_style="line", caption_words=6, caption_max_dur=2.2,
                    caption_pos=0.72, caption_color="#FFFFFF", caption_hl="#4FD8FF",
                    caption_upper=False, caption_size=0.05, caption_box="pill",
                    caption_anim="fade", caption_font="Helvetica"),
    "Cinematic": dict(caption_style="line", caption_words=7, caption_max_dur=2.5,
                      caption_pos=0.88, caption_color="#FFFFFF", caption_hl="#FFD400",
                      caption_upper=False, caption_size=0.048, caption_box="none",
                      caption_anim="fade", caption_font="Georgia"),
}


def load_user(raw: str) -> Dict[str, dict]:
    try:
        data = json.loads(raw or "{}")
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except ValueError:
        return {}


def dump_user(presets: Dict[str, dict]) -> str:
    return json.dumps(presets, ensure_ascii=False)
