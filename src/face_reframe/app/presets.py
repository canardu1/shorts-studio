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
    # ---- content-type presets ----
    "Gaming clip": dict(
        subject="main", noface="object", smoothing=0.4, punch=True, punch_zoom=1.2,
        cut_pauses=True, min_pause=0.35, cut_breaths=False, cut_fillers=True,
        cut_retakes=False, cinematic=False, enhance="fast", subtitles="burn",
        caption_style="word", caption_emoji=True, caption_sweep=True, hook="auto",
        mv_push=True, mv_crash=True, mv_crash_amount=0.5, mv_shake=True,
        mv_shake_amount=0.45, mv_transition="random", mv_every="alternate",
        beat_sync=True, progress_bar="top",
    ),
    "Tutorial / how-to": dict(
        subject="main", layout="single", smoothing=0.6, dead_zone=0.06, punch=False,
        cut_pauses=True, min_pause=0.5, cut_breaths=True, cut_fillers=True,
        cut_retakes=True, cinematic=True, enhance="fast", subtitles="burn",
        caption_style="line",
        mv_handheld=False, mv_push=False, mv_crash=False, mv_shake=False,
        mv_transition="off",
    ),
    "Course / webinar": dict(
        subject="main", layout="single", smoothing=0.7, punch=False,
        cut_pauses=True, min_pause=0.8, cut_breaths=False, cut_fillers=True,
        cut_retakes=False, cinematic=True, enhance="off", restore="off",
        subtitles="burn", caption_style="line", caption_box="pill",
        mv_handheld=False, mv_push=False, mv_crash=False, mv_shake=False,
        mv_transition="off", look="none",
    ),
    "Storytime": dict(
        subject="main", layout="single", smoothing=0.45, punch=True, punch_zoom=1.2,
        cut_pauses=True, min_pause=0.35, cut_breaths=True, cut_fillers=True,
        cut_retakes=True, cinematic=True, enhance="fast", subtitles="burn",
        caption_style="word", caption_emoji=True, caption_sweep=True, hook="auto",
        mv_push=True, mv_push_amount=0.4, mv_crash=True, mv_crash_amount=0.5,
        mv_shake=True, mv_shake_amount=0.35, mv_transition="zoom",
        mv_every="alternate", beat_sync=True, progress_bar="top",
    ),
    # pauses and breaths ARE the content — cut nothing, no loudness squashing
    "ASMR / soft voice": dict(
        subject="main", layout="single", smoothing=0.8, punch=False,
        cut_pauses=False, cut_breaths=False, cut_fillers=False, cut_retakes=False,
        cinematic=False, enhance="off", restore="off", subtitles="burn",
        caption_style="line", caption_anim="fade", caption_box="none",
        mv_handheld=False, mv_push=False, mv_crash=False, mv_shake=False,
        mv_transition="off", progress_bar="off",
    ),
    "Sports / action": dict(
        subject="main", noface="object", smoothing=0.35, punch=True, punch_zoom=1.25,
        cut_pauses=True, min_pause=0.35, cut_breaths=False, cut_fillers=True,
        cinematic=True, enhance="fast", stabilize=True, subtitles="burn",
        caption_style="word",
        mv_push=True, mv_crash=True, mv_crash_amount=0.6, mv_shake=True,
        mv_shake_amount=0.5, mv_transition="whip", mv_every="all", beat_sync=True,
    ),
    "B-roll / product": dict(
        subject="main", noface="object", smoothing=0.5, punch=False,
        cut_pauses=False, cut_breaths=False, cut_fillers=False, cut_retakes=False,
        cinematic=True, enhance="fast", stabilize=True, look="cine",
        subtitles="burn", caption_style="line", caption_box="pill",
        mv_push=True, mv_push_amount=0.3, mv_crash=False, mv_shake=False,
        mv_transition="off",
    ),
    # the restoration pack: deinterlace/VFR run automatically on ingest anyway
    "Archive restore": dict(
        subject="main", smoothing=0.6, punch=False,
        cut_pauses=False, cut_breaths=False, cut_fillers=False, cut_retakes=False,
        cinematic=True, denoise_engine="dfn", hum=True,
        enhance="ai", restore="gpen512", restore_blend=0.85, stabilize=True,
        lowlight="sg_llie", subtitles="srt",
        mv_handheld=False, mv_push=False, mv_crash=False, mv_shake=False,
        mv_transition="off",
    ),
    "Music beat edit": dict(
        subject="main", noface="object", smoothing=0.4, punch=False,
        cut_pauses=False, cut_breaths=False, cut_fillers=False, cut_retakes=False,
        cinematic=False, enhance="fast", subtitles="burn", caption_style="line",
        caption_anim="pop",
        mv_push=True, mv_push_amount=0.4, mv_crash=True, mv_crash_amount=0.5,
        mv_shake=True, mv_shake_amount=0.4, mv_transition="random",
        mv_every="all", beat_sync=True, progress_bar="top",
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
