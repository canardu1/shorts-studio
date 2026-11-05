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
                "caption_box", "caption_anim", "caption_sweep", "caption_emoji")

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
    "Karaoke neon": dict(caption_style="word", caption_words=2, caption_max_dur=1.2,
                         caption_pos=0.72, caption_color="#FFFFFF", caption_hl="#4FD8FF",
                         caption_upper=True, caption_size=0.09, caption_box="none",
                         caption_anim="pop", caption_sweep=True, caption_font="Arial Black"),
    "Podcast bold": dict(caption_style="line", caption_words=5, caption_max_dur=2.0,
                         caption_pos=0.75, caption_color="#FFFFFF", caption_hl="#FFD400",
                         caption_upper=True, caption_size=0.065, caption_box="pill",
                         caption_anim="off", caption_font="Arial Black"),
    "Emoji burst": dict(caption_style="word", caption_words=3, caption_max_dur=1.5,
                        caption_pos=0.7, caption_color="#FFFFFF", caption_hl="#7CFF6B",
                        caption_upper=True, caption_size=0.08, caption_box="none",
                        caption_anim="pop", caption_emoji=True, caption_font="Arial Black"),
    "Broadcast": dict(caption_style="line", caption_words=6, caption_max_dur=2.4,
                      caption_pos=0.82, caption_color="#FFFFFF", caption_hl="#4FD8FF",
                      caption_upper=False, caption_size=0.05, caption_box="pill",
                      caption_anim="fade", caption_font="Helvetica"),
}


# Audio-chain recipes: the controls in the Audio tab that together make a
# voice treatment.  "cinematic" is the master switch — every recipe assumes
# it on except where stated.
AUDIO_KEYS = ("cinematic", "denoise", "denoise_engine", "dereverb", "warmth",
              "presence", "air", "duck", "hum")

AUDIO_BUILTIN: Dict[str, dict] = {
    "Podcast voice": dict(cinematic=True, denoise=10, denoise_engine="rnn",
                          dereverb=8, warmth=1.6, presence=2.2, air=1.2,
                          duck=6.0, hum=False),
    "Clean studio": dict(cinematic=True, denoise=6, denoise_engine="rnn",
                         dereverb=4, warmth=1.6, presence=2.0, air=1.4,
                         duck=3.0, hum=False),
    "Noisy room": dict(cinematic=True, denoise=20, denoise_engine="dfn",
                       dereverb=18, warmth=0.0, presence=3.0, air=0.5,
                       duck=6.0, hum=True),
    "Phone recording": dict(cinematic=True, denoise=18, denoise_engine="dfn",
                            dereverb=15, warmth=0.8, presence=3.0, air=1.8,
                            duck=6.0, hum=True),
    "Voice memo / webcam": dict(cinematic=True, denoise=16, denoise_engine="fft",
                                dereverb=10, warmth=0.5, presence=2.5, air=1.0,
                                duck=6.0, hum=False),
    "Music + voice": dict(cinematic=True, denoise=8, denoise_engine="rnn",
                          dereverb=6, warmth=1.2, presence=2.0, air=1.2,
                          duck=12.0, hum=False),
}


# Colour-grade recipes: base look + the manual grade dials in the Colour tab.
GRADE_KEYS = ("look", "look_strength", "auto_wb", "color_temp", "color_tint",
              "color_exp", "color_con", "color_sat", "color_shad", "color_high",
              "color_vig")

GRADE_BUILTIN: Dict[str, dict] = {
    # color_con/color_sat are 0..2 multipliers centred on 1.0
    "Moody dark": dict(look="none", look_strength=1.0, auto_wb=False,
                       color_temp=-0.1, color_tint=0.0, color_exp=-0.1, color_con=1.15,
                       color_sat=0.9, color_shad=-0.15, color_high=0.0,
                       color_vig=0.2),
    "Bright clean": dict(look="none", look_strength=1.0, auto_wb=True,
                         color_temp=0.05, color_tint=0.0, color_exp=0.15, color_con=1.05,
                         color_sat=1.05, color_shad=0.1, color_high=0.0,
                         color_vig=0.0),
    "Warm film": dict(look="warm", look_strength=0.8, auto_wb=False,
                      color_temp=0.15, color_tint=0.0, color_exp=0.0, color_con=1.1,
                      color_sat=0.95, color_shad=0.0, color_high=-0.05,
                      color_vig=0.15),
    "Cinema punch": dict(look="cine", look_strength=1.0, auto_wb=False,
                         color_temp=0.0, color_tint=0.0, color_exp=0.0, color_con=1.12,
                         color_sat=1.1, color_shad=-0.08, color_high=0.0,
                         color_vig=0.18),
    "B&W studio": dict(look="bw", look_strength=1.0, auto_wb=False,
                       color_temp=0.0, color_tint=0.0, color_exp=0.05, color_con=1.2,
                       color_sat=1.0, color_shad=0.0, color_high=0.0,
                       color_vig=0.2),
    "Soft pastel": dict(look="none", look_strength=1.0, auto_wb=True,
                        color_temp=0.05, color_tint=0.0, color_exp=0.1, color_con=0.92,
                        color_sat=0.8, color_shad=0.08, color_high=0.0,
                        color_vig=0.0),
    "Neon night": dict(look="cool", look_strength=1.0, auto_wb=False,
                       color_temp=-0.15, color_tint=0.0, color_exp=0.0, color_con=1.12,
                       color_sat=1.15, color_shad=-0.1, color_high=0.0,
                       color_vig=0.25),
}


# Export/platform recipes: resolution + encoder + extras for each target.
# A platform implies its aspect too, so that travels in the recipe.
EXPORT_KEYS = ("aspect", "resolution", "preset", "crf", "fmt_45", "fmt_11",
               "fmt_169", "encoder", "slowmo", "slowmo_engine", "cover",
               "social", "progress_bar")

EXPORT_BUILTIN: Dict[str, dict] = {
    "TikTok": dict(aspect="9:16", resolution=1920, encoder="auto", preset="fast",
                   crf=18, fmt_45=True, fmt_11=False, fmt_169=False,
                   slowmo=1, slowmo_engine="rife", cover=True, social=True,
                   progress_bar="top"),
    "Instagram Reels": dict(aspect="9:16", resolution=1920, encoder="auto",
                            preset="fast", crf=18, fmt_45=True, fmt_11=False,
                            fmt_169=False, slowmo=1, slowmo_engine="rife",
                            cover=True, social=True, progress_bar="bottom"),
    "YouTube Shorts": dict(aspect="9:16", resolution=1920, encoder="auto",
                           preset="fast", crf=16, fmt_45=False, fmt_11=False,
                           fmt_169=True, slowmo=1, slowmo_engine="rife",
                           cover=True, social=True, progress_bar="off"),
    "Master archive": dict(aspect="9:16", resolution=1920, encoder="software",
                           preset="medium", crf=14, fmt_45=False, fmt_11=False,
                           fmt_169=False, slowmo=1, slowmo_engine="rife",
                           cover=False, social=False, progress_bar="off"),
    "Fast draft": dict(aspect="9:16", resolution=1280, encoder="auto",
                       preset="veryfast", crf=22, fmt_45=False, fmt_11=False,
                       fmt_169=False, slowmo=1, slowmo_engine="ffmpeg",
                       cover=False, social=False, progress_bar="off"),
}


# Camera-operator styles: how the virtual camera frames and moves —
# orthogonal to content type (Interview + Tight works, so does Gaming +
# Loose).  Who to follow stays a content choice, so subject/layout are
# left out.
CAMERA_KEYS = ("smoothing", "dead_zone", "max_speed", "zoom", "follow_vertical",
               "face_pos", "look_room", "look_room_amount", "punch",
               "punch_zoom", "wide_mode", "wide_threshold", "max_zoom")

CAMERA_BUILTIN: Dict[str, dict] = {
    "Tight podcast": dict(smoothing=0.5, dead_zone=0.03, max_speed=0.6, zoom=1.5,
                          follow_vertical=True, face_pos=0.42, look_room=False,
                          look_room_amount=0.1, punch=True, punch_zoom=1.2,
                          wide_mode="zoom", wide_threshold=0.1, max_zoom=2.2),
    "Loose cinematic": dict(smoothing=0.9, dead_zone=0.08, max_speed=0.35, zoom=1.15,
                            follow_vertical=False, face_pos=0.38, look_room=True,
                            look_room_amount=0.2, punch=False, punch_zoom=1.15,
                            wide_mode="keep", wide_threshold=0.15, max_zoom=1.8),
    "Sports follow": dict(smoothing=0.3, dead_zone=0.0, max_speed=1.6, zoom=1.3,
                          follow_vertical=True, face_pos=0.4, look_room=False,
                          look_room_amount=0.1, punch=False, punch_zoom=1.15,
                          wide_mode="zoom", wide_threshold=0.08, max_zoom=2.5),
    "Locked tripod": dict(smoothing=2.5, dead_zone=0.12, max_speed=0.25, zoom=1.1,
                          follow_vertical=False, face_pos=0.4, look_room=False,
                          look_room_amount=0.1, punch=False, punch_zoom=1.15,
                          wide_mode="keep", wide_threshold=0.12, max_zoom=1.5),
}


def load_user(raw: str) -> Dict[str, dict]:
    try:
        data = json.loads(raw or "{}")
        return {k: v for k, v in data.items() if isinstance(v, dict)}
    except ValueError:
        return {}


def dump_user(presets: Dict[str, dict]) -> str:
    return json.dumps(presets, ensure_ascii=False)
