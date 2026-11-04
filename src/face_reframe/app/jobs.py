"""Standalone shorts pipeline used by the desktop app — no Qt in here.

One source (optionally a time range of it) in, one or more vertical shorts out:

    trim range -> cut (pauses / breaths / fillers / retakes) -> cinematic audio
        -> per format: reframe (+ punch-in, split screen, enhance, captions)
        -> background music ducked under the voice

Whisper runs at most once per file (and is cached on disk): words are
remapped onto the trimmed/cut timeline, so fillers, retakes and captions
share one transcript. Face analyses are cached too, and reused whenever the
frames line up with the source (no cuts).
"""

from __future__ import annotations

import bisect
import dataclasses
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..cinematic import CinematicParams
from ..enhance import EnhanceOptions
from ..geometry import parse_aspect
from ..i18n import tr
from ..pipeline import Options
from ..silence import BreathParams, SpeechParams
from . import diskcache

Span = Tuple[float, float]
Progress = Callable[[str, float], None]
Log = Callable[[str], None]
CaptionRenderer = Callable[[Sequence[str], Optional[int], int], np.ndarray]
"""``(words, active_index_or_None, output_width) -> BGRA uint8 image``."""
Manual = List[Tuple[int, Optional[float], Optional[float]]]


class Cancelled(Exception):
    pass


@dataclass
class JobSettings:
    reframe: Options = field(default_factory=Options)
    output_height: int = 1920
    enhance: EnhanceOptions = field(default_factory=EnhanceOptions)
    beauty: "object" = None
    """:class:`beauty.BeautySettings` or None (off)."""
    skin: bool = False
    """Normalise skin tone (measured per video on cheeks/forehead)."""
    skin_strength: float = 0.7
    restore: str = "off"
    """Face restoration model: ``off`` | ``gpen256`` | ``gpen512`` | ``gfpgan`` | ``codeformer``."""
    restore_fidelity: float = 0.7
    restore_blend: float = 0.8
    formats: List[str] = field(default_factory=list)
    """Extra aspects rendered from the same analysis (e.g. ``["4:5", "1:1"]``)."""
    layout: str = "single"
    """``single`` or ``split`` (two people stacked when both are in shot)."""
    punch: bool = False
    punch_zoom: float = 1.15
    motion: "object" = None
    """:class:`motion.MotionSettings` (camera moves and cut transitions)."""

    cut_pauses: bool = False
    speech: SpeechParams = field(default_factory=SpeechParams)
    speaker_engine: str = "mouth"
    """``mouth`` = lip-motion heuristic; ``voice`` = ECAPA voice-prints —
    the speaker engine used when reframe.track.subject == "speaker"."""
    cut_breaths: bool = False
    cut_fillers: bool = False
    cut_retakes: bool = False

    cinematic: bool = False
    cinematic_params: CinematicParams = field(default_factory=CinematicParams)
    music_path: Optional[str] = None
    music_db: float = -18.0
    duck_ratio: float = 6.0

    subtitles: str = "off"
    """``off`` | ``srt`` (sidecar only) | ``burn`` (in the video + sidecar)."""
    caption_style: str = "word"
    """``line`` (whole caption) or ``word`` (current word highlighted)."""
    language: Optional[str] = None
    caption_words: int = 3
    caption_max_dur: float = 1.6
    caption_pos: float = 0.7

    person_by_file: Dict[str, List[np.ndarray]] = field(default_factory=dict)
    """Per-video "follow this person" picks (identity vectors), by abs path."""
    manual_by_file: Dict[str, Manual] = field(default_factory=dict)
    manual_cuts_by_file: Dict[str, List[Tuple[float, float]]] = field(default_factory=dict)
    """Hand-deleted ranges in SOURCE seconds per abs path (timeline edit)."""
    """Per-video hand corrections of the crop, in source frames/pixels."""

    overlays: "object" = None
    """:class:`overlays.OverlaySettings` (hook title, logo, closing CTA)."""
    hook_by_file: Dict[str, str] = field(default_factory=dict)
    """Custom hook text per video (abs path) when overlays.hook == 'custom'."""
    cover: bool = False
    social: bool = False
    sub_lang: str = "orig"
    """Caption language: ``orig``, ``en`` (Whisper translation) or ``both`` (two lines)."""
    look: str = "none"
    look_strength: float = 1.0
    auto_wb: bool = False
    grade: object = None
    """:class:`look.Grade` manual grading (temperature, exposure, contrast…), or None."""
    lowlight: str = "off"
    """Neural low-light lift: ``off`` | ``sg_llie`` (fast) | ``uretinex``."""
    lowlight_strength: float = 0.6
    stabilize: bool = False
    stab_amount: float = 0.8
    bg_blur: float = 0.0
    bg_engine: str = "rvm"
    """Matting engine for the background blur: ``rvm`` (fast, recurrent) or
    ``modnet`` (finer hair/edges, per-frame)."""
    """Blur strength behind the person (RVM matte); 0 = off."""
    bg_fill: str = "blur"
    """What replaces the backdrop: ``blur`` | ``color`` | ``image`` | ``video``.
    Non-blur fills use the same matte even at bg_blur == 0 (green screen)."""
    bg_color: str = "#00b140"
    bg_image: Optional[str] = None
    bg_video: Optional[str] = None
    encoder: str = "auto"
    """``auto`` (hardware on a Mac when available), ``hardware`` or ``software``."""
    preset: str = "veryfast"
    crf: int = 18
    slowmo: int = 1
    """Slow-motion factor: 1 = off, 2/4 = interpolated slow motion."""
    slowmo_engine: str = "rife"
    """``rife`` = neural interpolation, ``ffmpeg`` = minterpolate blend."""
    output_dir: Optional[str] = None
    suffix: str = "_short"

    @property
    def needs_words(self) -> bool:
        return self.cut_fillers or self.cut_retakes or self.subtitles != "off"

    @property
    def cuts(self) -> bool:
        return self.cut_pauses or self.cut_breaths or self.cut_fillers or self.cut_retakes

    def output_size(self, aspect: Optional[str] = None) -> Tuple[int, int]:
        return output_size(aspect or self.reframe.aspect, self.output_height)


def output_size(aspect: str, height: int) -> Tuple[int, int]:
    a = parse_aspect(aspect)
    if a <= 1.0:
        return int(round(height * a / 2.0)) * 2, height
    return height, int(round(height / a / 2.0)) * 2


# ---------------------------------------------------------------- helpers

def subtract(spans: List[Span], holes: List[Span], min_len: float = 0.05) -> List[Span]:
    out: List[Span] = []
    for a, b in spans:
        pieces = [(a, b)]
        for c, d in sorted(holes):
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                    continue
                if c > x:
                    nxt.append((x, c))
                if d < y:
                    nxt.append((d, y))
            pieces = nxt
        out.extend(pieces)
    return [(x, y) for x, y in out if y - x >= min_len]


class TimeMap:
    """Source seconds -> seconds on the cut timeline (kept spans glued)."""

    def __init__(self, spans: List[Span]):
        self.spans = spans
        self.starts = [a for a, _ in spans]
        self.offs, acc = [], 0.0
        for a, b in spans:
            self.offs.append(acc)
            acc += b - a
        self.length = acc

    def __call__(self, t: float) -> Optional[float]:
        k = bisect.bisect_right(self.starts, t) - 1
        if k < 0:
            return None
        a, b = self.spans[k]
        return self.offs[k] + (t - a) if t < b else None

    def inverse(self, t_out: float) -> float:
        """Cut-timeline seconds -> source seconds."""
        k = max(0, bisect.bisect_right(self.offs, t_out) - 1)
        return self.spans[k][0] + (t_out - self.offs[k])


def source_frames(n_out: int, fps: float, spans: Optional[List[Span]], offset_f: int) -> np.ndarray:
    """Source frame index for every frame of the trimmed/cut timeline."""
    if not spans:
        return offset_f + np.arange(n_out)
    tm = TimeMap(spans)
    t_out = np.arange(n_out) / fps
    k = np.clip(np.searchsorted(tm.offs, t_out, side="right") - 1, 0, len(spans) - 1)
    starts = np.array([a for a, _ in spans])
    return offset_f + np.round((starts[k] + (t_out - np.asarray(tm.offs)[k])) * fps).astype(int)


def remap_words(words, spans: List[Span]):
    """Move word timings onto the cut timeline; words in removed ranges go."""
    from ..transcribe import Word
    tm = TimeMap(spans)
    out = []
    for w in words:
        mid = 0.5 * (w.start + w.end)
        k = bisect.bisect_right(tm.starts, mid) - 1
        if k < 0:
            continue
        a, b = spans[k]
        if not (a <= mid < b):
            continue
        s = tm.offs[k] + max(0.0, w.start - a)
        e = tm.offs[k] + min(b - a, w.end - a)
        out.append(Word(s, max(s + 0.05, e), w.text))
    return out


def slice_words(words, start: float, end: float):
    from ..transcribe import Word
    return [Word(w.start - start, w.end - start, w.text) for w in words
            if w.start >= start - 0.05 and w.end <= end + 0.05]


def slice_analysis(a, f0: int, f1: int):
    """The part of a face analysis covering source frames [f0, f1)."""
    from ..detect import Analysis, FrameSample
    samples = [FrameSample(s.frame - f0, s.faces, cut=s.cut) for s in a.samples if f0 <= s.frame < f1]
    return Analysis(a.width, a.height, a.fps, a.start_frame + f0, max(0, min(f1, a.frame_count) - f0), samples)


def media_info(path: str) -> Tuple[float, float]:
    """(duration seconds, fps) via OpenCV."""
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return (n / fps if n else 0.0), fps


_HW = {}


def hw_encoder_available() -> bool:
    """Apple VideoToolbox H.264 encoder present (and working) in this ffmpeg."""
    if "vt" in _HW:
        return _HW["vt"]
    from ..silence import find_ffmpeg
    ok = False
    if sys.platform == "darwin":
        try:
            r = subprocess.run([find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-f", "lavfi", "-i",
                                "color=s=256x256:d=0.2", "-c:v", "h264_videotoolbox", "-f", "null", "-"],
                               capture_output=True, timeout=20)
            ok = r.returncode == 0
        except Exception:
            ok = False
    _HW["vt"] = ok
    return ok


def video_args(s: JobSettings, out_w: int, out_h: int) -> list:
    """Encoder arguments: VideoToolbox (3-5x faster on Apple Silicon) at a
    bitrate matched to the quality slider, or x264 CRF."""
    use_hw = s.encoder == "hardware" or (s.encoder == "auto" and hw_encoder_available())
    if use_hw and hw_encoder_available():
        # CRF 18 at 1080x1920 ~ 12 Mbit/s; scale with pixels and CRF (6 CRF ~ 2x)
        mbps = 12.0 * (out_w * out_h) / (1080 * 1920) * 2 ** ((18 - s.crf) / 6.0)
        mbps = max(2.0, min(40.0, mbps))
        return ["-c:v", "h264_videotoolbox", "-b:v", "%.1fM" % mbps, "-maxrate", "%.1fM" % (mbps * 1.5),
                "-bufsize", "%.1fM" % (mbps * 2), "-profile:v", "high", "-allow_sw", "1"]
    return ["-c:v", "libx264", "-crf", str(s.crf), "-preset", s.preset]


def has_audio(path: str) -> bool:
    from ..silence import find_ffmpeg
    raw = subprocess.run([find_ffmpeg(), "-hide_banner", "-i", path], capture_output=True, check=False)
    return "Audio:" in raw.stderr.decode(errors="replace")


def _ffmpeg(cmd: List[str], check: Callable[[], None], what: str) -> None:
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
    while True:
        try:
            proc.wait(timeout=0.3)
            break
        except subprocess.TimeoutExpired:
            try:
                check()
            except Cancelled:
                proc.kill()
                proc.wait()
                raise
    if proc.returncode != 0:
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise RuntimeError("ffmpeg %s failed: %s" % (what, err[-400:]))


INTERMEDIATE = ["-c:v", "libx264", "-crf", "10", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                "-c:a", "pcm_s16le"]


def trim(src: str, start: float, end: float, out: str, check) -> str:
    """Frame-accurate cut of [start, end) (re-encoded, near lossless)."""
    from ..silence import find_ffmpeg
    _ffmpeg([find_ffmpeg(), "-y", "-loglevel", "error", "-ss", "%.3f" % start, "-i", src,
             "-t", "%.3f" % (end - start), "-map", "0:v:0", "-map", "0:a:0?"] + INTERMEDIATE + [out],
            check, "trim")
    return out


def render_cut(src: str, spans: List[Span], out: str, check: Callable[[], None]) -> str:
    """Concatenate the kept spans (inline graph: -filter_complex_script is
    gone in ffmpeg 8; the graph is ~150 bytes per span)."""
    from ..cut import concat_filter
    from ..silence import find_ffmpeg
    _ffmpeg([find_ffmpeg(), "-y", "-loglevel", "error", "-i", src,
             "-filter_complex", concat_filter(spans), "-map", "[v]", "-map", "[a]"] + INTERMEDIATE + [out],
            check, "cut")
    return out


def mix_music(video: str, music: str, out: str, music_db: float, ratio: float, check) -> str:
    """Loop ``music`` under the video's audio, ducked by the voice."""
    from ..silence import find_ffmpeg
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    if has_audio(video):
        graph = ("[0:a]" + fmt + ",asplit=2[v][key];[1:a]" + fmt + ",volume=%.1fdB[m];"
                 "[m][key]sidechaincompress=threshold=0.02:ratio=%.1f:attack=20:release=350[d];"
                 "[v][d]amix=inputs=2:duration=first:normalize=0[a]" % (music_db, max(1.0, ratio)))
    else:  # silent video: the music is the whole soundtrack
        graph = "[1:a]" + fmt + ",volume=%.1fdB[a]" % music_db
    _ffmpeg([find_ffmpeg(), "-y", "-loglevel", "error", "-i", video, "-stream_loop", "-1", "-i", music,
             "-filter_complex", graph, "-map", "0:v", "-map", "[a]", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out], check, "music")
    return out


def preview_audio(src: str, start: float, dur: float, s: JobSettings, out: str) -> str:
    """A ``dur``-second window of the soundtrack as the export will sound:
    cinematic chain (+ loudness trim) and the ducked music bed. Written as
    WAV for the preview player; ffmpeg runs this far faster than real time."""
    from ..silence import extract_audio, find_ffmpeg
    fmt = "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo"
    chain = "anull"
    if s.cinematic:
        from ..cinematic import audio_chain, measure_processed, noise_floor_db, speech_level_db
        from ..loudness import gain_db
        pcm = extract_audio(src, start, dur)
        floor, loud = noise_floor_db(pcm), speech_level_db(pcm)
        try:
            gain = gain_db(measure_processed(src, start, dur, s.cinematic_params, floor, speech_db=loud))
        except Exception:
            gain = 0.0
        c = audio_chain(s.cinematic_params, floor, loud)
        chain = "%s,volume=%.2fdB,alimiter=limit=%.2f" % (
            c, gain, min(1.0, max(0.06, s.cinematic_params.limiter))) if c else "volume=%.2fdB" % gain
    cmd = [find_ffmpeg(), "-y", "-loglevel", "error", "-ss", "%.3f" % start, "-t", "%.3f" % dur, "-i", src]
    music = s.music_path if s.music_path and os.path.exists(s.music_path) else None
    if music:
        cmd += ["-stream_loop", "-1", "-i", music]
        graph = ("[0:a]%s,%s,asplit=2[v][key];[1:a]%s,volume=%.1fdB[m];"
                 "[m][key]sidechaincompress=threshold=0.02:ratio=%.1f:attack=20:release=350[d];"
                 "[v][d]amix=inputs=2:duration=first:normalize=0[a]"
                 % (chain, fmt, fmt, s.music_db, max(1.0, s.duck_ratio)))
    else:
        graph = "[0:a]%s,%s[a]" % (chain, fmt)
    cmd += ["-filter_complex", graph, "-map", "[a]", "-t", "%.3f" % dur, "-c:a", "pcm_s16le", out]
    raw = subprocess.run(cmd, capture_output=True, check=False)
    if raw.returncode != 0:
        raise RuntimeError(tr("audio preview: %s") % raw.stderr.decode(errors="replace")[-300:])
    return out


# ---------------------------------------------------------------- captions

def group_word_events(words, max_words: int = 3, max_dur: float = 1.6):
    """[(start, end, [Word, ...])] caption events, TikTok sized."""
    events, i = [], 0
    while i < len(words):
        j, start = i, words[i].start
        while j < len(words) and j - i < max_words and (j == i or words[j].end - start <= max_dur):
            j += 1
        grp = words[i:j]
        end = grp[-1].end
        if j < len(words):
            end = min(max(end, grp[-1].end + 0.15), words[j].start)
        events.append((start, end, grp))
        i = j
    return events


def _cv_caption(words: Sequence[str], active: Optional[int], out_w: int) -> np.ndarray:
    """Fallback caption renderer (no Qt): plain OpenCV text with outline."""
    import cv2
    text = " ".join(words).upper()
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, out_w / 600.0, max(2, out_w // 270)
    (tw, th), base = cv2.getTextSize(text, font, scale, thick)
    pad = thick * 3
    img = np.zeros((th + base + 2 * pad, tw + 2 * pad, 4), np.uint8)
    x = pad
    for k, word in enumerate(words):
        wtxt = word.upper() + (" " if k < len(words) - 1 else "")
        col = (0, 212, 255, 255) if active == k else (255, 255, 255, 255)
        cv2.putText(img, wtxt, (x, pad + th), font, scale, (0, 0, 0, 255), thick * 3, cv2.LINE_AA)
        cv2.putText(img, wtxt, (x, pad + th), font, scale, col, thick, cv2.LINE_AA)
        x += cv2.getTextSize(wtxt, font, scale, thick)[0][0]
    return img


def blend(frame: np.ndarray, img: np.ndarray, out_w: int, out_h: int, pos: float) -> np.ndarray:
    h, w = img.shape[:2]
    if w > out_w:
        import cv2
        s = out_w / float(w)
        img = cv2.resize(img, (out_w, max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
    x = (out_w - w) // 2
    y = max(0, min(out_h - h, int(out_h * pos - h / 2)))
    roi = frame[y:y + h, x:x + w].astype(np.float32)
    a = img[..., 3:4].astype(np.float32) / 255.0
    frame = frame if frame.flags.writeable else frame.copy()
    frame[y:y + h, x:x + w] = (roi * (1.0 - a) + img[..., :3].astype(np.float32) * a).astype(np.uint8)
    return frame


class CaptionBurner:
    """``post_frame`` hook: draws the caption active at frame ``i``."""

    def __init__(self, events, fps: float, out_w: int, out_h: int,
                 renderer: Optional[CaptionRenderer], pos: float, style: str = "word",
                 second_words=None):
        self.second_words = second_words
        """Words of a second language: their text under each caption, smaller."""
        self._second = {}
        self.events = events
        self.starts = [a for a, _, _ in events]
        self.fps, self.out_w, self.out_h = fps, out_w, out_h
        self.renderer = renderer or _cv_caption
        self.pos, self.style = pos, style
        self._key, self._img = None, None

    def image_at(self, t: float) -> Optional[np.ndarray]:
        k = bisect.bisect_right(self.starts, t) - 1
        if k < 0 or t >= self.events[k][1]:
            return None
        grp = self.events[k][2]
        active = None
        if self.style == "word":
            active = 0
            for j, w in enumerate(grp):
                if w.start <= t:
                    active = j
        key = (k, active)
        if key != self._key:
            texts = [w.text.strip() for w in grp]
            self._img, self._key = self.renderer(texts, active, self.out_w), key
        return self._img

    def _second_img(self, k: int):
        if k not in self._second:
            from .translate import overlap_text
            a, b, _ = self.events[k]
            txt = overlap_text(self.second_words, a - 0.2, b + 0.2)
            img = None
            if txt:
                img = self.renderer([txt], None, int(self.out_w * 0.62))
            self._second[k] = img
        return self._second[k]

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        t = i / self.fps
        img = self.image_at(t)
        if img is None:
            return frame
        frame = blend(frame, img, self.out_w, self.out_h, self.pos)
        if self.second_words:
            k = bisect.bisect_right(self.starts, t) - 1
            sec = self._second_img(k) if k >= 0 else None
            if sec is not None:
                frame = blend(frame, sec, self.out_w, self.out_h,
                              self.pos + (img.shape[0] + sec.shape[0]) / 2.0 / self.out_h)
        return frame


# ---------------------------------------------------------------- caches

class AnalysisCache:
    """Face analyses by (path, stride, min_score): memory, then disk."""

    def __init__(self, disk: bool = True):
        self._d = {}
        self._words = {}
        self._lock = threading.Lock()
        self.disk = disk

    @staticmethod
    def key(path: str, o: Options):
        return (os.path.abspath(path), o.stride, round(o.min_score, 3))

    def get(self, path: str, o: Options):
        k = self.key(path, o)
        with self._lock:
            a = self._d.get(k)
        if a is None and self.disk:
            a = diskcache.load(path, "faces", "%s|%s" % k[1:])
            if a is not None:
                with self._lock:
                    self._d[k] = a
        return a

    def put(self, path: str, o: Options, analysis) -> None:
        k = self.key(path, o)
        with self._lock:
            self._d[k] = analysis
        if self.disk:
            diskcache.save(path, "faces", analysis, "%s|%s" % k[1:])

    def words(self, path: str, language: Optional[str]):
        from ..transcribe import MODEL_NAME
        k = (os.path.abspath(path), language or "", MODEL_NAME)
        with self._lock:
            w = self._words.get(k)
        if w is None and self.disk:
            w = diskcache.load(path, "words", "%s|%s" % k[1:])
            if w is not None:
                with self._lock:
                    self._words[k] = w
        return w

    def put_words(self, path: str, language: Optional[str], words) -> None:
        from ..transcribe import MODEL_NAME
        k = (os.path.abspath(path), language or "", MODEL_NAME)
        with self._lock:
            self._words[k] = words
        if self.disk:
            diskcache.save(path, "words", words, "%s|%s" % k[1:])


def transcribe_file(path: str, language: Optional[str], cache: Optional[AnalysisCache],
                    progress=None, log: Log = print, pcm=None):
    """Word-level transcript of a whole file, from the cache when possible."""
    if cache is not None:
        w = cache.words(path, language)
        if w is not None:
            return w
    from ..silence import extract_audio
    from ..transcribe import transcribe
    if pcm is None:
        pcm = extract_audio(path)
    words, lang = transcribe(pcm, language, progress=progress, log=log)
    log(tr("  %d words (%s)") % (len(words), lang or "?"))
    if cache is not None:
        cache.put_words(path, language, words)
    return words


def cut_spans(pcm, duration: float, s: JobSettings, words, log: Log,
              manual_cuts: Optional[List[Span]] = None) -> Optional[List[Span]]:
    """Kept spans (seconds) for the cut settings; ``None`` = nothing to cut."""
    from .. import silence
    spans = [(0.0, duration)]
    if s.cut_pauses:
        found = silence.speech_intervals(pcm, s.speech)
        if found:
            spans = found
        else:
            log(tr("  no speech detected: pauses not cut"))
    holes: List[Span] = list(manual_cuts or [])
    if holes:
        log(tr("  %d manual cut%s") % (len(holes), "s" if len(holes) != 1 else ""))
    if s.cut_breaths:
        br = silence.breath_intervals(pcm, BreathParams(margin_db=s.speech.margin_db))
        holes += [(a - 0.03, b + 0.03) for a, b in br]
        log(tr("  %d breaths") % len(br))
    if words is not None and s.cut_fillers:
        from ..transcribe import filler_spans
        fs = filler_spans(words)
        holes += fs
        log(tr("  %d fillers (um, uh…)") % len(fs))
    if words is not None and s.cut_retakes:
        from ..retakes import retake_spans
        rs = retake_spans(words)
        holes += rs
        log(tr("  %d retakes") % len(rs))
    spans = subtract(spans, holes)
    if not spans or sum(b - a for a, b in spans) >= duration - 0.05:
        return None
    return spans


def output_path(src: str, s: JobSettings, stem: Optional[str] = None, aspect: Optional[str] = None) -> str:
    base = stem or os.path.splitext(os.path.basename(src))[0]
    d = s.output_dir or os.path.dirname(os.path.abspath(src))
    tag = ""
    if aspect and aspect != s.reframe.aspect:
        tag = "_" + aspect.replace(":", "x")
    return os.path.join(d, "%s%s%s.mp4" % (base, s.suffix, tag))


# ---------------------------------------------------------------- the job

def run_job(src: str, s: JobSettings, progress: Progress, log: Log,
            cancel: Optional[threading.Event] = None,
            cache: Optional[AnalysisCache] = None,
            caption_renderer: Optional[CaptionRenderer] = None,
            clip: Optional[Span] = None, stem: Optional[str] = None,
            title_renderer=None, thumb_renderer=None) -> List[str]:
    """Process one file (or the ``clip`` range of it); returns output paths."""
    from ..silence import extract_audio

    def check():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    def step(name):
        def cb(f):
            check()
            progress(name, float(f))
        return cb

    src_abs = os.path.abspath(src)
    aspects = [s.reframe.aspect] + [a for a in s.formats if a != s.reframe.aspect]
    outs = [output_path(src, s, stem, a) for a in aspects]
    os.makedirs(os.path.dirname(outs[0]), exist_ok=True)
    work = tempfile.mkdtemp(prefix="shorts_")
    try:
        src_dur, src_fps = media_info(src)
        cur = src
        offset_f = 0
        if clip is not None:
            c0, c1 = max(0.0, clip[0]), min(src_dur or clip[1], clip[1])
            log(tr("Clip %s-%s") % (_ts(c0), _ts(c1)))
            step(tr("Trim"))(0.0)
            cur = trim(src, c0, c1, os.path.join(work, "clip.mkv"), check)
            offset_f = int(round(c0 * src_fps))
        duration, fps = media_info(cur)

        # ---- 1. listen: one whisper pass (cached) + audio analysis
        words = None
        spans: Optional[List[Span]] = None
        pcm = None
        moves = s.motion if (s.motion is not None and s.motion.active) else None
        audio = has_audio(cur)
        if not audio:
            log(tr("  video has no audio: skipping cuts, captions, punch-in and cinematic audio"))
        wants_voice = s.reframe.track.subject == "speaker" and s.speaker_engine == "voice"
        if audio and (s.cuts or s.needs_words or s.punch or wants_voice
                      or (moves is not None and moves.needs_audio)):
            step(tr("Audio"))(0.0)
            pcm = extract_audio(cur)
            duration = len(pcm) / 16000.0
        if s.needs_words and audio:
            log(tr("Transcribing (Whisper)…"))
            try:
                cached = cache.words(src, s.language) if cache is not None else None
                if cached is not None and clip is not None:
                    words = slice_words(cached, clip[0], clip[1])
                    log(tr("  transcript from cache"))
                elif cached is not None:
                    words = cached
                    log(tr("  transcript from cache"))
                elif clip is None:
                    words = transcribe_file(src, s.language, cache, step(tr("Transcription")), log, pcm)
                else:
                    from ..transcribe import transcribe
                    words, _ = transcribe(pcm, s.language, progress=step(tr("Transcription")), log=log)
            except Cancelled:
                raise
            except Exception as exc:
                log(tr("  Whisper unavailable (%s): skipping fillers, retakes and captions") % exc)
                words = None
        check()
        mcuts = []
        for a, b in s.manual_cuts_by_file.get(src_abs) or []:
            # source seconds → seconds inside the (possibly trimmed) clip
            c, d = (a - c0, b - c0) if clip is not None else (a, b)
            if d > 0.02 and c < duration:
                mcuts.append((max(0.0, c), min(duration, d)))
        if (s.cuts and pcm is not None) or mcuts:
            if s.cuts and pcm is not None:
                spans = cut_spans(pcm, duration, s, words, log, manual_cuts=mcuts)
            else:
                spans = subtract([(0.0, duration)], mcuts)
                if not spans or sum(b - a for a, b in spans) >= duration - 0.05:
                    spans = None
            if spans is None:
                log(tr("  nothing to cut"))
            else:
                kept = sum(b - a for a, b in spans)
                log(tr("Cuts: %d segments, %.1fs of %.1fs") % (len(spans), kept, duration))
                step(tr("Cuts"))(0.0)
                cur = render_cut(cur, spans, os.path.join(work, "cut.mkv"), check)
                if words is not None:
                    words = remap_words(words, spans)
                pcm = extract_audio(cur) if (s.punch or moves is not None) else pcm

        # ---- 2. cinematic audio (video is stream-copied)
        if s.cinematic and audio:
            from ..cinematic import enhance as cine
            log(tr("Cinematic audio…"))
            nxt = os.path.join(work, "cine.mkv")
            # no cancel check inside: aborting mid-call would orphan its ffmpeg
            gain = cine(cur, nxt, s.cinematic_params,
                        progress=lambda f: progress(tr("Cinematic audio"), f))
            log(tr("  trim %+.1f dB") % gain)
            cur = nxt
        check()

        # ---- 3. faces (reused from the preview when frames line up)
        from ..detect import analyze_video
        from ..pipeline import plan_from_analysis
        o = s.reframe
        wants_object = o.track.noface == "object" or o.track.subject == "object"
        analysis = None
        if spans is None and cache is not None:
            full = cache.get(src, o)
            if full is not None:
                if wants_object and not getattr(full, "salient_pass", False):
                    from ..saliency import LazySaliency, backfill
                    sal = LazySaliency(log, step(tr("Model download")))
                    if any(not smp.faces for smp in full.samples):
                        log(tr("Salient objects on faceless frames…"))
                        backfill(src, full, sal, step(tr("Salient objects")))
                        if sal.ok:
                            cache.put(src, o, full)
                    else:
                        full.salient_pass = True
                analysis = slice_analysis(full, offset_f, offset_f + int(round(duration * fps))) \
                    if clip is not None else full
                log(tr("Faces: reusing the preview analysis"))
        if analysis is None:
            log(tr("Detecting faces…"))
            sal_fn = None
            if wants_object:
                from ..saliency import LazySaliency
                sal_fn = LazySaliency(log, step(tr("Model download")))
            analysis = analyze_video(cur, 0, None, stride=o.stride, min_score=o.min_score,
                                     progress=step(tr("Faces")), saliency=sal_fn)
            if cache is not None and spans is None and clip is None:
                cache.put(src, o, analysis)

        picks = s.person_by_file.get(src_abs) or None
        manual = _map_manual(s.manual_by_file.get(src_abs), offset_f, fps, spans)
        vsegs = None
        if wants_voice and pcm is not None:
            from ..silence import speech_intervals
            from ..speakerid import analyze as voice_analyze
            vsegs = voice_analyze(pcm, speech_intervals(pcm, s.speech), log,
                                  step(tr("Voice identification")))
        track = dataclasses.replace(o.track, person=picks, manual=manual, voice=vsegs)
        if picks:
            log(tr("Framing: follows the people picked in the preview"))
        if manual:
            log(tr("Framing: %d manual corrections") % len(manual))

        punch = None
        if s.punch and pcm is not None:
            from ..silence import speech_intervals
            from .compose import punch_curve
            punch = punch_curve(speech_intervals(pcm, s.speech), analysis.frame_count, fps, s.punch_zoom)

        enhancer = None
        if s.enhance.mode != "off":
            from ..enhance import Enhancer
            enhancer = Enhancer(s.enhance, log=log)
        lifter = None
        if s.lowlight != "off":
            from ..lowlight import build as build_lowlight
            lifter = build_lowlight(s.lowlight, s.lowlight_strength, log, step(tr("Model download")))
            if lifter is not None:
                log(tr("Low-light boost: %s, %.0f%%") % (s.lowlight, 100 * s.lowlight_strength))
        skin_fix = None
        if s.skin:
            from ..skin import measure_video
            skin_fix = measure_video(cur, analysis, strength=s.skin_strength)
            log(tr("Skin: %s") % (skin_fix.describe() if skin_fix
                                 else tr("no face to measure")))
        restorer = make_restorer(s, log, step(tr("Model download")))
        beautifier = None
        if s.beauty is not None and s.beauty.active:
            from ..beauty import build as build_beauty
            beautifier = build_beauty(s.beauty, log, step(tr("Model download")))
            log(tr("Beauty: smoothing %.0f%%, blemishes %.0f%%, wrinkles %.0f%%")
                % (100 * s.beauty.smooth, 100 * s.beauty.blemish, 100 * s.beauty.wrinkles))
        if restorer is not None:
            per = restorer.timing()
            n = analysis.frame_count * len(aspects)
            log(tr("Face restoration %s: %.2f s per face -> about %s for %d frames")
                % (s.restore, per, _dur(per * n), n))

        second = None
        if words and s.subtitles != "off" and s.sub_lang in ("en", "both"):
            try:
                from .translate import english_words
                log(tr("English translation (Whisper)…"))
                # whole source (cached for every later short/clip), then onto this timeline
                en = english_words(src, cache, step(tr("Translation")), log)
                if clip is not None:
                    en = slice_words(en, clip[0], clip[1])
                if spans is not None:
                    en = remap_words(en, spans)
                if s.sub_lang == "en":
                    words = en
                else:
                    second = en
            except Cancelled:
                raise
            except Exception as exc:
                log(tr("  translation unavailable (%s)") % exc)
        events = []
        if words and s.subtitles != "off":
            from ..transcribe import group_captions, to_srt
            srt = os.path.splitext(outs[0])[0] + ".srt"
            with open(srt, "w", encoding="utf-8") as fh:
                fh.write(to_srt(group_captions(words, s.caption_words, s.caption_max_dur)))
            events = group_word_events(words, s.caption_words, s.caption_max_dur)
            log(tr("Captions: %d lines -> %s") % (len(events), os.path.basename(srt)))

        look_obj = None
        grade_on = s.grade is not None and (s.grade.active or s.grade.vignette > 0)
        if s.look != "none" or s.auto_wb or grade_on:
            from .look import Look, auto_gains
            gains = auto_gains(cur) if s.auto_wb else None
            look_obj = Look(s.look, s.look_strength, gains, s.grade)
            log(tr("Colour: look %s%s%s") % (s.look, tr(", auto balance") if gains else "",
                                           tr(", grade") if grade_on else ""))
        stab = None
        if s.stabilize:
            from .stabilize import estimate
            full = cache.get(src, o) if cache is not None else None
            shot_cuts = [x.frame for x in full.samples if x.cut] if full is not None else []
            st = None
            from . import diskcache
            st = diskcache.load(src, "stab", "v1")
            if st is None:
                log(tr("Stabilization: motion analysis…"))
                st = estimate(src, shot_cuts, progress=step(tr("Stabilization")))
                if st is not None:
                    diskcache.save(src, "stab", st, "v1")
            if st is not None:
                idx = np.clip(source_frames(analysis.frame_count, fps, spans, offset_f), 0, len(st.dx) - 1)
                from .stabilize import Stabilization
                stab = Stabilization(st.dx[idx], st.dy[idx], st.rot[idx]).scaled(s.stab_amount)
        ov = s.overlays
        hook_text = ""
        if ov is not None and ov.hook != "off":
            from .overlays import first_sentence
            hook_text = s.hook_by_file.get(src_abs, "") if ov.hook == "custom" else first_sentence(words)

        # ---- 4. one render per format
        from ..render import render
        from .compose import Framer, build_split
        done = []
        for aspect, out in zip(aspects, outs):
            check()
            oa = dataclasses.replace(o, aspect=aspect, track=track)
            res = plan_from_analysis(analysis, oa)
            if aspect == aspects[0]:
                found = int(res.path.has_face.sum())
                log(tr("  face on %d/%d frames, %d shots")
                    % (found, analysis.frame_count, len(res.path.segments)))
            out_w, out_h = output_size(aspect, s.output_height)
            split = None
            if s.layout == "split" and parse_aspect(aspect) < 1.0:
                split = build_split(analysis, oa, out_w, out_h, picks)
                if split is not None and aspect == aspects[0]:
                    log(tr("  split screen on %d frames") % int(split.mask.sum()))
            curves = None
            if moves is not None:
                from ..silence import speech_intervals
                from .motion import build_motion, output_cuts
                cuts, shot_cuts = output_cuts(spans, res.path.segments, fps)
                phrases = speech_intervals(pcm, s.speech) if pcm is not None else []
                curves = build_motion(analysis.frame_count, fps, moves, cuts, shot_cuts, phrases,
                                      words, pcm, seed=_seed(src))
                if aspect == aspects[0]:
                    log(tr("  moves: %d cuts") % len(cuts))
            if stab is not None:
                from .motion import MotionCurves
                c2 = curves if curves is not None else MotionCurves.empty(analysis.frame_count)
                c2 = MotionCurves(c2.dx + stab.dx / np.maximum(1.0, res.plan.crop_w), c2.dy + stab.dy /
                                  np.maximum(1.0, res.plan.crop_h), c2.zoom * 1.04, c2.rot + stab.rot,
                                  c2.whip, c2.zblur, c2.glitch)
                curves = c2
            framer = Framer(res, out_w, out_h, enhancer, punch, split, restorer, curves, _seed(src), skin_fix,
                            beautifier, look_obj, lowlight=lifter)
            burner = None
            if events and s.subtitles == "burn":
                # split screen: captions sit on the seam, off both faces
                burner = CaptionBurner(events, fps, out_w, out_h, caption_renderer,
                                       0.5 if split is not None else s.caption_pos, s.caption_style,
                                       second_words=second)
            hooks = []
            if s.bg_blur > 0 or s.bg_fill not in ("blur", "off"):
                from .rvm import build as build_bg_blur
                bg = build_bg_blur(s.bg_blur, s.bg_engine, step(tr("Background blur")), log,
                                   size=(out_w, out_h), fill=s.bg_fill, color=s.bg_color,
                                   image=s.bg_image, video=s.bg_video, fps=fps)
                if bg is not None:
                    hooks.append(bg)          # matte first: don't soften captions/overlays
            if burner is not None:
                hooks.append(burner)
            if ov is not None:
                from .overlays import Chain, build as build_overlays
                hooks += build_overlays(ov, hook_text, analysis.frame_count, fps, out_w, out_h,
                                        title_renderer, log)
            if hooks:
                from .overlays import Chain
                burner = hooks[0] if len(hooks) == 1 else Chain(hooks)
            log(tr("Render %s %dx%d…") % (aspect, out_w, out_h))
            tmp = os.path.join(work, "out.mp4")
            vargs = video_args(s, out_w, out_h)
            if aspect == aspects[0]:
                log(tr("  encoder: %s") % ("hardware (VideoToolbox)" if "h264_videotoolbox" in vargs else "x264"))
            render(cur, res.plan, tmp, (out_w, out_h), fps, 0, crf=s.crf,
                   progress=step(tr("Render %s") % aspect), preset=s.preset,
                   post_frame=burner, frame_fn=framer, video_args=vargs)
            if s.music_path and os.path.exists(s.music_path):
                step(tr("Music"))(0.0)
                mixed = os.path.join(work, "music.mp4")
                mix_music(tmp, s.music_path, mixed, s.music_db, s.duck_ratio, check)
                tmp = mixed
            if s.slowmo > 1:
                from ..interp import ensure as rife_ensure, slowmo
                if s.slowmo_engine == "rife" and not rife_ensure(step(tr("Model download")), log):
                    s2 = dataclasses.replace(s, slowmo_engine="ffmpeg")
                else:
                    s2 = s
                log(tr("Slow motion %d× (%s)…") % (s.slowmo, s2.slowmo_engine))
                sl = os.path.join(work, "slow.mp4")
                if slowmo(tmp, sl, s.slowmo, s2.slowmo_engine,
                          progress=lambda f: step(tr("Slow motion"))(f), log=log):
                    tmp = sl
                else:
                    log(tr("  slow motion failed: keeping normal speed"))
            shutil.move(tmp, out)
            done.append(out)
            if aspect == aspects[0] and s.cover:
                try:
                    cp = make_cover(cur, analysis, framer, out_w, out_h, hook_text or (
                        __import__("face_reframe.app.overlays", fromlist=["x"]).first_sentence(words)
                        if words else ""), title_renderer, os.path.splitext(out)[0] + "_cover.jpg")
                    log(tr("Thumbnail: %s") % os.path.basename(cp))
                except Exception as exc:
                    log(tr("  thumbnail not created (%s)") % exc)
                try:
                    from . import thumbnail as T
                    import dataclasses as _dc
                    vs = T.prepare(cur, _dc.replace(analysis, start_frame=0), count=1, log=log)
                    if vs:
                        txt, hl = T.thumb_text(words or [], hook_text)
                        yp = os.path.splitext(out)[0] + "_youtube.jpg"
                        T.save(T.render(vs[0], txt, hl, "16:9", thumb_renderer), yp)
                        log(tr("YouTube thumbnail: %s") % os.path.basename(yp))
                except Exception as exc:
                    log(tr("  YouTube thumbnail not created (%s)") % exc)
            if aspect == aspects[0] and s.social:
                from .social import make_texts, to_text
                tp = os.path.splitext(out)[0] + "_texts.txt"
                with open(tp, "w", encoding="utf-8") as fh:
                    fh.write(to_text(make_texts(words or [], hook_text, s.language)))
                log(tr("Social texts: %s") % os.path.basename(tp))
        progress(tr("Done"), 1.0)
        return done
    finally:
        shutil.rmtree(work, ignore_errors=True)


def best_cover_frame(path: str, analysis, max_candidates: int = 24) -> Optional[int]:
    """Frame with the largest, sharpest, most frontal face."""
    import cv2
    cands = []
    for smp in analysis.samples:
        faces = [f for f in smp.faces if f.landmarks is not None]
        if faces:
            f = max(faces, key=lambda x: x.area)
            rex, rey, lex, ley, nx, ny = f.landmarks[:6]
            frontal = 1.0 - min(1.0, abs(nx - 0.5 * (rex + lex)) / max(1.0, abs(lex - rex)))
            cands.append((f.area * f.score * (0.5 + frontal), smp.frame, f))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[0])
    cap = cv2.VideoCapture(path)
    best, best_s = None, -1.0
    for score, fr, f in cands[:max_candidates]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, analysis.start_frame + fr)
        ok, img = cap.read()
        if not ok:
            continue
        x0, y0 = int(max(0, f.cx - f.w / 2)), int(max(0, f.cy - f.h / 2))
        crop = cv2.cvtColor(img[y0:int(y0 + f.h), x0:int(x0 + f.w)], cv2.COLOR_BGR2GRAY)
        if crop.size == 0:
            continue
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        sc = score * (sharp ** 0.5)
        if sc > best_s:
            best, best_s = fr, sc
    cap.release()
    return best


def make_cover(path: str, analysis, framer, out_w: int, out_h: int, title: str, title_renderer,
               out_path: str) -> str:
    """Cover image: best frame, framed like the short, with a big title."""
    import cv2
    from .overlays import cv_title
    fr = best_cover_frame(path, analysis)
    if fr is None:
        fr = analysis.frame_count // 3
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, analysis.start_frame + fr)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(tr("unreadable frame"))
    out = framer(fr, img)
    if title.strip():
        t = (title_renderer or cv_title)(title.strip(), out_w)
        out = blend(out, t, out_w, out_h, 0.18)
    cv2.imwrite(out_path, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return out_path


def make_restorer(s: JobSettings, log: Log, progress=None):
    """Face restorer for the settings (downloading the model once), or None."""
    if s.restore == "off":
        return None
    from .. import restore
    if not restore.is_available(s.restore):
        m = restore.MODELS[s.restore]
        log(tr("Downloading the %s model (%d MB, once)…") % (m.file, m.mb))
        restore.download(s.restore, progress)
    return restore.FaceRestorer(s.restore, s.restore_fidelity, s.restore_blend, log=log)


def _seed(path: str) -> int:
    """Stable per-video seed: preview and export draw the same random moves."""
    import zlib
    return zlib.crc32(os.path.basename(path).encode("utf-8")) & 0x7FFFFFFF


def _dur(sec: float) -> str:
    sec = int(sec)
    if sec < 90:
        return "%d s" % sec
    if sec < 5400:
        return "%d min" % round(sec / 60.0)
    return tr("%.1f hours") % (sec / 3600.0)


def _map_manual(manual: Optional[Manual], offset_f: int, fps: float,
                spans: Optional[List[Span]]) -> Optional[Manual]:
    """Source-frame corrections -> frames of the trimmed/cut file."""
    if not manual:
        return None
    tm = TimeMap(spans) if spans else None
    out = []
    for f, x, y in manual:
        f2 = f - offset_f
        if f2 < 0:
            # the correction still active when the clip starts holds from frame 0
            out = [k for k in out if k[0] != 0] + [(0, x, y)]
            continue
        if tm is not None:
            t = tm(f2 / fps)
            if t is None:
                continue
            f2 = int(round(t * fps))
        out.append((f2, x, y))
    return out or None


def _ts(sec: float) -> str:
    sec = int(sec)
    return "%d:%02d" % (sec // 60, sec % 60)
