"""Resolve integration for pause cutting: rebuilds the timeline keeping only
the spoken segments of each clip (jump cuts)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

from . import retakes, silence, transcribe
from .resolve_bridge import (Log, ResolveOptions, _float, source_range,
                             standalone_audio_items, target_items)


@dataclass
class CutOptions:
    scope: str = "current"
    track_index: int = 1
    speech: silence.SpeechParams = field(default_factory=silence.SpeechParams)
    keep_silent_clips: bool = True
    """Clips with no detected speech (e.g. b-roll) are kept whole."""
    remove_fillers: bool = False
    """Also cut filler words (uhm, ehm, ...), detected with whisper."""
    remove_breaths: bool = False
    """Also cut audible breaths sitting inside the kept pauses."""
    breath: silence.BreathParams = field(default_factory=silence.BreathParams)
    remove_retakes: bool = False
    """Also drop earlier takes of repeated phrases (whisper); last take wins."""
    retake_similarity: float = 0.65


def _subtract(spans: List[Tuple[int, int]], holes: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out = []
    for a, b in spans:
        pieces = [(a, b)]
        for c, d in holes:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append((x, y))
                else:
                    if c > x:
                        nxt.append((x, c))
                    if d < y:
                        nxt.append((d, y))
            pieces = nxt
        out.extend(pieces)
    return [(x, y) for x, y in out if y - x > 1]


def _filler_holes(path: str, start: int, count: int, fps: float, log: Log) -> List[Tuple[int, int]]:
    try:
        pcm = silence.extract_audio(path, start / fps, count / fps)
        words, _ = transcribe.transcribe(
            pcm, progress=lambda f: log("  transcribing %d%%" % int(f * 100)), log=log)
    except Exception as exc:
        log("  whisper failed (%s); keeping fillers" % exc)
        return []
    holes = [(start + int(round(a * fps)), start + int(round(b * fps)))
             for a, b in transcribe.filler_spans(words)]
    if holes:
        log("  %d filler word(s) to cut" % len(holes))
    return holes


def _breath_holes(path: str, start: int, count: int, fps: float, opts: CutOptions, log: Log) -> List[Tuple[int, int]]:
    try:
        pcm = silence.extract_audio(path, start / fps, count / fps)
        bp = silence.BreathParams(window_ms=opts.breath.window_ms,
                                  floor_percentile=opts.breath.floor_percentile,
                                  margin_db=opts.speech.margin_db,
                                  breath_db=opts.breath.breath_db,
                                  min_dur=opts.breath.min_dur,
                                  max_dur=opts.breath.max_dur)
        breaths = silence.breath_intervals(pcm, bp)
    except Exception as exc:
        log("  breath analysis failed (%s); keeping them" % exc)
        return []
    pad = int(round(0.03 * fps))
    holes = [(start + int(round(a * fps)) - pad, start + int(round(b * fps)) + pad)
             for a, b in breaths]
    if holes:
        log("  %d breath(s) to cut" % len(holes))
    return holes


def _retake_holes(path: str, start: int, count: int, fps: float, opts: CutOptions, log: Log) -> List[Tuple[int, int]]:
    try:
        pcm = silence.extract_audio(path, start / fps, count / fps)
        words, _ = transcribe.transcribe(
            pcm, progress=lambda f: log("  transcribing %d%%" % int(f * 100)), log=log)
        spans = retakes.retake_spans(words, similarity=opts.retake_similarity)
    except Exception as exc:
        log("  retake detection failed (%s); keeping takes" % exc)
        return []
    holes = [(start + int(round(a * fps)), start + int(round(b * fps)))
             for a, b in spans]
    if holes:
        log("  %d earlier take(s) to drop" % len(holes))
    return holes


def item_spans(item, path: str, opts: CutOptions, timeline_fps: float, log: Log) -> List[Tuple[int, int]]:
    """Source-frame (start, end] ranges to keep for one timeline item."""
    start, count = source_range(item)
    pool_item = item.GetMediaPoolItem()
    fps = _float(pool_item.GetClipProperty("FPS"), timeline_fps) if pool_item else timeline_fps
    try:
        spans = silence.clip_intervals(path, start, count, fps, opts.speech)
    except Exception as exc:
        log("  audio analysis failed for %s (%s); keeping it whole" % (item.GetName(), exc))
        return [(start, start + count)]
    if not spans:
        log("  no speech in %s%s" % (item.GetName(), "; kept whole" if opts.keep_silent_clips else "; dropped"))
        if not opts.keep_silent_clips:
            return []
        spans = [(start, start + count)]
    if opts.remove_fillers:
        spans = _subtract(spans, _filler_holes(path, start, count, fps, log))
    if opts.remove_breaths:
        spans = _subtract(spans, _breath_holes(path, start, count, fps, opts, log))
    if opts.remove_retakes:
        spans = _subtract(spans, _retake_holes(path, start, count, fps, opts, log))
    log("  %d speech segment(s) in %s" % (len(spans), item.GetName()))
    return [(max(a, start), min(b, start + count)) for a, b in spans]


def run_cut(resolve, opts: CutOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    items = target_items(timeline, ResolveOptions(scope=opts.scope, track_index=opts.track_index))
    if not items:
        log("No clip found.")
        return 0

    timeline_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    entries: List[dict] = []
    kept_frames = 0
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not pool_item:
            continue
        log("Analysing %s" % item.GetName())
        for a, b in item_spans(item, path, opts, timeline_fps, log):
            if b - a < 1:
                continue
            entries.append({"mediaPoolItem": pool_item, "startFrame": a, "endFrame": b})
            kept_frames += b - a
    if opts.scope == "all":
        # a music bed or lav kept at its original position would desync once
        # the video is packed — say so rather than silently dropping it
        n_audio = len(standalone_audio_items(timeline, items))
        if n_audio:
            log("Note: %d standalone audio item(s) are not carried into the "
                "cut timeline (they would lose sync)." % n_audio)
    if not entries:
        log("Nothing kept — check the silence settings.")
        return 0

    pool = project.GetMediaPool()
    name = "%s (no pauses)" % timeline.GetName()
    new_tl = pool.CreateEmptyTimeline(name)
    if new_tl is None:
        log("Could not create a new timeline.")
        return 0
    project.SetCurrentTimeline(new_tl)
    for key in ("useCustomSettings", "timelineResolutionWidth", "timelineResolutionHeight",
                "timelineFrameRate"):
        value = timeline.GetSetting(key)
        if value:
            new_tl.SetSetting(key, value)
    pool.AppendToTimeline(entries)
    log("Done: %d segments kept in '%s'." % (len(entries), name))
    return len(entries)
