"""Resolve integration: transcribe every clip and import the SRT as subtitles."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field
from typing import List, Tuple

from . import silence, transcribe
from .resolve_bridge import Log, ResolveOptions, _float, source_range, target_items


@dataclass
class SubtitleOptions:
    scope: str = "all"
    track_index: int = 1
    language: str = ""
    """Empty = auto-detect."""
    max_words: int = 3
    max_dur: float = 1.6
    srt_path: str = ""
    """Where to write the .srt (default: alongside the project, in /tmp)."""


def _item_timeline_start(item, log: Log) -> float:
    for meth in ("GetStart",):
        fn = getattr(item, meth, None)
        if fn:
            try:
                return float(fn())
            except Exception:
                pass
    try:
        return float(item.GetEnd()) - float(item.GetDuration())
    except Exception:
        log("  warning: cannot read the timeline position of %s; assuming 0" % item.GetName())
        return 0.0


def run_subtitles(resolve, opts: SubtitleOptions, log: Log = print) -> str:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return ""
    items = target_items(timeline, ResolveOptions(scope=opts.scope, track_index=opts.track_index))
    if not items:
        log("No clip found.")
        return ""
    fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    events: List[Tuple[float, float, str]] = []
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            continue
        start, count = source_range(item)
        tl_start = _item_timeline_start(item, log)
        log("Transcribing %s" % item.GetName())
        try:
            clip_fps = _float(pool_item.GetClipProperty("FPS"), fps)
            pcm = silence.extract_audio(path, start / clip_fps, count / clip_fps)
            words, lang = transcribe.transcribe(
                pcm, opts.language or None,
                progress=lambda f: log("  %d%%" % int(f * 100)), log=log)
        except Exception as exc:
            log("  transcription failed for %s (%s)" % (item.GetName(), exc))
            continue
        media_to_tl = (tl_start / fps) - (start / clip_fps)
        shifted = [transcribe.Word(w.start + media_to_tl, w.end + media_to_tl, w.text) for w in words]
        events += transcribe.group_captions(shifted, opts.max_words, opts.max_dur)
        log("  %d words (%s)" % (len(words), lang or "?"))
    if not events:
        log("No speech found.")
        return ""
    events.sort(key=lambda e: e[0])
    srt = transcribe.to_srt(events)
    srt_path = opts.srt_path or os.path.join(tempfile.gettempdir(), "face_reframe_subtitles.srt")
    with open(srt_path, "w", encoding="utf-8") as fh:
        fh.write(srt)
    log("Wrote %s (%d captions)" % (srt_path, len(events)))
    if _import_subtitles(timeline, project, srt_path, log):
        log("Subtitles imported into the timeline.")
    else:
        log("Import %s via File > Import > Subtitle (or drag it onto the timeline)." % srt_path)
    return srt_path


def _import_subtitles(timeline, project, srt_path: str, log: Log) -> bool:
    # fuscript exposes missing methods as None attributes — check callability,
    # not hasattr, and swallow anything the host build doesn't support.
    import_fn = getattr(timeline, "ImportSubtitles", None)
    if callable(import_fn):
        try:
            if import_fn(srt_path):
                return True
        except Exception as exc:
            log("  ImportSubtitles failed (%s)" % exc)
    # Fallback: import the .srt into the media pool and append it — Resolve
    # accepts subtitle files there and creates the subtitle track itself.
    try:
        pool = project.GetMediaPool()
        items = pool.ImportMedia([srt_path]) or []
        if items and pool.AppendToTimeline(items):
            return True
    except Exception as exc:
        log("  media-pool subtitle import failed (%s)" % exc)
    return False
