"""Resolve integration: drop timeline markers where the transcript says so.

Two marker families:
* keyword hits (default colour Red) — editor cues said out loud while
  recording ("taglia", "rifacciamo", ...), configurable;
* topic breaks (Blue) — speech resuming after a long pause, titled with
  the first words of the new block.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .markers import keyword_hits, topic_marks
from .resolve_bridge import (Log, ResolveOptions, _float, _int, source_range,
                             target_items)
from .silence import extract_audio
from .transcribe import transcribe


@dataclass
class MarkerOptions:
    scope: str = "current"
    track_index: int = 1
    keywords: str = "taglia,rifacciamo,retake,daccapo,ricominciamo"
    """Comma-separated cue words to mark in red."""
    topics: bool = True
    min_pause: float = 4.0
    title_words: int = 6
    keyword_color: str = "Red"
    topic_color: str = "Blue"


def run_markers(resolve, opts: MarkerOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    items = target_items(timeline, ResolveOptions(scope=opts.scope,
                                                track_index=opts.track_index))
    if not items:
        log("No clip found.")
        return 0
    timeline_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    keys = [k.strip() for k in opts.keywords.split(",") if k.strip()]
    added = 0
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            continue
        start, count = source_range(item)
        fps = _float(pool_item.GetClipProperty("FPS"), timeline_fps)
        try:
            pcm = extract_audio(path, start / fps, count / fps)
            words, _lang = transcribe(
                pcm, progress=lambda f: log("  transcribing %d%%" % int(f * 100)),
                log=log)
        except Exception as exc:
            log("  transcription failed on %s (%s); skipped" % (item.GetName(), exc))
            continue
        rec0 = _int(item.GetStart())

        def put(sec, color, name, note, rec0=rec0):
            try:
                frame = rec0 + int(round(sec * timeline_fps))
                return 1 if timeline.AddMarker(frame, color, name, note, 1) else 0
            except Exception:
                return 0

        n_key = sum(put(t, opts.keyword_color, "cue: %s" % k, k)
                    for t, k in keyword_hits(words, keys))
        n_top = (sum(put(t, opts.topic_color, title, "") for t, title in
                     topic_marks(words, opts.min_pause, opts.title_words))
                 if opts.topics else 0)
        log("  %s: %d cue marker(s), %d topic marker(s)" % (item.GetName(), n_key, n_top))
        added += n_key + n_top
    log("Done: %d marker(s) added." % added)
    return added
