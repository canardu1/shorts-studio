"""Resolve integration for loudness normalization: measure each clip's
integrated loudness and set its audio gain so everything sits at the target."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass

from .loudness import TARGET_I, bake_normalized, gain_db, measure_integrated
from .resolve_bridge import (Log, ResolveOptions, _float, record_frame,
                             source_range, standalone_audio_items,
                             target_items)


@dataclass
class LoudnessOptions:
    scope: str = "current"
    track_index: int = 1
    target: float = TARGET_I


def _apply_gain(item, pool_item, gain: float, log: Log) -> bool:
    """Set the clip gain through whichever API the Resolve version exposes."""
    for key in ("Volume", "Clip Gain"):
        try:
            if item.SetProperty(key, "%.1f" % gain):
                return True
        except Exception:
            pass
    for key in ("Clip Gain", "Audio Gain"):
        try:
            if pool_item is not None and pool_item.SetClipProperty(key, "%.1f" % gain):
                log("  (gain set on the media pool clip — it affects every use of it)")
                return True
        except Exception:
            pass
    return False


def run_loudness(resolve, opts: LoudnessOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    items = target_items(timeline, ResolveOptions(scope=opts.scope, track_index=opts.track_index))
    audio_ids = set()
    if opts.scope == "all":
        # standalone audio items too (e.g. separately recorded dialogue on
        # A2); the linked halves of A/V clips are covered by the video items
        audio = standalone_audio_items(timeline, items)
        audio_ids = {id(it) for it in audio}
        items += audio
    if not items:
        log("No clip found.")
        return 0
    timeline_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    done = 0
    entries = []      # timeline-app-ready segments for the fallback timeline
    pending = []      # items needing a baked (rewrapped) normalized copy
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            continue
        start, count = source_range(item)
        fps = _float(pool_item.GetClipProperty("FPS"), timeline_fps)
        try:
            measured = measure_integrated(path, start / fps, count / fps)
        except Exception as exc:
            log("  loudness measure failed on %s (%s); skipped" % (item.GetName(), exc))
            continue
        gain = gain_db(measured, opts.target)
        if _apply_gain(item, pool_item, gain, log):
            log("  %s: %.1f LUFS -> %+0.1f dB" % (item.GetName(), measured, gain))
            entries.append({"mediaPoolItem": pool_item, "startFrame": start,
                            "endFrame": start + count, "item": item})
            done += 1
        else:
            log("  %s: %.1f LUFS (%+0.1f dB) — no gain API on this Resolve, baking a normalized copy" % (item.GetName(), measured, gain))
            pending.append((item, pool_item, path, start, count, gain, measured))

    pool = project.GetMediaPool()
    baked = 0
    for item, pool_item, path, start, count, gain, measured in pending:
        base = os.path.splitext(path)[0]
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", item.GetName() or "clip")
        out_path = "%s_norm_%s.mp4" % (base, safe)
        try:
            bake_normalized(path, out_path, gain)
            clips = pool.ImportMedia([out_path]) or []
        except Exception as exc:
            log("  bake failed on %s (%s); keeping the original clip" % (item.GetName(), exc))
            entries.append({"mediaPoolItem": pool_item, "startFrame": start,
                            "endFrame": start + count, "item": item})
            continue
        if not clips:
            log("  import failed for %s; keeping the original clip" % out_path)
            entries.append({"mediaPoolItem": pool_item, "startFrame": start,
                            "endFrame": start + count, "item": item})
            continue
        # the copy mirrors the source file, so source frames carry over
        entries.append({"mediaPoolItem": clips[0], "startFrame": start,
                        "endFrame": start + count, "item": item})
        log("  %s: baked %s" % (item.GetName(), os.path.basename(out_path)))
        baked += 1

    if baked:
        name = "%s (normalized)" % timeline.GetName()
        new_tl = pool.CreateEmptyTimeline(name)
        if new_tl is None:
            log("Could not create the normalized timeline.")
            return done
        project.SetCurrentTimeline(new_tl)
        for key in ("useCustomSettings", "timelineResolutionWidth", "timelineResolutionHeight",
                    "timelineFrameRate"):
            value = timeline.GetSetting(key)
            if value:
                new_tl.SetSetting(key, value)
        clean = []
        for e in entries:
            d = {"mediaPoolItem": e["mediaPoolItem"], "startFrame": e["startFrame"],
                 "endFrame": e["endFrame"]}
            rec = record_frame(timeline, e["item"])
            if rec is not None:
                d["recordFrame"] = rec
            if id(e["item"]) in audio_ids:
                d["mediaType"] = 2
            clean.append(d)
        pool.AppendToTimeline(clean)
        log("Created timeline '%s' with %d baked clip(s)." % (name, baked))
    log("Done: %d clip(s) normalized to %.0f LUFS." % (done + baked, opts.target))
    return done + baked
