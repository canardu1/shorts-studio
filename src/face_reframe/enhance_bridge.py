"""Resolve integration for the standalone Enhance pass: sharpen/AI-restore
each clip at its own resolution and rebuild the timeline as '(enhanced)'.

Unlike the Enhance option inside Face Reframe (which acts on the reframe
crop), this tool works on the timeline items as they are — after a reframe
render, on native vertical clips, anywhere. One bake per source file;
standalone audio items are carried over untouched.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from .cinematic import output_dir_for
from .enhance import EnhanceOptions, Enhancer, bake_enhanced
from .resolve_bridge import (Log, ResolveOptions, _float, _int, record_frame,
                             source_range, standalone_audio_items,
                             target_items)

_MODES = {"fast": "fast", "ai": "ai", "ai-hq": "ai"}


@dataclass
class EnhanceRunOptions:
    scope: str = "current"
    track_index: int = 1
    mode: str = "fast"
    """``fast`` | ``ai`` (half-size model, quicker) | ``ai-hq`` (full)."""


def _params(mode: str) -> EnhanceOptions:
    return EnhanceOptions(mode=_MODES.get(mode, "fast"),
                          ai_fraction=1.0 if mode == "ai-hq" else 0.5)


def _entry(media_pool_item, start, count, media_type, rec):
    e = {"mediaPoolItem": media_pool_item, "startFrame": start,
         "endFrame": start + count}
    if media_type:
        e["mediaType"] = media_type
    if rec is not None:
        e["recordFrame"] = rec
    return e


def run_enhance(resolve, opts: EnhanceRunOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    video_items = target_items(
        timeline, ResolveOptions(scope=opts.scope,
                                 track_index=opts.track_index))
    if not video_items:
        log("No clip found.")
        return 0
    tl_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    try:
        enhancer = Enhancer(_params(opts.mode), log)
    except RuntimeError as exc:
        log(str(exc))
        return 0

    groups = {}
    entries = []
    for pos, item in enumerate(video_items):
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            continue
        groups.setdefault(path, []).append((pos, item, pool_item))

    # standalone audio (lav on A2, music) is carried over untouched at its
    # position — enhance doesn't change timing, so it stays in sync
    for it in standalone_audio_items(timeline, video_items):
        pool_item = it.GetMediaPoolItem()
        s, c = source_range(it)
        if pool_item is not None:
            total_f = _int(pool_item.GetClipProperty("Frames"))
            if total_f > 0:
                s = max(0, min(s, total_f - 1))
                c = max(1, min(c, total_f - s))
            entries.append((len(video_items) + len(entries),
                            _entry(pool_item, s, c, 2,
                                   record_frame(timeline, it))))

    if not groups:
        log("Nothing to enhance.")
        return 0
    log("Enhance (%s): %d clip(s) on %d source file(s)."
        % (opts.mode, len(video_items), len(groups)))

    pool = project.GetMediaPool()
    baked = 0
    for gi, (path, members) in enumerate(sorted(groups.items())):
        base = os.path.splitext(os.path.basename(path))[0]
        log("[%d/%d] %s" % (gi + 1, len(groups), os.path.basename(path)))
        keeps = [(pos, _entry(pool_item, *source_range(item), None,
                              record_frame(timeline, item)))
                 for pos, item, pool_item in members]
        out_path = os.path.join(output_dir_for(path),
                                "%s_enh_%s.mp4" % (base, opts.mode))
        try:
            seen = [-1]

            def pct(frac):
                p = int(frac * 100) // 10 * 10
                if p != seen[0]:
                    seen[0] = p
                    log("  enhancing %d%%" % p)

            bake_enhanced(path, out_path, enhancer, progress=pct)
            clips = pool.ImportMedia([out_path]) or []
        except Exception as exc:
            log("  bake failed (%s); keeping the original" % exc)
            entries += keeps
            continue
        if not clips:
            log("  import failed for %s; keeping the original" % out_path)
            entries += keeps
            continue
        # the bake mirrors the source frame-for-frame, ranges carry over
        entries += [(pos, _entry(clips[0], *source_range(item), None,
                                 record_frame(timeline, item)))
                    for pos, item, pool_item in members]
        log("  -> %s" % os.path.basename(out_path))
        baked += len(members)

    if baked:
        name = "%s (enhanced)" % timeline.GetName()
        new_tl = pool.CreateEmptyTimeline(name)
        if new_tl is None:
            log("Could not create the enhanced timeline.")
            return baked
        project.SetCurrentTimeline(new_tl)
        for key in ("useCustomSettings", "timelineResolutionWidth",
                    "timelineResolutionHeight", "timelineFrameRate"):
            value = timeline.GetSetting(key)
            if value:
                new_tl.SetSetting(key, value)
        entries.sort(key=lambda e: e[0])
        pool.AppendToTimeline([e for _, e in entries])
        log("Created timeline '%s'." % name)
    log("Done: %d clip(s) enhanced." % baked)
    return baked
