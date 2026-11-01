"""Resolve integration for cinematic audio: each clip's audio is cleaned and
sweetened with ffmpeg, then the timeline is rebuilt from the baked copies.

Resolve (free and Studio) exposes no per-clip audio-FX scripting API, so —
unlike Normalize Audio — there is no in-place path: every clip is baked.
Segments sharing one source file are processed once, not once per segment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .cinematic import (CinematicParams, bake_cinematic, combine_loudness,
                        measure_processed, noise_floor_db, output_dir_for,
                        speech_level_db)
from .loudness import TARGET_I, gain_db
from .resolve_bridge import (Log, ResolveOptions, _float, _int, record_frame,
                             source_range, standalone_audio_items,
                             target_items)
from .silence import extract_audio


@dataclass
class CinematicOptions:
    scope: str = "current"
    track_index: int = 1
    params: CinematicParams = field(default_factory=CinematicParams)
    target: float = TARGET_I


def _pct(log: Log, label: str):
    """Throttle ffmpeg fractions to one Console line per 10% step."""
    seen = [-1]

    def cb(frac):
        p = int(frac * 100) // 10 * 10
        if p != seen[0]:
            seen[0] = p
            log("  %s %d%%" % (label, p))
    return cb


def _entry(media_pool_item, start, count, media_type, rec):
    """Timeline-append dict.

    ``recordFrame`` restores the item's position in the edit (gaps and
    track-bumping included); older Resolves ignore it and pack items in
    order, which is the previous behaviour. ``mediaType`` is only ever set
    for standalone audio items — passing it (or trackIndex) on A/V entries
    has produced muted/blank items on some versions.
    """
    e = {"mediaPoolItem": media_pool_item, "startFrame": start,
         "endFrame": start + count}
    if media_type:
        e["mediaType"] = media_type
    if rec is not None:
        e["recordFrame"] = rec
    return e


def _targets(timeline, opts: CinematicOptions):
    """[(item, media_type)] — video items, plus standalone audio for 'all'.

    An A/V clip exists as a video item plus a *linked* audio item — those
    halves are skipped or each clip would be processed (and re-appended)
    twice. Only unattached audio (e.g. a lav recorded to A2) is picked up.
    """
    video_items = target_items(
        timeline, ResolveOptions(scope=opts.scope, track_index=opts.track_index))
    targets = [(it, None) for it in video_items]
    if opts.scope == "all":
        targets += [(it, 2)
                    for it in standalone_audio_items(timeline, video_items)]
    return targets


def run_cinematic(resolve, opts: CinematicOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    targets = _targets(timeline, opts)
    if not targets:
        log("No clip found.")
        return 0
    timeline_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)

    # group by source file: a (no pauses) timeline holds dozens of segments
    # of the same file — one bake serves them all
    groups = {}
    skipped = 0
    for pos, (item, mtype) in enumerate(targets):
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            skipped += 1
            continue
        start, count = source_range(item)
        if mtype:
            # audio items can report source ranges in units of their own —
            # clamp to the clip's frame count so entries never reference
            # audio that isn't there (muted items result otherwise)
            total_f = _int(pool_item.GetClipProperty("Frames"))
            if total_f > 0:
                start = max(0, min(start, total_f - 1))
                count = max(1, min(count, total_f - start))
        fps = _float(pool_item.GetClipProperty("FPS"), timeline_fps)
        groups.setdefault(path, []).append({
            "item": item, "mtype": mtype, "pool_item": pool_item,
            "start": start, "count": count, "fps": fps, "pos": pos,
            "rec": record_frame(timeline, item)})
    if not groups:
        log("Nothing to do.")
        return 0
    log("Cinematic Audio: %d clip(s) on %d source file(s)."
        % (len(targets) - skipped, len(groups)))

    entries = []   # (pos, timeline-append dict) — re-sorted to edit order
    baked = 0
    pool = project.GetMediaPool()
    for gi, (path, members) in enumerate(sorted(groups.items())):
        log("[%d/%d] %s (%d segment%s)"
            % (gi + 1, len(groups), os.path.basename(path), len(members),
               "s" if len(members) > 1 else ""))
        keeps = [(m["pos"], _entry(m["pool_item"], m["start"], m["count"],
                                 m["mtype"], m["rec"]))
                 for m in members]
        try:
            # the whole source, not just the used ranges: after Cut Pauses
            # the true noise bed only survives in the untouched file
            pcm = extract_audio(path)
        except Exception as exc:
            log("  %s — kept whole" % exc)
            entries += keeps
            continue
        floor = noise_floor_db(pcm)
        loud = speech_level_db(pcm)
        log("  noise floor %.0f dB, speech %.0f dB" % (floor, loud))
        # loudness of what will actually be heard: the union of used ranges
        ranges = sorted({(m["start"] / m["fps"], m["count"] / m["fps"])
                         for m in members})
        total = max(1e-9, sum(d for _, d in ranges))
        parts, done_d = [], 0.0
        pct = _pct(log, "measuring")
        try:
            for a, d in ranges:
                lufs = measure_processed(
                    path, a, d, opts.params, floor, speech_db=loud,
                    progress=lambda f, a=a, d=d: pct(
                        min(1.0, (done_d + f * d) / total)))
                parts.append((lufs, d))
                done_d += d
        except Exception as exc:
            log("  loudness measure failed (%s); keeping the original" % exc)
            entries += keeps
            continue
        measured = combine_loudness(parts)
        gain = gain_db(measured, opts.target)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir_for(path), "%s_cine.mp4" % base)
        try:
            bake_cinematic(path, out_path, opts.params, floor, gain,
                           speech_db=loud, progress=_pct(log, "baking"))
            clips = pool.ImportMedia([out_path]) or []
        except Exception as exc:
            log("  bake failed (%s); keeping the original" % exc)
            entries += keeps
            continue
        if not clips:
            log("  import failed for %s; keeping the original" % out_path)
            entries += keeps
            continue
        # the copy mirrors the source file, so source frames carry over
        entries += [(m["pos"], _entry(clips[0], m["start"], m["count"],
                                    m["mtype"], m["rec"]))
                    for m in members]
        log("  %.1f LUFS -> %+0.1f dB (%s)"
            % (measured, gain, os.path.basename(out_path)))
        baked += len(members)

    if baked:
        name = "%s (cinematic)" % timeline.GetName()
        new_tl = pool.CreateEmptyTimeline(name)
        if new_tl is None:
            log("Could not create the cinematic timeline.")
            return baked
        project.SetCurrentTimeline(new_tl)
        for key in ("useCustomSettings", "timelineResolutionWidth", "timelineResolutionHeight",
                    "timelineFrameRate"):
            value = timeline.GetSetting(key)
            if value:
                new_tl.SetSetting(key, value)
        entries.sort(key=lambda e: e[0])
        pool.AppendToTimeline([e for _, e in entries])
        log("Created timeline '%s' — run this tool before Face Reframe, Punch In "
            "and grading (comps/effects aren't carried over)." % name)
    log("Done: %d clip(s) sweetened to %.0f LUFS." % (baked, opts.target))
    return baked
