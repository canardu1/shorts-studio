"""Resolve integration for music ducking: every standalone audio item gets
a ducked copy (voice as the sidechain key) relinked in place."""

from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .cinematic import output_dir_for
from .duck import DuckParams, bake_ducked, build_voice_key, slice_key
from .resolve_bridge import (Log, ResolveOptions, _float, _int, source_range,
                             standalone_audio_items, target_items)

_AUDIO_EXTS = {".wav", ".aif", ".aiff", ".m4a", ".mp3", ".flac", ".ogg", ".bwf"}


@dataclass
class DuckOptions:
    params: DuckParams = field(default_factory=DuckParams)


def _out_ext(path: str) -> str:
    """Keep a WAV for WAV sources; containers with video get an m4a."""
    return ".wav" if os.path.splitext(path)[1].lower() in _AUDIO_EXTS else ".m4a"


def run_duck(resolve, opts: DuckOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    video_items = target_items(timeline, ResolveOptions(scope="all"))
    audio_items = standalone_audio_items(timeline, video_items)
    if not audio_items:
        log("No standalone audio item found (nothing to duck — the music "
            "should sit on its own audio track, e.g. A2).")
        return 0
    tl_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    tl_start = _int(timeline.GetStartFrame())
    end_s = 0.0
    segments = []
    for v in video_items:
        pool = v.GetMediaPoolItem()
        vpath = pool.GetClipProperty("File Path") if pool else ""
        if not vpath or not os.path.exists(vpath):
            continue
        vrec, vdur = _int(v.GetStart()) - tl_start, _int(v.GetDuration())
        vs, vc = source_range(v)
        fps = _float(pool.GetClipProperty("FPS"), tl_fps)
        segments.append((vpath, vs / fps, vc / fps, vrec / tl_fps))
        end_s = max(end_s, (vrec + vdur) / tl_fps)
    for it in audio_items:
        end_s = max(end_s, (_int(it.GetStart()) - tl_start
                            + _int(it.GetDuration())) / tl_fps)
    log("Duck Music: %d audio item(s) under %d video clip(s)."
        % (len(audio_items), len(video_items)))
    key = build_voice_key(segments, end_s, log)
    if not np.any(key):
        log("No voice content on the video tracks; nothing to duck against.")
        return 0

    ducked = 0
    for it in audio_items:
        pool_item = it.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % it.GetName())
            continue
        name = it.GetName()
        rec, dur = _int(it.GetStart()) - tl_start, _int(it.GetDuration())
        s0, _c = source_range(it)
        item_fps = _float(pool_item.GetClipProperty("FPS"), tl_fps)
        total_f = _int(pool_item.GetClipProperty("Frames")) or int(dur * item_fps / tl_fps)
        # where file frame 0 sits on the record axis -> the key slice the
        # ducked file must align to
        off_s = (rec - s0 * tl_fps / item_fps) / tl_fps
        item_key = slice_key(key, off_s, total_f / item_fps)
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir_for(path),
                                "%s_ducked%s" % (base, _out_ext(path)))
        try:
            bake_ducked(path, item_key, out_path, opts.params,
                        total_s=total_f / item_fps)
            ok = pool_item.ReplaceClip(out_path) if hasattr(pool_item, "ReplaceClip") else False
        except Exception as exc:
            log("  %s: duck bake failed (%s); skipped" % (name, exc))
            continue
        if ok:
            log("  %s: ducked -> relinked to %s (affects every use of that "
                "clip)" % (name, os.path.basename(out_path)))
            ducked += 1
        else:
            log("  %s: ducked copy written to %s — relink the clip to it "
                "manually" % (name, out_path))
    log("Done: %d music item(s) ducked under the voice." % ducked)
    return ducked
