"""Resolve integration: sync standalone audio items to the camera track.

The free version of Resolve has no audio auto-sync — this is the PluralEyes
use case. For each unattached audio item (a lav/boom recording on A2+), the
camera audio under the item's timeline span is rebuilt, cross-correlated
with the recorder file (GCC-PHAT), and the media pool clip is relinked to a
shifted copy — so every use of it stays aligned without touching the edit.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from .cinematic import output_dir_for
from .resolve_bridge import (Log, ResolveOptions, _float, _int, source_range,
                             standalone_audio_items, target_items)
from .silence import SAMPLE_RATE, extract_audio
from .syncaudio import aligned_copy, find_offset

MAX_REF_S = 120.0
"""The correlation only needs a couple of minutes of camera audio."""


@dataclass
class SyncOptions:
    min_score: float = 0.03
    """GCC-PHAT peak below which a match is reported as not found."""
    min_shift_s: float = 0.02
    """Corrections smaller than this are reported as already in sync."""


def _camera_ref(video_items, rec: int, dur: int, tl_fps: float,
                log: Log) -> np.ndarray:
    """Camera audio laid out at timeline positions, over [rec, rec+dur]."""
    n = min(dur, int(MAX_REF_S * tl_fps))
    ref = np.zeros(int(n / tl_fps * SAMPLE_RATE), dtype=np.float32)
    for v in video_items:
        pool = v.GetMediaPoolItem()
        vpath = pool.GetClipProperty("File Path") if pool else ""
        if not vpath or not os.path.exists(vpath):
            continue
        vrec, vdur = _int(v.GetStart()), _int(v.GetDuration())
        vs, _vc = source_range(v)
        fps = _float(pool.GetClipProperty("FPS"), tl_fps)
        lo, hi = max(rec, vrec), min(rec + n, vrec + vdur)
        if hi <= lo:
            continue
        off_s = (lo - vrec) / tl_fps            # seconds into the item
        dur_s = (hi - lo) / tl_fps
        try:
            pcm = extract_audio(vpath, vs / fps + off_s, dur_s)
        except Exception as exc:
            log("  camera audio decode failed on %s (%s)" % (v.GetName(), exc))
            continue
        at = int((lo - rec) / tl_fps * SAMPLE_RATE)
        ref[at:at + len(pcm)] += pcm[:max(0, len(ref) - at)]
    return ref


def run_sync(resolve, opts: SyncOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    video_items = target_items(timeline, ResolveOptions(scope="all"))
    audio_items = standalone_audio_items(timeline, video_items)
    if not audio_items:
        log("No standalone audio item found (nothing on the audio tracks "
            "that isn't already linked to a video clip).")
        return 0
    tl_fps = _float(timeline.GetSetting("timelineFrameRate"), 25.0)
    tl_start = _int(timeline.GetStartFrame())
    synced = 0
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
        ref = _camera_ref(video_items, rec, dur, tl_fps, log)
        if not np.any(ref):
            log("  %s: no camera audio under it; skipped" % name)
            continue
        try:
            ext = extract_audio(path)
            lag_s, score = find_offset(
                ref, ext,
                progress=lambda f: log("  correlating %d%%" % int(f * 100)))
        except Exception as exc:
            log("  %s: correlation failed (%s); skipped" % (name, exc))
            continue
        if score < opts.min_score:
            log("  %s: no reliable match (score %.3f); skipped — is the "
                "right take on the timeline?" % (name, score))
            continue
        shift_s = lag_s - s0 / item_fps
        if abs(shift_s) < opts.min_shift_s:
            log("  %s: already in sync (%.0f ms off)" % (name, shift_s * 1000))
            synced += 1
            continue
        base = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(output_dir_for(path), "%s_synced.wav" % base)
        try:
            aligned_copy(path, out_path, shift_s)
            ok = pool_item.ReplaceClip(out_path) if hasattr(pool_item, "ReplaceClip") else False
        except Exception as exc:
            log("  %s: align bake failed (%s); skipped" % (name, exc))
            continue
        if ok:
            log("  %s: shifted %+0.2fs -> relinked to %s (affects every use "
                "of that clip)" % (name, shift_s, os.path.basename(out_path)))
            synced += 1
        else:
            log("  %s: aligned copy written to %s — relink the clip to it "
                "manually (this Resolve can't ReplaceClip)" % (name, out_path))
    log("Done: %d audio item(s) in sync." % synced)
    return synced
