"""Resolve integration for punch-in zooms: a Fusion Transform node with
keyframed Size, inserted just before MediaOut on each target clip."""

from __future__ import annotations

from dataclasses import dataclass, field

import os

from . import silence
from .punch import PunchParams, curve_keys, punch_curve, punch_curve_synced
from .resolve_bridge import Log, ResolveOptions, _float, _int, _mark_comp_dirty, source_range, target_items

PUNCH_NAME = "PunchInZoom"


@dataclass
class PunchOptions:
    scope: str = "current"
    track_index: int = 1
    params: PunchParams = field(default_factory=PunchParams)
    sync: bool = True
    """Toggle the zoom at each speech-phrase start; falls back to the fixed
    interval when no speech is detected."""
    speech: silence.SpeechParams = field(default_factory=silence.SpeechParams)


def _curve_for(item, count: int, fps: float, opts: PunchOptions, log: Log):
    """Zoom curve: voice-synced when possible, timed otherwise."""
    if opts.sync:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if path and os.path.exists(path):
            start, _ = source_range(item)
            clip_fps = _float(pool_item.GetClipProperty("FPS"), fps)
            try:
                pcm = silence.extract_audio(path, start / clip_fps, count / clip_fps)
                spans = silence.speech_intervals(pcm, opts.speech)
            except Exception as exc:
                spans = []
                log("  speech analysis failed (%s); using the timer" % exc)
            if len(spans) >= 2:
                log("  voice-synced: zoom toggles at %d phrase starts" % len(spans))
                return punch_curve_synced([a for a, _ in spans], count, fps, opts.params)
    return punch_curve(count, fps, opts.params)


def apply_punch_in(item, timeline_fps: float, opts: PunchOptions, log: Log) -> None:
    params = opts.params
    comp = item.GetFusionCompByIndex(1) if _int(item.GetFusionCompCount()) > 0 else item.AddFusionComp()
    if comp is None:
        raise RuntimeError("Could not create a Fusion composition on %s" % item.GetName())
    comp.Lock()
    comp.StartUndo("Punch In")
    node = None
    try:
        media_out = (comp.GetToolList(False, "MediaOut") or {}).get(1)
        if media_out is None:
            raise RuntimeError("No MediaOut node in the Fusion comp of %s" % item.GetName())
        attrs = comp.GetAttrs() or {}
        t0 = _int(attrs.get("COMPN_RenderStart"), 0)
        _, count = source_range(item)

        node = comp.FindTool(PUNCH_NAME)
        if node is None:
            upstream = media_out.Input.GetConnectedOutput()
            if upstream is None:
                raise RuntimeError("Nothing feeds MediaOut on %s" % item.GetName())
            node = comp.AddTool("Transform", -32768, -32768)
            node.SetAttrs({"TOOLS_Name": PUNCH_NAME})
            node.Input = upstream
            media_out.Input = node.Output

        curve = _curve_for(item, count, timeline_fps or 25.0, opts, log)
        keys = curve_keys(curve)
        node.Size = None
        node.AddModifier("Size", "BezierSpline")
        for f, v in keys:
            node.SetInput("Size", v, t0 + f)
        log("  %d zoom keyframes on %s (%.0f%% every %.1fs)"
            % (len(keys), PUNCH_NAME, params.zoom * 100.0, params.interval))
    finally:
        comp.EndUndo(True)
        comp.Unlock()
    if node is not None:
        _mark_comp_dirty(comp, node, log)


def run_punch(resolve, opts: PunchOptions, log: Log = print) -> int:
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
    done = 0
    for item in items:
        try:
            apply_punch_in(item, timeline_fps, opts, log)
            done += 1
        except Exception as exc:
            log("  punch-in failed on %s: %s" % (item.GetName(), exc))
    log("Done: %d clip(s) with punch-in." % done)
    return done
