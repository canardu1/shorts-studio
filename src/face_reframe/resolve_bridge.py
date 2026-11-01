"""DaVinci Resolve integration.

Two ways to apply a reframe to timeline clips:

* ``fusion`` (default, non-destructive): adds a Fusion comp to the clip with a
  ``Background`` at timeline resolution and a ``Merge`` whose ``Center`` is
  keyframed so the face stays in the middle of the vertical frame.
* ``render``: bakes a reframed file with ffmpeg next to the source, imports it
  and builds a new vertical timeline from it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np

from .geometry import parse_aspect
from .pipeline import Options, Result, reframe, simplify_keys

MERGE_NAME = "FaceReframeMerge"
BG_NAME = "FaceReframeCanvas"

Log = Callable[[str], None]


@dataclass
class ResolveOptions:
    reframe: Options = field(default_factory=Options)
    scope: str = "current"
    """``current`` = clip under the playhead, ``track`` = every clip on ``track_index``,
    ``all`` = every clip on every video track."""
    track_index: int = 1
    mode: str = "fusion"
    set_timeline_resolution: bool = True
    output_height: int = 1920
    key_tolerance: float = 0.5
    enhance: str = "off"
    """Render mode only: ``off`` | ``fast`` | ``ai`` — see ``enhance.py``."""


def get_resolve():
    try:
        import DaVinciResolveScript as dvr  # type: ignore
    except ImportError:
        return None
    return dvr.scriptapp("Resolve")


def _int(value, default=0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def timeline_size(timeline):
    return _int(timeline.GetSetting("timelineResolutionWidth"), 1920), _int(timeline.GetSetting("timelineResolutionHeight"), 1080)


def _set_vertical(timeline, width: int, height: int) -> bool:
    timeline.SetSetting("useCustomSettings", "1")
    return (timeline.SetSetting("timelineResolutionWidth", str(width))
            and timeline.SetSetting("timelineResolutionHeight", str(height)))


def make_timeline_vertical(project, timeline, aspect: float, height: int, log: Log):
    """Ensure a ``width x height`` canvas. Some Resolve builds refuse
    ``SetSetting`` on the current timeline but accept it on a fresh one —
    fall back to a duplicated ``<name> (vertical)`` timeline."""
    width = int(round(height * aspect / 2.0)) * 2 if aspect <= 1 else height
    height = height if aspect <= 1 else int(round(width / aspect / 2.0)) * 2
    if timeline_size(timeline) == (width, height):
        return timeline
    if _set_vertical(timeline, width, height):
        log("Timeline resolution set to %dx%d" % (width, height))
        return timeline
    dup = None
    try:
        pool = project.GetMediaPool() if project is not None else None
        dup = pool.DuplicateTimeline(timeline) if pool is not None else None
    except Exception:
        dup = None
    if dup is not None and _set_vertical(dup, width, height):
        project.SetCurrentTimeline(dup)
        log("Timeline resolution can't change on '%s' — working on the "
            "duplicate '%s' (%dx%d)" % (timeline.GetName(), dup.GetName(),
                                      width, height))
        return dup
    log("Could not change timeline resolution; set it to %dx%d in Timeline > "
        "Timeline Settings" % (width, height))
    return timeline


def target_items(timeline, opts: ResolveOptions) -> List:
    if opts.scope == "track":
        return list(timeline.GetItemListInTrack("video", opts.track_index) or [])
    if opts.scope == "all":
        items = []
        for track in range(1, _int(timeline.GetTrackCount("video")) + 1):
            items.extend(timeline.GetItemListInTrack("video", track) or [])
        return items
    item = timeline.GetCurrentVideoItem()
    return [item] if item else []


def source_range(item):
    """(first source frame, frame count) of a timeline item."""
    start = None
    if hasattr(item, "GetSourceStartFrame"):
        start = item.GetSourceStartFrame()
    if start is None:
        start = item.GetLeftOffset()
    return max(0, _int(start)), _int(item.GetDuration())


def record_frame(timeline, item):
    """Timeline frame where ``item`` sits, relative to the timeline start —
    the value ``AppendToTimeline`` expects in ``recordFrame``. ``None`` when
    the API can't tell."""
    try:
        return max(0, _int(item.GetStart()) - _int(timeline.GetStartFrame()))
    except Exception:
        return None


def standalone_audio_items(timeline, video_items):
    """Audio-track items that are NOT the linked half of an A/V clip.

    Every A/V clip exists as a video-track item plus a linked audio-track
    item covering the same source range — sweeping audio tracks blindly
    would process those twice. Only items with no links (or no ``video``
    counterpart at the same source range) are returned.
    """
    seen = set()
    for it in video_items:
        pool = it.GetMediaPoolItem()
        path = pool.GetClipProperty("File Path") if pool else ""
        seen.add((path,) + source_range(it))
    out = []
    try:
        tracks = _int(timeline.GetTrackCount("audio"))
    except Exception:
        return out
    for t in range(1, tracks + 1):
        try:
            items = timeline.GetItemListInTrack("audio", t) or []
        except Exception:
            continue
        for it in items:
            try:
                if it.GetLinkedItems():
                    continue
            except Exception:
                pass
            pool = it.GetMediaPoolItem()
            path = pool.GetClipProperty("File Path") if pool else ""
            if (path,) + source_range(it) in seen:
                continue
            out.append(it)
    return out


def comp_values(result: Result, out_w: int, out_h: int, media_w: Optional[float] = None):
    """Per-frame Merge ``Center`` (x, y) and ``Size`` arrays for the Fusion comp.

    Fusion keeps the foreground's pixel size relative to the background, so
    scaling by ``out_w / crop_w`` maps the crop window onto the canvas.
    """
    plan = result.plan
    src_w, src_h = plan.src_w, plan.src_h
    pixel_scale = (media_w / float(src_w)) if media_w else 1.0
    scale = out_w / plan.crop_w
    size = scale / pixel_scale
    xs = 0.5 - (plan.cx - src_w / 2.0) * scale / out_w
    ys = 0.5 + (plan.cy - src_h / 2.0) * scale / out_h
    return xs, ys, size


def _media_width(tool, time) -> Optional[float]:
    try:
        img = tool.Output[time]
        return float(img.Width) if img and img.Width else None
    except Exception:
        return None


def _mark_comp_dirty(comp, tool, log: Log) -> None:
    """Nudge the comp so the timeline sees it as modified and re-renders.

    Script-only edits don't flag the comp dirty; the timeline keeps showing
    cached frames until something changes the comp (like dragging a node).
    """
    try:
        attrs = tool.GetAttrs() or {}
        pos = attrs.get("TOOLS_LayerOriginX")
        if pos is not None:
            tool.SetAttrs({"TOOLS_LayerOriginX": pos})
        tool.SetAttrs({"TOOLB_PassThrough": True})
        tool.SetAttrs({"TOOLB_PassThrough": False})
    except Exception as exc:
        log("  (refresh nudge failed: %s)" % exc)


def apply_fusion(item, result: Result, out_w: int, out_h: int, opts: ResolveOptions, log: Log) -> None:
    comp = item.GetFusionCompByIndex(1) if _int(item.GetFusionCompCount()) > 0 else item.AddFusionComp()
    if comp is None:
        raise RuntimeError("Could not create a Fusion composition on %s" % item.GetName())
    comp.Lock()
    comp.StartUndo("Face Reframe")
    merge = None
    try:
        media_out = (comp.GetToolList(False, "MediaOut") or {}).get(1)
        media_in = (comp.GetToolList(False, "MediaIn") or {}).get(1)
        if media_out is None:
            raise RuntimeError("No MediaOut node in the Fusion comp of %s" % item.GetName())
        attrs = comp.GetAttrs() or {}
        t0 = _int(attrs.get("COMPN_RenderStart"), 0)

        merge = comp.FindTool(MERGE_NAME)
        bg = comp.FindTool(BG_NAME)
        if merge is None:
            upstream = media_out.Input.GetConnectedOutput()
            if upstream is None and media_in is not None:
                upstream = media_in.Output
            bg = comp.AddTool("Background", -32768, -32768)
            bg.SetAttrs({"TOOLS_Name": BG_NAME})
            merge = comp.AddTool("Merge", -32768, -32768)
            merge.SetAttrs({"TOOLS_Name": MERGE_NAME})
            merge.Background = bg.Output
            merge.Foreground = upstream
            media_out.Input = merge.Output
        bg.UseFrameFormatSettings = 0
        bg.Width = out_w
        bg.Height = out_h
        bg.TopLeftRed, bg.TopLeftGreen, bg.TopLeftBlue, bg.TopLeftAlpha = 0, 0, 0, 1

        media_w = _media_width(media_in, t0) if media_in is not None else None
        xs, ys, size = comp_values(result, out_w, out_h, media_w)
        cuts = tuple(a for a, _ in result.path.segments[1:])
        frames = simplify_keys(xs * out_w, ys * out_h, opts.key_tolerance, cuts)
        merge.Size = None
        if np.ptp(size) < 1e-6:
            merge.Size = float(size[0])
        else:
            merge.AddModifier("Size", "BezierSpline")
            for f in frames:
                merge.SetInput("Size", float(size[f]), t0 + f)
        merge.Center = None
        merge.AddModifier("Center", "XYPath")
        for f in frames:
            merge.SetInput("Center", {1: float(xs[f]), 2: float(ys[f])}, t0 + f)
        log("  %d keyframes on %s (size %.3f-%.3f)" % (len(frames), MERGE_NAME, size.min(), size.max()))
    finally:
        comp.EndUndo(True)
        comp.Unlock()
    if merge is not None:
        _mark_comp_dirty(comp, merge, log)


def apply_render(project, item, path: str, result: Result, out_w: int, out_h: int, log: Log, enhancer=None):
    from .render import render

    start, _ = source_range(item)
    base = os.path.splitext(path)[0]
    out = "%s_reframed_%dx%d_%d.mp4" % (base, out_w, out_h, start)
    log("  rendering %s" % out)
    render(path, result.plan, out, (out_w, out_h), result.analysis.fps, start,
           enhancer=enhancer)
    pool = project.GetMediaPool()
    clips = pool.ImportMedia([out]) or []
    return clips[0] if clips else None


def run(resolve, opts: ResolveOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    aspect = parse_aspect(opts.reframe.aspect)
    if opts.mode == "fusion" and opts.set_timeline_resolution:
        timeline = make_timeline_vertical(project, timeline, aspect,
                                          opts.output_height, log)
    items = target_items(timeline, opts)
    if not items:
        log("No clip found (put the playhead over a clip, or use the 'track'/'all' scope).")
        return 0
    out_w, out_h = timeline_size(timeline) if opts.mode == "fusion" else (
        int(round(opts.output_height * aspect / 2.0)) * 2, opts.output_height)

    timeline_fps = _float(timeline.GetSetting("timelineFrameRate"), 0)
    enhancer = None
    if opts.mode == "render" and opts.enhance != "off":
        from .enhance import EnhanceOptions, Enhancer
        enhancer = Enhancer(EnhanceOptions(mode=opts.enhance), log)
    rendered = []
    done = 0
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            continue
        clip_fps = _float(pool_item.GetClipProperty("FPS"), 0)
        if clip_fps and timeline_fps and abs(clip_fps - timeline_fps) > 0.01:
            log("Warning: %s is %.3f fps on a %.3f fps timeline; keyframes may drift" % (item.GetName(), clip_fps, timeline_fps))
        start, count = source_range(item)
        log("Analysing %s (frames %d-%d)" % (item.GetName(), start, start + count - 1))
        result = reframe(path, start, count, opts.reframe)
        found = int(result.path.has_face.sum())
        log("  face found on %d/%d frames, %d shot(s)" % (found, result.analysis.frame_count, len(result.path.segments)))
        for shot in result.path.shots:
            if shot.kind == "wide":
                log("  wide shot at frames %d-%d -> zoom %.2fx" % (start + shot.start, start + shot.end - 1, shot.zoom))
        if opts.mode == "render":
            clip = apply_render(project, item, path, result, out_w, out_h, log, enhancer)
            if clip:
                rendered.append(clip)
        else:
            apply_fusion(item, result, out_w, out_h, opts, log)
        done += 1

    if rendered:
        pool = project.GetMediaPool()
        name = "%s (Face Reframe)" % timeline.GetName()
        new_tl = pool.CreateTimelineFromClips(name, rendered)
        if new_tl:
            new_tl.SetSetting("useCustomSettings", "1")
            new_tl.SetSetting("timelineResolutionWidth", str(out_w))
            new_tl.SetSetting("timelineResolutionHeight", str(out_h))
            log("Created timeline '%s'" % name)
    log("Done: %d clip(s) reframed." % done)
    return done
