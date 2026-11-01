"""Resolve integration for scene-cut splitting: detect hard cuts inside each
clip and rebuild the timeline with one item per shot."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import cv2

from .detect import CutDetector, iter_frames
from .resolve_bridge import Log, ResolveOptions, _float, source_range, target_items


@dataclass
class SplitOptions:
    scope: str = "current"
    track_index: int = 1
    cut_threshold: float = 0.45
    min_shot: float = 0.5
    """Shots shorter than this (seconds) are merged back into the previous one."""


def shot_boundaries(path: str, start: int, count: int, opts: SplitOptions) -> List[Tuple[int, int]]:
    """Contiguous (start, end] source-frame ranges, one per detected shot."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError("Cannot open video: %s" % path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    detector = CutDetector(hist_threshold=opts.cut_threshold)
    cuts: List[int] = []
    for i, frame in enumerate(iter_frames(cap, start, count)):
        if detector(frame):
            cuts.append(start + i)
    cap.release()
    bounds = [start] + cuts + [start + count]
    min_len = max(1, int(round(opts.min_shot * fps)))
    ranges: List[Tuple[int, int]] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b - a < min_len and ranges:
            prev_a, _ = ranges[-1]
            ranges[-1] = (prev_a, b)
        else:
            ranges.append((a, b))
    return ranges


def run_split(resolve, opts: SplitOptions, log: Log = print) -> int:
    project = resolve.GetProjectManager().GetCurrentProject()
    timeline = project.GetCurrentTimeline() if project else None
    if timeline is None:
        log("Open a timeline first.")
        return 0
    items = target_items(timeline, ResolveOptions(scope=opts.scope, track_index=opts.track_index))
    if not items:
        log("No clip found.")
        return 0

    entries: List[dict] = []
    shot_count = 0
    for item in items:
        pool_item = item.GetMediaPoolItem()
        path = pool_item.GetClipProperty("File Path") if pool_item else ""
        if not path or not os.path.exists(path):
            log("Skipping %s: source file not found" % item.GetName())
            continue
        start, count = source_range(item)
        log("Detecting cuts in %s (frames %d-%d)" % (item.GetName(), start, start + count - 1))
        try:
            ranges = shot_boundaries(path, start, count, opts)
        except Exception as exc:
            log("  cut detection failed (%s); keeping the clip whole" % exc)
            ranges = [(start, start + count)]
        if len(ranges) > 1:
            log("  %d shots" % len(ranges))
        for a, b in ranges:
            entries.append({"mediaPoolItem": pool_item, "startFrame": a, "endFrame": b})
            shot_count += 1
    if not entries:
        log("Nothing to split.")
        return 0

    pool = project.GetMediaPool()
    name = "%s (split)" % timeline.GetName()
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
    log("Done: %d shots in '%s'." % (shot_count, name))
    return shot_count
