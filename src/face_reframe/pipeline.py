"""Analysis -> camera path -> crop plan, shared by the CLI and the Resolve script."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np

from .detect import Analysis, analyze_video
from .geometry import CropPlan, crop_size, parse_aspect, plan_crops
from .track import CameraPath, ShotSettings, TrackSettings, build_camera_path


@dataclass
class Options:
    aspect: str = "9:16"
    zoom: float = 1.0
    follow_vertical: bool = False
    face_height_pos: float = 0.4
    stride: int = 2
    min_score: float = 0.6
    track: TrackSettings = field(default_factory=TrackSettings)
    shots: ShotSettings = field(default_factory=ShotSettings)


@dataclass
class Result:
    analysis: Analysis
    path: CameraPath
    plan: CropPlan


def reframe(
    video: str,
    start_frame: int = 0,
    frame_count: Optional[int] = None,
    options: Optional[Options] = None,
    progress: Optional[Callable[[float], None]] = None,
) -> Result:
    o = options or Options()
    analysis = analyze_video(video, start_frame, frame_count, stride=o.stride, min_score=o.min_score, progress=progress)
    return plan_from_analysis(analysis, o)


def plan_from_analysis(analysis: Analysis, options: Optional[Options] = None) -> Result:
    """Camera path + crop plan from an existing face analysis.

    Cheap compared to detection, so a UI can re-run it live while the user
    tweaks smoothing / zoom / aspect without re-detecting faces."""
    o = options or Options()
    aspect = parse_aspect(o.aspect)
    size = crop_size(analysis.width, analysis.height, aspect, o.zoom)
    path = build_camera_path(analysis, o.track, size, o.shots)
    plan = plan_crops(
        analysis.width, analysis.height, path.x, path.y, aspect,
        zoom=o.zoom * path.zoom,
        follow_vertical=np.logical_or(o.follow_vertical, path.zoom > 1.0),
        face_height_pos=o.face_height_pos,
    )
    return Result(analysis, path, plan)


def simplify_keys(x: np.ndarray, y: np.ndarray, tolerance: float = 0.5, cuts: Tuple[int, ...] = ()) -> List[int]:
    """Frames to keep so linear interpolation stays within ``tolerance`` px.

    Frames on either side of a cut are always kept so jumps stay hard.
    """
    n = len(x)
    if n <= 2:
        return list(range(n))
    anchors = sorted({0, n - 1} | {c for c in cuts if 0 < c < n} | {c - 1 for c in cuts if 0 < c < n})
    keep = set(anchors)
    stack = list(zip(anchors[:-1], anchors[1:]))
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        t = (np.arange(a + 1, b) - a) / float(b - a)
        ex = np.abs(x[a + 1:b] - (x[a] + (x[b] - x[a]) * t))
        ey = np.abs(y[a + 1:b] - (y[a] + (y[b] - y[a]) * t))
        err = np.maximum(ex, ey)
        i = int(np.argmax(err))
        if err[i] > tolerance:
            m = a + 1 + i
            keep.add(m)
            stack.append((a, m))
            stack.append((m, b))
    return sorted(keep)
