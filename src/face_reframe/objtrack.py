"""Track any picked object with NanoTrackV2 (OpenCV's ``TrackerNano``).

The user drags a box on the preview; the tracker follows that object
forward and backward over the video (~30 ms/frame) and stores the boxes in
the analysis samples' ``salient`` slot — the same slot the neural saliency
pass uses, so the "follow object" camera logic applies unchanged.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

from . import models
from .i18n import tr

MIN_SIDE = 8          # px — a tracked box smaller than this is considered lost


def available() -> bool:
    import cv2
    return (hasattr(cv2, "TrackerNano_create")
            and models.available("nanotrack_backbone") and models.available("nanotrack_head"))


def ensure(progress=None, log=print) -> bool:
    import cv2
    if not hasattr(cv2, "TrackerNano_create"):
        log(tr("Object tracking needs OpenCV with NanoTrack support"))
        return False
    ok = (models.ensure("nanotrack_backbone", progress, log) is not None
          and models.ensure("nanotrack_head", progress, log) is not None)
    return ok


def _tracker():
    import cv2
    p = cv2.TrackerNano_Params()
    p.backbone = models.path("nanotrack_backbone")
    p.neckhead = models.path("nanotrack_head")
    return cv2.TrackerNano_create(p)


def _box(r) -> Tuple[float, float, float, float]:
    return (float(r[0] + r[2] / 2), float(r[1] + r[3] / 2), float(r[2]), float(r[3]))


def _forward(cap, grid_set, f0: int, last: int, first_box,
             off: int, progress=None) -> Dict[int, Tuple[float, float, float, float]]:
    """Track forward from ``f0`` with sequential reads (no per-frame seeks —
    each set() on a long-GOP file can cost a keyframe re-decode)."""
    import cv2
    out = {}
    tr = _tracker()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, frame = cap.read()
    if not ok:
        return out
    tr.init(frame, tuple(int(round(v)) for v in first_box))
    if f0 - off in grid_set:
        out[f0 - off] = first_box
    f = f0
    while f < last:
        ok, frame = cap.read()
        if not ok:
            break
        f += 1
        good, r = tr.update(frame)
        if not good or min(r[2], r[3]) < MIN_SIDE:
            break
        rel = f - off
        if rel in grid_set:
            out[rel] = _box(r)
        if progress and f % 25 == 0:
            progress(f)
    return out


def _backward(cap, grid_set, f0: int, first_box, off: int,
              progress=None) -> Dict[int, Tuple[float, float, float, float]]:
    """Track backward from ``f0``: cv2 can't step backwards, so this seeks
    one frame at a time (each set() can re-decode a GOP on heavy files)."""
    import cv2
    out = {}
    tr = _tracker()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, frame = cap.read()
    if not ok:
        return out
    tr.init(frame, tuple(int(round(v)) for v in first_box))
    f = f0
    while f > 0:
        f -= 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        good, r = tr.update(frame)
        if not good or min(r[2], r[3]) < MIN_SIDE:
            break
        rel = f - off
        if rel in grid_set:
            out[rel] = _box(r)
        if progress and f % 25 == 0:
            progress(f)
    return out


def track(path: str, start_abs: int, box: Tuple[float, float, float, float],
          analysis, log=print, progress=None) -> int:
    """Fill ``sample.salient`` with the tracked box, forward and backward
    from ``start_abs`` (absolute frame number, as the slider reports).
    Returns how many samples got a box."""
    import cv2
    if not ensure(progress, log):
        return 0
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    off = int(getattr(analysis, "start_frame", 0))
    grid = {smp.frame for smp in analysis.samples}
    if not grid or total <= 0:
        cap.release()
        return 0
    start_abs = min(max(0, start_abs), total - 1)
    lo, hi = off, off + getattr(analysis, "frame_count", total) - 1
    if not (lo <= start_abs <= hi):
        log(tr("Pick a frame inside the analysed range"))
        cap.release()
        return 0
    last = min(total - 1, hi)
    fwd_span = max(1, last - start_abs)
    bwd_span = max(0, start_abs - off)
    span = fwd_span + bwd_span

    def fwd_prog(f):
        if progress:
            progress(min(0.99, (f - start_abs) / span))

    def bwd_prog(f):
        if progress:
            progress(min(0.99, (fwd_span + start_abs - f) / span))

    if progress:
        progress(0.0)
    boxes = _forward(cap, grid, start_abs, last, box, off, fwd_prog)
    if bwd_span:
        boxes.update(_backward(cap, grid, start_abs, box, off, bwd_prog))
    cap.release()
    if progress:
        progress(1.0)

    by = {smp.frame: smp for smp in analysis.samples}
    for rel, b in boxes.items():
        if rel in by:
            by[rel].salient = b
    n = len([rel for rel in boxes if rel in by])
    covered = n * (sorted(grid)[1] - sorted(grid)[0] if len(grid) > 1 else 1)
    log(tr("Tracked %d sample%s (~%.1f s of video)") % (
        n, "s" if n != 1 else "", covered / (getattr(analysis, "fps", 25.0) or 25.0)))
    return n
