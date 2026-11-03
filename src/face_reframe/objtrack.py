"""Track any picked object with NanoTrackV2 (OpenCV's ``TrackerNano``).

The user drags a box on the preview; the tracker follows that object
forward and backward over the video (~30 ms/frame) and stores the boxes in
the analysis samples' ``salient`` slot — the same slot the neural saliency
pass uses, so the "follow object" camera logic applies unchanged.
"""

from __future__ import annotations

from typing import Dict, Tuple

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


def _run(cap, grid_set, f0: int, f1: int, step: int, first_box,
         total: int) -> Dict[int, Tuple[float, float, float, float]]:
    """Track frame-by-frame from f0 to f1 (exclusive), recording the box at
    every frame in ``grid_set``. Stops when the tracker loses the object."""
    import cv2
    out = {}
    tr = _tracker()
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    ok, frame = cap.read()
    if not ok:
        return out
    # TrackerNano wants an int (x, y, w, h) rect — floats fail the binding
    tr.init(frame, tuple(int(round(v)) for v in first_box))
    if f0 in grid_set:
        out[f0] = first_box
    f = f0
    while True:
        f += step
        if not (0 <= f <= f1 if step > 0 else f1 <= f < total):
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        good, r = tr.update(frame)
        if not good or min(r[2], r[3]) < MIN_SIDE:
            break
        if f in grid_set:
            out[f] = (float(r[0] + r[2] / 2), float(r[1] + r[3] / 2),
                      float(r[2]), float(r[3]))
    return out


def track(path: str, start_frame: int, box: Tuple[float, float, float, float],
          samples, log=print, progress=None) -> int:
    """Fill ``sample.salient`` with the tracked box, forward and backward
    from ``start_frame`` (relative frame numbers, as the analysis uses).
    Returns how many samples got a box."""
    import cv2
    if not ensure(progress, log):
        return 0
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    grid = {smp.frame for smp in samples}
    if not grid or total <= 0:
        cap.release()
        return 0
    start_frame = min(max(0, start_frame), total - 1)
    boxes = _run(cap, grid, start_frame, total - 1, 1, box, total)
    if start_frame > 0:
        boxes.update(_run(cap, grid, start_frame, 0, -1, box, total))
    cap.release()
    by = {smp.frame: smp for smp in samples}
    for f, b in boxes.items():
        by[f].salient = b
    return len(boxes)
