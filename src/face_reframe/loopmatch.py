"""Seamless-loop clip bounds.

Shorts replay automatically: if the last frame looks like the first, the
loop is invisible and watch time doubles.  ``suggest_loop_end`` searches a
few seconds around the proposed end for the frame that most closely matches
the clip's opening — low-resolution grayscale, cheap enough to run per
suggestion.  Returns the original ``end`` when no better boundary exists.
"""
import cv2
import numpy as np

_SIG = (32, 18)


def _sig(frame) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, _SIG, interpolation=cv2.INTER_AREA).astype(np.float32)


def _grab_sig(cap, t: float) -> np.ndarray:
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
    ok, fr = cap.read()
    return _sig(fr) if ok else None


def _avg_sig(cap, t: float, n: int = 3, step: float = 0.12):
    acc, cnt = None, 0
    for i in range(n):
        s = _grab_sig(cap, t + i * step)
        if s is None:
            break
        acc = s if acc is None else acc + s
        cnt += 1
    return acc / cnt if cnt else None


def suggest_loop_end(path: str, start: float, end: float,
                     span: float = 4.0, step: float = 0.25) -> float:
    """Best loop point for a clip ``start``–``end``: scan ``end ± span`` and
    pick the time whose frames look most like the opening.  Cheap: ~40
    tiny-grayscale reads, no decoding of the whole file."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return end
    try:
        dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1)
        ref = _avg_sig(cap, start)
        if ref is None:
            return end
        best, best_t = None, end
        t = max(start + 5.0, end - span)
        hi = min(dur - 0.1, end + span)
        while t <= hi:
            s = _avg_sig(cap, t, n=2, step=0.1)
            if s is not None:
                d = float(np.abs(s - ref).mean())
                if best is None or d < best:
                    best, best_t = d, t
            t += step
        return round(best_t, 2)
    finally:
        cap.release()
