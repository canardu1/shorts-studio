"""Source frame -> output frame, shared by the render and the live preview:
single crop (optionally punched in) or a two-person split screen."""

from __future__ import annotations

import dataclasses
from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..pipeline import Options, Result, plan_from_analysis


def crop_box(plan, i: int, zoom: float = 1.0, dx: float = 0.0, dy: float = 0.0) -> Tuple[int, int, int, int]:
    """Integer crop box for frame ``i``, shrunk by ``zoom`` around its centre
    and shifted by (dx, dy) fractions of the crop size (camera moves)."""
    if zoom <= 1.0001 and dx == 0.0 and dy == 0.0:
        return plan.box(i)
    w = int(round(plan.crop_w[i] / max(1.0, zoom)))
    h = int(round(plan.crop_h[i] / max(1.0, zoom)))
    x = int(round(plan.cx[i] + dx * plan.crop_w[i] - w / 2.0))
    y = int(round(plan.cy[i] + dy * plan.crop_h[i] - h / 2.0))
    x = max(0, min(plan.src_w - w, x))
    y = max(0, min(plan.src_h - h, y))
    return x, y, w, h


def _scale(img, w: int, h: int):
    import cv2
    up = img.shape[1] < w
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_LANCZOS4 if up else cv2.INTER_AREA)


@dataclasses.dataclass
class Split:
    top: Result
    bottom: Result
    mask: np.ndarray
    """True on frames shown as split screen."""


def build_split(analysis, opts: Options, out_w: int, out_h: int,
                picks: Optional[Sequence[np.ndarray]] = None) -> Optional[Split]:
    """Two people stacked (top/bottom) wherever both are in the shot.

    The two are the user's picks when they chose two different people,
    otherwise the two most-seen people who share a shot."""
    from ..track import _segments, find_people, match_person
    people = find_people(analysis)
    chosen = []
    for sig in picks or []:
        p = match_person(people, np.asarray(sig, np.float32))
        if p is not None and p not in chosen:
            chosen.append(p)
    if len(chosen) < 2:
        for p in people:
            if p in chosen:
                continue
            if not chosen or any(si in p.tracks for si in chosen[0].tracks):
                chosen.append(p)
            if len(chosen) == 2:
                break
    if len(chosen) < 2:
        return None
    a, b = chosen[:2]
    segs = _segments(analysis)
    shared = [si for si in a.tracks if si in b.tracks]
    if not shared:
        return None
    mask = np.zeros(analysis.frame_count, bool)
    for si in shared:
        s0, s1 = segs[si]
        mask[s0:s1] = True
    # the person on the left of the wide shot goes on top
    ax = np.mean([f.cx for si in shared for _, f in a.tracks[si]])
    bx = np.mean([f.cx for si in shared for _, f in b.tracks[si]])
    if bx < ax:
        a, b = b, a
    half = "%d:%d" % (out_w, out_h // 2)

    def plan_for(p):
        o = dataclasses.replace(opts, aspect=half, follow_vertical=True,
                                track=dataclasses.replace(opts.track, person=[p.sig], manual=None))
        return plan_from_analysis(analysis, o)

    return Split(plan_for(a), plan_for(b), mask)


class Framer:
    """Callable ``(i, src_frame) -> out_frame`` for one output size."""

    def __init__(self, result: Result, out_w: int, out_h: int, enhancer=None,
                 punch: Optional[np.ndarray] = None, split: Optional[Split] = None,
                 restorer=None, motion=None, seed: int = 0, skin=None, beauty=None, look=None):
        self.r, self.out_w, self.out_h = result, out_w, out_h
        self.enhancer, self.punch, self.split = enhancer, punch, split
        self.restorer = restorer
        self.motion, self.seed = motion, seed
        self.skin = skin
        self.beauty = beauty
        self.look = look
        """:class:`look.Look` (grade / auto white balance), after beauty."""
        """:class:`beauty.Beautifier`, after restoration (it refines skin the
        restorer rebuilt)."""
        """:class:`skin.SkinCorrection` applied to the whole output frame (its
        mask only touches skin-coloured pixels)."""
        """Camera moves (:class:`motion.MotionCurves`): crop offset/zoom here,
        rotation and transition effects after restoration."""
        """Face restoration runs last, on the finished frame (works for split too)."""

    def _zoom(self, i: int) -> float:
        p = self.punch
        z = float(p[i]) if p is not None and i < len(p) else 1.0
        m = self.motion
        if m is not None and i < len(m.zoom):
            z *= float(m.zoom[i])
        return z

    def _offset(self, i: int):
        m = self.motion
        if m is None or i >= len(m.dx):
            return 0.0, 0.0
        return float(m.dx[i]), float(m.dy[i])

    def _one(self, plan, i, frame, w, h, zoom):
        dx, dy = self._offset(i)
        x, y, cw, ch = crop_box(plan, i, zoom, dx, dy)
        crop = frame[y:y + ch, x:x + cw]
        if self.enhancer is not None:
            face = ((plan.cx[i] - x) * w / cw, (plan.cy[i] - y) * h / ch)
            return self.enhancer(crop, w, h, face)
        return _scale(crop, w, h)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        out = self._compose(i, frame)
        if self.skin is not None:
            from ..skin import apply as skin_apply
            out = skin_apply(out, self.skin)
        if self.restorer is not None:
            out = self.restorer(out)
        if self.beauty is not None:
            out = self.beauty(out)
        if self.look is not None:
            from .look import apply as look_apply
            out = look_apply(out, self.look)
        if self.motion is not None:
            from .motion import apply_effects
            out = apply_effects(out, self.motion, i, self.seed)
        return out

    def _compose(self, i: int, frame: np.ndarray) -> np.ndarray:
        z = self._zoom(i)
        sp = self.split
        if sp is not None and i < len(sp.mask) and sp.mask[i]:
            h1 = self.out_h // 2
            top = self._one(sp.top.plan, i, frame, self.out_w, h1, z)
            bot = self._one(sp.bottom.plan, i, frame, self.out_w, self.out_h - h1, z)
            out = np.vstack([top, bot])
            out[h1 - 2:h1 + 2] = (18, 18, 18)
            return out
        return self._one(self.r.plan, i, frame, self.out_w, self.out_h, z)


def punch_curve(speech: List[Tuple[float, float]], n_frames: int, fps: float,
                zoom: float = 1.15) -> np.ndarray:
    """Zoom toggles at each new phrase (speech span start), eased."""
    from ..punch import PunchParams, punch_curve_synced
    starts = [a for a, _ in speech[1:]]  # the first phrase opens unzoomed
    return punch_curve_synced(starts, n_frames, fps, PunchParams(zoom=zoom, ease=0.15))


def scaled(result, k: float):
    """The same framing for a frame scaled by ``k`` (preview proxies): crop
    plan in proxy pixels; the camera path keeps its shots/flags."""
    if result is None or abs(k - 1.0) < 1e-6:
        return result
    from types import SimpleNamespace
    from ..geometry import CropPlan
    p = result.plan
    plan = CropPlan(int(round(p.src_w * k)), int(round(p.src_h * k)),
                    p.crop_w * k, p.crop_h * k, p.cx * k, p.cy * k)
    return SimpleNamespace(plan=plan, path=result.path, analysis=getattr(result, "analysis", None))


def scaled_split(split, k: float):
    if split is None or abs(k - 1.0) < 1e-6:
        return split
    return Split(scaled(split.top, k), scaled(split.bottom, k), split.mask)
