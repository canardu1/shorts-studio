"""Graphic overlays burned on the finished frames: opening hook title,
channel logo, closing call-to-action card. Qt-free (renderers are injected;
OpenCV fallbacks below)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from ..i18n import tr

from .jobs import blend

TitleRenderer = Callable[[str, int], np.ndarray]
"""``(text, output_width) -> BGRA`` — wrapped, big, on a rounded pill."""


def cv_title(text: str, out_w: int) -> np.ndarray:
    """Fallback title renderer (no Qt): wrapped OpenCV text on a dark box."""
    import cv2
    font, scale, thick = cv2.FONT_HERSHEY_DUPLEX, out_w / 560.0, max(2, out_w // 300)
    words, lines, cur = text.upper().split(), [], ""
    for w in words:
        t = (cur + " " + w).strip()
        if cv2.getTextSize(t, font, scale, thick)[0][0] > out_w * 0.84 and cur:
            lines.append(cur)
            cur = w
        else:
            cur = t
    if cur:
        lines.append(cur)
    lines = lines[:3]
    lh = cv2.getTextSize("Ag", font, scale, thick)[0][1] + int(18 * scale)
    tw = max([cv2.getTextSize(l, font, scale, thick)[0][0] for l in lines] or [10])
    pad = int(24 * scale)
    img = np.zeros((lh * len(lines) + 2 * pad, tw + 2 * pad, 4), np.uint8)
    img[..., 3] = 200
    for k, l in enumerate(lines):
        w = cv2.getTextSize(l, font, scale, thick)[0][0]
        cv2.putText(img, l, ((img.shape[1] - w) // 2, pad + lh * (k + 1) - int(10 * scale)), font, scale,
                    (255, 255, 255, 255), thick, cv2.LINE_AA)
    return img


@dataclass
class OverlaySettings:
    hook: str = "off"
    """``off`` | ``auto`` (first sentence) | ``custom`` (text per video)."""
    hook_dur: float = 2.8
    cta: str = ""
    """Closing call-to-action text ('' = off)."""
    cta_dur: float = 2.5
    logo_path: Optional[str] = None
    logo_corner: str = "tr"
    logo_size: float = 0.16
    logo_opacity: float = 0.85
    progress: str = "off"
    """``off`` | ``top`` | ``bottom`` — retention progress bar along an edge."""
    card_in_path: Optional[str] = None
    """Intro card: a full-frame image shown at the start."""
    card_in_dur: float = 1.5
    card_out_path: Optional[str] = None
    """Outro card: a full-frame image shown at the end."""
    card_out_dur: float = 2.0


def first_sentence(words, max_words: int = 9) -> str:
    """Opening line for the hook: the first sentence, capped to a few words."""
    out = []
    for w in words or []:
        t = w.text.strip()
        if not t:
            continue
        out.append(t)
        if t.endswith((".", "?", "!")) or len(out) >= max_words:
            break
    s = " ".join(out).strip()
    return s.rstrip(".,;:") + ("…" if out and len(out) >= max_words and not s.endswith(("?", "!")) else "")


class TimedImage:
    """Shows ``img`` between frames [a, b) with short fades, at vertical ``pos``."""

    def __init__(self, img: np.ndarray, a: int, b: int, pos: float, fade: int = 6):
        self.img, self.a, self.b, self.pos, self.fade = img, a, b, pos, max(1, fade)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        if not (self.a <= i < self.b):
            return frame
        # no fade-in at the very start: the first frame is what feeds show as a still
        k_in = 1.0 if self.a == 0 else (i - self.a + 1) / self.fade
        k = min(1.0, k_in, (self.b - i) / self.fade)
        img = self.img
        if k < 1.0:
            img = img.copy()
            img[..., 3] = (img[..., 3].astype(np.float32) * k).astype(np.uint8)
        h, w = frame.shape[:2]
        return blend(frame, img, w, h, self.pos)


class Logo:
    def __init__(self, path: str, out_w: int, out_h: int, corner: str, size: float, opacity: float):
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise IOError(tr("unreadable logo: %s") % path)
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        tw = max(8, int(out_w * size))
        th = max(8, int(img.shape[0] * tw / img.shape[1]))
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        img[..., 3] = (img[..., 3].astype(np.float32) * opacity).astype(np.uint8)
        m = int(0.045 * out_w)
        x = m if corner in ("tl", "bl") else out_w - tw - m
        y = int(0.06 * out_h) if corner in ("tl", "tr") else out_h - th - int(0.06 * out_h)
        self.img, self.x, self.y = img, x, y

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        h, w = self.img.shape[:2]
        roi = frame[self.y:self.y + h, self.x:self.x + w].astype(np.float32)
        a = self.img[..., 3:4].astype(np.float32) / 255.0
        frame = frame if frame.flags.writeable else frame.copy()
        frame[self.y:self.y + h, self.x:self.x + w] = (roi * (1 - a) + self.img[..., :3] * a).astype(np.uint8)
        return frame


class ProgressBar:
    """Thin bar along the top/bottom edge that fills with playback time —
    the "watch to the end" cue viewers got used to on TikTok/Reels."""

    COLOR = (77, 72, 229)  # app accent #e5484d, BGR

    def __init__(self, n: int, out_w: int, out_h: int, pos: str = "top", frac: float = 0.007):
        self.n, self.w = n, out_w
        th = max(3, int(out_h * frac))
        self.y0, self.y1 = (0, th) if pos == "top" else (out_h - th, out_h)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        import cv2
        frame = frame if frame.flags.writeable else frame.copy()
        cv2.rectangle(frame, (0, self.y0), (int(self.w * (i + 1) / self.n), self.y1), self.COLOR, -1)
        return frame


class Card:
    """Full-frame image for a few seconds (intro/outro card).

    Cover-fitted to the output size; cross-fades with the video at the edges,
    fully replaces the frame in between. The audio keeps running."""

    def __init__(self, path: str, a: int, b: int, out_w: int, out_h: int, fade: int = 8):
        import cv2
        img = cv2.imread(path)
        if img is None:
            raise IOError(tr("unreadable card image: %s") % path)
        sc = max(out_w / img.shape[1], out_h / img.shape[0])          # cover, not fit
        img = cv2.resize(img, (int(img.shape[1] * sc + 0.5), int(img.shape[0] * sc + 0.5)),
                         interpolation=cv2.INTER_AREA if sc < 1.0 else cv2.INTER_LANCZOS4)
        x0 = max(0, (img.shape[1] - out_w) // 2)
        y0 = max(0, (img.shape[0] - out_h) // 2)
        self.img = img[y0:y0 + out_h, x0:x0 + out_w]
        self.a, self.b, self.fade = a, b, max(1, fade)

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        if not (self.a <= i < self.b):
            return frame
        k = min(1.0, (i - self.a + 1) / self.fade, (self.b - i) / self.fade)
        if k >= 1.0:
            return self.img.copy()
        import cv2
        return cv2.addWeighted(self.img, k, frame, 1.0 - k, 0)


class Chain:
    """Several ``post_frame`` hooks in order (captions, hook, logo, CTA...)."""

    def __init__(self, parts: Sequence):
        self.parts = [p for p in parts if p is not None]

    def __call__(self, i: int, frame: np.ndarray) -> np.ndarray:
        for p in self.parts:
            frame = p(i, frame)
        return frame

    def __bool__(self):
        return bool(self.parts)


def build(o: OverlaySettings, hook_text: str, n: int, fps: float, out_w: int, out_h: int,
          title_renderer: Optional[TitleRenderer], log=print) -> List:
    """The overlay hooks for one output (hook, logo, CTA)."""
    render = title_renderer or cv_title
    parts = []
    if o.hook != "off" and hook_text.strip():
        parts.append(TimedImage(render(hook_text.strip(), out_w), 0, min(n, int(o.hook_dur * fps)), 0.24))
    if o.logo_path and os.path.exists(o.logo_path):
        try:
            parts.append(Logo(o.logo_path, out_w, out_h, o.logo_corner, o.logo_size, o.logo_opacity))
        except IOError as exc:
            log("  %s" % exc)
    if o.cta.strip():
        a = max(0, n - int(o.cta_dur * fps))
        parts.append(TimedImage(render(o.cta.strip(), out_w), a, n, 0.5, fade=8))
    if o.progress in ("top", "bottom"):
        parts.append(ProgressBar(n, out_w, out_h, o.progress))
    for path, dur, a0 in ((o.card_in_path, o.card_in_dur, 0),
                          (o.card_out_path, o.card_out_dur, None)):
        if path and os.path.exists(path):
            a = a0 if a0 is not None else max(0, n - int(dur * fps))
            try:
                parts.append(Card(path, a, min(n, a + int(dur * fps)), out_w, out_h))
            except IOError as exc:
                log("  %s" % exc)
    return parts
