"""TikTok-style caption images drawn with Qt (real fonts, accents, outline),
optionally with the word being spoken highlighted."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPainterPath, QPen


def make_renderer(color: str = "#FFFFFF", uppercase: bool = True, highlight: str = "#FFD400",
                  size_frac: float = 0.075, family: str = "Arial Black", box: str = "none",
                  sweep: bool = False):
    """Returns ``(words, active_index, out_w, emojis=()) -> BGRA ndarray``; safe
    to call from a worker thread (QImage painting doesn't need the GUI thread).
    ``sweep`` is the karaoke look: every word already spoken stays highlighted.
    ``emojis`` are drawn after the text via font fallback (colour on macOS)."""

    def render(words: Sequence[str], active: Optional[int], out_w: int,
               emojis: Sequence[str] = ()) -> np.ndarray:
        toks = [(w.upper() if uppercase else w).strip() for w in words]
        text = " ".join(toks)
        font = QFont(family)
        font.setStyleHint(QFont.SansSerif)
        font.setWeight(QFont.Black)
        font.setPixelSize(max(12, int(out_w * size_frac)))
        fm = QFontMetricsF(font)
        while fm.horizontalAdvance(text) > out_w * 0.9 and font.pixelSize() > 12:
            font.setPixelSize(int(font.pixelSize() * 0.92))
            fm = QFontMetricsF(font)
        stroke = max(3.0, font.pixelSize() * 0.16)
        pad = int(stroke * 2) if box == "none" else int(font.pixelSize() * 0.42)
        # the active word pops a little: leave room for it
        pop = 1.12 if active is not None else 1.0
        em_w = int(len(emojis) * font.pixelSize() * 1.25)
        w = int(fm.horizontalAdvance(text) * pop) + 2 * pad + em_w
        h = int(fm.height() * pop) + 2 * pad
        img = QImage(w, h, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        if box == "pill":
            from PySide6.QtCore import QRectF
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 120))
            p.drawRoundedRect(QRectF(0, 0, w, h), h * 0.32, h * 0.32)
        x = float(pad)
        space = fm.horizontalAdvance(" ")
        base = pad + fm.ascent() * pop
        for k, tok in enumerate(toks):
            f = QFont(font)
            if k == active:
                f.setPixelSize(int(font.pixelSize() * pop))
            path = QPainterPath()
            path.addText(QPointF(x, base), f, tok)
            p.setPen(QPen(QColor(0, 0, 0, 230), stroke, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)
            p.setPen(Qt.NoPen)
            lit = k == active or (sweep and active is not None and k < active)
            p.setBrush(QColor(highlight if lit else color))
            p.drawPath(path)
            x += QFontMetricsF(f).horizontalAdvance(tok) + space
        if emojis:
            p.setFont(font)
            p.setPen(QColor("#FFFFFF"))
            for e in emojis:
                p.drawText(QPointF(x, base), e)
                x += font.pixelSize() * 1.25
        p.end()
        arr = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
        arr = arr.reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
        return arr[..., [2, 1, 0, 3]].copy()  # -> BGRA to match OpenCV frames

    return render


def make_title_renderer(color: str = "#FFFFFF", box: str = "#000000", box_alpha: int = 190,
                        size_frac: float = 0.078, uppercase: bool = True):
    """``(text, out_w) -> BGRA``: big title wrapped on up to 3 lines, on a
    rounded semi-transparent box (hook titles, closing card, covers)."""
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QTextOption

    def render(text: str, out_w: int) -> np.ndarray:
        text = (text.upper() if uppercase else text).strip()
        font = QFont("Arial Black")
        font.setStyleHint(QFont.SansSerif)
        font.setWeight(QFont.Black)
        font.setPixelSize(max(14, int(out_w * size_frac)))
        maxw = out_w * 0.84
        while True:
            fm = QFontMetricsF(font)
            # greedy wrap
            lines, cur = [], ""
            for w in text.split():
                t = (cur + " " + w).strip()
                if fm.horizontalAdvance(t) > maxw and cur:
                    lines.append(cur)
                    cur = w
                else:
                    cur = t
            if cur:
                lines.append(cur)
            if len(lines) <= 3 or font.pixelSize() <= 14:
                break
            font.setPixelSize(int(font.pixelSize() * 0.9))
        lines = lines[:3]
        fm = QFontMetricsF(font)
        lh = fm.height() * 1.05
        pad = int(font.pixelSize() * 0.55)
        w = int(max(fm.horizontalAdvance(l) for l in lines)) + 2 * pad
        h = int(lh * len(lines)) + 2 * pad
        img = QImage(w, h, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        c = QColor(box)
        c.setAlpha(box_alpha)
        p.setPen(Qt.NoPen)
        p.setBrush(c)
        p.drawRoundedRect(QRectF(0, 0, w, h), pad * 0.8, pad * 0.8)
        p.setPen(QColor(color))
        p.setFont(font)
        for k, l in enumerate(lines):
            p.drawText(QRectF(0, pad + k * lh, w, lh), Qt.AlignHCenter | Qt.AlignVCenter, l)
        p.end()
        arr = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
        arr = arr.reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
        return arr[..., [2, 1, 0, 3]].copy()

    return render


def make_thumb_text_renderer():
    """``(words, highlighted_index, max_w, max_h, style) -> BGRA``: huge
    thumbnail words (line breaks chosen for the biggest type), thick
    outline, one word in an accent colour."""
    accents = {"colore": "#FF2A2A", "neon": "#00E5FF"}

    def best_layout(words, max_w, max_h):
        from .thumbnail import partitions
        font = QFont("Arial Black")
        font.setStyleHint(QFont.SansSerif)
        font.setWeight(QFont.Black)
        font.setPixelSize(100)
        fm = QFontMetricsF(font)
        best = None
        for lines in partitions(words, 4):
            wmax = max(fm.horizontalAdvance(l) for l in lines) / 100.0
            hsum = fm.height() * 0.95 * len(lines) / 100.0
            size = min(max_w * 0.94 / max(1e-3, wmax), max_h / max(1e-3, hsum))
            if best is None or size > best[0] * 1.04:
                best = (size, lines)
        return int(best[0]), best[1]

    def render(words, hl, max_w, max_h, style="pop"):
        if isinstance(words, str):
            words = words.split()
        size, lines = best_layout(list(words), max_w, max_h)
        size = max(10, min(size, int(max_h * 0.5)))
        font = QFont("Arial Black")
        font.setStyleHint(QFont.SansSerif)
        font.setWeight(QFont.Black)
        font.setPixelSize(size)
        fm = QFontMetricsF(font)
        n = len(lines)
        stroke = max(4.0, size * 0.12)
        pad = int(stroke * 1.5)
        lh = fm.height() * 0.95
        w = int(max(fm.horizontalAdvance(l) for l in lines)) + 2 * pad
        h = int(lh * n + fm.descent() * 0.3) + 2 * pad
        img = QImage(w, h, QImage.Format_RGBA8888)
        img.fill(Qt.transparent)
        p = QPainter(img)
        p.setRenderHint(QPainter.Antialiasing)
        accent = QColor(accents.get(style, "#FFD400"))
        idx = 0
        space = fm.horizontalAdvance(" ")
        for k, line in enumerate(lines):
            x = float(pad)
            base = pad + k * lh + fm.ascent()
            for tok in line.split():
                path = QPainterPath()
                path.addText(QPointF(x, base), font, tok)
                p.setPen(QPen(QColor(0, 0, 0), stroke * 2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
                p.setBrush(Qt.NoBrush)
                p.drawPath(path)
                p.setPen(Qt.NoPen)
                p.setBrush(accent if idx == hl else QColor("#FFFFFF"))
                p.drawPath(path)
                x += fm.horizontalAdvance(tok) + space
                idx += 1
        p.end()
        arr = np.frombuffer(img.constBits(), np.uint8, count=img.sizeInBytes())
        arr = arr.reshape(h, img.bytesPerLine() // 4, 4)[:, :w]
        return arr[..., [2, 1, 0, 3]].copy()

    return render
