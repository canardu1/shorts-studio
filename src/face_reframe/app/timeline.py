"""Timeline strip under the preview: removed (cut) ranges, shot changes,
hand corrections, the clip range, the playhead. Click/drag to seek."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..i18n import tr


class Timeline(QWidget):
    seek = Signal(int)
    menu = Signal(float, object)
    """Right-click: (seconds at cursor, global pos) for the edit menu."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.n, self.fps = 1, 25.0
        self.removed: List[Tuple[float, float]] = []
        self.mcuts: List[Tuple[float, float]] = []
        """Hand-cut ranges in seconds — painted orange, kept separate from
        the automatic (red) ones so the user sees what they deleted."""
        self.sel: Optional[Tuple[float, float]] = None
        """Selected range in seconds — drag to set, click elsewhere clears."""
        self._press_t: Optional[float] = None
        self._moved = False
        self.cuts: Sequence[int] = ()
        self.manual: Sequence[int] = ()
        self.events: Sequence[float] = ()
        """Event times in seconds (voice changes, laughter…) — cyan dots."""
        self.clip: Optional[Tuple[int, int]] = None
        self.words = False
        self.pos = 0
        self.blade: Optional[float] = None
        """Pending blade mark in seconds — orange line; a second B cuts blade->playhead."""
        self.snaps: List[float] = []
        """Snap targets in seconds (shot cuts, markers, events, clip edges)."""

    def set_data(self, n: int, fps: float, removed=None, cuts=(), manual=(), clip=None,
                 words=False, events=(), mcuts=None):
        self.n, self.fps = max(1, n), fps or 25.0
        self.removed = list(removed or [])
        self.mcuts = list(mcuts or [])
        self.cuts, self.manual, self.clip, self.words = list(cuts), list(manual), clip, words
        self.events = list(events)
        pts = [c / self.fps for c in self.cuts] + [m / self.fps for m in self.manual] + list(self.events)
        if self.clip is not None:
            pts += [self.clip[0] / self.fps, self.clip[1] / self.fps]
        self.snaps = sorted(set(pts))
        self.update()

    def set_pos(self, f: int):
        self.pos = f
        self.update()

    def _x(self, f: float) -> float:
        return 6 + (self.width() - 12) * f / float(self.n)

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        bar = QRectF(6, 9, self.width() - 12, 12)
        track = QPainterPath()
        track.addRoundedRect(bar, 6, 6)
        p.fillPath(track, QColor("#1e1e1e"))
        p.save()
        p.setClipPath(track)
        if self.clip is not None:
            a, b = self.clip
            p.fillRect(QRectF(self._x(a), 9, self._x(b) - self._x(a), 12), QColor("#323232"))
        for a, b in self.removed:
            x0, x1 = self._x(a * self.fps), self._x(b * self.fps)
            p.fillRect(QRectF(x0, 9, max(1.0, x1 - x0), 12), QColor(229, 72, 77, 170))
        for a, b in self.mcuts:
            x0, x1 = self._x(a * self.fps), self._x(b * self.fps)
            p.fillRect(QRectF(x0, 9, max(1.0, x1 - x0), 12), QColor(229, 160, 10, 200))
        if self.sel is not None:
            x0, x1 = self._x(self.sel[0] * self.fps), self._x(self.sel[1] * self.fps)
            p.fillRect(QRectF(x0, 9, max(1.0, x1 - x0), 12), QColor(90, 200, 250, 90))
            p.setPen(QPen(QColor("#5ac8fa"), 1))
            p.drawRect(QRectF(x0, 9, max(1.0, x1 - x0), 12))
        p.restore()
        p.setPen(QPen(QColor("#8d8d8d"), 1))
        for c in self.cuts:
            x = self._x(c)
            p.drawLine(QPointF(x, 6), QPointF(x, 24))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#f5c542"))
        for m in self.manual:
            x = self._x(m)
            p.drawPolygon(QPolygonF([QPointF(x, 1), QPointF(x + 4, 5), QPointF(x, 9), QPointF(x - 4, 5)]))
        p.setBrush(QColor("#5ac8fa"))
        for t in self.events:
            x = self._x(t * self.fps)
            p.drawEllipse(QPointF(x, 25), 2.4, 2.4)
        if self.blade is not None:
            x = self._x(self.blade * self.fps)
            pen = QPen(QColor("#e5a00a"), 2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.drawLine(QPointF(x, 2), QPointF(x, h - 2))
        x = self._x(self.pos)
        pen = QPen(QColor("#ffffff"), 2)
        pen.setCapStyle(Qt.RoundCap)
        p.setPen(pen)
        p.drawLine(QPointF(x, 4), QPointF(x, h - 4))

    def _t(self, ev) -> float:
        f = (ev.position().x() - 6) / max(1.0, self.width() - 12) * self.n
        t = max(0.0, min(self.n - 1, f)) / self.fps
        if self.snaps:
            thr = 6.0 * self.n / max(1.0, self.width() - 12) / self.fps   # ~6 px
            near = min(self.snaps, key=lambda s: abs(s - t))
            if abs(near - t) <= thr:
                t = near
        return t

    def _emit(self, ev):
        self.seek.emit(int(self._t(ev) * self.fps))

    def mousePressEvent(self, ev):
        if ev.button() == Qt.RightButton:
            self.menu.emit(self._t(ev), ev.globalPosition().toPoint())
            return
        if ev.button() == Qt.LeftButton:
            self._press_t = self._t(ev)
            self._moved = False

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton and self._press_t is not None:
            t = self._t(ev)
            if abs(t - self._press_t) * self.fps * (self.width() - 12) / self.n > 4:
                lo, hi = min(t, self._press_t), max(t, self._press_t)
                if self.sel != (lo, hi):
                    self.sel = (lo, hi)
                    self.update()
                self._moved = True
            return
        t = self._t(ev)
        if any(a <= t < b for a, b in self.mcuts):
            self.setToolTip(tr("Manual cut — right-click to restore"))
        elif any(a <= t < b for a, b in self.removed):
            self.setToolTip(tr("Cut part"))
        else:
            self.setToolTip(tr("Drag to select · right-click to edit"))

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton or self._press_t is None:
            return
        if self._moved:
            pass                # selection stays: the edit menu / keys act on it
        else:
            self.clear_sel()    # a click, not a drag: seek + drop any selection
            self._emit(ev)
        self._press_t = None
        self._moved = False

    def clear_sel(self):
        self.sel = None
        self.update()
