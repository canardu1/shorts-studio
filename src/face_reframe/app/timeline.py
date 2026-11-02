"""Timeline strip under the preview: removed (cut) ranges, shot changes,
hand corrections, the clip range, the playhead. Click/drag to seek."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtCore import QPointF
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..i18n import tr


class Timeline(QWidget):
    seek = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(30)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.n, self.fps = 1, 25.0
        self.removed: List[Tuple[float, float]] = []
        self.cuts: Sequence[int] = ()
        self.manual: Sequence[int] = ()
        self.clip: Optional[Tuple[int, int]] = None
        self.words = False
        self.pos = 0

    def set_data(self, n: int, fps: float, removed=None, cuts=(), manual=(), clip=None, words=False):
        self.n, self.fps = max(1, n), fps or 25.0
        self.removed = list(removed or [])
        self.cuts, self.manual, self.clip, self.words = list(cuts), list(manual), clip, words
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
        p.fillRect(bar, QColor("#2a2f38"))
        if self.clip is not None:
            a, b = self.clip
            p.fillRect(QRectF(self._x(a), 9, self._x(b) - self._x(a), 12), QColor("#3b4352"))
        for a, b in self.removed:
            x0, x1 = self._x(a * self.fps), self._x(b * self.fps)
            p.fillRect(QRectF(x0, 9, max(1.0, x1 - x0), 12), QColor(229, 72, 77, 170))
        p.setPen(QPen(QColor("#9aa3b0"), 1))
        for c in self.cuts:
            x = self._x(c)
            p.drawLine(QPointF(x, 6), QPointF(x, 24))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#f5c542"))
        for m in self.manual:
            x = self._x(m)
            p.drawPolygon(QPolygonF([QPointF(x, 1), QPointF(x + 4, 5), QPointF(x, 9), QPointF(x - 4, 5)]))
        x = self._x(self.pos)
        p.setPen(QPen(QColor("#ffffff"), 2))
        p.drawLine(QPointF(x, 3), QPointF(x, h - 3))

    def _emit(self, ev):
        f = (ev.position().x() - 6) / max(1.0, self.width() - 12) * self.n
        self.seek.emit(int(max(0, min(self.n - 1, f))))

    def mousePressEvent(self, ev):
        self._emit(ev)

    def mouseMoveEvent(self, ev):
        if ev.buttons() & Qt.LeftButton:
            self._emit(ev)
        else:
            t = (ev.position().x() - 6) / max(1.0, self.width() - 12) * self.n / self.fps
            inside = any(a <= t < b for a, b in self.removed)
            self.setToolTip(tr("Cut part") if inside else "")
