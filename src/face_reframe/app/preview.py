"""Preview widgets: source frame with the moving crop box, and the 9:16 result."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


def to_qimage(bgr: np.ndarray) -> QImage:
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, bgr.strides[0], QImage.Format_BGR888).copy()


class VideoReader:
    """Random access over a video with a fast path for sequential reads."""

    def __init__(self, path: str):
        import cv2
        self.cv2 = cv2
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError("Impossibile aprire %s" % path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0
        self.max_grab = int(3 * self.fps)
        self.sequential = False
        """Set while playing: tolerate tiny backward steps without seeking."""
        self._last: Optional[np.ndarray] = None
        self._last_i = -1

    def read(self, i: int) -> Optional[np.ndarray]:
        i = max(0, min(self.count - 1, i))
        if i == self._last_i:
            return self._last
        if self.sequential and 0 < self._last_i - i <= 3:
            return self._last  # tiny step back while playing: not worth a seek
        if 0 < i - self._next <= self.max_grab:
            # jump forward (playback dropping frames, a skipped cut): decoding
            # forward is far cheaper than a seek, which on long-GOP camera files
            # re-decodes from the previous keyframe (up to seconds back)
            for _ in range(i - self._next):
                self.cap.grab()
        elif i != self._next:
            self.cap.set(self.cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = self.cap.read()
        self._next = i + 1
        if not ok:
            return self._last
        self._last, self._last_i = frame, i
        return frame

    def close(self):
        self.cap.release()


class FrameView(QWidget):
    """Paints an image fitted to the widget, optionally with a crop overlay
    (everything outside the crop is dimmed) and a face marker. With
    ``draggable`` the crop box can be dragged: ``crop_dragged(dx, dy)`` is
    emitted on release, in image pixels."""

    crop_dragged = Signal(float, float)
    object_picked = Signal(float, float, float, float)
    """Emitted when a pick-mode drag ends: ``(x, y, w, h)`` in image pixels."""

    def __init__(self, placeholder: str = "", parent=None, draggable: bool = False):
        super().__init__(parent)
        self.draggable = draggable
        self.pick_mode = False
        self._pick_from: Optional[QPointF] = None
        self._pick_rect: Optional[QRectF] = None
        self._scale = 1.0
        self._target = QRectF()
        self._drag_from: Optional[QPointF] = None
        self._drag_d = (0.0, 0.0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(160, 160)
        self._img: Optional[QImage] = None
        self._crop: Optional[Tuple[float, float, float, float]] = None
        self._face: Optional[Tuple[float, float]] = None
        self._placeholder = placeholder
        self._badge = ""
        self.zones = None
        """Safe-zone rectangles as (x, y, w, h) fractions of the image, or None."""
        self.setMouseTracking(True)

    def set_frame(self, img: Optional[QImage], crop=None, face=None, badge: str = ""):
        self._img, self._crop, self._face, self._badge = img, crop, face, badge
        self.update()

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.setRenderHint(QPainter.Antialiasing)
        rr = QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5)
        clip = QPainterPath()
        clip.addRoundedRect(rr, 9, 9)
        p.fillPath(clip, QColor("#0e0e0e"))
        p.setClipPath(clip)
        if self._img is None or self._img.isNull():
            p.setPen(QColor("#585858"))
            p.drawText(self.rect(), Qt.AlignCenter | Qt.TextWordWrap, self._placeholder)
            p.setClipping(False)
            p.setPen(QPen(QColor("#232323"), 1))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(rr, 9, 9)
            return
        iw, ih = self._img.width(), self._img.height()
        s = min(self.width() / iw, self.height() / ih)
        dw, dh = iw * s, ih * s
        target = QRectF((self.width() - dw) / 2, (self.height() - dh) / 2, dw, dh)
        p.drawImage(target, self._img)
        self._scale, self._target = s, target
        if self._crop is not None:
            x, y, w, h = self._crop
            x += self._drag_d[0]
            y += self._drag_d[1]
            r = QRectF(target.x() + x * s, target.y() + y * s, w * s, h * s)
            dim = QPainterPath()
            dim.addRect(target)
            hole = QPainterPath()
            hole.addRect(r)
            p.fillPath(dim.subtracted(hole), QColor(0, 0, 0, 150))
            p.setPen(QPen(QColor("#f5c542"), 2))
            p.drawRect(r)
        if self.zones:
            p.setPen(QPen(QColor(255, 80, 80, 200), 1, Qt.DashLine))
            p.setBrush(QColor(255, 60, 60, 55))
            for zx, zy, zw, zh in self.zones:
                p.drawRect(QRectF(target.x() + zx * dw, target.y() + zy * dh, zw * dw, zh * dh))
        if self._face is not None:
            fx, fy = self._face
            p.setPen(QPen(QColor("#4ade80"), 2))
            p.drawEllipse(target.x() + fx * s - 5, target.y() + fy * s - 5, 10, 10)
        if self._pick_rect is not None:
            p.setPen(QPen(QColor("#e5484d"), 2, Qt.DashLine))
            p.setBrush(QColor(229, 72, 77, 40))
            p.drawRect(self._pick_rect)
        if self._badge:
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 170))
            fm = p.fontMetrics()
            bw = fm.horizontalAdvance(self._badge) + 14
            p.drawRoundedRect(QRectF(target.x() + 8, target.y() + 8, bw, 22), 6, 6)
            p.setPen(QColor("#e6e6e6"))
            p.drawText(QRectF(target.x() + 8, target.y() + 8, bw, 22), Qt.AlignCenter, self._badge)
        p.setClipping(False)
        p.setPen(QPen(QColor("#232323"), 1))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(rr, 9, 9)

    # ------------------------------------------------------------ dragging
    def _in_crop(self, pos) -> bool:
        if self._crop is None or self._img is None:
            return False
        x, y, w, h = self._crop
        t, s = self._target, self._scale
        return QRectF(t.x() + x * s, t.y() + y * s, w * s, h * s).contains(pos)

    def mousePressEvent(self, ev):
        if self.pick_mode and ev.button() == Qt.LeftButton and self._img is not None:
            self._pick_from = ev.position()
            self._pick_rect = None
            self.update()
            return
        if self.draggable and ev.button() == Qt.LeftButton and self._in_crop(ev.position()):
            self._drag_from = ev.position()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, ev):
        if self._pick_from is not None:
            self._pick_rect = QRectF(self._pick_from, ev.position()).normalized()
            self.update()
        elif self._drag_from is not None:
            d = ev.position() - self._drag_from
            self._drag_d = (d.x() / self._scale, d.y() / self._scale)
            self.update()
        elif self.pick_mode:
            self.setCursor(Qt.CrossCursor)
        elif self.draggable:
            self.setCursor(Qt.OpenHandCursor if self._in_crop(ev.position()) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, ev):
        if self._pick_from is not None:
            r, t, s = self._pick_rect, self._target, self._scale
            self._pick_from, self._pick_rect = None, None
            self.update()
            if r is not None and r.width() * r.height() > 4:
                x = (r.x() - t.x()) / s
                y = (r.y() - t.y()) / s
                self.object_picked.emit(x, y, r.width() / s, r.height() / s)
            return
        if self._drag_from is None:
            return
        dx, dy = self._drag_d
        self._drag_from, self._drag_d = None, (0.0, 0.0)
        self.setCursor(Qt.OpenHandCursor)
        if abs(dx) + abs(dy) > 2:
            self.crop_dragged.emit(dx, dy)
        self.update()
