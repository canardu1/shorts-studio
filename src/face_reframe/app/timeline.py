"""Timeline strip under the preview: waveform lane, removed (cut) ranges,
shot changes, hand corrections, the clip range, the playhead.
Click/drag to seek, scroll to zoom, right-click for the edit menu."""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..i18n import tr

SAMPLE_RATE = 16000
"""extract_audio() always decodes to mono 16 kHz float32."""


class _ThumbTip(QWidget):
    """Borderless hover card: source frame + timecode, like an NLE scrub."""

    def __init__(self):
        super().__init__(None, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "QWidget { background: #17171b; border: 1px solid #3a3a44; border-radius: 4px; }"
            "QLabel { border: none; color: #9a9aad; }"
        )
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(3)
        self.img = QLabel()
        self.img.setAlignment(Qt.AlignCenter)
        self.lab = QLabel()
        self.lab.setAlignment(Qt.AlignCenter)
        self.lab.setFont(QFont("IBM Plex Mono, monospace"))
        lay.addWidget(self.img)
        lay.addWidget(self.lab)

    def show_for(self, img, text: str, gpos: QPoint):
        self.img.setPixmap(QPixmap.fromImage(img))
        self.lab.setText(text)
        self.adjustSize()
        screen = self.screen().availableGeometry() if self.screen() else None
        pos = gpos + QPoint(12, 16)
        if screen is not None:
            pos.setX(min(pos.x(), screen.right() - self.width() - 4))
            pos.setY(min(pos.y(), screen.bottom() - self.height() - 4))
        self.move(pos)
        self.show()


class Timeline(QWidget):
    seek = Signal(int)
    menu = Signal(float, object)
    """Right-click: (seconds at cursor, global pos) for the edit menu."""
    blade_at = Signal(float)
    """Blade mode: left-click time — window marks or completes the cut."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(56)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.n, self.fps = 1, 25.0
        self.wave: Optional[np.ndarray] = None
        """Raw 16 kHz PCM of the source — the lane slices it per zoom level,
        so dead spots keep reading as flat stretches however deep you zoom."""
        self._wave_src = None       # id() of the array the peak was taken on
        self._wave_peak = 1.0
        self._env_cache = None      # (s0, s1, cols) -> np.ndarray
        self.zoom = 1.0
        self.v0 = 0.0               # left edge of the visible window, in frames
        self.removed: List[Tuple[float, float]] = []
        self.mcuts: List[Tuple[float, float]] = []
        """Hand-cut ranges in seconds — painted orange, kept separate from
        the automatic (red) ones so the user sees what they deleted."""
        self.sel: Optional[Tuple[float, float]] = None
        """Selected range in seconds — drag to set, click elsewhere clears."""
        self._press_t: Optional[float] = None
        self._moved = False
        self._pan_x: Optional[float] = None
        self._pan_v0 = 0.0
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
        """Snap targets in seconds (shot cuts, markers, events, clip edges,
        speech/silence boundaries)."""
        self.snaps_on = True
        self.blade_mode = False
        """Armed by the toolbar razor: left-click marks/cuts instead of seeking."""
        self.thumb_fn: Optional[Callable[[int], object]] = None
        """frame index -> QImage thumbnail, set by the window (None in tests)."""
        self._thumb_tip: Optional[_ThumbTip] = None
        self._thumb_f = 0
        self._thumb_pos = QPoint()
        self._thumb_timer = QTimer(self)
        self._thumb_timer.setSingleShot(True)
        self._thumb_timer.setInterval(130)
        self._thumb_timer.timeout.connect(self._show_thumb)

    def set_data(self, n: int, fps: float, removed=None, cuts=(), manual=(), clip=None,
                 words=False, events=(), mcuts=None, wave=None, snaps=()):
        if abs(n - self.n) > 1:                 # different file: reset the view
            self.zoom, self.v0 = 1.0, 0.0
            self._env_cache = None
        self.n, self.fps = max(1, n), fps or 25.0
        self.removed = list(removed or [])
        self.mcuts = list(mcuts or [])
        self.cuts, self.manual, self.clip, self.words = list(cuts), list(manual), clip, words
        self.events = list(events)
        if wave is not self._wave_src:
            self._wave_src = wave
            self._env_cache = None
            if wave is not None and len(wave):
                self._wave_peak = float(np.abs(wave).max()) or 1.0
        self.wave = wave
        pts = [c / self.fps for c in self.cuts] + [m / self.fps for m in self.manual] + list(self.events)
        if self.clip is not None:
            pts += [self.clip[0] / self.fps, self.clip[1] / self.fps]
        pts += [t for t in (snaps or ()) if 0.0 <= t <= self.n / self.fps]
        self.snaps = sorted(set(pts))
        self._clamp_view()
        self.update()

    def set_pos(self, f: int):
        self.pos = f
        if self.zoom > 1.0:
            span = self._span()
            if f < self.v0 or f > self.v0 + span:   # follow the playhead
                self.v0 = max(0.0, min(self.n - span, f - span * 0.5))
        self.update()

    # ------------------------------------------------------------------ view
    def _span(self) -> float:
        return self.n / self.zoom

    def _clamp_view(self):
        if self._span() >= self.n:
            self.zoom, self.v0 = 1.0, 0.0
        else:
            self.v0 = max(0.0, min(self.n - self._span(), self.v0))

    def zoom_fit(self):
        self.zoom, self.v0 = 1.0, 0.0
        self.update()

    def zoom_center(self, steps: float):
        """Toolbar zoom: anchor at the playhead when it's inside the view."""
        px = self._x(self.pos)
        if not (6 <= px <= self.width() - 6):
            px = self.width() / 2.0
        self.zoom_at(steps, px)

    def set_blade_mode(self, on: bool):
        self.blade_mode = on
        self.setCursor(Qt.CrossCursor if on else Qt.ArrowCursor)
        if on:
            self._hide_thumb()
        self.update()

    def zoom_at(self, steps: float, anchor_x: float):
        """steps > 0 zooms in (1.25 per wheel notch), anchored at cursor x."""
        max_zoom = max(2.0, self.n / self.fps / 5.0)
        zoom = max(1.0, min(max_zoom, self.zoom * (1.25 ** steps)))
        if zoom == self.zoom:
            return
        anchor_f = self._f(anchor_x)
        self.zoom = zoom
        frac = max(0.0, min(1.0, (anchor_x - 6) / max(1.0, self.width() - 12)))
        self.v0 = anchor_f - frac * self._span()
        self._clamp_view()
        self.update()

    def pan_by(self, dx_px: float):
        if self.zoom <= 1.0:
            return
        self.v0 -= dx_px / max(1.0, self.width() - 12) * self._span()
        self._clamp_view()
        self.update()

    def wheelEvent(self, ev):
        self._hide_thumb()
        if self.n <= 1:
            return
        d = ev.angleDelta()
        if abs(d.x()) > abs(d.y()):               # trackpad two-finger pan
            self.pan_by(d.x() / 3.0)
        elif d.y():
            mods = ev.modifiers()
            if mods & Qt.ShiftModifier:           # shift-wheel: pan too
                self.pan_by(d.y() / 3.0)
            else:
                self.zoom_at(d.y() / 120.0, ev.position().x())
        ev.accept()

    def _x(self, f: float) -> float:
        return 6 + (self.width() - 12) * (f - self.v0) / self._span()

    def _f(self, x: float) -> float:
        return self.v0 + (x - 6) / max(1.0, self.width() - 12) * self._span()

    # -------------------------------------------------------------- painting
    def _wave_cols(self, s0: int, s1: int, cols: int) -> np.ndarray:
        """Peak envelope of wave[s0:s1] in ``cols`` buckets, normalised by the
        file peak (not the window's — a quiet stretch must stay quiet zoomed in)."""
        key = (s0, s1, cols)
        if self._env_cache is not None and self._env_cache[0] == key:
            return self._env_cache[1]
        seg = np.abs(self.wave[s0:s1])
        if len(seg) > cols:
            k = len(seg) // cols
            env = seg[:cols * k].reshape(cols, k).max(1)
        else:
            env = seg
        env = env / self._wave_peak if self._wave_peak else env
        self._env_cache = (key, env)
        return env

    def paintEvent(self, _ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        h = self.height()
        W = self.width() - 12
        # layout: diamonds 1-7 · wave lane 9-33 · track 36-48 · ticks 34-50 · events ~53
        wave_rect = QRectF(6, 9, W, 24)
        wpath = QPainterPath()
        wpath.addRoundedRect(wave_rect, 5, 5)
        p.fillPath(wpath, QColor("#141416"))
        if self.wave is not None and len(self.wave):
            p.save()
            p.setClipPath(wpath)
            p.setPen(QPen(QColor("#7c7c92"), 1))
            mid = 21.0
            f1 = min(float(self.n), self.v0 + self._span())
            s0 = int(self.v0 / self.fps * SAMPLE_RATE)
            s1 = max(s0, min(int(f1 / self.fps * SAMPLE_RATE), len(self.wave)))
            env = self._wave_cols(s0, s1, max(1, int(W)))
            nw = len(env)
            for i in range(nw):
                v = min(1.0, float(env[i])) * 11.0
                x = 6.0 + (i + 0.5) * W / nw
                p.drawLine(QPointF(x, mid - v), QPointF(x, mid + max(v, 0.5)))
            p.restore()
        if self.zoom > 1.0:
            pen = QPen(QColor("#5a5a6c"), 1)
            p.setPen(pen)
            f = QFont(p.font())
            f.setPointSizeF(7.5)
            p.setFont(f)
            p.drawText(wave_rect.adjusted(0, 0, -6, -3),
                       Qt.AlignRight | Qt.AlignBottom, "%.1f×" % self.zoom)
        bar = QRectF(6, 36, W, 12)
        track = QPainterPath()
        track.addRoundedRect(bar, 6, 6)
        p.fillPath(track, QColor("#1e1e1e"))
        p.save()
        p.setClipPath(track)
        if self.clip is not None:
            a, b = self.clip
            p.fillRect(QRectF(self._x(a), 36, self._x(b) - self._x(a), 12), QColor("#323232"))
        for a, b in self.removed:
            x0, x1 = self._x(a * self.fps), self._x(b * self.fps)
            p.fillRect(QRectF(x0, 36, max(1.0, x1 - x0), 12), QColor(229, 72, 77, 170))
        for a, b in self.mcuts:
            x0, x1 = self._x(a * self.fps), self._x(b * self.fps)
            p.fillRect(QRectF(x0, 36, max(1.0, x1 - x0), 12), QColor(229, 160, 10, 200))
        if self.sel is not None:
            x0, x1 = self._x(self.sel[0] * self.fps), self._x(self.sel[1] * self.fps)
            p.fillRect(QRectF(x0, 36, max(1.0, x1 - x0), 12), QColor(90, 200, 250, 90))
            p.setPen(QPen(QColor("#5ac8fa"), 1))
            p.drawRect(QRectF(x0, 36, max(1.0, x1 - x0), 12))
        p.restore()
        p.setPen(QPen(QColor("#8d8d8d"), 1))
        for c in self.cuts:
            x = self._x(c)
            p.drawLine(QPointF(x, 34), QPointF(x, 50))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#f5c542"))
        for m in self.manual:
            x = self._x(m)
            p.drawPolygon(QPolygonF([QPointF(x, 1), QPointF(x + 3, 4), QPointF(x, 7), QPointF(x - 3, 4)]))
        p.setBrush(QColor("#5ac8fa"))
        for t in self.events:
            x = self._x(t * self.fps)
            p.drawEllipse(QPointF(x, 52.5), 2.4, 2.4)
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

    # ----------------------------------------------------------------- mouse
    def _t(self, ev) -> float:
        f = self._f(ev.position().x())
        t = max(0.0, min(self.n - 1, f)) / self.fps
        if self.snaps_on and self.snaps:
            thr = 6.0 * self._span() / max(1.0, self.width() - 12) / self.fps   # ~6 px
            near = min(self.snaps, key=lambda s: abs(s - t))
            if abs(near - t) <= thr:
                t = near
        return t

    def _emit(self, ev):
        self.seek.emit(int(self._t(ev) * self.fps))

    def mousePressEvent(self, ev):
        self._hide_thumb()
        if ev.button() == Qt.RightButton:
            self.menu.emit(self._t(ev), ev.globalPosition().toPoint())
            return
        if ev.button() == Qt.MiddleButton:
            self._pan_x = ev.position().x()
            self._pan_v0 = self.v0
            return
        if ev.button() == Qt.LeftButton:
            if self.blade_mode:
                self.blade_at.emit(self._t(ev))
                return
            self._press_t = self._t(ev)
            self._moved = False

    def mouseMoveEvent(self, ev):
        if self._pan_x is not None:
            self.pan_by(ev.position().x() - self._pan_x)
            self._pan_x = ev.position().x()
            return
        if ev.buttons() & Qt.LeftButton and self._press_t is not None:
            self._hide_thumb()
            t = self._t(ev)
            if abs(t - self._press_t) * self.fps * (self.width() - 12) / self._span() > 4:
                lo, hi = min(t, self._press_t), max(t, self._press_t)
                if self.sel != (lo, hi):
                    self.sel = (lo, hi)
                    self.update()
                self._moved = True
            return
        t = self._t(ev)
        if self.blade_mode:
            self.setToolTip(tr("Blade: click to mark, click again to cut"))
            return
        if any(a <= t < b for a, b in self.mcuts):
            self.setToolTip(tr("Manual cut — right-click to restore"))
        elif any(a <= t < b for a, b in self.removed):
            self.setToolTip(tr("Cut part"))
        else:
            self.setToolTip(tr("Drag to select · scroll to zoom · right-click to edit"))
        if self.thumb_fn is not None:
            self._thumb_f = int(t * self.fps)
            self._thumb_pos = ev.globalPosition().toPoint()
            self._thumb_timer.start()

    def _show_thumb(self):
        if self.thumb_fn is None:
            return
        img = self.thumb_fn(self._thumb_f)
        if img is None:
            return
        if self._thumb_tip is None:
            self._thumb_tip = _ThumbTip()
        f, fps = self._thumb_f, self.fps
        tc = "%d:%02d.%02d" % (f / fps / 60, f / fps % 60, f % max(1, int(round(fps))))
        self._thumb_tip.show_for(img, tc, self._thumb_pos)

    def _hide_thumb(self):
        self._thumb_timer.stop()
        if self._thumb_tip is not None:
            self._thumb_tip.hide()

    def leaveEvent(self, ev):
        self._hide_thumb()
        super().leaveEvent(ev)

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.MiddleButton:
            self._pan_x = None
            return
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
