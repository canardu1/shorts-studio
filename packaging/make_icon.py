"""Draws the app icon (a red rounded square with a vertical frame) into an
.iconset folder for macOS ``iconutil``."""
import os
import sys

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format_ARGB32)
    img.fill(Qt.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.Antialiasing)
    m = size * 0.08
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#e5484d"))
    p.drawRoundedRect(QRectF(m, m, size - 2 * m, size - 2 * m), size * 0.2, size * 0.2)
    w, h = size * 0.34, size * 0.6
    r = QRectF((size - w) / 2, (size - h) / 2, w, h)
    p.setPen(QPen(QColor("white"), size * 0.05))
    p.setBrush(Qt.NoBrush)
    p.drawRoundedRect(r, size * 0.05, size * 0.05)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("white"))
    p.drawEllipse(r.center(), size * 0.07, size * 0.07)
    p.end()
    return img


if __name__ == "__main__":
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    app = QGuiApplication([])
    for s in (16, 32, 64, 128, 256, 512):
        draw(s).save(os.path.join(out, "icon_%dx%d.png" % (s, s)))
        draw(2 * s).save(os.path.join(out, "icon_%dx%d@2x.png" % (s, s)))
