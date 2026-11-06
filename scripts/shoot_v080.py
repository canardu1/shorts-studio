#!/usr/bin/env python3
"""Regenerate the v0.8.0 feature shots: docs/media/moments.png (queue panel
with speaker-labelled moments + watch folder) and docs/media/templates.png
(spectrum rail + Crop tab template row).

    python scripts/shoot_v080.py

Same offscreen trick as shoot_ui.py: a synthetic clip, fake moments, no
external footage or models needed.
"""

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SHORTS_NO_RESTORE"] = "1"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, os.path.join(HERE, "src"))

from PySide6.QtCore import QPointF, QRect, QSettings, Qt  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# keep the shot self-contained: _watch_set would otherwise persist the demo
# folder into the user's real preferences
import tempfile as _tempfile  # noqa: E402
_ini = os.path.join(_tempfile.mkdtemp(prefix="shoot080_ini_"), "s.ini")
import face_reframe.app.window as _win  # noqa: E402
_win.QSettings = lambda *a, **k: QSettings(_ini, QSettings.IniFormat)

from make_sample import make  # noqa: E402
from face_reframe.app.highlights import Highlight  # noqa: E402
from face_reframe.app.window import MainWindow  # noqa: E402
from face_reframe.transcribe import Word  # noqa: E402

OUT = os.path.join(HERE, "docs", "media")


def grab_region(w, left_w, top_w, width, height, path):
    org = left_w.mapTo(w, QPointF(0, 0).toPoint())
    img = w.grab(QRect(org.x(), org.y(), width, height))
    img.save(path)
    print("wrote", path)


def main():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="shoot080_")
    clip = os.path.join(tmp, "podcast.mp4")
    make(clip, n=200, fps=25)

    app = QApplication.instance() or QApplication([])
    from face_reframe.app.window import apply_theme
    apply_theme(app)
    w = MainWindow()
    w.resize(1480, 920)
    w.add_files([clip])
    src = w._items()[0]
    key = os.path.abspath(clip)
    w._voice[key] = [(0.0, 25.0, 0), (25.0, 60.0, 1), (60.0, 120.0, 0)]
    w.cache.put_words(clip, w._val("language"),
                      [Word(0.2 + 0.4 * k, 0.5 + 0.4 * k, "w%d" % k) for k in range(40)])
    # two speakers' moments, one already renamed
    w._speaker_names[key] = {1: "Marco"}
    hls = [Highlight(4.0, 38.0, 0.92, "the part nobody tells you about editing"),
           Highlight(41.0, 76.0, 0.88, "how we actually shot the whole thing"),
           Highlight(82.0, 118.0, 0.81, "why vertical video changed everything")]
    w._apply_highlights(src, hls)
    watch_dir = os.path.join(tmp, "incoming")
    os.makedirs(watch_dir)
    w._watch_set(watch_dir)
    w._select(w.queue.item(0))
    for _ in range(25):
        app.processEvents()

    # ---- moments.png: the whole left column (queue + moments + watch) ------
    q = w.queue
    org = q.mapTo(w, QPointF(0, 0).toPoint())
    right = q.mapTo(w, QPointF(q.width(), 0).toPoint()).x()
    grab_region(w, q, w, right - 6 + 10, w.height() - org.y() - 4,
                os.path.join(OUT, "moments.png"))

    # ---- templates.png: spectrum rail + Crop tab ---------------------------
    w.tabs.setCurrentIndex(0)
    for _ in range(10):
        app.processEvents()
    col = w._settings_col
    org = col.mapTo(w, QPointF(0, 0).toPoint())
    img = w.grab(QRect(org.x(), org.y(), col.width(), w.height() - org.y() - 4))
    img.save(os.path.join(OUT, "templates.png"))
    print("wrote", os.path.join(OUT, "templates.png"))

    w._watch_timer.stop()
    w.close()
    w.deleteLater()
    app.processEvents()


if __name__ == "__main__":
    main()
