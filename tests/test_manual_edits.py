"""Manual timeline editing: drag-select, cut/restore, undo, projects, export spans."""

import os
import subprocess
import time

import numpy as np
import pytest

from face_reframe.silence import find_ffmpeg
from face_reframe.transcribe import Word

pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    from make_sample import make
    d = tmp_path_factory.mktemp("edits")
    silent = str(d / "v.mp4")
    make(silent, n=150)
    out = str(d / "talk.mp4")
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i",
                    "sine=f=220:d=6", "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", "-shortest", out], check=True)
    return out


WORDS = [Word(0.2 + 0.4 * k, 0.5 + 0.4 * k, w) for k, w in enumerate(
    "uno due tre quattro cinque sei sette otto nove dieci undici dodici".split())]


@pytest.fixture
def win(clip, tmp_path, monkeypatch):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SHORTS_NO_RESTORE", "1")
    from PySide6.QtWidgets import QApplication, QMessageBox
    from face_reframe.app.window import MainWindow
    from face_reframe.detect import analyze_video
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    w.add_files([clip])
    opts = w.reframe_options()
    w.cache.put(clip, opts, analyze_video(clip, stride=opts.stride))
    w.cache.put_words(clip, w._val("language"), WORDS)
    yield w
    w.close()
    w.replan_timer.stop()
    w.shutdown_workers()
    if w.reader:
        w.reader.close()
    w.deleteLater()
    app.processEvents()


def _key(win):
    return os.path.abspath(win.current)


def test_drag_selects_range(win):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    tl = win.timeline
    tl.resize(600, 30)

    def xof(frac):
        return 6 + (tl.width() - 12) * frac

    def ev(tp, x):
        return QMouseEvent(tp, QPointF(x, 15), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)

    tl.mousePressEvent(ev(QMouseEvent.MouseButtonPress, xof(0.2)))
    tl.mouseMoveEvent(ev(QMouseEvent.MouseMove, xof(0.5)))
    tl.mouseReleaseEvent(ev(QMouseEvent.MouseButtonRelease, xof(0.5)))
    assert tl.sel is not None
    a, b = tl.sel
    assert b - a > 1.0


def test_cut_restore_and_merge(win):
    key = _key(win)
    win.timeline.sel = (1.0, 2.0)
    win._cut_sel()
    assert win.manual_cuts[key] == [(1.0, 2.0)]
    win.timeline.sel = (3.0, 4.0)
    win._cut_sel()
    assert win.manual_cuts[key] == [(1.0, 2.0), (3.0, 4.0)]
    # overlapping cut merges
    win.timeline.sel = (1.5, 3.5)
    win._cut_sel()
    assert win.manual_cuts[key] == [(1.0, 4.0)]
    # restore the middle splits the range
    win.timeline.sel = (2.0, 3.0)
    win._restore_sel()
    assert win.manual_cuts[key] == [(1.0, 2.0), (3.0, 4.0)]
    # restored spans feed the skip list (source seconds)
    assert any(abs(a - 1.0) < 1e-6 and abs(b - 2.0) < 1e-6 for a, b in win.removed)


def test_undo_manual_cuts(win):
    key = _key(win)
    win.timeline.sel = (1.0, 2.0)
    win._cut_sel()
    win.timeline.sel = (4.0, 5.0)
    win._cut_sel()
    assert len(win.manual_cuts[key]) == 2
    win._undo_action()
    assert win.manual_cuts[key] == [(1.0, 2.0)]
    win._undo_action()
    assert win.manual_cuts.get(key) in (None, [])
    win._redo_action()
    win._redo_action()
    assert len(win.manual_cuts[key]) == 2


def test_project_roundtrip_manual_cuts(win):
    key = _key(win)
    win.timeline.sel = (1.0, 2.5)
    win._cut_sel()
    d = win._project_data()
    assert d["manual_cuts"][key] == [[1.0, 2.5]]
    win.manual_cuts.clear()
    win._apply_project(d)
    assert win.manual_cuts[key] == [(1.0, 2.5)]


def test_job_settings_passes_manual_cuts(win):
    key = _key(win)
    win.manual_cuts[key] = [(1.0, 2.0)]
    s = win.job_settings()
    assert s.manual_cuts_by_file[key] == [(1.0, 2.0)]


def test_cut_spans_merges_manual():
    from face_reframe.app.jobs import JobSettings, cut_spans
    rng = np.random.default_rng(0)
    pcm = (rng.normal(0, 0.02, 16000 * 10)).astype(np.float32)
    s = JobSettings(cut_pauses=True)
    spans = cut_spans(pcm, 10.0, s, WORDS, lambda *_: None,
                      manual_cuts=[(8.0, 9.5)])
    assert spans is not None
    # the manual hole (8.0–9.5) is honoured inside the kept spans
    for a, b in spans:
        assert b <= 8.0 + 1e-6 or a >= 9.5 - 1e-6 or not (a < 9.5 and b > 8.0)


def test_click_seeks_not_selects(win):
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    tl = win.timeline
    tl.resize(600, 30)
    tl.sel = (1.0, 2.0)
    x = 6 + (tl.width() - 12) * 0.7 * tl.n / tl.n
    ev = lambda tp: QMouseEvent(tp, QPointF(x, 15), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
    seen = []
    tl.seek.connect(lambda f: seen.append(f))
    tl.mousePressEvent(ev(QMouseEvent.MouseButtonPress))
    tl.mouseReleaseEvent(ev(QMouseEvent.MouseButtonRelease))
    assert tl.sel is None and seen
