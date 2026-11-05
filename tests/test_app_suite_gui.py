"""GUI integration of the suite extras: titles/logo/CTA, safe zones, looks,
stabilisation, undo/redo, projects + autosave, social texts."""

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
    d = tmp_path_factory.mktemp("suite")
    silent = str(d / "v.mp4")
    make(silent, n=150)
    out = str(d / "talk.mp4")
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i",
                    "sine=f=220:d=6", "-map", "0:v", "-map", "1:a", "-c:v", "libx264",
                    "-preset", "ultrafast", "-c:a", "aac", "-shortest", out], check=True)
    return out


WORDS = [Word(0.2 + 0.4 * k, 0.5 + 0.4 * k, w) for k, w in enumerate(
    "ecco il trucco che nessuno ti dice. funziona davvero con la luce naturale".split())]


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
    w._set("subtitles", "burn")
    for k in ("stabilize", "auto_wb"):      # QSettings persist between runs
        w._set(k, False)
    w._set("look", "none")
    yield w
    w.close()
    w.replan_timer.stop()
    w.shutdown_workers()
    if w.reader:
        w.reader.close()
    w.deleteLater()
    app.processEvents()


def _type(win, text):
    win.hook_edit.setText(text)
    win.hook_edit.textEdited.emit(text)   # as if typed


def _wait(cond, secs=60):
    from PySide6.QtWidgets import QApplication
    t0 = time.time()
    while time.time() - t0 < secs:
        QApplication.processEvents()
        if cond():
            return True
        time.sleep(0.02)
    return False


def test_preview_titles_logo_zones_look(win, tmp_path):
    import cv2
    from face_reframe.app.overlays import Chain
    logo = str(tmp_path / "logo.png")
    img = np.zeros((80, 160, 4), np.uint8)
    img[..., 2] = 255
    img[..., 3] = 255
    cv2.imwrite(logo, img)
    win._set_logo(logo)
    win._set("hook", "auto")
    win.cta_edit.setText("Seguimi per altri trucchi")
    win._set("look", "warm")
    win._set("aspect", "9:16")
    win._replan()
    assert isinstance(win._burner, Chain) and len(win._burner.parts) >= 3   # captions+hook+logo+cta
    assert win._look is not None and win._look.look == "warm"
    win._set_zones("tiktok")
    win._show_frame(5)
    assert win.out_view.zones
    # custom hook typed in the field wins over the first sentence
    _type(win, "Il segreto della luce")
    assert win._val("hook") == "custom"
    assert win.hook_text[os.path.abspath(win.current)] == "Il segreto della luce"


def test_auto_wb_and_stabilisation_run_in_background(win):
    win._set("auto_wb", True)
    win._set("stabilize", True)
    win._replan()
    key = os.path.abspath(win.current)
    assert _wait(lambda: key in getattr(win, "_stab", {}) and key in getattr(win, "_gains", {}))
    win._replan()
    assert win._motion is not None and win._motion.zoom.min() >= 1.03
    win._show_frame(20)


def test_undo_redo_manual_and_person(win):
    win._replan()
    key = os.path.abspath(win.current)
    win.slider.setValue(10)
    win._crop_dragged(120.0, 0.0)
    assert win.manual.get(key)
    win._undo_action()
    assert not win.manual.get(key)
    win._redo_action()
    assert win.manual.get(key)
    win._manual_clear()
    assert not win.manual.get(key)
    win._undo_action()
    assert win.manual.get(key)
    _type(win, "a")
    _type(win, "ab")          # one burst = one undo step
    win._undo_action()
    assert win.hook_text.get(key, "") == ""


def test_project_roundtrip_and_autosave(win, clip, tmp_path):
    key = os.path.abspath(clip)
    win._replan()
    win.slider.setValue(8)
    win._crop_dragged(80.0, 0.0)
    win.hook_text[key] = "Titolo prova"
    win._set("look", "cine")
    win._set("stab_amount", 0.5)
    p = str(tmp_path / "x.shorts")
    win.save_project(p)
    man = list(win.manual[key])
    win.manual.clear()
    win.hook_text.clear()
    win._set("look", "none")
    win.load_project(p)
    assert win.manual[key] == man
    assert win.hook_text[key] == "Titolo prova"
    assert win._val("look") == "cine" and abs(win._val("stab_amount") - 0.5) < 1e-6
    assert win.queue.count() == 1 and win.current == clip
    win._autosave()
    assert os.path.exists(win.autosave_path())


def test_social_dialog(win):
    win._show_social()
    dlg = win._social_dlg
    from PySide6.QtWidgets import QLineEdit
    texts = [e.text() for e in dlg.findChildren(QLineEdit)]
    assert any(t.startswith("#") for t in texts)
    assert any(t.strip() and not t.startswith("#") for t in texts)
    dlg.close()


def test_thumbnails_dialog_and_text(win, tmp_path, monkeypatch):
    from face_reframe.app import thumbnail as T
    text, hl = T.thumb_text(WORDS, "")
    assert 2 <= len(text.split()) <= 4 and hl is not None
    assert "NESSUNO" in text or "TRUCCO" in text
    # offline: no download, soft head-and-shoulders matte instead of the model
    monkeypatch.setenv("SHORTS_STUDIO_MODELS", str(tmp_path / "models"))
    monkeypatch.setattr(T, "download_seg", lambda *a, **k: (_ for _ in ()).throw(OSError("offline")))
    win.out_dir.setText(str(tmp_path / "out"))
    win._replan()
    win._open_thumbs()
    assert _wait(lambda: getattr(win, "_thumb_dlg", None) is not None)
    dlg = win._thumb_dlg
    assert len(dlg.variants) >= 1
    dlg.text.setText("TRUCCO SEGRETO")
    dlg.refresh()
    p = dlg.save_variant(0)
    import cv2
    assert cv2.imread(p).shape[:2] == (720, 1280)
    dlg.aspect.setCurrentIndex(1)
    assert cv2.imread(dlg.save_variant(0)).shape[:2] == (1920, 1080)
    dlg.close()


def test_thumbnail_partitions_and_cv_fallback():
    from face_reframe.app import thumbnail as T
    parts = list(T.partitions(["A", "B", "C"], 4))
    assert ["A B C"] in parts and ["A", "B", "C"] in parts and len(parts) == 4
    img = T.cv_text(["CIAO", "MONDO"], 1, 600, 300, "pop")
    assert img.shape[2] == 4 and img[..., 3].max() == 255
