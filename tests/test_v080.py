"""v0.8.0 features: CoreML provider selection, moment dedupe, loop bounds,
watch folder, speaker rename, remove-undo, chapter export, CLI v2."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ----------------------------------------------------------------- providers
def test_providers_cpu_only_env(monkeypatch):
    monkeypatch.setenv("SHORTS_CPU_ONLY", "1")
    from face_reframe import models
    assert models.providers() == ["CPUExecutionProvider"]


def test_providers_prefers_accelerator(monkeypatch):
    monkeypatch.delenv("SHORTS_CPU_ONLY", raising=False)
    import onnxruntime as ort
    from face_reframe import models
    have = set(ort.get_available_providers())
    prov = models.providers()
    if "CoreMLExecutionProvider" in have:
        assert prov[0] == "CoreMLExecutionProvider"
    assert prov[-1] == "CPUExecutionProvider"          # CPU always the fallback


def test_session_uses_central_providers(monkeypatch):
    """session() must not regress to CPU-only behind the helper's back."""
    monkeypatch.delenv("SHORTS_CPU_ONLY", raising=False)
    from face_reframe import models
    seen = {}
    import onnxruntime as ort

    class FakeSess:
        def __init__(self, path, so, providers=None):
            seen["providers"] = providers

    monkeypatch.setattr(ort, "InferenceSession", FakeSess)
    monkeypatch.setattr(models, "path", lambda k: "/tmp/x.onnx")
    models._sessions.pop("fake", None)
    models.session("fake")
    assert seen["providers"] == models.providers()
    models._sessions.pop("fake", None)


# ------------------------------------------------------------------ dedupe
def test_dedupe_jaccard_keeps_stronger():
    from face_reframe.app.highlights import Highlight, dedupe_highlights
    from face_reframe.transcribe import Word

    def words_for(text, t0):
        return [Word(t0 + 0.3 * i, t0 + 0.3 * i + 0.25, w)
                for i, w in enumerate(text.split())]

    a = "the secret trick nobody tells you about editing videos fast"
    b = "the secret trick nobody tells you about editing videos quickly"   # ~same
    c = "completely different story about cooking pasta tonight"
    words = words_for(a, 0) + words_for(b, 40) + words_for(c, 80)
    hls = [Highlight(0.0, 17.0, 0.9, "a"), Highlight(40.0, 57.0, 0.6, "b"),
           Highlight(80.0, 96.0, 0.7, "c")]
    out = dedupe_highlights(hls, words)
    titles = [h.title for h in out]
    assert "a" in titles and "c" in titles and "b" not in titles  # weaker dup dropped


def test_dedupe_embeddings():
    import numpy as np
    from face_reframe.app.highlights import Highlight, dedupe_highlights
    from face_reframe.transcribe import Word
    words = [Word(0.0, 1.0, "x")]
    hls = [Highlight(0, 10, 0.9, "a"), Highlight(30, 40, 0.5, "b"),
           Highlight(60, 70, 0.7, "c")]
    emb = np.zeros((3, 4), np.float32)
    emb[0, 0] = emb[1, 0] = 1.0          # a≈b semantically
    emb[2, 1] = 1.0
    out = dedupe_highlights(hls, words, embedder=lambda texts: emb)
    assert [h.title for h in out] == ["a", "c"]


# -------------------------------------------------------------- loop bounds
def test_loopmatch_snaps_to_matching_frame(tmp_path):
    """A clip whose colour field repeats at t≈8 s should snap its end there."""
    import cv2
    import numpy as np
    from face_reframe.loopmatch import suggest_loop_end
    p = str(tmp_path / "loop.mp4")
    w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 90))
    for i in range(100):                                   # 10 s
        v = 30 if i < 10 or 80 <= i < 90 else 200          # dark, bright, dark, bright
        w.write(np.full((90, 160, 3), v, np.uint8))
    w.release()
    end = suggest_loop_end(p, 0.2, 7.0, span=4.0, step=0.2)
    assert 8.0 <= end <= 9.5                               # found the dark stretch


# ------------------------------------------------------- GUI: watch/speaker/undo/chapters
@pytest.fixture
def win(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_NO_RESTORE", "1")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    from PySide6.QtWidgets import QApplication
    from face_reframe.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    w = MainWindow()
    yield w
    w._watch_timer.stop()
    w.close()
    w.deleteLater()
    app.processEvents()


def test_watch_folder_queues_stable_file(win, tmp_path):
    win.watch_auto.setChecked(False)                       # queue only, no render
    src = tmp_path / "src.mp4"
    src.write_bytes(b"\x00" * 4096)
    win._watch_set(str(tmp_path))
    win._watch_scan()
    assert str(src) in win._watch_pending
    win._watch_tick()                                      # first stable check
    win._watch_tick()                                      # second → queue
    assert any(it["path"] == str(src) for it in win._items())
    # still-copying file is held back
    grow = tmp_path / "grow.mp4"
    grow.write_bytes(b"\x00" * 1000)
    win._watch_scan()
    win._watch_tick()
    assert not any(it["path"] == str(grow) for it in win._items())


def test_speaker_rename_persists(win, tmp_path):
    from face_reframe.app.highlights import Highlight
    vid = tmp_path / "x.mp4"
    vid.write_bytes(b"\x00" * 4096)
    win._add_item({"path": str(vid), "label": "x.mp4"})
    src = win._items()[0]
    win._voice[os.path.abspath(str(vid))] = [(0.0, 5.0, 0), (5.0, 10.0, 1)]
    win._apply_highlights(src, [Highlight(0.5, 4.0, 0.8, "take one"),
                                Highlight(5.5, 9.0, 0.7, "take two")])
    win.hl_filter.setCurrentIndex(win.hl_filter.findData(1))
    from unittest import mock
    with mock.patch("face_reframe.app.window.QInputDialog.getText",
                    return_value=("Marco", True)):
        win._rename_speaker()
    labels = [it["label"] for it in win._items() if it.get("clip")]
    assert any("Marco" in l for l in labels)
    data = win._project_data()
    assert data["speaker_names"][os.path.abspath(str(vid))] == {"1": "Marco"}
    win2 = win.__class__()
    win2._apply_project(data)
    assert any("Marco" in it["label"] for it in win2._items() if it.get("clip"))
    win2._watch_timer.stop()
    win2.close()
    win2.deleteLater()


def test_remove_undo_restores_order(win):
    win._add_item({"path": "/a.mp4", "label": "a.mp4"})
    win._add_item({"path": "/b.mp4", "label": "b.mp4"})
    win._add_item({"path": "/c.mp4", "label": "c.mp4"})
    win.queue.item(0).setSelected(True)
    win._remove_selected()
    assert [it["label"] for it in win._items()] == ["b.mp4", "c.mp4"]
    win._undo_action()
    assert [it["label"] for it in win._items()] == ["a.mp4", "b.mp4", "c.mp4"]
    win._redo_action()
    assert [it["label"] for it in win._items()] == ["b.mp4", "c.mp4"]


def test_chapters_export_txt(win, tmp_path):
    win._add_item({"path": "/v/x.mp4", "label": "x.mp4"})
    win._add_item({"path": "/v/x.mp4", "clip": (12.0, 45.0), "stem": "x_clip1",
                   "score": 80, "speaker": 1, "title": "the hook",
                   "label": "   ↳ 00:12–00:45  🔥80  👤1  the hook"})
    from unittest import mock
    out = str(tmp_path / "ch.txt")
    with mock.patch("face_reframe.app.window.QFileDialog.getSaveFileName",
                    return_value=(out, "YouTube chapters (*.txt)")):
        win._export_chapters()
    text = open(out).read()
    assert "0:00 Intro" in text and "the hook" in text and "12" in text


# ------------------------------------------------------------------- CLI v2
def test_cli_render_list(capsys):
    from face_reframe.cli import main
    assert main(["render", "--list"]) == 0
    out = capsys.readouterr().out
    assert "Vlog" in out and "TikTok" in out


# ------------------------------------------------------------- timeline wave
def test_wave_envelope(win):
    import numpy as np
    pcm = np.zeros(16000 * 4, dtype=np.float32)
    pcm[16000:16000 * 2] = 0.8                      # loud second
    pcm[16000 * 3:16000 * 3 + 400] = 0.5            # short blip
    win.pcm["/v/a.mp4"] = pcm
    env = win._wave_env("/v/a.mp4")
    assert env is not None and len(env) <= 1600
    assert float(env.max()) == 1.0                  # normalised to peak
    # the dead first second must read near zero in the envelope
    dead = float(env[: len(env) // 4].max())
    loud = float(env[len(env) // 4: len(env) // 2].max())
    assert dead < 0.05 and loud > 0.9
    assert win._wave_env("/v/missing.mp4") is None


def test_timeline_accepts_wave(win):
    import numpy as np
    win.timeline.set_data(100, 25.0, wave=np.linspace(0, 1, 400))
    assert win.timeline.wave is not None and len(win.timeline.wave) == 400
    win.timeline.set_data(100, 25.0)
    assert win.timeline.wave is None
