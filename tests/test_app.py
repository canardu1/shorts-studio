"""Standalone app: Qt-free pipeline (jobs.py) + an offscreen GUI smoke test."""

import os
import subprocess

import numpy as np
import pytest

from face_reframe.app.jobs import (AnalysisCache, CaptionBurner, JobSettings, _cv_caption,
                                   remap_words, run_job, subtract)
from face_reframe.silence import find_ffmpeg
from face_reframe.transcribe import Word


def test_subtract_and_remap():
    spans = subtract([(0.0, 10.0)], [(2.0, 3.0), (5.0, 6.0)])
    assert spans == [(0.0, 2.0), (3.0, 5.0), (6.0, 10.0)]
    words = [Word(1.0, 1.4, "uno"), Word(2.2, 2.6, "via"), Word(3.5, 3.9, "due"), Word(6.5, 7.0, "tre")]
    out = remap_words(words, spans)
    assert [w.text for w in out] == ["uno", "due", "tre"]
    assert out[1].start == pytest.approx(2.5)   # 3.5 - 1.0 removed
    assert out[2].start == pytest.approx(4.5)   # 6.5 - 2.0 removed


def test_caption_burner_draws_only_inside_event():
    ev = [(0.0, 1.0, [Word(0.0, 0.3, "ciao"), Word(0.3, 0.6, "a"), Word(0.6, 1.0, "tutti")])]
    burn = CaptionBurner(ev, 25.0, 360, 640, _cv_caption, 0.7)
    blank = np.zeros((640, 360, 3), np.uint8)
    assert burn(0, blank.copy()).any()
    assert not burn(40, blank.copy()).any()


@pytest.fixture(scope="module")
def talking_clip(tmp_path_factory):
    from make_sample import make
    d = tmp_path_factory.mktemp("app")
    silent = str(d / "v.mp4")
    make(silent, n=200)
    out = str(d / "talk.mp4")
    # 8 s: speech-like tone 0-2 s and 4-6 s, silence in between
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i",
                    "aevalsrc='0.3*sin(2*PI*220*t)*lt(mod(t\\,4)\\,2)+0.001*random(0)':s=48000:d=8",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", out], check=True)
    return out


def _probe(path):
    import cv2
    cap = cv2.VideoCapture(path)
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 25))
    cap.release()
    return res


def test_run_job_cut_enhance(talking_clip, tmp_path):
    s = JobSettings(output_height=1280, cut_pauses=True, output_dir=str(tmp_path))
    s.speech.vad = False  # synthetic tone: exercise the RMS engine
    s.enhance.mode = "fast"
    stages, logs = [], []
    out = run_job(talking_clip, s, lambda st, f: stages.append(st), logs.append,
                  cache=AnalysisCache(disk=False))[0]
    assert os.path.exists(out) and out.endswith("talk_short.mp4")
    w, h, dur = _probe(out)
    assert (w, h) == (720, 1280)
    assert dur < 7.0, logs          # the pauses were cut out of 8 s
    assert any(st.startswith("Render") for st in stages)


def test_preview_cache_reused(talking_clip, tmp_path):
    from face_reframe.detect import analyze_video
    s = JobSettings(output_height=1280, output_dir=str(tmp_path))
    cache = AnalysisCache(disk=False)
    cache.put(talking_clip, s.reframe, analyze_video(talking_clip, stride=s.reframe.stride))
    logs = []
    run_job(talking_clip, s, lambda *_: None, logs.append, cache=cache)
    assert any("preview" in line for line in logs)


def test_gui_smoke(talking_clip, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SHORTS_STUDIO_NO_VAD", "1")  # synthetic tone: exercise the RMS engine
    from PySide6.QtWidgets import QApplication
    from face_reframe.app.captions_qt import make_renderer
    from face_reframe.app.window import MainWindow, apply_theme
    from face_reframe.detect import analyze_video
    from face_reframe.silence import extract_audio
    app = QApplication.instance() or QApplication([])
    apply_theme(app)
    win = MainWindow()
    win.add_files([talking_clip])
    assert win.current == talking_clip
    opts = win.reframe_options()
    win.cache.put(talking_clip, opts, analyze_video(talking_clip, stride=opts.stride))
    win.pcm = {os.path.abspath(talking_clip): extract_audio(talking_clip)}
    # pretend Whisper ran: words in the cache feed fillers, captions, highlights
    words = [Word(0.2 + 0.4 * k, 0.5 + 0.4 * k, w) for k, w in enumerate(
        "perché nessuno ti dice questo trucco? ehm funziona davvero".split())]
    win.cache.put_words(talking_clip, None, words)
    win.preset_box.setCurrentIndex(win.preset_box.findText("Vlog"))
    win._apply_preset(win.preset_box.currentIndex())
    assert win._val("cut_pauses") and win._val("punch") and win._val("caption_style") == "word"
    win._replan()
    assert win.result is not None
    assert win.removed, "the 2 s silences should show as cut ranges"
    assert win._burner is not None and win._punch is not None
    win.slider.setValue(10)
    # drag the crop 200 px right: a hand correction appears and moves the crop
    x_before = float(win.result.plan.cx[10])
    win._crop_dragged(200.0, 0.0)
    assert win.manual[os.path.abspath(talking_clip)][0][0] == 10
    assert float(win.result.plan.cx[40]) > x_before + 100
    win._manual_clear()
    img = make_renderer()(["perché", "è", "così"], 1, 1080)
    assert img.shape[2] == 4 and img[..., 3].max() == 255
    shot = os.environ.get("APP_SCREENSHOT")
    if shot:
        win.resize(1480, 900)
        win.show()
        app.processEvents()
        win._show_frame(12)
        app.processEvents()
        win.grab().save(shot)
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()


def test_gui_playback_with_audio(talking_clip, tmp_path, monkeypatch):
    """Play drives the frames from the audio clock (or silently falls back)."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    import time
    from PySide6.QtWidgets import QApplication
    from face_reframe.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.add_files([talking_clip])
    win.skip_cuts.setChecked(False)
    win._toggle_play()
    assert win.play_timer.isActive()
    t0 = time.time()
    while time.time() - t0 < 1.5:
        app.processEvents()
        time.sleep(0.01)
    assert win.slider.value() > 10          # ~25 fps for 1.5 s
    win._toggle_mute()
    win._seek(100)
    win._toggle_play()                       # pause
    assert not win.play_timer.isActive()
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()


def test_gui_preview_plays_treated_audio(talking_clip, tmp_path, monkeypatch):
    """With cinematic audio on, play switches to the treated WAV; A/B goes back
    to the original; editing a setting rebuilds the treated audio."""
    pytest.importorskip("PySide6")
    try:
        import PySide6.QtMultimedia  # noqa: F401
    except ImportError:
        pytest.skip("QtMultimedia not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    import time
    from PySide6.QtWidgets import QApplication
    from face_reframe.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.add_files([talking_clip])
    win.skip_cuts.setChecked(False)
    win._set("cinematic", True)

    def wait(cond, secs=15):
        t0 = time.time()
        while time.time() - t0 < secs and not cond():
            app.processEvents()
            time.sleep(0.02)
        return cond()

    def treated():
        return os.path.basename(str(win._player_src)).startswith("pa_")

    win._toggle_play()
    assert wait(treated), "treated audio never started"
    first = win._player_src
    win.ab_btn.setChecked(True)          # A/B -> original (the source, or its audio-only WAV)
    assert wait(lambda: win._player_src == win._orig_src(), 3)
    assert win._player_src in (talking_clip, win._orig_wav.get("path"))
    win.ab_btn.setChecked(False)
    assert wait(treated, 5)
    # edit -> rebuilt with the new sound (settings persist between runs: change for real)
    win._set("warmth", 3.0 if win._val("warmth") == 4.0 else 4.0)
    assert wait(lambda: treated() and win._player_src != first, 15)
    assert win.slider.value() > 0
    win._toggle_play()
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()


def test_gui_skip_cuts_with_treated_audio_never_sticks(talking_clip, tmp_path, monkeypatch):
    """Regression: with 'Salta i tagli' + treated audio, playback used to bounce
    back into the skipped part (stale player position after a seek) and stick."""
    pytest.importorskip("PySide6")
    try:
        import PySide6.QtMultimedia  # noqa: F401
    except ImportError:
        pytest.skip("QtMultimedia not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SHORTS_STUDIO_NO_VAD", "1")  # synthetic tone: exercise the RMS engine
    import time
    from PySide6.QtWidgets import QApplication
    from face_reframe.app.window import MainWindow
    from face_reframe.silence import extract_audio
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.add_files([talking_clip])
    win.pcm = {os.path.abspath(talking_clip): extract_audio(talking_clip)}
    win._set("cut_pauses", True)
    win._set("cinematic", True)
    win.skip_cuts.setChecked(True)
    win._replan()
    assert win.removed, "the test clip has 2 s pauses to skip"
    fps = win.reader.fps
    seen, t0, last_change, stuck = [], time.time(), time.time(), 0.0
    prev = win.slider.value()
    win._toggle_play()
    while time.time() - t0 < 9:
        app.processEvents()
        time.sleep(0.01)
        v = win.slider.value()
        if v != prev:
            stuck = max(stuck, time.time() - last_change)
            last_change, prev = time.time(), v
            seen.append(v)
    win._toggle_play()
    assert stuck < 0.8, "playback froze for %.2fs" % stuck
    inside = [f for f in seen if any(a + 0.1 <= f / fps < b - 0.1 for a, b in win.removed)]
    assert len(inside) <= 2, "played skipped frames: %s" % inside[:10]
    assert max(seen) > 150, "didn't get through the clip"
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()


def test_gui_quit_during_analysis_exits_cleanly(talking_clip, tmp_path):
    """Regression (crash report): closing the app while 'Analizza volti' runs
    aborted Python with 'QThread: Destroyed while thread is still running'."""
    pytest.importorskip("PySide6")
    import subprocess
    import sys
    import textwrap
    script = textwrap.dedent('''
        import os, sys, time
        from PySide6.QtWidgets import QApplication
        from face_reframe.app.window import MainWindow
        app = QApplication([])
        win = MainWindow()
        win.add_files([sys.argv[1]])
        win._analyze()
        t = time.time()
        while time.time() - t < 0.4:
            app.processEvents(); time.sleep(0.01)
        assert win.threads_running(), "analysis should still be running"
        win.close()
        print("running after close:", win.threads_running(), flush=True)
        if win.threads_running():
            os._exit(0)
        win.reader.close(); win.deleteLater(); del win
    ''')
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen", SHORTS_STUDIO_CACHE=str(tmp_path / "c"),
               PYTHONPATH=os.pathsep.join(["src", os.environ.get("PYTHONPATH", "")]))
    r = subprocess.run([sys.executable, "-c", script, talking_clip], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr[-800:]
    assert "running after close: False" in r.stdout, r.stdout + r.stderr[-400:]


def test_gui_heavy_source_plays_from_proxy(talking_clip, tmp_path, monkeypatch):
    """Heavy camera files: a light proxy is built and used while playing;
    stills (paused) still come from the original."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    import time
    from PySide6.QtWidgets import QApplication
    from face_reframe.app import proxy
    from face_reframe.app.window import MainWindow
    from face_reframe.detect import analyze_video
    monkeypatch.setattr(proxy, "needs_proxy", lambda reader: True)
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.add_files([talking_clip])
    t0 = time.time()
    while win.proxy is None and time.time() - t0 < 30:
        app.processEvents()
        time.sleep(0.02)
    assert win.proxy is not None and win.proxy.height <= proxy.PROXY_HEIGHT
    assert win.proxy.count == win.reader.count
    opts = win.reframe_options()
    win.cache.put(talking_clip, opts, analyze_video(talking_clip, stride=opts.stride))
    win._replan()
    assert win._view_reader() is win.reader           # paused: original
    win._toggle_play()
    assert win._view_reader() is win.proxy            # playing: proxy
    t0 = time.time()
    while time.time() - t0 < 1.0:
        app.processEvents()
        time.sleep(0.01)
    assert win.out_view._img is not None and abs(win._view_k - win.proxy.width / win.reader.width) < 1e-6
    win._crop_dragged(100.0, 0.0)                     # view px -> source px
    man = win.manual[os.path.abspath(talking_clip)][-1]
    assert man[1] > 0
    win._toggle_play()
    assert win._view_reader() is win.reader
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()


def test_proxy_build_retries_a_decoder_that_drops_frames(talking_clip, tmp_path, monkeypatch):
    """If a decoder silently loses frames, the next attempt fixes the proxy."""
    from face_reframe.app import proxy
    real = proxy._frames(talking_clip)
    seq = iter([real - 30, real])     # first decode drops, second is faithful
    monkeypatch.setattr(proxy, "_frames", lambda p: next(seq, real))
    out = proxy.build(talking_clip, str(tmp_path / "p.mp4"), expected=real)
    assert proxy.verified_count(out) == real


def test_proxy_build_accepts_the_decoders_consensus(talking_clip, tmp_path, monkeypatch):
    """When every decoder reports the same count that differs from cv2's
    metadata estimate, the proxy is right and cv2 was wrong (VFR, edit
    lists): keep it instead of discarding on every open."""
    from face_reframe.app import proxy
    real = proxy._frames(talking_clip)
    monkeypatch.setattr(proxy, "_frames", lambda p: real - 5)
    out = proxy.build(talking_clip, str(tmp_path / "p2.mp4"), expected=real)
    assert proxy.verified_count(out) == real - 5


def test_proxy_build_fails_when_decoders_disagree(talking_clip, tmp_path, monkeypatch):
    from face_reframe.app import proxy
    real = proxy._frames(talking_clip)
    seq = iter([real - 30, real - 5, real - 30, real - 5])
    monkeypatch.setattr(proxy, "_frames", lambda p: next(seq, real - 5))
    with pytest.raises(RuntimeError, match="frame count"):
        proxy.build(talking_clip, str(tmp_path / "p3.mp4"), expected=real)


def test_gui_subtitle_editor(talking_clip, tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    from PySide6.QtWidgets import QApplication, QMessageBox
    from face_reframe.app.window import MainWindow
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "question", lambda *a, **k: QMessageBox.Yes)
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.add_files([talking_clip])
    lang = win._val("language")
    words = [Word(0.3 + 0.5 * i, 0.7 + 0.5 * i, t) for i, t in enumerate(
        "oggi parliamo di fischi per fiaschi con rossi".split())]
    win.cache.put_words(talking_clip, lang, words)
    win._set("caption_words", 3)
    win._open_subeditor()
    ed = win._subeditor
    assert ed.table.rowCount() == 3
    ed.table.item(1, 2).setText("fiaschi per fiaschi")          # fix a misheard line
    app.processEvents()
    got = [w.text for w in win.cache.words(talking_clip, lang)]
    assert got[3:6] == ["fiaschi", "per", "fiaschi"]
    ed.find.setText("rossi")
    ed.repl.setText("Russo")
    ed._replace_all()
    assert "Russo" in [w.text for w in win.cache.words(talking_clip, lang)]
    ed._clicked(2, 0)                                          # jump the preview to the line
    assert abs(win.slider.value() / win.reader.fps - ed._rows[2][0]) < 0.1
    ed._restore()
    assert [w.text for w in win.cache.words(talking_clip, lang)] == [w.text for w in words]
    shot = os.environ.get("APP_SCREENSHOT_SUBS")
    if shot:
        ed.table.item(1, 2).setText("fiaschi per fiaschi")
        app.processEvents()
        ed.show()
        app.processEvents()
        ed.grab().save(shot)
    ed.close()
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()


def test_gui_audio_behind_picture_resyncs_instead_of_freezing(talking_clip, tmp_path, monkeypatch):
    """Regression: when the audio clock fell far behind the picture (a slow
    source load, a lost seek), the picture froze until the sound caught up.
    Now the audio is moved to the picture and playback keeps going."""
    pytest.importorskip("PySide6")
    try:
        import PySide6.QtMultimedia  # noqa: F401
    except ImportError:
        pytest.skip("QtMultimedia not installed")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "cache"))
    import time
    from PySide6.QtWidgets import QApplication
    from face_reframe.app.window import MainWindow
    app = QApplication.instance() or QApplication([])
    win = MainWindow()
    win.add_files([talking_clip])
    t0 = time.time()
    while not win._orig_wav and time.time() - t0 < 20:      # original soundtrack as WAV
        app.processEvents()
        time.sleep(0.02)
    assert win._orig_wav and win._orig_src().endswith(".wav")
    win.skip_cuts.setChecked(False)
    win._toggle_play()
    t0 = time.time()
    while time.time() - t0 < 1.5:
        app.processEvents()
        time.sleep(0.01)
    if not win._audio_active():
        win._toggle_play()
        pytest.skip("no audio device: nothing to resync")
    # knock the audio 2 s behind the picture, as a slow/lost seek would
    win.player.setPosition(max(0, win.player.position() - 2000))
    prev, last, stuck = win.slider.value(), time.time(), 0.0
    t0 = time.time()
    while time.time() - t0 < 3:
        app.processEvents()
        time.sleep(0.01)
        v = win.slider.value()
        if v != prev:
            stuck = max(stuck, time.time() - last)
            prev, last = v, time.time()
    stuck = max(stuck, time.time() - last)
    win._toggle_play()
    assert stuck < 0.6, "picture froze %.2fs waiting for the audio" % stuck
    win.close()
    win.reader.close()
    win.deleteLater()
    app.processEvents()
