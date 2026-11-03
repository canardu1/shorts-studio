#!/usr/bin/env python3
"""Regenerate README UI media: docs/media/app.png and docs/media/timeline.gif.

    python scripts/shoot_ui.py

Runs the real window offscreen on a synthetic clip (moving face + hard cut +
bursty audio so the auto-cuts light up), then captures the populated UI and a
short timeline-editing sequence. No external footage needed.
"""

import os
import subprocess
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ["SHORTS_NO_RESTORE"] = "1"
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "tests"))
sys.path.insert(0, os.path.join(HERE, "src"))

import numpy as np  # noqa: E402
from make_sample import make  # noqa: E402

from PySide6.QtCore import QPointF, QRect, Qt  # noqa: E402
from PySide6.QtGui import QImage, QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from face_reframe.app.window import MainWindow  # noqa: E402
from face_reframe.detect import analyze_video  # noqa: E402
from face_reframe.silence import extract_audio, find_ffmpeg  # noqa: E402
from face_reframe.transcribe import Word  # noqa: E402

OUT = os.path.join(HERE, "docs", "media")


def demo_clip(path: str) -> None:
    """Synthetic 12 s clip + audio with speech-like bursts (pauses → red cuts)."""
    raw = path + ".silent.mp4"
    make(raw, n=300, fps=25)
    # bursts of a "voice-ish" wobble separated by real silences
    dur = 12.0
    sr = 16000
    t = np.arange(int(sr * dur)) / sr
    voice = np.sin(2 * np.pi * 220 * t) * np.sin(2 * np.pi * 7 * t)
    bursts = np.zeros_like(voice)
    for a, b in [(0.2, 2.1), (2.9, 5.3), (6.6, 9.4), (10.0, 11.6)]:
        m = (t >= a) & (t < b)
        bursts[m] = voice[m] * 0.5
    import wave
    wav = path + ".wav"
    with wave.open(wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((bursts * 30000).astype("<i2").tobytes())
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", raw, "-i", wav,
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-crf", "20",
                    "-c:a", "aac", "-shortest", path], check=True)


WORDS = [Word(0.35 + 0.22 * k, 0.5 + 0.22 * k, w) for k, w in enumerate(
    "questa è la parte che nessuno ti racconta mai eppure cambia davvero tutto "
    "quando fai i tuoi video corti".split())]


def mouse(tp, x, btn=Qt.LeftButton):
    return QMouseEvent(tp, QPointF(x, 15), btn, btn, Qt.NoModifier)


def shot(w, path):
    img = w.grab().toImage()
    img.save(path)


def band_frame(w, tl):
    """Timeline + play controls + a slice of the previews above it.

    Stops at the right edge of the output preview so the settings rail
    stays out of the shot.
    """
    org = tl.mapTo(w, QPointF(0, 0).toPoint())
    left = w.src_view.mapTo(w, QPointF(0, 0).toPoint()).x()
    right = (w.out_view.mapTo(w, QPointF(w.out_view.width(), 0).toPoint())).x()
    top = max(0, org.y() - 8)
    img = w.grab(QRect(left - 6, top, right - left + 12, 168)).toImage()
    img = img.convertToFormat(QImage.Format_RGBA8888)
    img = img.scaled(int(img.width() * 0.7), int(img.height() * 0.7),
                     Qt.KeepAspectRatio, Qt.SmoothTransformation)
    return np.asarray(img.constBits()).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 4]


def main():
    import tempfile
    tmp = tempfile.mkdtemp(prefix="shoot_")
    clip = os.path.join(tmp, "demo.mp4")
    print("generating demo clip…")
    demo_clip(clip)

    app = QApplication.instance() or QApplication([])
    from face_reframe.app.window import apply_theme
    apply_theme(app)
    w = MainWindow()
    w.add_files([clip])
    opts = w.reframe_options()
    print("analysing…")
    w.cache.put(clip, opts, analyze_video(clip, stride=opts.stride))
    w.cache.put_words(clip, w._val("language"), WORDS)
    key = os.path.abspath(clip)
    w.pcm[key] = extract_audio(clip)
    w.manual[key] = [(80, None, None), (185, None, None)]   # yellow diamonds
    w.manual_cuts[key] = [(9.6, 10.4)]                      # an amber manual cut
    w.hook_text[key] = "Nobody tells you this part"
    w._set("hook", "auto")
    w._set("subtitles", "burn")
    w._set("speaker_engine", "mouth")      # no neural job in the shot
    w._select(w.queue.item(0))
    w.slider.setValue(int(w.reader.count * 0.45))
    for _ in range(30):                    # let background jobs settle
        app.processEvents()

    os.makedirs(OUT, exist_ok=True)
    app_png = os.path.join(OUT, "app.png")
    shot(w, app_png)
    print("wrote", app_png)

    # ---- timeline-edit GIF -------------------------------------------------
    # clear the demo mcut so the gif starts clean
    w.manual_cuts.clear()
    w._replan()
    tl = w.timeline

    from PIL import Image
    frames = []

    def grab(times=1):
        bgra = band_frame(w, tl)
        im = Image.frombytes("RGBA", (bgra.shape[1] // 4, bgra.shape[0]),
                             bgra.tobytes()).convert("RGB")
        for _ in range(times):
            frames.append(im)

    def x(frac):
        return 6 + (tl.width() - 12) * frac

    grab(3)
    w.slider.setValue(int(w.reader.count * 0.30))
    app.processEvents()
    grab(3)
    # drag-select from 55% to 78%
    tl.mousePressEvent(mouse(QMouseEvent.MouseButtonPress, x(0.55)))
    for f in np.linspace(0.56, 0.78, 9):
        tl.mouseMoveEvent(mouse(QMouseEvent.MouseMove, x(f)))
        grab()
    tl.mouseReleaseEvent(mouse(QMouseEvent.MouseButtonRelease, x(0.78)))
    grab(4)
    w._cut_sel()
    grab(6)                                        # amber band appears
    # drag-select over it and restore
    tl.mousePressEvent(mouse(QMouseEvent.MouseButtonPress, x(0.60)))
    for f in np.linspace(0.62, 0.74, 5):
        tl.mouseMoveEvent(mouse(QMouseEvent.MouseMove, x(f)))
        grab()
    tl.mouseReleaseEvent(mouse(QMouseEvent.MouseButtonRelease, x(0.74)))
    grab(3)
    w._restore_sel()
    grab(6)

    gif_path = os.path.join(OUT, "timeline.gif")
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                   duration=280, loop=0, optimize=True)
    print("wrote", gif_path, f"({len(frames)} frames)")

    w.close()
    w.shutdown_workers()
    if w.reader:
        w.reader.close()


if __name__ == "__main__":
    main()
