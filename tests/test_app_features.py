"""Shorts Studio features: clips, formats, split screen, punch-in, music,
word captions, hand corrections, highlights, disk cache."""

import os
import subprocess

import numpy as np
import pytest

from face_reframe.app import diskcache
from face_reframe.app.jobs import (AnalysisCache, CaptionBurner, JobSettings, TimeMap, _cv_caption,
                                   _map_manual, group_word_events, run_job, slice_words)
from face_reframe.silence import find_ffmpeg
from face_reframe.transcribe import Word


def _probe(path):
    import cv2
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    res = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
           cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps)
    cap.release()
    return res


@pytest.fixture(scope="module")
def two_people(tmp_path_factory):
    """6 s: close-up on B (blue shirt) for 2 s, then a wide shot with A (red) and B."""
    import cv2
    from skimage import data
    d = tmp_path_factory.mktemp("feat")
    face = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)[0:300, 100:380]
    small = cv2.resize(face, None, fx=0.5, fy=0.5)
    sh, sw = small.shape[:2]
    silent = str(d / "v.mp4")
    out = cv2.VideoWriter(silent, cv2.VideoWriter_fourcc(*"mp4v"), 25, (1920, 1080))
    for i in range(150):
        fr = np.full((1080, 1920, 3), (70, 80, 90), np.uint8)
        if i < 50:
            big = cv2.resize(face, None, fx=1.4, fy=1.4)
            bh, bw = big.shape[:2]
            fr[120:120 + bh, 760:760 + bw] = big
            cv2.rectangle(fr, (700, 120 + bh), (1250, 1080), (170, 60, 20), -1)
        else:
            for x, col in ((450, (30, 30, 190)), (1300, (170, 60, 20))):
                fr[300:300 + sh, x:x + sw] = small
                cv2.rectangle(fr, (x - 40, 300 + sh), (x + sw + 40, 900), col, -1)
        out.write(fr)
    out.release()
    path = str(d / "two.mp4")
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-i", silent, "-f", "lavfi", "-i",
                    "aevalsrc='0.3*sin(2*PI*220*t)*lt(mod(t\\,2)\\,1.2)+0.001*random(0)':s=48000:d=6",
                    "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast",
                    "-c:a", "aac", "-shortest", path], check=True)
    music = str(d / "music.wav")
    subprocess.run([find_ffmpeg(), "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    "sine=f=660:d=2", music], check=True)
    return path, music


def test_clip_formats_split_punch_music(two_people, tmp_path):
    path, music = two_people
    s = JobSettings(output_height=640, output_dir=str(tmp_path), formats=["1:1"],
                    layout="split", punch=True, music_path=music)
    logs = []
    outs = run_job(path, s, lambda *_: None, logs.append, cache=AnalysisCache(disk=False),
                   clip=(1.0, 5.0), stem="two_clip1")
    assert [os.path.basename(o) for o in outs] == ["two_clip1_short.mp4", "two_clip1_short_1x1.mp4"]
    w, h, dur = _probe(outs[0])
    assert (w, h) == (360, 640) and 3.6 < dur < 4.4, logs
    assert _probe(outs[1])[:2] == (640, 640)
    assert any("split screen" in l for l in logs), logs


def test_split_framer_stacks_two_people(two_people):
    from face_reframe.app.compose import Framer, build_split
    from face_reframe.detect import analyze_video
    from face_reframe.pipeline import Options, plan_from_analysis
    import cv2
    path, _ = two_people
    a = analyze_video(path)
    o = Options()
    sp = build_split(a, o, 360, 640)
    assert sp is not None and sp.mask[100] and not sp.mask[10]
    fr = Framer(plan_from_analysis(a, o), 360, 640, split=sp)
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 100)
    ok, frame = cap.read()
    out = fr(100, frame)
    assert out.shape == (640, 360, 3)
    # top half shows the red shirt person, bottom the blue one
    top, bot = out[:320].reshape(-1, 3).mean(0), out[320:].reshape(-1, 3).mean(0)
    assert top[2] > bot[2] and bot[0] > top[0]


def test_word_captions_highlight_active_word():
    words = [Word(0.0, 0.4, "ciao"), Word(0.4, 0.8, "a"), Word(0.8, 1.2, "tutti"), Word(1.3, 1.6, "oggi")]
    ev = group_word_events(words, 3, 1.6)
    assert [len(e[2]) for e in ev] == [3, 1]
    seen = []
    burn = CaptionBurner(ev, 25.0, 360, 640, lambda w, a, ow: (seen.append(a), _cv_caption(w, a, ow))[1], 0.7)
    for i in (2, 12, 22):
        burn(i, np.zeros((640, 360, 3), np.uint8))
    assert seen == [0, 1, 2]


def test_timemap_and_manual_mapping():
    tm = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    assert tm(0.5) == 0.5 and tm(2.5) == 1.5 and tm(1.5) is None
    man = [(10, 500.0, None), (60, 800.0, None), (40, 300.0, None)]
    # clip starts at frame 25: the correction at 10 holds from 0; 40 -> 15;
    # 60 -> 35 = 1.4 s falls in the removed 1.0-1.5 s gap
    got = _map_manual(sorted(man), 25, 25.0, [(0.0, 1.0), (1.5, 3.0)])
    assert got == [(0, 500.0, None), (15, 300.0, None)]


def test_manual_correction_moves_camera():
    from face_reframe.detect import Analysis, Face, FrameSample
    from face_reframe.track import TrackSettings, build_camera_path
    samples = [FrameSample(f, [Face(600, 400, 80, 90, 0.9)]) for f in range(0, 200, 2)]
    an = Analysis(1920, 1080, 25.0, 0, 200, samples)
    path = build_camera_path(an, TrackSettings(dead_zone=0.0, manual=[(100, 1300.0, None), (160, None, None)]))
    assert abs(path.x[50] - 600) < 5
    assert abs(path.x[130] - 1300) < 60
    assert abs(path.x[199] - 600) < 60


def test_slice_words():
    w = [Word(1.0, 1.2, "a"), Word(5.0, 5.3, "b"), Word(9.0, 9.2, "c")]
    got = slice_words(w, 4.0, 8.0)
    assert [x.text for x in got] == ["b"] and got[0].start == pytest.approx(1.0)


def test_highlights_prefers_dense_hooky_sentence():
    from face_reframe.app.highlights import find_highlights
    words, t = [], 0.0
    # 0-40 s: slow, filler-heavy rambling
    while t < 40:
        for tok in ("ehm", "allora", "niente"):
            words.append(Word(t, t + 0.3, tok)); t += 1.2
    words[-1].text += "."
    t += 1.0
    # 41-81 s: lively, opens with a question
    first = True
    while t < 81:
        for tok in ("perché", "nessuno", "ti", "dice", "questo", "trucco?") if first else ("e", "funziona", "davvero", "bene", "sempre."):
            words.append(Word(t, t + 0.25, tok)); t += 0.3
        first = False
        t += 0.2
    hl = find_highlights(words, t, count=1)
    assert hl and hl[0].start > 39


def test_diskcache_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("SHORTS_STUDIO_CACHE", str(tmp_path / "c"))
    f = tmp_path / "x.bin"
    f.write_bytes(b"abc")
    diskcache.save(str(f), "faces", {"k": 1}, "2|0.6")
    assert diskcache.load(str(f), "faces", "2|0.6") == {"k": 1}
    assert diskcache.load(str(f), "faces", "3|0.6") is None
    f.write_bytes(b"abcd")  # file changed -> stale entry ignored
    assert diskcache.load(str(f), "faces", "2|0.6") is None


def test_job_with_face_restoration(two_people, tmp_path):
    from face_reframe import restore
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        pytest.skip("onnxruntime missing")
    if not restore.is_available("gpen256"):
        pytest.skip("GPEN-256 model not downloaded")
    path, _ = two_people
    s = JobSettings(output_height=640, output_dir=str(tmp_path), restore="gpen256")
    logs = []
    outs = run_job(path, s, lambda *_: None, logs.append, cache=AnalysisCache(disk=False), clip=(0.0, 1.0))
    assert os.path.exists(outs[0])
    assert any("Restauro volti gpen256" in l for l in logs), logs


def test_job_with_all_moves(two_people, tmp_path):
    from face_reframe.app.motion import MotionSettings
    path, _ = two_people
    m = MotionSettings(handheld=True, push=True, crash=True, shake=True, transition="random", min_gap=0.3)
    s = JobSettings(output_height=640, output_dir=str(tmp_path), motion=m, cut_pauses=True)
    logs = []
    outs = run_job(path, s, lambda *_: None, logs.append, cache=AnalysisCache(disk=False))
    assert os.path.exists(outs[0]) and any("movimenti" in l for l in logs), logs


def test_job_with_skin_fix(two_people, tmp_path):
    path, _ = two_people
    s = JobSettings(output_height=640, output_dir=str(tmp_path), skin=True)
    logs = []
    outs = run_job(path, s, lambda *_: None, logs.append, cache=AnalysisCache(disk=False), clip=(0.0, 1.0))
    assert os.path.exists(outs[0]) and any("Pelle:" in l for l in logs), logs


def test_job_with_extras(two_people, tmp_path):
    from face_reframe.app.overlays import OverlaySettings
    path, _ = two_people
    s = JobSettings(output_height=640, output_dir=str(tmp_path), cover=True, social=True,
                    overlays=OverlaySettings(hook="custom", cta="Seguimi!"), look="cine", auto_wb=True,
                    stabilize=True, encoder="software")
    s.hook_by_file[os.path.abspath(path)] = "Il trucco che nessuno ti dice"
    logs = []
    outs = run_job(path, s, lambda *_: None, logs.append, cache=AnalysisCache(disk=False))
    base = os.path.splitext(outs[0])[0]
    assert os.path.exists(base + "_copertina.jpg") and os.path.exists(base + "_testi.txt"), logs
    assert any("Colore: look cine" in l for l in logs) and any("encoder: x264" in l for l in logs)
