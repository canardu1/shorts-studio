"""Overlays, social texts, colour looks, stabilisation, cover, HW encode args."""

import os
import subprocess

import numpy as np
import pytest

from face_reframe.app import look, overlays, social
from face_reframe.app.jobs import JobSettings, TimeMap, source_frames, video_args
from face_reframe.transcribe import Word


def _w(text, t0=0.0, step=0.4):
    return [Word(t0 + i * step, t0 + i * step + 0.3, tok) for i, tok in enumerate(text.split())]


def test_first_sentence_and_social_texts():
    words = _w("Perché nessuno ti dice questo trucco? La cucina italiana è semplice. Basta poco olio.")
    assert overlays.first_sentence(words) == "Perché nessuno ti dice questo trucco?"
    t = social.make_texts(words, lang="it")
    assert t["title"].startswith("Perché") and "#shorts" in t["hashtags"] and "#cucina" in t["hashtags"]
    assert "Seguimi" in t["description"]
    assert "TITLE" in social.to_text(t)


def test_overlays_hook_logo_cta(tmp_path):
    import cv2
    logo = np.zeros((40, 80, 4), np.uint8)
    logo[..., 2] = 255
    logo[..., 3] = 255
    lp = str(tmp_path / "logo.png")
    cv2.imwrite(lp, logo)
    o = overlays.OverlaySettings(hook="auto", hook_dur=1.0, cta="Seguimi!", cta_dur=1.0, logo_path=lp)
    parts = overlays.build(o, "Il trucco che nessuno ti dice", 100, 25.0, 360, 640, None)
    chain = overlays.Chain(parts)
    blank = np.zeros((640, 360, 3), np.uint8)
    first, mid, last = chain(5, blank.copy()), chain(50, blank.copy()), chain(95, blank.copy())
    top = slice(0, 220)
    assert first[top].any() and not mid[top][:, :200].any()        # hook only at the start
    assert mid[..., 2].max() > 200                                 # logo always (red, 85% opacity)
    centre = slice(250, 390)
    assert last[centre].any() and not mid[centre].any()            # CTA only at the end


def test_look_tables():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
    assert look.apply(img, look.Look("none")) is img
    bw = look.apply(img, look.Look("bw"))
    assert np.abs(bw[..., 0].astype(int) - bw[..., 2]).max() <= 2  # grey
    warm = look.apply(img, look.Look("warm")).astype(int)
    assert warm[..., 2].mean() > img[..., 2].mean() and warm[..., 0].mean() < img[..., 0].mean()
    half = look.apply(img, look.Look("pop", 0.5)).astype(int)
    full = look.apply(img, look.Look("pop", 1.0)).astype(int)
    assert np.abs(half - img).mean() < np.abs(full - img).mean()
    wb = look.apply(img, look.Look("none", 1.0, (1.2, 1.0, 0.8))).astype(int)
    assert wb[..., 2].mean() > img[..., 2].mean()


def test_timemap_inverse_and_source_frames():
    tm = TimeMap([(0.0, 1.0), (2.0, 3.0)])
    assert tm.inverse(0.5) == pytest.approx(0.5) and tm.inverse(1.5) == pytest.approx(2.5)
    f = source_frames(50, 25.0, [(0.0, 1.0), (2.0, 3.0)], 10)
    assert f[0] == 10 and f[30] == 10 + 2.2 * 25


def test_stabilization_cancels_jitter(tmp_path):
    import cv2
    from face_reframe.app.stabilize import estimate
    rng = np.random.default_rng(1)
    base = (rng.random((400, 700, 3)) * 255).astype(np.uint8)
    base = cv2.GaussianBlur(base, (0, 0), 2)
    path = str(tmp_path / "shaky.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    jit = []
    for i in range(80):
        dx, dy = int(rng.integers(-8, 9)), int(rng.integers(-6, 7))
        jit.append((dx, dy))
        out.write(base[20 + dy:380 + dy, 30 + dx:670 + dx])
    out.release()
    st = estimate(path, smooth_s=1.0)
    assert st is not None and len(st.dx) == 80
    # each frame is a window shifted by +dx into a still scene: to keep looking at
    # the same scene, the crop must move by -dx (minus the smooth part)
    jx = np.array([j[0] for j in jit], float)
    jx -= np.convolve(np.pad(jx, 12, mode="edge"), np.ones(25) / 25, mode="valid")
    corr = np.corrcoef(st.dx, jx)[0, 1]
    assert corr < -0.8, corr


def test_stabilization_rotation_sign(tmp_path):
    import cv2
    from face_reframe.app.stabilize import estimate
    rng = np.random.default_rng(2)
    base = cv2.GaussianBlur((rng.random((360, 640, 3)) * 255).astype(np.uint8), (0, 0), 2)
    path = str(tmp_path / "rot.mp4")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    ang = rng.uniform(-2, 2, 60)
    for a in ang:
        out.write(cv2.warpAffine(base, cv2.getRotationMatrix2D((320, 180), float(a), 1.0), (640, 360),
                                 borderMode=cv2.BORDER_REFLECT))
    out.release()
    st = estimate(path, smooth_s=1.0)
    hp = ang - np.convolve(np.pad(ang, 12, mode="edge"), np.ones(25) / 25, mode="valid")
    # output rotation (getRotationMatrix2D convention) must undo the frame's
    assert np.corrcoef(st.rot, hp)[0, 1] < -0.8


def test_video_args_software_and_hardware(monkeypatch):
    from face_reframe.app import jobs
    s = JobSettings(encoder="software", crf=20)
    assert video_args(s, 1080, 1920)[:2] == ["-c:v", "libx264"]
    monkeypatch.setattr(jobs, "hw_encoder_available", lambda: True)
    a = video_args(JobSettings(encoder="auto", crf=18), 1080, 1920)
    assert a[1] == "h264_videotoolbox" and a[3] == "12.0M"
