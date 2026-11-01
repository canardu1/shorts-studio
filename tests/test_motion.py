"""Social camera moves and cut transitions."""

import numpy as np

from face_reframe.app.motion import (MotionSettings, apply_effects, build_motion, loud_peaks,
                                     output_cuts, strong_words)
from face_reframe.transcribe import Word

FPS = 25.0


def test_inactive_is_identity():
    c = build_motion(100, FPS, MotionSettings())
    assert np.all(c.zoom == 1) and not c.dx.any() and not c.rot.any()


def test_handheld_is_small_smooth_and_deterministic():
    m = MotionSettings(handheld=True, handheld_amount=1.0)
    a = build_motion(500, FPS, m, seed=3)
    b = build_motion(500, FPS, m, seed=3)
    assert np.array_equal(a.dx, b.dx)
    assert np.abs(a.dx).max() < 0.03 and np.abs(a.rot).max() < 2.0
    assert np.abs(np.diff(a.dx)).max() < 0.004          # no jumps
    assert a.zoom.min() > 1.0                            # margin covers the drift


def test_push_ramps_within_phrases():
    c = build_motion(200, FPS, MotionSettings(push=True), phrases=[(0.0, 3.0), (4.0, 7.0)])
    assert c.zoom[10] < c.zoom[60] and c.zoom[60] > 1.02
    assert abs(c.zoom[100] - 1.0) < 1e-6 or c.zoom[100] < c.zoom[60]


def test_crash_zoom_on_strong_words_with_min_gap():
    words = [Word(1.0, 1.3, "mai"), Word(1.5, 1.8, "nessuno"), Word(5.0, 5.3, "davvero?")]
    assert strong_words(words) == [1.0, 1.5, 5.0]
    c = build_motion(200, FPS, MotionSettings(crash=True, min_gap=1.5), words=words)
    assert c.zoom[30] > 1.1 and c.zoom[130] > 1.1
    assert c.zoom[10] == 1.0 and c.zoom[100] == 1.0


def test_loud_peaks_and_shake():
    sr = 16000
    pcm = (0.01 * np.random.default_rng(0).standard_normal(sr * 6)).astype(np.float32)
    pcm[sr * 3:sr * 3 + sr // 5] *= 30                   # a shout at 3 s
    peaks = loud_peaks(pcm)
    assert peaks and abs(peaks[0] - 3.0) < 0.2
    c = build_motion(150, FPS, MotionSettings(shake=True), pcm=pcm)
    assert np.abs(c.dx[75:85]).max() > 0.005 and not c.dx[:60].any()


def test_transitions_around_cuts():
    for style, curve in (("whip", "whip"), ("zoom", "zblur"), ("glitch", "glitch")):
        c = build_motion(100, FPS, MotionSettings(transition=style), cuts=[(50, 50)])
        v = getattr(c, curve)
        assert np.abs(v[47:53]).max() > 0.5 and not np.abs(v[:40]).any(), style
    # jump-cut preview: effect before the removed range and after it, none inside
    c = build_motion(200, FPS, MotionSettings(transition="whip"), cuts=[(60, 120)], skip=[(60, 120)])
    assert np.abs(c.whip[56:60]).max() > 0.3 and np.abs(c.whip[120:124]).max() > 0.3
    assert not c.whip[70:110].any()


def test_every_alternate_and_shots_only():
    cuts = [(20, 20), (60, 60), (100, 100), (140, 140)]
    c = build_motion(200, FPS, MotionSettings(transition="glitch", transition_every="alternate", min_gap=0.1), cuts=cuts)
    hits = [f for f, _ in cuts if c.glitch[f] > 0]
    assert hits == [20, 100]
    c = build_motion(200, FPS, MotionSettings(transition="glitch", transition_every="shots", min_gap=0.1),
                     cuts=cuts, shot_cuts=[60])
    assert [f for f, _ in cuts if c.glitch[f] > 0] == [60]


def test_output_cuts_merges_jump_and_shot():
    cuts, shots = output_cuts([(0.0, 2.0), (3.0, 5.0)], [(0, 49), (49, 100)], FPS)
    assert cuts == [(49, 49)] and shots == [49]          # jump at 50 ~ shot at 49 -> one edit


def test_apply_effects_changes_frame_and_keeps_size():
    img = np.random.default_rng(1).integers(0, 255, (640, 360, 3), dtype=np.uint8)
    for style in ("whip", "zoom", "glitch"):
        c = build_motion(100, FPS, MotionSettings(transition=style), cuts=[(50, 50)])
        out = apply_effects(img, c, 50)
        assert out.shape == img.shape and not np.array_equal(out, img), style
    c = build_motion(100, FPS, MotionSettings(handheld=True, handheld_amount=1.0), seed=1)
    i = int(np.argmax(np.abs(c.rot)))
    assert apply_effects(img, c, i).shape == img.shape
