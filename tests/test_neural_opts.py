"""Selectable neural engines: fallbacks and option plumbing (no models needed)."""
import numpy as np
import pytest

from face_reframe.cinematic import CinematicParams, _denoiser, audio_chain


def test_dfn_engine_falls_back_to_afftdn_in_chain():
    """DeepFilterNet runs ahead of the filter graph: inside the ffmpeg chain
    itself the 'dfn' engine degrades to the spectral denoiser, not RNNoise."""
    p = CinematicParams(denoise_db=12, denoise_engine="dfn")
    chain = audio_chain(p, floor_db=-55.0)
    assert "afftdn" in chain
    assert "arnndn" not in chain


def test_dfn_params_do_not_break_chain():
    p = CinematicParams(denoise_db=0, denoise_engine="dfn")
    assert "afftdn" not in audio_chain(p, floor_db=-55.0)


def test_atempo_chain_splits_out_of_range_factors():
    from face_reframe.interp import _atempo_chain
    assert _atempo_chain(0.5) == [0.5]
    assert _atempo_chain(0.25) == [0.5, 0.5]
    prod = float(np.prod(_atempo_chain(0.125)))
    assert prod == pytest.approx(0.125)


def test_pick_frames_salient_fallback(tmp_path, monkeypatch):
    """Faceless analyses still get thumbnail candidates from salient boxes."""
    import cv2
    from face_reframe.app.thumbnail import pick_frames
    from face_reframe.detect import Analysis, FrameSample

    rng = np.random.default_rng(1)
    img = str(tmp_path / "v.mp4")
    w = cv2.VideoWriter(img, cv2.VideoWriter_fourcc(*"mp4v"), 25, (160, 90))
    for _ in range(10):
        w.write(rng.integers(0, 255, (90, 160, 3), dtype=np.uint8))
    w.release()

    a = Analysis(width=160, height=90, fps=25.0, frame_count=10, start_frame=0,
                 samples=[FrameSample(frame=i, faces=[], salient=(80, 45, 40, 40))
                          for i in range(10)])
    shots = pick_frames(img, a, count=3, min_gap_s=0.0, max_read=5)
    assert shots and all(s.face is not None for s in shots)


def test_objtrack_reports_availability(monkeypatch):
    from face_reframe import models, objtrack
    monkeypatch.setattr(models, "available", lambda k: False)
    assert not objtrack.available()


def test_lowlight_lifter_blends_strength(monkeypatch):
    """Strength 0 returns the input, 1 the full model output."""
    from face_reframe import lowlight, models

    class _Inp:
        name = "in"
        shape = [1, 3, 32, 32]

    class FakeSess:
        def get_inputs(self):
            return [_Inp()]

        def run(self, _out, feed):
            return [np.ones((1, 3, 32, 32), np.float32)]

    monkeypatch.setattr(models, "session", lambda key: FakeSess())
    monkeypatch.setattr(models, "available", lambda key: True)
    src = np.full((32, 32, 3), 100, np.uint8)
    l0 = lowlight.Lifter("sg_llie", 0.0)
    l1 = lowlight.Lifter("sg_llie", 1.0)
    assert np.array_equal(l0(src), src)
    assert (l1(src) == 255).all()


def test_saliency_feeds_float32(monkeypatch):
    """The normalization step must not promote the input to float64 — ONNX
    rejects tensor(double). Regression test for the tuple-subtraction bug."""
    from face_reframe import models, saliency

    fed = {}

    class _Inp:
        name = "in"

    class FakeSess:
        def get_inputs(self):
            return [_Inp()]

        def run(self, _out, feed):
            fed["x"] = feed["in"]
            return [np.zeros((1, 1, 320, 320), np.float32)]

    monkeypatch.setattr(models, "session", lambda key: FakeSess())
    s = saliency.Saliency()
    s.map(np.zeros((90, 160, 3), np.uint8))
    assert fed["x"].dtype == np.float32
