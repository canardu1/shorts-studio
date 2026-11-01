"""Tests for the render-path Enhancer (fast unsharp + optional AI upscale)."""

import numpy as np
import pytest

from face_reframe.enhance import EnhanceOptions, Enhancer, model_path


def _edge_energy(img):
    """High-frequency energy: how 'sharp' the frame is (Laplacian variance)."""
    import cv2
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def _crop(seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((1080, 608, 3), 120, np.uint8)
    img[300:700, 200:400] = 200                     # bright face-ish block
    img = (img.astype(np.float32)
           + rng.standard_normal(img.shape) * 8).clip(0, 255).astype(np.uint8)
    return img


def test_off_mode_is_plain_resize():
    import cv2
    enh = Enhancer(EnhanceOptions(mode="off"))
    out = enh(_crop(), 1080, 1920)
    assert out.shape == (1920, 1080, 3)
    plain = cv2.resize(_crop(), (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
    assert np.array_equal(out, plain)               # no sharpening in off mode


def test_fast_mode_sharpens():
    enh = Enhancer(EnhanceOptions(mode="fast"))
    crop = _crop()
    out = enh(crop, 1080, 1920)
    assert out.shape == (1920, 1080, 3)
    import cv2
    plain = cv2.resize(crop, (1080, 1920), interpolation=cv2.INTER_LANCZOS4)
    assert _edge_energy(out) > _edge_energy(plain) * 1.05


def test_face_boost_targets_face_region():
    enh = Enhancer(EnhanceOptions(mode="fast", sharpen=0.4, face_boost=0.5))
    crop = _crop()
    face = (1080 * 0.5, 1920 * 0.30)                # centre-x, upper third
    out = enh(crop, 1080, 1920, face=face)
    # energy in the face box vs a far corner: the boost must live near the face
    face_box = out[200:900, 350:750]
    corner = out[1500:1900, 50:250]
    plain = enh(crop, 1080, 1920, face=None)
    gain_face = _edge_energy(face_box) - _edge_energy(plain[200:900, 350:750])
    gain_corner = _edge_energy(corner) - _edge_energy(plain[1500:1900, 50:250])
    assert gain_face > gain_corner


def test_ai_mode_falls_back_without_onnxruntime(monkeypatch):
    import builtins, sys
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "onnxruntime":
            raise ImportError("no ort")
        return real_import(name, *a, **k)

    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    enh = Enhancer(EnhanceOptions(mode="ai"), log=lambda *_: None)
    assert enh.mode == "fast"                        # graceful degrade
    assert enh(_crop(), 1080, 1920).shape == (1920, 1080, 3)


def test_bake_enhanced_e2e(tmp_path):
    pytest.importorskip("cv2")
    import shutil, subprocess, cv2
    if shutil.which("ffmpeg") is None:
        pytest.skip("needs ffmpeg")
    src = str(tmp_path / "src.mp4")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "testsrc2=size=320x240:duration=2:rate=25",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    src], check=True)
    out = str(tmp_path / "src_enh.mp4")
    from face_reframe.enhance import bake_enhanced
    bake_enhanced(src, out, Enhancer(EnhanceOptions(mode="fast"),
                                   log=lambda *_: None))
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                            "stream=codec_type", "-of", "csv=p=0", out],
                           capture_output=True, text=True, check=True)
    kinds = probe.stdout.split()
    assert "video" in kinds and "audio" in kinds   # audio carried over
    cap = cv2.VideoCapture(out)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape == (240, 320, 3)     # same resolution kept


def test_ai_mode_smoke(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    if not __import__("os").path.exists(model_path()):
        pytest.skip("model not vendored")
    enh = Enhancer(EnhanceOptions(mode="ai", ai_fraction=0.25),
                   log=lambda *_: None)
    out = enh(_crop(), 540, 960)
    assert out.shape == (960, 540, 3)
    # the model output must look like an image, not noise/garbage
    assert 0 < float(out.mean()) < 255
