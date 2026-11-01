import cv2
import numpy as np
import pytest

from face_reframe.detect import Analysis, CutDetector, Face, FrameSample
from face_reframe.geometry import crop_size, parse_aspect, plan_crops
from face_reframe.pipeline import simplify_keys
from face_reframe.track import ShotSettings, TrackSettings, build_camera_path, classify_shot


def analysis(xs, cut_at=None, w=1920, h=1080, fps=25.0, face_h=None):
    samples = []
    for i, x in enumerate(xs):
        fh = 200 if face_h is None else face_h[i]
        faces = [] if x is None else [Face(x, 400, fh, fh, 0.9)]
        samples.append(FrameSample(i, faces, cut=(i == cut_at)))
    return Analysis(w, h, fps, 0, len(xs), samples)


def test_parse_aspect():
    assert parse_aspect("9:16") == pytest.approx(0.5625)
    assert parse_aspect("1/1") == 1.0
    assert parse_aspect("1.25") == 1.25


def test_crop_size_fits_source():
    assert crop_size(1920, 1080, 9 / 16) == pytest.approx((607.5, 1080))
    w, h = crop_size(1080, 1920, 16 / 9)
    assert w == 1080 and h == pytest.approx(607.5)
    assert crop_size(1920, 1080, 9 / 16, zoom=2) == pytest.approx((303.75, 540))


def test_crop_is_clamped_inside_frame():
    plan = plan_crops(1920, 1080, np.array([0.0, 960.0, 5000.0]), np.zeros(3), 9 / 16)
    assert plan.box(0)[0] == 0
    assert plan.box(2)[0] + plan.box(2)[2] == 1920
    assert plan.cx[1] == 960


def test_follow_vertical_puts_face_at_requested_height():
    plan = plan_crops(1920, 1080, np.array([960.0]), np.array([540.0]), 9 / 16, zoom=2,
                      follow_vertical=True, face_height_pos=0.4)
    top = plan.cy[0] - plan.crop_h / 2
    assert (540 - top) / plan.crop_h == pytest.approx(0.4)


def test_static_face_gives_static_camera():
    path = build_camera_path(analysis([700.0] * 50))
    assert np.allclose(path.x, 700.0)
    assert path.has_face.all()


def test_outlier_detection_is_ignored():
    xs = [800.0] * 60
    xs[30] = 1700.0
    path = build_camera_path(analysis(xs), TrackSettings(dead_zone=0))
    assert np.abs(path.x - 800).max() < 1


def test_missing_frames_are_interpolated_and_edges_held():
    xs = [None] * 10 + [600.0] * 20 + [None] * 10 + [600.0] * 10
    path = build_camera_path(analysis(xs))
    assert np.allclose(path.x, 600.0)


def test_cut_makes_hard_jump():
    xs = [400.0] * 50 + [1500.0] * 50
    path = build_camera_path(analysis(xs, cut_at=50))
    assert path.segments == [(0, 50), (50, 100)]
    assert path.x[49] == pytest.approx(400) and path.x[50] == pytest.approx(1500)


def test_no_cut_means_smooth_pan_with_speed_limit():
    xs = [400.0] * 50 + [1500.0] * 50
    s = TrackSettings(max_speed=0.5)
    path = build_camera_path(analysis(xs), s)
    step = np.abs(np.diff(path.x)).max()
    assert step <= 0.5 * 1920 / 25 + 1e-6
    assert path.x[-1] > 1400


def test_dead_zone_keeps_camera_still_for_small_moves():
    xs = list(960 + 20 * np.sin(np.linspace(0, 6, 100)))
    path = build_camera_path(analysis(xs), TrackSettings(dead_zone=0.1, smoothing=0), crop_size=(607.5, 1080))
    assert np.ptp(path.x) < 1


def test_simplify_keys_respects_tolerance_and_cuts():
    x = np.concatenate([np.linspace(0, 100, 50), np.full(50, 500.0)])
    y = np.zeros(100)
    keys = simplify_keys(x, y, 0.5, cuts=(50,))
    assert {0, 49, 50, 99} <= set(keys)
    assert len(keys) <= 6
    rebuilt = np.interp(np.arange(100), keys, x[keys])
    assert np.abs(rebuilt[:50] - x[:50]).max() <= 0.5


def test_classify_shot():
    s = ShotSettings()
    assert classify_shot(None, s) == ("noface", 1.0)
    assert classify_shot(0.3, s) == ("close", 1.0)
    assert classify_shot(0.06, s) == ("wide", 2.0)
    assert classify_shot(0.1, s) == ("wide", pytest.approx(1.8))
    assert classify_shot(0.06, ShotSettings(wide_mode="keep")) == ("wide", 1.0)


def test_wide_shot_gets_zoomed_and_followed_vertically():
    from face_reframe.pipeline import Options
    from face_reframe.track import build_camera_path as bcp
    a = analysis([960.0] * 100, cut_at=50, face_h=[300] * 50 + [65] * 50)
    path = bcp(a, TrackSettings(), (607.5, 1080), Options().shots)
    assert [s.kind for s in path.shots] == ["close", "wide"]
    assert np.allclose(path.zoom[:50], 1.0) and np.allclose(path.zoom[50:], 2.0)
    plan = plan_crops(1920, 1080, path.x, path.y, 9 / 16, zoom=path.zoom, follow_vertical=path.zoom > 1)
    assert plan.crop_h[0] == pytest.approx(1080) and plan.crop_h[99] == pytest.approx(540)
    # face (y=400) sits at 40% from the top of the zoomed crop
    assert (400 - (plan.cy[99] - 270)) / 540 == pytest.approx(0.4)


def _scene(close: bool):
    img = np.full((1080, 1920, 3), (60, 90, 120), np.uint8)
    if close:
        cv2.rectangle(img, (700, 150), (1200, 1080), (180, 170, 160), -1)
    else:
        cv2.rectangle(img, (900, 500), (1000, 900), (180, 170, 160), -1)
        cv2.rectangle(img, (300, 700), (1600, 760), (30, 30, 30), -1)
    return img


def test_cut_detector_catches_same_set_cut_but_not_motion():
    det = CutDetector()
    close = _scene(True)
    flags = []
    for i in range(20):
        frame = np.roll(close, i * 4, axis=1)
        flags.append(det(frame))
    flags.append(det(_scene(False)))
    assert not any(flags[:-1])
    assert flags[-1]
