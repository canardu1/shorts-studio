"""Skin-tone normalisation: measure on cheeks/forehead, fix only skin."""

import numpy as np
import pytest

from face_reframe import skin


@pytest.fixture(scope="module")
def red_face():
    import cv2
    from skimage import data
    from face_reframe.detect import FaceDetector
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    faces = FaceDetector().detect(img)
    o = skin.measure([(img, faces)])
    # the "Sony lobster": skin pushed towards red and over-saturated
    red = skin.apply_exact(img, skin.SkinCorrection(o.a, o.b, o.L, 14, -18, 1.45, 1.0))
    return img, red, faces, o


def test_measure_finds_red_skin(red_face):
    img, red, faces, o = red_face
    m = skin.measure([(red, faces)])
    assert m.hue < o.hue - 10 and m.chroma > o.chroma * 1.2
    assert m.rot > 3 and m.scale < 1.0


def test_fix_moves_skin_back_and_spares_the_red_suit(red_face):
    img, red, faces, _ = red_face
    m = skin.measure([(red, faces)])
    fixed = skin.apply(red, m.with_strength(1.0))
    after = skin.measure([(fixed, faces)])
    assert after.hue > m.hue + 3 and after.hue >= skin.NATURAL_HUE[0] - 3
    suit = np.s_[420:500, 120:260]   # orange spacesuit
    cheek = np.s_[160:200, 220:260]
    d = lambda s: float(np.abs(fixed[s].astype(int) - red[s].astype(int)).mean())  # noqa: E731
    assert d(cheek) > 2 * d(suit)


def test_fast_matches_exact(red_face):
    _, red, faces, _ = red_face
    m = skin.measure([(red, faces)])
    assert np.abs(skin.apply(red, m).astype(int) - skin.apply_exact(red, m)).max() <= 2


def test_no_faces_no_correction():
    assert skin.measure([(np.zeros((100, 100, 3), np.uint8), [])]) is None
    frame = np.full((50, 50, 3), 120, np.uint8)
    assert skin.apply(frame, None) is frame


def test_cube_file(red_face, tmp_path):
    _, red, faces, _ = red_face
    m = skin.measure([(red, faces)])
    p = skin.write_cube(m, str(tmp_path / "s.cube"), size=17)
    lines = open(p).read().splitlines()
    assert lines[1] == "LUT_3D_SIZE 17" and len(lines) == 2 + 17 ** 3
    assert lines[2].split() == ["0.000000", "0.000000", "0.000000"]


def test_natural_skin_left_alone(red_face):
    img, _, faces, o = red_face
    assert not o.needed and "already natural" in o.describe()
    assert skin.apply(img, o) is img
