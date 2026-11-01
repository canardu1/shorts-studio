"""Face beauty: spots and lines softened on skin; eyes and background kept."""

import numpy as np
import pytest

from face_reframe.beauty import Beautifier, BeautySettings


@pytest.fixture(scope="module")
def blemished():
    import cv2
    from skimage import data
    from face_reframe.detect import FaceDetector
    img = cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    face = cv2.resize(img[0:300, 100:380], None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    fr = np.full((1920, 1080, 3), 70, np.uint8)
    fr[300:1200, 120:960] = face
    f = FaceDetector().detect(fr)[0]
    lm = np.array(f.landmarks).reshape(5, 2)
    ed = float(np.linalg.norm(lm[1] - lm[0]))
    bad = fr.copy()
    spots = [(lm[0][0] - 10, lm[0][1] + 0.6 * ed), (lm[1][0] + 20, lm[1][1] + 0.7 * ed),
             (lm[2][0] - 0.5 * ed, lm[2][1] - 0.1 * ed)]
    for x, y in spots:
        cv2.circle(bad, (int(x), int(y)), 7, (60, 60, 150), -1)
    return bad, spots, lm


def _contrast(im, x, y):
    import cv2
    L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    return L[int(y) - 14:int(y) + 15, int(x) - 14:int(x) + 15].std()


def test_spots_reduced_eyes_and_background_kept(blemished):
    bad, spots, lm = blemished
    out = Beautifier(BeautySettings(0.6, 0.8, 0.5))(bad)
    for x, y in spots:
        assert _contrast(out, x, y) < 0.75 * _contrast(bad, x, y)
    eye = np.s_[int(lm[0][1]) - 12:int(lm[0][1]) + 12, int(lm[0][0]) - 20:int(lm[0][0]) + 20]
    assert np.abs(out[eye].astype(int) - bad[eye]).mean() < 4
    assert np.array_equal(out[1500:], bad[1500:])          # below the face: untouched


def test_off_and_no_face():
    frame = np.full((640, 360, 3), 90, np.uint8)
    assert Beautifier(BeautySettings(0.5, 0.5, 0.5))(frame) is frame
    assert not BeautySettings(0, 0, 0).active
